import torch
import torch.nn as nn
import torch.nn.functional as F
from udw.toolbox.model.module.MLPDecoder import DecoderHead
from sam2.build_sam import build_sam2
from udw.toolbox.model.module.Test import UnderwaterFeatureFusionModule
from udw.toolbox.model.module.edge_enhancement_filters import EdgeEnhanceFilter


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            # nn.Mish(),
            nn.GELU(),
            nn.Linear(32, dim),
            # nn.Mish()
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class EnDecoderModel(nn.Module):
    def __init__(self,
                 num_classes=8,
                 checkpoint_path='/media/wby/shuju/Seg_Water/Under2/Sam_prepth2.0/sam2_hiera_large.pt'
                 ):
        super().__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            modelR = build_sam2(model_cfg, checkpoint_path)
            modelD = build_sam2(model_cfg, checkpoint_path)
        else:
            modelR = build_sam2(model_cfg, checkpoint_path)
            modelD = build_sam2(model_cfg, checkpoint_path)

        # 保留提示编码器
        self.prompt_encoder = modelR.sam_prompt_encoder
        # 动态获取 embed_dim
        self.embed_dim = getattr(self.prompt_encoder, 'embed_dim', 256)  # 默认 256
        # print(f"Initialized embed_dim: {self.embed_dim}")  # 调试

        del modelR.sam_mask_decoder
        del modelR.memory_encoder
        del modelR.memory_attention
        del modelR.mask_downsample
        del modelR.obj_ptr_tpos_proj
        del modelR.obj_ptr_proj
        del modelR.image_encoder.neck

        del modelD.sam_mask_decoder
        del modelD.sam_prompt_encoder
        del modelD.memory_encoder
        del modelD.memory_attention
        del modelD.mask_downsample
        del modelD.obj_ptr_tpos_proj
        del modelD.obj_ptr_proj
        del modelD.image_encoder.neck

        self.encodeR = modelR.image_encoder.trunk
        self.encodeD = modelD.image_encoder.trunk

        for param in self.encodeR.parameters():
            param.requires_grad = False
        for param in self.encodeD.parameters():
            param.requires_grad = False
        blocksR = []
        for block in self.encodeR.blocks:
            blocksR.append(Adapter(block))
        self.encodeR.blocks = nn.Sequential(*blocksR)
        blocksD = []
        for block in self.encodeD.blocks:
            blocksD.append(Adapter(block))
        self.encodeD.blocks = nn.Sequential(*blocksD)

        factor = 1
        self.conv1 = BasicConv2d(1152 * 2, 1152 // factor, 1)
        self.conv2 = BasicConv2d(576 * 2, 576 // factor, 1)
        self.conv3 = BasicConv2d(288 * 2, 288 // factor, 1)
        self.conv4 = BasicConv2d(144 * 2, 144 // factor, 1)

        self.UDF1 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=144, depth_feature_channels=144, eta=0.1, depth_scale=5.0)
        self.UDF2 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=288, depth_feature_channels=288, eta=0.1, depth_scale=5.0)
        self.UDF3 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=576, depth_feature_channels=576, eta=0.1, depth_scale=5.0)
        self.UDF4 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=1152, depth_feature_channels=1152, eta=0.1, depth_scale=5.0)
        self.edge = EdgeEnhanceFilter()

        # 提示嵌入融合层
        fuse_channels = [144 // factor, 288 // factor, 576 // factor, 1152 // factor]
        self.prompt_fusion = nn.ModuleList([
            BasicConv2d(ch + self.embed_dim, ch, 1) for ch in fuse_channels
        ])
        # 调试：打印 prompt_fusion 权重形状
        # for i, fusion in enumerate(self.prompt_fusion):
        #     print(f"Prompt fusion {i} weight shape: {fusion.conv.weight.shape}")

        self.mlpdecoder = DecoderHead(in_channels=fuse_channels, num_classes=8)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, rgb, dep, points=None, point_labels=None):
        # print(points.shape)
        features_rgb = self.encodeR(rgb)
        features_dep = self.encodeD(dep)
        features_rlist = features_rgb
        features_dlist = features_dep

        rf1, rf2, rf3, rf4 = features_rlist
        df1, df2, df3, df4 = features_dlist

        fuse4 = self.UDF4(rf4, df4)
        fuse3 = self.UDF3(rf3, df3)
        fuse2 = self.UDF2(rf2, df2)
        fuse1 = self.UDF1(rf1, df1)


        # fuse1 = torch.cat([rf1, df1], dim=1)
        # fuse1 = self.conv4(fuse1)
        # fuse2 = torch.cat([rf2, df2], dim=1)
        # fuse2 = self.conv3(fuse2)
        # fuse3 = torch.cat([rf3, df3], dim=1)
        # fuse3 = self.conv2(fuse3)
        # fuse4 = torch.cat([rf4, df4], dim=1)
        # fuse4 = self.conv1(fuse4)


        fuse_list = [fuse1, fuse2, fuse3, fuse4]
        prompt_embeding = []

        # 处理提示
        if points is not None and point_labels is not None:
            for i in range(4):
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(points[i], point_labels[i]),
                    boxes=None,
                    masks=None
                )
                # print(f"Sparse embeddings shape: {sparse_embeddings.shape}")
                # print(f"Dense embeddings shape: {dense_embeddings.shape}")
                prompt_embed = sparse_embeddings.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
                prompt_embed = prompt_embed.expand(-1, -1, fuse_list[i].shape[2], fuse_list[i].shape[3])
                # print(f"Prompt embed shape before add: {prompt_embed.shape}")
                if dense_embeddings.shape[1] > 0:
                    dense_embeddings = F.interpolate(
                        dense_embeddings,
                        size=(fuse_list[i].shape[2], fuse_list[i].shape[3]),
                        mode='bilinear',
                        align_corners=True
                    )
                    # print(f"Dense embeddings shape after interp: {dense_embeddings.shape}")
                    prompt_embed = prompt_embed + dense_embeddings
                    prompt_embeding.append(prompt_embed)
                # print(f"Prompt embed shape after add: {prompt_embed.shape}")
        else:
            # 修复：使用正确的 embed_dim 初始化
            for i in range(4):
                prompt_embed = torch.zeros(rgb.shape[0], self.embed_dim, fuse_list[i].shape[2], fuse_list[i].shape[3],
                                           device=fuse1.device)
                prompt_embeding.append(prompt_embed)
            # print(f"Prompt embed shape (no points): {prompt_embed.shape}")  # 调试

        # 融合提示嵌入
        fuse_list = [fuse1, fuse2, fuse3, fuse4]
        for i in range(4):
            prompt_resized = F.interpolate(prompt_embeding[i], size=fuse_list[i].shape[2:], mode='bilinear',
                                           align_corners=True)
            # print(f"Prompt resized shape for fuse {i + 1}: {prompt_resized.shape}")
            # print(f"Fuse {i + 1} shape before concat: {fuse_list[i].shape}")
            combined = torch.cat([fuse_list[i], prompt_resized], dim=1)
            # print(f"Combined shape for fuse {i + 1}: {combined.shape}")
            fuse_list[i] = self.prompt_fusion[i](combined)

        out = self.mlpdecoder(fuse_list)
        out = self.upsample4(out)
        return out, fuse_list


if __name__ == '__main__':
    a = torch.randn(1, 3, 480, 640).cuda()
    b = torch.randn(1, 3, 480, 640).cuda()
    points = torch.tensor([[[500, 375]]], dtype=torch.float32).cuda()
    point_labels = torch.tensor([[1]], dtype=torch.float32).cuda()

    model = EnDecoderModel().cuda()
    output, features = model(a, b, points=points, point_labels=point_labels)
    print(output.shape)

    from thop import profile

    flops, params = profile(model, inputs=(a, b, points, point_labels))
    print(f"FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")