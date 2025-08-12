import torch
import torch.nn as nn
import torch.nn.functional as F
from udw.toolbox.model.module.MLPDecoder import DecoderHead
from sam2.build_sam import build_sam2
from udw.toolbox.Backbone.SegFormer.mix_transformer import mit_b0
from udw.toolbox.model.module.Test import UnderwaterFeatureFusionModule
# from udw.toolbox.model.module.UnderwaterFeatureFusionModule import UnderwaterFeatureFusionModule
# from udw.toolbox.model.module.edge_enhancement import apply_edge_enhancement
from udw.toolbox.model.module.edge_enhancement_filters import EdgeEnhanceFilter


###########encoder###########################
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
        del modelR.sam_mask_decoder
        del modelR.sam_prompt_encoder
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
            blocksR.append(
                Adapter(block)
            )
        self.encodeR.blocks = nn.Sequential(
            *blocksR
        )
        blocksD = []
        for block in self.encodeD.blocks:
            blocksD.append(
                Adapter(block)
            )
        self.encodeD.blocks = nn.Sequential(
            *blocksD
        )
        # self.backboned = mit_b0()
        factor = 1

        # [768, 384, 192, 96] tiny
        # [896, 448, 224, 112] b+
        # [1152, 576, 288, 144] L
        # [32, 64, 160, 256] b0
        self.conv1 = BasicConv2d(1152*2, 1152//factor, 1)
        self.conv2 = BasicConv2d(576*2, 576//factor, 1)
        self.conv3 = BasicConv2d(288*2, 288//factor, 1)
        self.conv4 = BasicConv2d(144*2, 144//factor, 1)

        self.UDF1 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=144,
            depth_feature_channels=144,
            eta=0.1,
            depth_scale=5.0
        )
        self.UDF2 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=288,
            depth_feature_channels=288,
            eta=0.1,
            depth_scale=5.0
        )
        self.UDF3 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=576,
            depth_feature_channels=576,
            eta=0.1,
            depth_scale=5.0
        )
        self.UDF4 = UnderwaterFeatureFusionModule(
            rgb_feature_channels=1152,
            depth_feature_channels=1152,
            eta=0.1,
            depth_scale=5.0
        )

        self.mlpdecoder = DecoderHead(in_channels=[144//factor, 288//factor, 576//factor, 1152//factor], num_classes=8)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    def forward(self, rgb, dep):
        features_rgb = self.encodeR(rgb)
        features_dep = self.encodeD(dep)
        features_rlist = features_rgb
        features_dlist = features_dep

        rf1 = features_rlist[0]
        rf2 = features_rlist[1]
        rf3 = features_rlist[2]
        rf4 = features_rlist[3]

        df1 = features_dlist[0]
        df2 = features_dlist[1]
        df3 = features_dlist[2]
        df4 = features_dlist[3]

        # fuse1 = rf1 + df1
        # fuse1 = self.conv4(fuse1)
        # fuse2 = rf2 + df2
        # fuse2 = self.conv3(fuse2)
        # fuse3 = rf3 + df3
        # fuse3 = self.conv2(fuse3)
        # fuse4 = rf4 + df4
        # fuse4 = self.conv1(fuse4)

        fuse4 = self.UDF4(rf4, df4)
        # print(fuse4.shape)
        fuse3 = self.UDF3(rf3, df3)
        fuse2 = self.UDF2(rf2, df2)
        fuse1 = self.UDF1(rf1, df1)

        # fuse1 = torch.cat([rf1, df1], dim=1)
        # fuse1 = self.conv4(fuse1)
        # # fuse1 = self.SGE(fuse1)
        # fuse2 = torch.cat([rf2, df2], dim=1)
        # fuse2 = self.conv3(fuse2)
        # # fuse2 = self.SGE(fuse2)
        # fuse3 = torch.cat([rf3, df3], dim=1)
        # fuse3 = self.conv2(fuse3)
        # # fuse3 = self.SGE(fuse3)
        # fuse4 = torch.cat([rf4, df4], dim=1)
        # fuse4 = self.conv1(fuse4)
        # # fuse4 = self.SGE(fuse4)

        # fuse1 = self.edge(fuse1)
        # fuse2 = self.edge(fuse2)
        # fuse3 = self.edge(fuse3)
        # fuse4 = self.edge(fuse4)

        list = []
        list.append(fuse1)
        list.append(fuse2)
        list.append(fuse3)
        list.append(fuse4)

        out = self.mlpdecoder(list)
        out = self.upsample4(out)
        return out, list

    def load_pre(self, pre_model):
        # save_model = torch.load(pre_model)
        # model_dict_r = self.backboner.state_dict()
        # state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        # model_dict_r.update(state_dict_r)
        # self.backboner.load_state_dict(model_dict_r)
        # print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.backboned.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.backboned.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")



if __name__ == '__main__':

    a = torch.randn(1, 3, 480, 640).cuda()
    b = torch.randn(1, 3, 480, 640).cuda()
    model = EnDecoderModel()
    # # model.load_pre('/workspace/projects/ULNet_10/Backbone_Pretrain/ckpt_B.pth')
    #
    model.cuda()
    output = model(a, b)
    print(output[0].size())
    # print(output[2].size())
    # print(output[3].size())
    # print(output.size())
    # from torchsummary import summary
    # summary(model, (3, 512, 512))
    from thop import profile
    flops, params = profile(model, inputs=(a, b))
    # print(flops, params)
    print('Flops', flops / 1e9, 'G')
    print('Params: ', params / 1e6, 'M')

