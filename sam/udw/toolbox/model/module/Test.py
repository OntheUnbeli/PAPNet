import torch
import torch.nn as nn

class UnderwaterFeatureFusionModule(nn.Module):
    def __init__(self, rgb_feature_channels, depth_feature_channels, eta=0.1, depth_scale=5.0):
        """
        初始化水下特征融合模块，融合骨干网络处理后的RGB和深度特征
        Args:
            rgb_feature_channels (int): RGB特征图的通道数
            depth_feature_channels (int): 深度特征图的通道数
            eta (float): 衰减系数，默认0.1，可根据水质调整
            depth_scale (float): 深度缩放因子，默认5.0米
        """
        super(UnderwaterFeatureFusionModule, self).__init__()
        self.rgb_feature_channels = rgb_feature_channels
        self.depth_feature_channels = depth_feature_channels
        self.eta = nn.Parameter(torch.tensor(eta), requires_grad=True)  # 可学习的衰减系数
        # self.depth_scale = depth_scale
        self.depth_scale = nn.Parameter(torch.tensor(depth_scale), requires_grad=True)
        
        # 环境光A，按RGB特征通道独立定义
        self.ambient_light = nn.Parameter(torch.ones(1, rgb_feature_channels, 1, 1) * 0.8, requires_grad=True)
        
        # 深度特征加权融合的卷积层，将深度特征压缩到单通道
        self.depth_weight = nn.Conv2d(depth_feature_channels, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.depth_weight.weight, 1.0)

    def forward(self, rgb_feature, depth_feature):
        """
        前向传播，融合RGB特征图和深度特征图
        Args:
            rgb_feature (torch.Tensor): RGB特征图，形状 (B, C_rgb, H, W)
            depth_feature (torch.Tensor): 深度特征图，形状 (B, C_depth, H, W)
        Returns:
            torch.Tensor: 融合后的特征张量，形状 (B, C_rgb, H, W)
        """
        B, C_rgb, H, W = rgb_feature.shape
        _, C_depth, _, _ = depth_feature.shape
        
        # 处理深度特征：压缩到单通道
        depth = self.depth_weight(depth_feature)  # (B, 1, H, W)
        depth = torch.sigmoid(depth) * torch.abs(self.depth_scale)

        # 归一化深度并映射到物理范围

        depth = torch.sigmoid(depth) * self.depth_scale  # 值域 [0, depth_scale]

        # 计算衰减项 e^(-ηd)
        attenuation = torch.exp(-self.eta * depth)  # (B, 1, H, W)
        
        # 后向散射项 A * (1 - e^(-ηd))
        backscatter = self.ambient_light * (1 - attenuation)  # (B, C_rgb, H, W)
        
        # 直接衰减项 J * e^(-ηd)，假设RGB特征图近似为J
        direct = rgb_feature * attenuation  # (B, C_rgb, H, W)
        
        # 融合结果
        output = direct + backscatter
        
        # 不强制限制值域，因为特征图可能超出[0, 1]
        return output

# 示例用法
if __name__ == "__main__":
    # 假设骨干网络输出的RGB特征和深度特征
    batch_size, rgb_feature_channels, height, width = 4, 64, 32, 32  # RGB特征通道数64
    depth_feature_channels = 32  # 深度特征通道数32
    
    rgb_feature = torch.rand(batch_size, rgb_feature_channels, height, width)  # 随机RGB特征
    depth_feature = torch.rand(batch_size, depth_feature_channels, height, width)  # 随机深度特征
    
    # 初始化模块
    model = UnderwaterFeatureFusionModule(
        rgb_feature_channels=rgb_feature_channels, 
        depth_feature_channels=depth_feature_channels, 
        eta=0.1, 
        depth_scale=5.0
    )
    
    # 前向传播
    output = model(rgb_feature, depth_feature)
    print(f"RGB Feature shape: {rgb_feature.shape}")
    print(f"Depth Feature shape: {depth_feature.shape}")
    print(f"Output shape: {output.shape}")
