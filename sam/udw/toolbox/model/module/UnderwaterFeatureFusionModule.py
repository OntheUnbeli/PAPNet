import torch
import torch.nn as nn

class UnderwaterFeatureFusionModule(nn.Module):
    def __init__(self, rgb_feature_channels, depth_feature_channels, eta_direct_init=0.1, eta_backscatter_init=0.08,
                 depth_scale_init=5.0, preferred_distance=5.0):
        """
        初始化水下特征融合模块，结合Akkaynak & Treibitz (2018)的优化，自适应处理水体类型
        Args:
            rgb_feature_channels (int): RGB特征图的通道数
            depth_feature_channels (int): 深度特征图的通道数
            eta_direct_init (float): 直接信号衰减系数初始值，默认0.1
            eta_backscatter_init (float): 背景散射衰减系数初始值，默认0.08
            depth_scale_init (float): 深度缩放因子初始值，默认5.0米
            preferred_distance (float): 优选距离，用于衰减系数估计，默认5.0米
        """
        super(UnderwaterFeatureFusionModule, self).__init__()
        self.rgb_feature_channels = rgb_feature_channels
        self.depth_feature_channels = depth_feature_channels
        self.preferred_distance = preferred_distance

        # 区分直接信号和背景散射的衰减系数 (Akkaynak & Treibitz, 2018)
        self.eta_direct = nn.Parameter(torch.tensor(eta_direct_init), requires_grad=True)  # 对应 beta_c^D
        self.eta_backscatter = nn.Parameter(torch.tensor(eta_backscatter_init), requires_grad=True)  # 对应 beta_c^B

        # 自适应水体类型：可学习权重因子，范围 [0, 1]
        self.scatter_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 初始值为中性
        self.scatter_threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 背景散射强度阈值

        # 环境光，按RGB特征通道独立定义，随深度衰减
        self.ambient_light_base = nn.Parameter(torch.ones(1, rgb_feature_channels, 1, 1) * 0.8, requires_grad=True)
        self.ambient_decay_rate = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # 环境光随深度衰减率

        # 深度特征加权融合的卷积层
        self.depth_weight = nn.Conv2d(depth_feature_channels, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.depth_weight.weight, 1.0)

        self.depth_scale = nn.Parameter(torch.tensor(depth_scale_init), requires_grad=True)

    def forward(self, rgb_feature, depth_feature):
        """
        前向传播，融合RGB特征图和深度特征图，自适应处理水体类型
        Args:
            rgb_feature (torch.Tensor): RGB特征图，形状 (B, C_rgb, H, W)
            depth_feature (torch.Tensor): 深度特征图，形状 (B, C_depth, H, W)
        Returns:
            torch.Tensor: 融合后的特征张量，形状 (B, C_rgb, H, W)
        """
        B, C_rgb, H, W = rgb_feature.shape

        # 处理深度特征：压缩到单通道
        depth = self.depth_weight(depth_feature)  # (B, 1, H, W)
        depth = torch.sigmoid(depth) * torch.abs(self.depth_scale)  # 值域 [0, depth_scale]

        # 计算初始衰减项
        attenuation_direct = torch.exp(-self.eta_direct * depth)  # 直接信号衰减 e^(-beta_c^D * z)
        attenuation_backscatter = torch.exp(-self.eta_backscatter * depth)  # 背景散射衰减 e^(-beta_c^B * z)

        # 环境光随深度衰减 (Akkaynak & Treibitz, 2018)
        ambient_light = self.ambient_light_base * torch.exp(-self.ambient_decay_rate * depth)

        # 后向散射项 B_c^∞ * (1 - e^(-beta_c^B * z))
        backscatter = ambient_light * (1 - attenuation_backscatter)

        # 自适应调整水体类型：基于背景散射强度
        backscatter_intensity = backscatter.mean(dim=[1, 2, 3], keepdim=True)  # 计算平均背景散射强度
        scatter_weight = torch.sigmoid((backscatter_intensity - self.scatter_threshold) * 10.0)  # 映射到 [0, 1]
        scatter_weight = scatter_weight.expand_as(depth)  # 扩展到 (B, 1, H, W)

        # 动态调整衰减系数：吸收主导时增强 eta_direct，散射主导时增强 eta_backscatter
        effective_eta_direct = self.eta_direct * (1.0 + (1.0 - scatter_weight) * 0.2)  # 吸收主导时增加 20%
        effective_eta_backscatter = self.eta_backscatter * (1.0 + scatter_weight * 0.5)  # 散射主导时增加 50%

        # 优选距离调整：避免近距离过补偿 (Akkaynak & Treibitz, 2018)
        distance_factor = torch.clamp(depth / self.preferred_distance, min=0.5, max=2.0)
        effective_eta_direct = effective_eta_direct * distance_factor

        # 重新计算衰减项
        attenuation_direct = torch.exp(-effective_eta_direct * depth)
        attenuation_backscatter = torch.exp(-effective_eta_backscatter * depth)

        # 重新计算背景散射
        backscatter = ambient_light * (1 - attenuation_backscatter)

        # 直接衰减项 J * e^(-beta_c^D * z)
        direct = rgb_feature * attenuation_direct

        # 融合结果
        output = direct + backscatter

        return output

# 示例用法
if __name__ == "__main__":
    batch_size, rgb_feature_channels, height, width = 4, 64, 32, 32
    depth_feature_channels = 32

    rgb_feature = torch.rand(batch_size, rgb_feature_channels, height, width)
    depth_feature = torch.rand(batch_size, depth_feature_channels, height, width)

    model = UnderwaterFeatureFusionModule(
        rgb_feature_channels=rgb_feature_channels,
        depth_feature_channels=depth_feature_channels,
        eta_direct_init=0.1,
        eta_backscatter_init=0.08,
        depth_scale_init=5.0,
        preferred_distance=5.0
    )

    output = model(rgb_feature, depth_feature)
    print(f"RGB Feature shape: {rgb_feature.shape}")
    print(f"Depth Feature shape: {depth_feature.shape}")
    print(f"Output shape: {output.shape}")