import torch
import torch.nn as nn

class UnderwaterFeatureFusionModule(nn.Module):
    def __init__(self, rgb_feature_channels, depth_feature_channels, eta_init=0.1, 
                 cdom_absorption_init=0.05, chlorophyll_init=1.0, depth_scale_init=5.0):
        """
        初始化水下特征融合模块，结合三参数模型优化
        Args:
            rgb_feature_channels (int): RGB特征图的通道数
            depth_feature_channels (int): 深度特征图的通道数
            eta_init (float): 初始总衰减系数，默认0.1
            cdom_absorption_init (float): CDOM吸收初始值，默认0.05 m^-1
            chlorophyll_init (float): 叶绿素浓度初始值，默认1.0 mg/m^3
            depth_scale_init (float): 深度缩放因子初始值，默认5.0米
        """
        super(UnderwaterFeatureFusionModule, self).__init__()
        self.rgb_feature_channels = rgb_feature_channels
        self.depth_feature_channels = depth_feature_channels
        
        # 可学习参数：总衰减系数初始值
        self.eta = nn.Parameter(torch.tensor(eta_init), requires_grad=True)
        
        # 三参数模型的生物学参数，可学习
        self.cdom_absorption = nn.Parameter(torch.tensor(cdom_absorption_init), requires_grad=True)  # CDOM吸收
        self.chlorophyll = nn.Parameter(torch.tensor(chlorophyll_init), requires_grad=True)  # 叶绿素浓度
        
        # 环境光，按RGB特征通道独立定义
        self.ambient_light = nn.Parameter(torch.ones(1, rgb_feature_channels, 1, 1) * 0.8, requires_grad=True)
        
        # 深度特征加权融合的卷积层
        self.depth_weight = nn.Conv2d(depth_feature_channels, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.depth_weight.weight, 1.0)
        
        self.depth_scale = nn.Parameter(torch.tensor(depth_scale_init), requires_grad=True)  # 可学习深度范围

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
        
        # 处理深度特征：压缩到单通道
        depth = self.depth_weight(depth_feature)  # (B, 1, H, W)
        
        # 归一化深度并映射到物理范围
        depth = torch.sigmoid(depth) * torch.abs(self.depth_scale)  # 值域 [0, depth_scale]
        
        # 近似衰减系数：结合CDOM和叶绿素的影响
        # 文章中a(λ) = a_w(λ) + a_Φ(λ) + a_CDOM(λ) + a_NAP(λ)，这里简化
        # 假设a_Φ(λ) ≈ 0.0378 * C^0.627 (文章公式10)，a_CDOM(λ) ≈ a_CDOM,440 * exp(-0.014*(λ-440))
        # 由于特征图无波长信息，近似用chlorophyll和cdom_absorption加权eta
        absorption_factor = (0.0378 * torch.pow(self.chlorophyll, 0.627) + self.cdom_absorption) / 10.0  # 归一化因子10为近似
        effective_eta = self.eta * (1.0 + absorption_factor)  # 总衰减系数受生物参数增强
        
        # 计算衰减项 e^(-ηd)
        attenuation = torch.exp(-effective_eta * depth)  # (B, 1, H, W)
        
        # 后向散射项 A * (1 - e^(-ηd))
        backscatter = self.ambient_light * (1 - attenuation)  # (B, C_rgb, H, W)
        
        # 直接衰减项 J * e^(-ηd)
        direct = rgb_feature * attenuation  # (B, C_rgb, H, W)
        
        # 融合结果
        output = direct + backscatter
        
        return output

# 示例用法
if __name__ == "__main__":
    batch_size, rgb_feature_channels, height, width = 4, 64, 32, 32
    depth_feature_channels = 64
    
    rgb_feature = torch.rand(batch_size, rgb_feature_channels, height, width)
    depth_feature = torch.rand(batch_size, depth_feature_channels, height, width)
    
    model = UnderwaterFeatureFusionModule(
        rgb_feature_channels=rgb_feature_channels,
        depth_feature_channels=depth_feature_channels,
        eta_init=0.1,
        cdom_absorption_init=0.05,
        chlorophyll_init=1.0,
        depth_scale_init=5.0
    )
    
    output = model(rgb_feature, depth_feature)
    print(f"RGB Feature shape: {rgb_feature.shape}")
    print(f"Depth Feature shape: {depth_feature.shape}")
    print(f"Output shape: {output.shape}")
