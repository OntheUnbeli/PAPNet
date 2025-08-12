import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class OptimizedEdgeEnhancement(nn.Module):
    def __init__(self, gaussian_sigma=2.0, scharr_weight=1.0, laplacian_weight=0.1):
        super().__init__()
        self.scharr_weight = scharr_weight
        self.laplacian_weight = laplacian_weight

        # 高斯核 (5x5，sigma调大以增强去噪)
        self.gaussian_kernel = self._create_gaussian_kernel(sigma=gaussian_sigma, size=5)
        self.gaussian_kernel = nn.Parameter(self.gaussian_kernel.view(1, 1, 5, 5), requires_grad=False)

        # Scharr核 (仅0°和90°，简化计算)
        scharr_0 = torch.tensor([[3, 0, -3],
                                 [10, 0, -10],
                                 [3, 0, -3]], dtype=torch.float32)
        scharr_90 = torch.tensor([[3, 10, 3],
                                  [0, 0, 0],
                                  [-3, -10, -3]], dtype=torch.float32)
        laplacian = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32)

        self.scharr_kernels = nn.ParameterDict({
            '0': nn.Parameter(scharr_0.view(1, 1, 3, 3), requires_grad=False),
            '90': nn.Parameter(scharr_90.view(1, 1, 3, 3), requires_grad=False)
        })
        self.laplacian_kernel = nn.Parameter(laplacian.view(1, 1, 3, 3), requires_grad=False)

        # 适配层：调整输入特征
        self.input_adapter = nn.Conv2d(256, 64, 1)  # 假设SAM2输出256通道，降维到64
        self.attention = ChannelAttention(64)
        self.output_adapter = nn.Conv2d(64, 64, 1)  # 输出适配

    def _create_gaussian_kernel(self, sigma, size):
        x = torch.arange(-size // 2 + 1., size // 2 + 1.)
        x, y = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x):
        # x: (B, C, H, W)，假设SAM2 encoder输出
        B, C, H, W = x.size()

        # 适配输入特征
        x = self.input_adapter(x)  # 降维到64通道

        # 高斯平滑
        smoothed_x = torch.zeros_like(x)
        for c in range(x.size(1)):
            smoothed_x[:, c:c+1, :, :] = F.conv2d(
                x[:, c:c+1, :, :], self.gaussian_kernel, padding=2, groups=1)

        # Scharr滤波（简化到2方向）
        scharr_grads = []
        for angle, kernel in self.scharr_kernels.items():
            grad = torch.zeros_like(smoothed_x)
            for c in range(smoothed_x.size(1)):
                grad[:, c:c+1, :, :] = F.conv2d(
                    smoothed_x[:, c:c+1, :, :], kernel.to(x.device), padding=1, groups=1)
            scharr_grads.append(grad)
        
        scharr_magnitude = torch.stack(scharr_grads, dim=0).abs().max(dim=0)[0]
        scharr_magnitude = self.scharr_weight * scharr_magnitude

        # Laplacian滤波（权重降低）
        laplacian_grad = torch.zeros_like(smoothed_x)
        for c in range(smoothed_x.size(1)):
            laplacian_grad[:, c:c+1, :, :] = F.conv2d(
                smoothed_x[:, c:c+1, :, :], self.laplacian_kernel.to(x.device), padding=1, groups=1)
        laplacian_grad = self.laplacian_weight * laplacian_grad.abs()

        # 组合输出
        edge_map = scharr_magnitude + laplacian_grad
        edge_map = (edge_map / (edge_map.max() + 1e-8)).clamp(0, 1)

        # 注意力融合
        edge_map = self.attention(edge_map)
        edge_map = self.output_adapter(edge_map)

        return edge_map

def apply_edge_enhancement(feature_map, gaussian_sigma=2.0, scharr_weight=1.0, laplacian_weight=0.1):
    edge_module = OptimizedEdgeEnhancement(gaussian_sigma, scharr_weight, laplacian_weight)
    edge_module = edge_module.to(feature_map.device)
    with torch.no_grad():
        edge_map = edge_module(feature_map)
    return edge_map

# 示例用法
if __name__ == "__main__":
    B, C, H, W = 4, 256, 256, 256  # 假设SAM2 encoder输出
    feature_map = torch.randn(B, C, H, W).cuda()
    edge_map = apply_edge_enhancement(
        feature_map,
        gaussian_sigma=2.0,  # 增强去噪
        scharr_weight=1.0,
        laplacian_weight=0.1  # 降低Laplacian影响
    )
    print(f"Input shape: {feature_map.shape}")
    print(f"Edge map shape: {edge_map.shape}")