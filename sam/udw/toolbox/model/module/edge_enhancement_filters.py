import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhanceFilter(nn.Module):
    def __init__(self, scharr_weight=0.7, laplacian_weight=0.3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        混合Scharr和Laplacian滤波器用于增强特征图边界。

        参数:
            scharr_weight (float): Scharr滤波器的权重，控制边界梯度强度。
            laplacian_weight (float): Laplacian滤波器的权重，控制边界锐化强度。
            device (str): 计算设备。
        """
        super(EdgeEnhanceFilter, self).__init__()
        self.scharr_weight = scharr_weight
        self.laplacian_weight = laplacian_weight
        self.device = device

        # 定义Scharr滤波器核 (3x3)
        scharr_x = torch.tensor([[-3, 0, 3],
                                 [-10, 0, 10],
                                 [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3],
                                 [0, 0, 0],
                                 [3, 10, 3]], dtype=torch.float32)

        # 定义Laplacian滤波器核 (3x3)
        laplacian = torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=torch.float32)

        # 将滤波器核扩展为 (out_channels, in_channels, height, width) 形式
        # 假设输入特征图的通道数为 C，滤波器对每个通道独立处理
        self.scharr_x = nn.Parameter(scharr_x.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.scharr_y = nn.Parameter(scharr_y.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.laplacian = nn.Parameter(laplacian.unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.to(device)

    def forward(self, x):
        """
        对输入特征图进行混合滤波。

        参数:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。

        返回:
            torch.Tensor: 增强后的特征图，形状与输入相同。
        """
        B, C, H, W = x.shape

        # 扩展滤波器核到与输入通道数匹配
        scharr_x = self.scharr_x.repeat(C, 1, 1, 1)
        scharr_y = self.scharr_y.repeat(C, 1, 1, 1)
        laplacian = self.laplacian.repeat(C, 1, 1, 1)

        # 使用卷积操作进行滤波 (group=C 表示逐通道处理)
        scharr_x_out = F.conv2d(x, scharr_x, padding=1, groups=C)
        scharr_y_out = F.conv2d(x, scharr_y, padding=1, groups=C)
        scharr_out = torch.sqrt(scharr_x_out ** 2 + scharr_y_out ** 2 + 1e-8)  # 梯度幅度

        laplacian_out = F.conv2d(x, laplacian, padding=1, groups=C)

        # 混合两种滤波结果
        edge_enhanced = self.scharr_weight * scharr_out + self.laplacian_weight * laplacian_out

        # 对结果进行归一化，防止数值过大
        edge_enhanced = torch.tanh(edge_enhanced)

        # 将增强结果与原始特征图结合（可以根据需求调整）
        output = x + edge_enhanced

        return output


# 使用示例
def example_usage():
    # 假设有一个特征图，形状为 (B, C, H, W)
    batch_size, channels, height, width = 4, 3, 64, 64
    feature_map = torch.randn(batch_size, channels, height, width).cuda()

    # 实例化滤波器
    edge_filter = EdgeEnhanceFilter(scharr_weight=0.6, laplacian_weight=0.4)

    # 应用滤波
    enhanced_feature_map = edge_filter(feature_map)

    print(f"Input shape: {feature_map.shape}")
    print(f"Output shape: {enhanced_feature_map.shape}")
    print(f"Input min/max: {feature_map.min().item():.4f}/{feature_map.max().item():.4f}")
    print(f"Output min/max: {enhanced_feature_map.min().item():.4f}/{enhanced_feature_map.max().item():.4f}")


if __name__ == "__main__":
    example_usage()