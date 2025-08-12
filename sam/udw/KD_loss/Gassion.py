import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSimilarityLossWithGaussian(nn.Module):
    def __init__(self, initial_alpha=0.5, gamma=0.1):
        """
        初始化动态可学习的 alpha 参数，使用高斯核作为相似度度量。
        Args:
            initial_alpha (float): 初始 alpha 值（介于 0 和 1 之间）。
            gamma (float): 高斯核的超参数，控制范围。
        """
        super(FeatureSimilarityLossWithGaussian, self).__init__()
        # 定义 alpha 为一个可学习的参数，初始化为给定值
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
        self.gamma = gamma  # 高斯核超参数

    def forward(self, feature_a, feature_b):
        """
        计算特征图之间的相似性损失。
        Args:
            feature_a (torch.Tensor): 模型 A 的特征图，形状 (B, C, H, W)。
            feature_b (torch.Tensor): 模型 B 的特征图，形状 (B, C, H, W)。
        Returns:
            loss (torch.Tensor): 综合损失值。
        """
        # 确保输入特征图维度一致
        assert feature_a.shape == feature_b.shape, "Feature maps must have the same shape."

        # 1. 均方误差（MSE）损失
        mse_loss = F.mse_loss(feature_a, feature_b)

        # 2. 高斯核相似性损失
        diff = feature_a - feature_b
        squared_dist = torch.sum(diff ** 2, dim=(1, 2, 3))  # 欧几里得距离的平方
        gaussian_similarity = torch.exp(-self.gamma * squared_dist)  # 高斯核相似度
        gaussian_loss = 1 - gaussian_similarity.mean()  # 转化为损失（1 表示完全相似）

        # 3. 结合动态 alpha 的综合损失
        alpha = torch.sigmoid(self.alpha)  # 将 alpha 映射到 [0, 1] 区间
        loss = alpha * mse_loss + (1 - alpha) * gaussian_loss

        return loss
