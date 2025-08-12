import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSimilarityLoss(nn.Module):
    def __init__(self, initial_alpha=0.5):
        """
        初始化动态可学习的 alpha 参数。
        Args:
            initial_alpha (float): 初始 alpha 值（介于 0 和 1 之间）。
        """
        super(FeatureSimilarityLoss, self).__init__()
        # 定义 alpha 为一个可学习的参数，初始化为给定值
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))

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

        # 2. 余弦相似度损失
        feature_a_flat = feature_a.view(feature_a.size(0), -1)  # 展平为 (B, C*H*W)
        feature_b_flat = feature_b.view(feature_b.size(0), -1)
        cosine_similarity = F.cosine_similarity(feature_a_flat, feature_b_flat, dim=1)
        cosine_loss = 1 - cosine_similarity.mean()

        # 3. 结合动态 alpha 的综合损失
        alpha = torch.sigmoid(self.alpha)  # 将 alpha 映射到 [0, 1] 区间
        loss = alpha * mse_loss + (1 - alpha) * cosine_loss

        return loss