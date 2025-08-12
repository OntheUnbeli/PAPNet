import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureOptimization(nn.Module):
    def __init__(self, in_channels):
        super(FeatureOptimization, self).__init__()
        # 定义特征优化模块，这里使用残差连接和注意力机制
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_teacher, feat_student):
        # 对学生特征进行残差连接
        residual_feat = self.residual_conv(feat_student)
        residual_feat = self.relu(residual_feat)

        # 计算注意力权重
        attention_weights = self.sigmoid(self.attention_conv(feat_teacher))

        # 将注意力权重应用到残差特征上
        optimized_feat = feat_teacher + residual_feat * attention_weights

        return optimized_feat