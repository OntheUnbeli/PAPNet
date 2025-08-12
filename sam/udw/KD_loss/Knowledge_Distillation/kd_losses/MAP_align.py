import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureRemappingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureRemappingModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class FeatureAlignmentModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureAlignmentModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_student, feat_teacher):
        # 对学生特征进行卷积变换
        feat_student_mapped = self.conv(feat_student)

        # 对学生特征和教师特征进行相似性计算
        # similarity_map = torch.matmul(feat_student_mapped, feat_teacher)

        # 使用相似性映射对学生特征进行加权
        alignment_feat_student = torch.mul(feat_student_mapped, feat_student_mapped)

        # 使用 Sigmoid 函数进行归一化
        alignment_feat_student_normalized = self.sigmoid(alignment_feat_student)

        # 返回对齐后的学生特征
        return alignment_feat_student_normalized