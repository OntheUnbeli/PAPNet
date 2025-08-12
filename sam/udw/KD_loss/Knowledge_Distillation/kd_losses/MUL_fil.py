import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFilter(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7], nonlinearity='relu'):
        super(MultiScaleFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.nonlinearity = nonlinearity

        # 初始化各个尺度的滤波器
        self.filters = nn.ModuleList()
        for scale in scales:
            # 使用不同的滤波器核大小和类型，这里示例使用3x3和5x5的高斯滤波器
            self.filters.append(nn.Conv2d(in_channels, out_channels, kernel_size=scale, padding=scale//2))

        # 非线性激活函数
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        filtered_outputs = []
        for i, scale in enumerate(self.scales):
            # 使用不同尺度的滤波器对输入特征进行滤波
            filtered_output = self.filters[i](x)

            # 应用非线性激活函数
            filtered_output = self.activation(filtered_output)

            # 将滤波后的结果保存到列表中
            filtered_outputs.append(filtered_output)

        # 将不同尺度的滤波结果进行合并
        combined_output = sum(filtered_outputs)

        return combined_output