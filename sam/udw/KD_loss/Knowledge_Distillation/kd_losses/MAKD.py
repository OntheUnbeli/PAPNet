import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    def __init__(self, channels):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        conv7x7 = self.conv7x7(x)
        return conv3x3 + conv5x5 + conv7x7

def apply_freq_filter_per_channel(feature_map, filter_type='high_pass', cutoff_freq=0.7):
    f_transform = torch.fft.fft2(feature_map)
    f_shift = torch.fft.fftshift(f_transform)
    _, channels, rows, cols = feature_map.shape
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros_like(feature_map)
    r = int(cutoff_freq * crow)
    c = int(cutoff_freq * ccol)

    if filter_type == 'high_pass':
        mask[:, :, crow-r:crow+r, ccol-c:ccol+c] = 1
    elif filter_type == 'low_pass':
        mask[:, :, crow-r:crow+r, ccol-c:ccol+c] = 0
        mask = 1 - mask

    f_filtered = f_shift * mask
    f_ishift = torch.fft.ifftshift(f_filtered)
    filtered_feature_map = torch.fft.ifft2(f_ishift)
    return torch.abs(filtered_feature_map)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x

class FeatureEnhancementModule(nn.Module):
    def __init__(self, channels,out_planes):
        super(FeatureEnhancementModule, self).__init__()
        self.multi_scale_conv = MultiScaleConv(out_planes)
        self.spatial_attention = SpatialAttention()
        self.conv = nn.Conv2d(channels, out_planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1,x2 ):
        x1 = F.interpolate(input=x1, scale_factor= 2, mode='bilinear', align_corners=True)
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        combined_features = x1 + x2
        multi_scale_features = self.multi_scale_conv(combined_features)
        multi_scale_features =multi_scale_features.to(dtype=torch.float32)
        filtered_features = apply_freq_filter_per_channel(multi_scale_features, 'low_pass', 0.7)
        attention_features = self.spatial_attention(filtered_features)
        return attention_features