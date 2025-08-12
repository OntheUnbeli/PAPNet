import torch
import torch.nn as nn
import torch.nn.functional as F
class MGUI(nn.Module):
    def __init__(self, in_planes,out_planes, ratio=16):
        super(MGUI, self).__init__()
        self.dim = out_planes
        self.max_pool = nn.AdaptiveMaxPool2d(1,)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        x1 = F.interpolate(input=x1, scale_factor= 1/2, mode='bilinear',align_corners=True)
        _,_,h,w = x1.shape
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        out = x1 + x2
        out_avg = self.avg_pool(out).view(-1,self.dim)
        out_max = self.max_pool(out).view(-1,self.dim)
        out_am = torch.cat([out_avg,out_max],dim=1)
        # print(out_am.shape)


        return out_am