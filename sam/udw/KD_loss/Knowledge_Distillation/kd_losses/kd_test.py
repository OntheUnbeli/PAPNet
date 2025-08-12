import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math

class SW(nn.Module):
    def __init__(self, dim, reduction=1):
        self.dim = dim
        super(SW, self).__init__()
        self.localattention = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(self.dim, self.dim , 1),
            nn.BatchNorm2d(self.dim ),
            nn.ReLU()
        )

        self.globalattention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim, self.dim , 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU()
        )
        self.sig = nn.Sigmoid()

        # '1:将x + localAtt + globalAtt变成localAtt + globalAtt；对深度进行filter-x3'
        # '1:将'self.globalattention(x2)变成self.globalattention(x)；localAtt = self.localattention(x1)变成localAtt = self.localattention(x)
    def forward(self, x1,):
        # x = torch.cat((x1, x2), dim=1) # B 2C H W
        localAtt = self.localattention(x1)
        globalAtt = self.globalattention(x1)

        Fusion = localAtt + globalAtt + x1
        # Fusion = x + localAtt + globalAtt
        Fusion = self.sig(Fusion)
        spatial_weights = x1 * Fusion

        return spatial_weights