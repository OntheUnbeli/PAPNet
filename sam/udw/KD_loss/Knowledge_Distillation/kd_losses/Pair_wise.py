# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
#     def __init__(self, scale):
#         super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
#         self.criterion = sim_dis_compute
#         self.scale = scale
#         self.maxpool = None  # 延迟初始化
#
#     def forward(self, preds_S, preds_T):
#         feat_S = preds_S
#         feat_T = preds_T
#         feat_T.detach()
#
#         if self.maxpool is None:  # 按需创建
#             total_w, total_h = feat_T.shape[2], feat_T.shape[3]
#             patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
#             self.maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
#
#         loss = self.criterion(self.maxpool(feat_S), self.maxpool(feat_T))
#         return loss
#
# def sim_dis_compute(f_S, f_T):
#     sim_err = ((similarity(f_T) - similarity(f_S))**2) / ((f_T.shape[-1]*f_T.shape[-2])**2) / f_T.shape[0]
#     sim_dis = sim_err.sum()
#     return sim_dis
#
# def similarity(feat):
#     feat = feat.float()
#     norm = L2(feat).detach()
#     feat_normalized = feat / norm
#     feat_flat = feat_normalized.view(feat.shape[0], feat.shape[1], -1)
#     return torch.einsum('icm,icn->imn', [feat_flat, feat_flat])
#
# def L2(f_):
#     return torch.sqrt((f_**2).sum(dim=1, keepdim=True)) + 1e-8
#
# # Example usage
# if __name__ == '__main__':
#     # Example tensors simulating student and teacher features
#     x = torch.randn(2, 3, 480, 640)  # Student features
#     y = torch.randn(2, 3, 480, 640)  # Teacher features
#     criterion = CriterionPairWiseforWholeFeatAfterPool(scale=0.75)
#     loss = criterion(x, y)
#     print("Calculated Loss:", loss)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, initial_scale=0.5):
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        # 将 scale 设置为可学习的参数
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32, requires_grad=True))
        self.maxpool = None  # 延迟初始化

    def forward(self, preds_S, preds_T):
        # Detach teacher features to avoid gradient flow
        feat_T = preds_T.detach()
        feat_S = preds_S

        # Clamp scale to prevent invalid values
        scale_clamped = torch.clamp(self.scale, min=0.1, max=1.0).item()

        # Dynamically initialize the maxpool layer
        if self.maxpool is None or self.scale.requires_grad:
            total_w, total_h = feat_T.size(2), feat_T.size(3)
            patch_w = max(1, int(total_w * scale_clamped))
            patch_h = max(1, int(total_h * scale_clamped))
            self.maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)

        # Apply maxpool and compute the loss
        pooled_feat_S = self.maxpool(feat_S)
        pooled_feat_T = self.maxpool(feat_T)
        loss = sim_dis_compute(pooled_feat_S, pooled_feat_T)
        return loss

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2).mean()
    return sim_err

def similarity(feat):
    feat_normalized = F.normalize(feat.view(feat.size(0), feat.size(1), -1), p=2, dim=1)
    return torch.einsum('icm,icn->imn', [feat_normalized, feat_normalized])

# Example usage
if __name__ == '__main__':
    # Simulate student and teacher feature maps
    x = torch.randn(2, 3, 480, 640)  # Student features
    y = torch.randn(2, 3, 480, 640)  # Teacher features

    # Initialize the loss criterion with a learnable scale
    criterion = CriterionPairWiseforWholeFeatAfterPool(initial_scale=0.5)
    optimizer = torch.optim.SGD(criterion.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        loss = criterion(x, y)
        loss.backward()
        optimizer.step()

        # Print the updated scale and loss
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}, Scale = {criterion.scale.item()}")