import torch
import torch.nn as nn
import torch.nn.functional as F

class LossWrapper(nn.Module):
    def __init__(self, ce_loss, kd_loss,pw_loss, rampup_start=5, rampup_end=80):
        super(LossWrapper, self).__init__()
        self.ce_loss = ce_loss
        self.kd_loss = kd_loss
        self.pw_loss = pw_loss
        self.rampup_start = rampup_start
        self.rampup_end = rampup_end

    def compute_rampup_weight(self, epoch):
        if epoch < self.rampup_start:
            return 0
        elif epoch <= self.rampup_end:
            return (epoch - self.rampup_start) / (self.rampup_end - self.rampup_start)
        else:
            return 1

    def forward(self, pred_A, pred_B,pw_A1,pw_B1, pw_A2,pw_B2,target, epoch):
        # Compute losses
        loss_A = self.ce_loss(pred_A, target)
        loss_B = self.ce_loss(pred_B, target)

        # Knowledge Distillation loss
        kd_loss_A = self.kd_loss(pred_A, pred_B.detach())
        kd_loss_B = self.kd_loss(pred_B, pred_A.detach())

        pw_loss_A_B = self.pw_loss(pw_A1.detach(), pw_B1.detach()) + self.pw_loss(pw_A2.detach(), pw_B2.detach())
        pw_loss_B_A = self.pw_loss(pw_B1.detach(), pw_A1.detach()) + self.pw_loss(pw_B2.detach(), pw_A2.detach())

        # Compute ramp-up weight for KD loss
        rampup_weight = self.compute_rampup_weight(epoch)

        # Total loss for each model
        total_loss_A = loss_A +  rampup_weight * kd_loss_A +  pw_loss_A_B
        total_loss_B = loss_B +  rampup_weight * kd_loss_B +  pw_loss_B_A


        return total_loss_A, total_loss_B
