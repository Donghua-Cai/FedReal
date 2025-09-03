import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivergenceWithTemperature(nn.Module):
    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean"):
        super().__init__()
        self.T = float(temperature)
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_logits, teacher_logits):
        # student: log_softmax / teacher: softmax
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        return (self.T * self.T) * self.kl(s, t)