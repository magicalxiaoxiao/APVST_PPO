import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatitudeAdaptiveModule(nn.Module):
    def __init__(self, c_size):
        super(LatitudeAdaptiveModule, self).__init__()
        
        self.fc1 = nn.Linear(c_size, c_size)
        self.fc2 = nn.Linear(c_size, c_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, px):
        B, C, H, W_ = px.shape
        hx = torch.clamp_min(-torch.log(px) / math.log(2.0), 0)

        hx_mean = hx.mean(dim=3)
        hx_mean_2d = hx_mean.permute(0, 2, 1).reshape(-1, C)

        x = self.fc1(hx_mean_2d)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        yita = x.reshape(B, H, C).permute(0, 2, 1)

        return yita
    
def calculate_weight_latitude(H, W, yita):
    device = yita.device
    dtype = yita.dtype
    
    indices = torch.arange(H, device=device, dtype=dtype)
    cos_factors = torch.cos((0.5 - (indices + 0.5) / H) * math.pi)
    cos_factors = cos_factors.view(1, 1, H).unsqueeze(-1).repeat(1, 1, 1, W)
    cos_factors = F.adaptive_avg_pool2d(cos_factors, output_size=(H // 16, W // 16)).sum(dim=-1)

    weight = yita * cos_factors + 1 - yita
    weight = weight.unsqueeze(-1).repeat(1, 1, 1, W // 16).mean(dim=1)
    
    return weight