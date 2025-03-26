import torch
from torch import nn
import math

def calculate_weight(H, W, batch_size, yita=0.9):
    weight = torch.zeros(H)
    for i in range(H):
        weight[i] = yita * math.cos((0.5 - (i + 0.5) / H) * math.pi) + 1 - yita
    weight = weight.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, W)
    return weight


class Weight(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(channel + 2, channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weight, feature):
        maxpool_weight = self.maxpool(self.maxpool(self.maxpool(self.maxpool(weight))))
        avgpool_weight = self.avgpool(self.avgpool(self.avgpool(self.avgpool(weight))))
        weight_tensor = torch.cat((maxpool_weight, avgpool_weight, feature), dim=1)
        weight_tensor = self.conv(weight_tensor)
        weight_tensor = self.sigmoid(weight_tensor)
        feature = feature * weight_tensor
        return feature
    
    def feature_weight(self, weight):
        return self.avgpool(self.avgpool(self.avgpool(self.avgpool(weight))))