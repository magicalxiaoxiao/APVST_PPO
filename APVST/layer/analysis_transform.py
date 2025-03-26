import torch
import torch.nn as nn
from layer.video_net import *

class AnlaysisTransform(nn.Module):
    def __init__(self, out_channel_M = 96, out_channel_N = 64):
        super().__init__()

        self.out_channel_M = out_channel_M
        
        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N + 3, out_channel_N, 3, stride=2, padding=1),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 3, stride=2, padding=1),
        )


        self.apply(self._init_weights)  
        
    def forward(self, input_image, context_c):
        feature = self.contextualEncoder(torch.cat((input_image, context_c), dim=1))
        return feature

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)

