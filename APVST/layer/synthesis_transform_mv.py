import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.video_net import *
from layer.transform_layer.layers import *

class SynthesisTransformMV(nn.Module):
    def __init__(self, out_channel_mv = 128):
        super().__init__()
        self.out_channel_mv = out_channel_mv
        
        self.mvDecoder_part1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        )

        self.mvDecoder_part2 = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )

        self.apply(self._init_weights)

    def mv_refine(self, ref, mv):
        return self.mvDecoder_part2(torch.cat((mv, ref), 1)) + mv

    
    def forward(self, mvfeature, referframe):
        quant_mv_upsample = self.mvDecoder_part1(mvfeature)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        return quant_mv_upsample_refine
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)