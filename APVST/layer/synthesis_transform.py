import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.video_net import *
from layer.transform_layer.layers import *

class SynthesisTransform(nn.Module):
    def __init__(self, out_channel_M = 96, out_channel_N = 64):
        super().__init__()
        self.out_channel_M = out_channel_M

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, kernel_size=3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine_c = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N * 2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.apply(self._init_weights)

    def motioncompensation_c(self, ref, mv):
        ref_feature = self.feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context = self.context_refine_c(prediction_init)

        return context


    def forward(self, feature_hat, context_c):
        recon_image_feature = self.contextualDecoder_part1(feature_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context_c), dim=1))
    
        return recon_image

    def context(self, referframe, quant_mv_upsample_refine):
        context_c = self.motioncompensation_c(referframe, quant_mv_upsample_refine)
        return context_c

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)