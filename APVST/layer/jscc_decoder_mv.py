import math
import torch.nn as nn
import torch
from layer.layers import BasicLayerDec
from timm.models.layers import trunc_normal_
import numpy as np



class RateAdaptionDecoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionDecoder, self).__init__()
        self.C = channel_num
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.weight = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice), self.C))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, self.C))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.rate_num)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, max(self.rate_choice), self.C)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, self.C)
        x_BLC = torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b  # BLN
        out = x_BLC.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class JSCCDecoderMV(nn.Module):
    def __init__(self, embed_dim_mv, depths=[1, 1, 1], input_resolution=(16, 16),
                 num_heads=[8, 8, 8], window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, rate_choice_mv=[0, 128, 256],
                 out_channel_mv = 128):
        super(JSCCDecoderMV, self).__init__()
        self.layers_mv = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayerDec(dim=embed_dim_mv, out_dim=embed_dim_mv, input_resolution=input_resolution,
                                  depth=depths[i_layer], num_heads=num_heads[i_layer],
                                  window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, upsample=None)
            self.layers_mv.append(layer)


        self.upsample_mv = nn.ConvTranspose2d(embed_dim_mv, out_channel_mv, stride=1, kernel_size=3, padding=1)
        self.conv_mv = nn.Conv2d(out_channel_mv, out_channel_mv, stride=1, kernel_size=3, padding=1)
        
        self.embed_dim_mv = embed_dim_mv

        self.rate_adaption_mv = RateAdaptionDecoder(embed_dim_mv, rate_choice_mv)
        self.rate_choice_mv = rate_choice_mv
        self.rate_num_mv = len(rate_choice_mv)
        self.register_buffer("rate_choice_tensor_mv", torch.tensor(np.asarray(rate_choice_mv)))
        self.rate_token_mv = nn.Parameter(torch.zeros(self.rate_num_mv, embed_dim_mv))
        trunc_normal_(self.rate_token_mv, std=.02)

        self.apply(self._init_weights)


    def forward(self, mv, indexes_mv):
        B, _, H, W = mv.size()
        mv = self.rate_adaption_mv(mv, indexes_mv)

        mv_BLC = mv.flatten(2).permute(0, 2, 1)
        rate_token_mv = torch.index_select(self.rate_token_mv, 0, indexes_mv)  # BL, N
        rate_token_mv = rate_token_mv.reshape(B, H * W, self.embed_dim_mv)

        mv_BLC = mv_BLC + rate_token_mv
        for layer in self.layers_mv:
            mv_BLC = layer(mv_BLC.contiguous())
        mv_BCHW = mv_BLC.reshape(B, H, W, self.embed_dim_mv).permute(0, 3, 1, 2)
        mv_BCHW = self.upsample_mv(mv_BCHW)

        return mv_BCHW

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers_mv):
            layer.update_resolution(H, W)
