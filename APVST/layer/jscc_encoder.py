import math
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from layer.layers import Mlp, BasicLayerEnc
from layer.weight import *
import numpy as np


class RateAdaptionEncoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionEncoder, self).__init__()
        self.C, self.H, self.W = (channel_num, 16, 16)
        self.rate_num = len(rate_choice)
        self.rate_choice = rate_choice
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        print("CONFIG RATE", self.rate_choice_tensor)
        self.weight = nn.Parameter(torch.zeros(self.rate_num, self.C, max(self.rate_choice)))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice)))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.w, std=.02)
        mask = torch.arange(0, max(self.rate_choice)).repeat(self.H * self.W, 1)
        self.register_buffer("mask", mask)

    def forward(self, x, indexes):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        if H != self.H or W != self.W:
            self.update_resolution(H, W, x.get_device())
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, self.C, -1)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, -1)
        mask = self.mask.repeat(B, 1, 1)
        rate_constraint = self.rate_choice_tensor[indexes].reshape(B, H * W, 1).repeat(1, 1, max(self.rate_choice))
        mask_new = torch.zeros_like(mask)
        mask_new[mask < rate_constraint] = 1
        mask_new[mask >= rate_constraint] = 0
        x_BLC_masked = (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b) * mask_new
        x_masked = x_BLC_masked.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mask_BCHW = mask_new.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x_masked, mask_BCHW

    def update_resolution(self, H, W, device):
        self.H = H
        self.W = W
        self.num_patches = H * W
        self.mask = torch.arange(0, max(self.rate_choice)).repeat(self.num_patches, 1)
        self.mask = self.mask.to(device)


class JSCCEncoder(nn.Module):
    def __init__(self, embed_dim, depths=[1, 1, 1], input_resolution=(16, 16),
                 num_heads=[8, 8, 8], window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, rate_choice=[0, 128, 256],
                 out_channel_M = 96):
        super(JSCCEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList()
        self.downsample = nn.Conv2d(out_channel_M , embed_dim, stride=1, kernel_size=3, padding=1)
        self.bn = norm_layer(embed_dim)
        for i_layer in range(len(depths)):
            layer = BasicLayerEnc(dim=embed_dim, out_dim=embed_dim, input_resolution=input_resolution,
                               depth=depths[i_layer], num_heads=num_heads[i_layer],
                               window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, downsample=None)
            self.layers.append(layer)
        self.rate_adaption = RateAdaptionEncoder(embed_dim, rate_choice)
        
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)

        self.refine = Mlp(embed_dim * 2, embed_dim * 8, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def forward(self, feature, px, eta, feature_weight_tensor=None):
        x = feature
        x = self.downsample(x)
        B, C, H, W = x.size()
        #px = torch.clamp(px, 1e-10, 1e10)
        hx = torch.clamp_min(-torch.log(px) / math.log(2.0), 0)
        
        if isinstance(feature_weight_tensor, torch.Tensor):
            symbol_num = torch.sum(hx, dim=1).flatten(0) * eta * feature_weight_tensor.flatten()
        else:
            symbol_num = torch.sum(hx, dim=1).flatten(0) * eta
        x_BLC = x.flatten(2).permute(0, 2, 1)
        x_BLC = self.bn(x_BLC)
        px_BLC = px.flatten(2).permute(0, 2, 1)
        x_BLC = x_BLC + self.refine(torch.cat([1 - px_BLC, x_BLC], dim=-1))
        indexes = torch.searchsorted(self.rate_choice_tensor, symbol_num).clamp(0, self.rate_num - 1)  # B*H*W
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, C)
        x_BLC = x_BLC + rate_token
        for layer in self.layers:
            x_BLC = layer(x_BLC.contiguous())
        x_BLC = self.norm(x_BLC)
        x_BCHW = x_BLC.reshape(B, H, W, C).permute(0, 3, 1, 2)
        s_masked, mask = self.rate_adaption(x_BCHW, indexes)
        return s_masked, mask, indexes


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

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * 2, W * 2)
