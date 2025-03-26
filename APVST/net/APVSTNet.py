import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.distortion import Distortion
from layer.jscc_encoder import JSCCEncoder
from layer.jscc_encoder_mv import JSCCEncoderMV
from layer.jscc_decoder import JSCCDecoder
from layer.jscc_decoder_mv import JSCCDecoderMV
from layer.analysis_transform import AnlaysisTransform
from layer.analysis_transform_mv import AnlaysisTransformMV
from layer.synthesis_transform import SynthesisTransform
from layer.synthesis_transform_mv import SynthesisTransformMV
from layer.entropy_model import EntropyModel
from layer.entropy_model_mv import EntropyModelMV
from layer.weight import Weight, calculate_weight
from layer.latitude_adaptive import LatitudeAdaptiveModule, calculate_weight_latitude
from channel.channel import *


class APVST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ga_mv = AnlaysisTransformMV(**config.ga_mv_kwargs)
        self.gs_mv = SynthesisTransformMV(**config.gs_mv_kwargs)
        self.entropy_mv = EntropyModelMV(**config.entropy_mv_kwargs)
        self.fe_mv = JSCCEncoderMV(**config.fe_mv_kwargs)
        self.fd_mv = JSCCDecoderMV(**config.fd_mv_kwargs)
        self.weight_mv = Weight(channel=config.out_channel_mv, kernel_size=3)
        
        self.ga = AnlaysisTransform(**config.ga_kwargs)
        self.gs = SynthesisTransform(**config.gs_kwargs)
        self.entropy = EntropyModel(**config.entropy_kwargs)
        self.fe = JSCCEncoder(**config.fe_kwargs)
        self.fd = JSCCDecoder(**config.fd_kwargs)
        self.latitude_adaptive = LatitudeAdaptiveModule(config.out_channel_M)
        self.weight = Weight(channel=config.out_channel_M, kernel_size=3)
        
        self.channel = Channel(config)
        self.distortion_mse = Distortion(distortion_metric='MSE')
        self.distortion_ssim = Distortion(distortion_metric='SSIM')
        self.H = self.W = 0

    def update_resolution(self, H, W):
        if H != self.H or W != self.W:
            self.fe_mv.update_resolution(H // 16, W // 16)
            self.fd_mv.update_resolution(H // 16, W // 16)
            self.fe.update_resolution(H // 16, W // 16)
            self.fd.update_resolution(H // 16, W // 16)
            self.H = H
            self.W = W
    
    def forward(self, input_image, referframe):
        B, C, H, W = input_image.shape
        self.update_resolution(H, W)

        latitude_weight = calculate_weight(H, W, B)
        latitude_weight = latitude_weight.unsqueeze(1).to(input_image.device)
        latitude_weight_tensor = latitude_weight.repeat(1, C, 1, 1)

        mvfeature = self.ga_mv(input_image, referframe)
        _, _, pmv = self.entropy_mv(mvfeature)
        
        mv_feature = self.weight_mv(latitude_weight, mvfeature)
        
        s_masked_mv, mask_mv, indexes_mv = self.fe_mv(mv_feature, pmv, self.config.eta)

        mask_mv = mask_mv.bool()
        channel_input_mv = torch.masked_select(s_masked_mv, mask_mv)
        channel_output_mv = self.channel.forward(channel_input_mv)
        s_hat_mv = torch.zeros_like(s_masked_mv)
        s_hat_mv[mask_mv] = channel_output_mv

        y_hat_mv = self.fd_mv(s_masked_mv, indexes_mv)
        mv_hat = self.gs_mv(y_hat_mv, referframe)
        context_c = self.gs.context(referframe, mv_hat)

        feature = self.ga(input_image, context_c)
        _, _, px = self.entropy(feature, context_c)
        
        feature_scale = self.weight(latitude_weight, feature)
        
        yita = self.latitude_adaptive(px)
        feature_length_tensor = calculate_weight_latitude(H, W, yita=yita).to(input_image.device)
        feature_length_tensor = feature_length_tensor.clip(0, 1)

        s_masked, mask_BCHW, indexes = self.fe(feature_scale, px, self.config.eta, feature_length_tensor)
        
        mask_BCHW = mask_BCHW.bool()
        channel_input = torch.masked_select(s_masked, mask_BCHW)
        channel_output = self.channel.forward(channel_input)
        s_hat = torch.zeros_like(s_masked)
        s_hat[mask_BCHW] = channel_output

        y_hat = self.fd(s_hat, indexes)
        x_hat = self.gs(y_hat, context_c).clip(0, 1)
        
        mse_loss = self.distortion_mse(input_image, x_hat)
        ssim_loss = 1 - self.distortion_ssim(input_image, x_hat)
        wmse_loss = self.distortion_mse(input_image, x_hat, latitude_weight_tensor)
        wssim_loss = 1 - self.distortion_ssim(input_image, x_hat, latitude_weight_tensor)
    
        return x_hat, mse_loss, wmse_loss, ssim_loss, wssim_loss