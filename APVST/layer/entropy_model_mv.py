import torch
import math
import torch.nn as nn
from layer.video_net import *
from layer.transform_layer.layers import *
from layer.entropy_models.video_entropy_models import *
from channel.channel import *
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class EntropyModelMV(nn.Module):
    def __init__(self, out_channel_mv = 128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(out_channel_mv)
        self.gaussian_conditional = GaussianConditional(None)
        
        #self.bitEstimator_z_mv = BitEstimator(out_channel_mv)

        self.mvpriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        )

        self.mvpriorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv * 2, 3, stride=1, padding=1)
        )

        self.auto_regressive_mv = MaskedConv2d(
            out_channel_mv, 2 * out_channel_mv, kernel_size=3, padding=1, stride=1
        )

        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 12 // 3, out_channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 10 // 3, out_channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 8 // 3, out_channel_mv * 6 // 3, 1),
        )

        self.apply(self._init_weights)

    def feature_probs_based_sigma(self, feature, mean, sigma):
        values = feature
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mean, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs  
    
    def iclr18_estrate_bits_z_mv(self, z_mv):
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - self.bitEstimator_z_mv(z_mv - 0.5)
        prob = torch.clamp(prob, 1e-10, 1)
        total_bits = torch.sum(-1.0 * torch.log(prob) / math.log(2.0))
        return total_bits, prob   
   
    def forward(self, mvfeature):
        z_mv = self.mvpriorEncoder(mvfeature)
        compressed_z_mv = z_mv
        params_mv = self.mvpriorDecoder(compressed_z_mv)

        quant_mv = mvfeature
        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.entropy_parameters_mv(torch.cat((params_mv, ctx_params_mv), dim=1))
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)
        _, pmv = self.gaussian_conditional(quant_mv, means_hat_mv, scales_hat_mv)
        _, pmv_z = self.entropy_bottleneck(compressed_z_mv)
        total_bits_mv = torch.log(pmv).sum() / (-math.log(2))
        total_bits_z_mv = torch.log(pmv_z).sum() / (-math.log(2))

        return total_bits_mv, total_bits_z_mv, pmv

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)