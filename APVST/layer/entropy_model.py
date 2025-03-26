import torch
import math
import torch.nn as nn
from layer.video_net import *
from layer.transform_layer.layers import *
from layer.entropy_models.video_entropy_models import *
from channel.channel import *
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class EntropyModel(nn.Module):
    def __init__(self, out_channel_M = 96, out_channel_N = 64):
        super().__init__()
        
        self.entropy_bottleneck = EntropyBottleneck(out_channel_N)
        self.gaussian_conditional = GaussianConditional(None)

        #self.bitEstimator_z = BitEstimator(out_channel_N)

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        )


        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )


        self.auto_regressive = MaskedConv2d(
            out_channel_M,  out_channel_M * 2, kernel_size=3, padding=1, stride=1
        )


        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M , 3, stride=2, padding=1),
        ) 

        self.apply(self._init_weights)

    def feature_probs_based_sigma(self, feature, mean, sigma):
        values = feature
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mean, sigma)
        # 计算y(t,i)的概率值（y服从拉普拉斯变换，并加入噪声服从均匀分布）
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        # 通过y(t,i)的概率值计算yt的信息熵，即信息速率
        probs = torch.clamp(probs, 1e-10, 1e10)
        total_bits = torch.sum(-1.0 * torch.log(probs) / math.log(2.0))
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        prob = torch.clamp(prob, 1e-10, 1e10)
        total_bits = torch.sum(-1.0 * torch.log(prob) / math.log(2.0))
        return total_bits, prob


    def forward(self, feature, context_c):
        
        temporal_prior_params = self.temporalPriorEncoder(context_c)
        z = self.priorEncoder(feature)
        compressed_z = z
        params = self.priorDecoder(compressed_z)

        compressed_y_renorm = feature

        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(torch.cat((temporal_prior_params, params, ctx_params), dim=1))
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        _, px = self.gaussian_conditional(compressed_y_renorm, means_hat, scales_hat)
        _, px_z = self.entropy_bottleneck(compressed_z)
        total_bits_y = torch.log(px).sum() / (-math.log(2))
        total_bits_z = torch.log(px_z).sum() / (-math.log(2))
        return total_bits_y, total_bits_z, px
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)