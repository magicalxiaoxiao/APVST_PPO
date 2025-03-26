import torch.nn as nn
from layer.video_net import *

class AnlaysisTransformMV(nn.Module):
    def __init__(self, out_channel_mv = 128):
        super().__init__()
        self.out_channel_mv = out_channel_mv
        
        self.mvEncoder = nn.Sequential(
            nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        )

        self.opticFlow = ME_Spynet()
        
        self.apply(self._init_weights)

    def forward(self, input_image, referframe):
        estmv =self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        return mvfeature
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias, 0)


