import torch
import imageio
import numpy as np
import random
from utils import *
from torch.autograd import Variable
from net.APVSTNet import APVST
from config.config import config

def Var(x: torch.Tensor, deivce):
    return Variable(x.to(deivce))


def test(net, input_image, referframe):
    net.eval()
    with torch.no_grad():
        x_hat, mse_loss, wmse_loss, ssim_loss, wssim_loss = net(input_image, referframe)
        
        psnr = 10 * (torch.log(255. * 255. / mse_loss) / np.log(10))
        wpsnr = 10 * (torch.log(255. * 255. / wmse_loss) / np.log(10))
        ssim = 1 - ssim_loss
        wssim = 1 - wssim_loss
        
        print(f"Test PSNR: {psnr.item():.2f} dB")
        print(f"Test WS-PSNR: {wpsnr.item():.2f} dB")
        print(f"Test SSIM: {ssim.item():.4f}")
        print(f"Test WS-SSIM: {wssim.item():.4f}")


def main():
    device = config.device

    torch.manual_seed(1024)
    random.seed(1024)

    checkpoint = "ckpt/train_APVST_best_model.model"
    net = APVST(config).to(device)
    load_weights(net, checkpoint, device)
    
    referframe_file = "test_image/00000.png"
    input_image_file = "test_image/00001.png"
    
    input_image = (imageio.v2.imread(input_image_file).transpose(2, 0, 1)).astype(np.float32) / 255.0
    referframe = (imageio.v2.imread(referframe_file).transpose(2, 0, 1)).astype(np.float32) / 255.0
    
    input_image = torch.from_numpy(np.array(input_image)).float().unsqueeze(0)
    referframe = torch.from_numpy(np.array(referframe)).float().unsqueeze(0)
                   
    input_image = input_image.to(device)
    referframe = referframe.to(device)
    test(net, input_image, referframe)

if __name__ == "__main__":
    main()