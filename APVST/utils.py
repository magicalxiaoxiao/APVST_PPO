import numpy as np
import torch
import torchvision
import os
import logging


def logger_configuration(filename, phase, save_log=True):
    logger = logging.getLogger(filename)
    workdir = './history/{}'.format(filename)
    if phase == 'test':
        workdir += '_test'
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    makedirs(workdir)
    makedirs(samples)
    makedirs(models)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    return workdir, logger


def single_plot(epoch, global_step, real, gen, config):
    images = [real, gen]
    filename = "{}/NTSCCModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step)
    torchvision.utils.save_image(images, filename)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def bpp_snr_to_kdivn(bpp, SNR):
    snr = 10 ** (SNR / 10)
    kdivn = bpp / 3 / np.log2(1 + snr)
    return kdivn


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)


def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)


def CalcuPSNR_int(img1, img2, max_val=255.):
    float_type = 'float64'
    img1 = np.round(torch.clamp(img1, 0, 1).detach().cpu().numpy() * 255)
    img2 = np.round(torch.clamp(img2, 0, 1).detach().cpu().numpy() * 255)

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def load_weights(net, model_path, device):
    pretrained = torch.load(model_path, map_location=device)
    result_dict = {}
    for key, weight in pretrained.items():
        if 'attn_mask' not in key and 'rate_adaption.mask' not in key and 'rate_adaption_mv.mask' not in key:
            result_dict[key] = weight

    print(net.load_state_dict(result_dict, strict=False))
    del result_dict, pretrained

def add_gaussian_noise(signal, snr_db):
    # 计算信号功率
    if torch.is_complex(signal):
        signal_power = torch.mean(torch.abs(signal)**2)
    else:
        signal_power = torch.mean(signal**2)
    
    # 将SNR从dB转换为线性尺度
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    # 根据信号类型添加噪声
    if torch.is_complex(signal):
        noise_real = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std
        noise_imag = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std
        noise = torch.complex(noise_real, noise_imag)
    else:
        noise = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std

    # 添加噪声到信号
    noisy_signal = signal + noise
    return noisy_signal


def add_rayleigh_noise(signal, snr_db):
    # 计算信号功率
    if torch.is_complex(signal):
        signal_power = torch.mean(torch.abs(signal)**2)
    else:
        signal_power = torch.mean(signal**2)

    # 将SNR从dB转换为线性尺度
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    # 生成复数散射分量
    std_dev = 0.7  # 可以调整这个值以减小h的波动
    scatter_real = torch.normal(mean=0.0, std=std_dev, size=signal.size(), dtype=torch.float32, device=signal.device)
    scatter_imag = torch.normal(mean=0.0, std=std_dev, size=signal.size(), dtype=torch.float32, device=signal.device)
    h = torch.sqrt(scatter_real**2 + scatter_imag**2) / torch.sqrt(2)

    # 归一化信道增益以保持平均功率为1
    h_power = h.pow(2).mean()
    h = h / torch.sqrt(h_power)

    # 生成噪声
    if torch.is_complex(signal):
        noise_real = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std
        noise_imag = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std
        noise = torch.complex(noise_real, noise_imag)
    else:
        noise = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std

    # 应用信道增益和添加噪声到信号
    noisy_signal = signal * h + noise
    return noisy_signal

def add_rician_noise(signal, snr_db, s=3):
    # 计算信号功率
    if torch.is_complex(signal):
        signal_power = torch.mean(torch.abs(signal)**2)
    else:
        signal_power = torch.mean(signal**2)

    # 将SNR从dB转换为线性尺度
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power / 2)  # 除以2因为噪声分布在实部和虚部

    # 生成噪声
    if torch.is_complex(signal):
        noise_real = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std
        noise_imag = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std
        noise = torch.complex(noise_real, noise_imag)
    else:
        noise = torch.randn(signal.size(), dtype=torch.float32, device=signal.device) * noise_std

    # Rician分布的直射分量
    s_real = torch.full(signal.shape, s / np.sqrt(2), dtype=torch.float32, device=signal.device)
    s_imag = torch.full(signal.shape, s / np.sqrt(2), dtype=torch.float32, device=signal.device)

    # 散射分量
    scatter_real = torch.normal(mean=0.0, std=1, size=signal.shape, device=signal.device)
    scatter_imag = torch.normal(mean=0.0, std=1, size=signal.shape, device=signal.device)
    h_scatter = torch.sqrt(scatter_real**2 + scatter_imag**2) / np.sqrt(2)

    # 计算总的信道增益
    h = torch.sqrt((s_real + scatter_real)**2 + (s_imag + scatter_imag)**2)

    # 归一化信道增益使其功率为1
    h_power = h.pow(2).mean()  # 计算平均功率
    h = h / torch.sqrt(h_power)  # 归一化

    # 应用信道增益和添加噪声
    noisy_signal = signal * h + noise
    return noisy_signal



def bpsk_modulation_vector(data_tensor):
    """BPSK调制 - 批量处理"""
    modulated_data = (data_tensor * 2 - 1).float()
    return modulated_data

def qpsk_modulation_vector(data_vector):
    """QPSK调制 - 单一一维向量处理"""
    if data_vector.numel() % 2 != 0:
        raise ValueError("QPSK调制需要偶数长度的Tensor。")

    reshaped_data = data_vector.view(-1, 2)
    modulated_data = torch.zeros(reshaped_data.size(0), dtype=torch.complex64)
    modulated_data.real = (reshaped_data[:, 0] * 2 - 1).float()
    modulated_data.imag = (reshaped_data[:, 1] * 2 - 1).float()
    
    return modulated_data


def qam16_modulation_vector(data_vector):
    """16QAM调制 - 单一一维向量处理"""
    length = data_vector.numel()
    if length % 4 != 0:
        raise ValueError("16QAM调制需要长度能被4整除的Tensor。")

    bits_to_symbol = torch.tensor([-3, -1, 1, 3], dtype=torch.float32)
    reshaped = data_vector.view(-1, 4)
    modulated_symbols = torch.zeros(reshaped.shape[0], dtype=torch.complex64)
    
    for j in range(reshaped.shape[0]):
        real_idx = reshaped[j, 0]*2 + reshaped[j, 1]
        imag_idx = reshaped[j, 2]*2 + reshaped[j, 3]
        modulated_symbols[j] = torch.complex(bits_to_symbol[real_idx], bits_to_symbol[imag_idx])

    return modulated_symbols

def bpsk_demodulation_vector(modulated_data):
    """BPSK解调 - 批量处理"""
    demodulated_data = (modulated_data >= 0).int()
    return demodulated_data

def qpsk_demodulation_vector(modulated_data):
    """QPSK解调 - 单一一维向量处理"""
    length = modulated_data.numel()
    demodulated_data = torch.zeros(length * 2, dtype=torch.uint8)
    
    for i in range(length):
        real_part = modulated_data[i].real
        imag_part = modulated_data[i].imag
        
        # 实部大于0则为1，否则为0
        demodulated_data[i*2] = (real_part >= 0).int()
        # 虚部大于0则为1，否则为0
        demodulated_data[i*2+1] = (imag_part >= 0).int()

    return demodulated_data


def qam16_demodulation_vector(modulated_data):
    """16QAM解调 - 单一一维向量处理，考虑噪声"""
    length = modulated_data.numel()
    demodulated_data = torch.zeros(length * 4, dtype=torch.uint8)
    
    def symbol_to_bit(value):
        """根据值返回对应的比特。"""
        if value < -2:
            return (0, 0)
        elif value < 0:
            return (0, 1)
        elif value < 2:
            return (1, 0)
        else:
            return (1, 1)
    
    for i in range(length):
        symbol = modulated_data[i]
        real_part = symbol.real
        imag_part = symbol.imag
        real_bits = symbol_to_bit(real_part.item())
        imag_bits = symbol_to_bit(imag_part.item())
        demodulated_data[i*4:i*4+2] = torch.tensor(real_bits, dtype=torch.uint8)
        demodulated_data[i*4+2:i*4+4] = torch.tensor(imag_bits, dtype=torch.uint8)

    return demodulated_data


def bpsk_modulation_batch(data_tensor):
    """BPSK调制 - 批量处理"""
    modulated_data = (data_tensor * 2 - 1).float()
    return modulated_data

def qpsk_modulation_batch(data_tensor):
    """QPSK调制 - 批量处理"""
    n, _ = data_tensor.shape
    if data_tensor.numel() % 2 != 0:
        raise ValueError("QPSK调制需要偶数长度的Tensor。")

    reshaped_data = data_tensor.view(n, -1, 2)

    modulated_data = torch.zeros((n, reshaped_data.size(1)), dtype=torch.complex64)
    modulated_data.real = (reshaped_data[:, :, 0] * 2 - 1).float()
    modulated_data.imag = (reshaped_data[:, :, 1] * 2 - 1).float()   
    
    return modulated_data


def qam16_modulation_batch(data_tensor):
    """16QAM调制 - 批量处理"""
    n, length = data_tensor.shape
    if length % 4 != 0:
        raise ValueError("16QAM调制需要长度能被4整除的Tensor。")

    bits_to_symbol = torch.tensor([-3, -1, 1, 3], dtype=torch.float32)

    reshaped = data_tensor.view(n, -1, 4)
    modulated_symbols = torch.zeros((n, reshaped.shape[1]), dtype=torch.complex64)
    
    for i in range(n):
        for j in range(reshaped.shape[1]):
            real_idx = reshaped[i, j, 0]*2 + reshaped[i, j, 1]
            imag_idx = reshaped[i, j, 2]*2 + reshaped[i, j, 3]
            modulated_symbols[i, j] = torch.complex(bits_to_symbol[real_idx], bits_to_symbol[imag_idx])

    return modulated_symbols


def bpsk_demodulation_batch(modulated_data):
    """BPSK解调 - 批量处理"""
    demodulated_data = (modulated_data >= 0).int()
    return demodulated_data


def qpsk_demodulation_batch(modulated_data):
    """QPSK解调 - 批量处理"""
    n, length = modulated_data.shape
    demodulated_data = torch.zeros((n, length * 2), dtype=torch.uint8)
    
    for i in range(length):
        real_part = modulated_data[:, i].real
        imag_part = modulated_data[:, i].imag
        
        # 实部大于0则为1，否则为0
        demodulated_data[:, i*2] = (real_part >= 0).int()
        # 虚部大于0则为1，否则为0
        demodulated_data[:, i*2+1] = (imag_part >= 0).int()

    return demodulated_data

def qam16_demodulation_batch(modulated_data):
    """16QAM解调 - 批量处理，考虑噪声"""
    n, length = modulated_data.shape
    demodulated_data = torch.zeros((n, length * 4), dtype=torch.uint8)
    
    def symbol_to_bit(value):
        """根据值返回对应的比特。"""
        if value < -2:
            return (0, 0)
        elif value < 0:
            return (0, 1)
        elif value < 2:
            return (1, 0)
        else:
            return (1, 1)
    
    for i in range(n):
        for j in range(length):
            symbol = modulated_data[i, j]
            real_part = symbol.real
            imag_part = symbol.imag
            real_bits = symbol_to_bit(real_part.item())
            imag_bits = symbol_to_bit(imag_part.item())
            demodulated_data[i, j*4:j*4+2] = torch.tensor(real_bits, dtype=torch.uint8)
            demodulated_data[i, j*4+2:j*4+4] = torch.tensor(imag_bits, dtype=torch.uint8)

    return demodulated_data
