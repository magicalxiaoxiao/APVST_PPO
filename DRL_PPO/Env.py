import torch 
import torch.nn.functional as F
import joblib
import numpy as np

class RSMA_Env:
    def __init__(self, config):
        super(RSMA_Env, self).__init__()

        self.config = config
        self.num_users = config.num_users
        self.frame_dims = config.frame_dims
        self.bs_transmist_power = config.bs_transmist_power
        self.noise_power =  config.noise_power
        self.bandwidth = config.bandwidth

        if config.immerssive_quality_mode == "PSNR":
            self.predict_model = joblib.load(config.wspsnr_predict_model_path) # This model will be released later
            self.immerssive_quality_min = config.wspsnr_min
            self.immerssive_quality_max = config.wspsnr_max
        elif config.immerssive_quality_mode == "SSIM":
            self.predict_model = joblib.load(config.wsssim_predict_model_path) # This model will be released later 
            self.immerssive_quality_min = config.wsssim_min
            self.immerssive_quality_max = config.wsssim_max

    def reset(self, channel_gain, user_request):

        batch_size, _ = channel_gain.shape
        
        power_user_ratio = torch.full((batch_size, self.num_users + 1), 1.0 / (self.num_users + 1))
        commen_rate_ratio = torch.full((batch_size, self.num_users), 1.0 / self.num_users)
        
        common_rate_min, private_rate_user = self.rate(self.bs_transmist_power, channel_gain, power_user_ratio[:, 0], power_user_ratio[:, 1:], self.bandwidth)
        common_rate_user = common_rate_min.unsqueeze(1) * commen_rate_ratio
        cbr = self.config.cbr_scale_normalization / 2 * torch.ones_like(channel_gain)
        
        state = torch.concat([channel_gain * self.config.channel_gain_scale, user_request, power_user_ratio, commen_rate_ratio, common_rate_user * self.config.rate_scale, private_rate_user * self.config.rate_scale, cbr * self.config.cbr_scale], dim=-1)

        return state

    def step(self, action, time, channel_gain, user_request):
        
        action[:, :self.num_users + 1] = F.softmax(action[:, :self.num_users + 1], dim=-1)
        action[:, self.num_users + 1:2 * self.num_users + 1] = F.softmax(action[:, self.num_users + 1:2 * self.num_users + 1], dim=-1)
        
        action[:, 2 * self.num_users + 1:] = F.sigmoid(action[:, 2 * self.num_users + 1:]) * self.config.cbr_scale_normalization

        power_user_ratio = action[:, :self.num_users + 1]
        commen_power_ratio = power_user_ratio[:, 0]
        private_power_ratio = power_user_ratio[:, 1:]
        commen_rate_ratio = action[:, self.num_users + 1:2 * self.num_users + 1]
        cbr = action[:, 2 * self.num_users + 1:]

       
        common_rate_min, private_rate_user = self.rate(self.bs_transmist_power, channel_gain, commen_power_ratio, private_power_ratio, self.bandwidth)
        common_rate_user = common_rate_min.unsqueeze(1) * commen_rate_ratio
    
        transmit_bits = self.frame_dims[0] * self.frame_dims[1] * self.frame_dims[2] * cbr * 8
        latency_user = transmit_bits / (common_rate_user + private_rate_user) * 1000

        latency_user_score = torch.where(latency_user > self.config.latency_max, torch.zeros_like(latency_user), 1 / (1 + torch.exp(-self.config.latency_socre_scale * (self.config.latency_max - latency_user))))

        snr_db_user = 10 * torch.log10((2 ** ((common_rate_user + private_rate_user) / self.bandwidth) - 1))

        X_new = torch.cat((cbr.flatten().unsqueeze(1), snr_db_user.flatten().unsqueeze(1)), dim=1).numpy()
        y_new_pred = self.predict_model.predict(X_new)
        immerssive_quality_user = torch.tensor(y_new_pred).view_as(cbr).float()

        immerssive_quality_user_score = torch.zeros_like(immerssive_quality_user)
        immerssive_quality_user_score = torch.where(immerssive_quality_user <= self.immerssive_quality_min, torch.zeros_like(immerssive_quality_user), immerssive_quality_user_score)
        immerssive_quality_user_score = torch.where(immerssive_quality_user >= self.immerssive_quality_max, torch.ones_like(immerssive_quality_user), immerssive_quality_user_score)
        immerssive_quality_user_score = torch.where((immerssive_quality_user > self.immerssive_quality_min) & (immerssive_quality_user < self.immerssive_quality_max), (immerssive_quality_user - self.immerssive_quality_min) / (self.immerssive_quality_max - self.immerssive_quality_min), immerssive_quality_user_score)

        reward_user = self.config.latency_weight * latency_user_score + self.config.immerssive_quality_weight * immerssive_quality_user_score
        
        reward = torch.mean(reward_user, dim=-1)
       
        state = torch.concat([channel_gain * self.config.channel_gain_scale, user_request, power_user_ratio, commen_rate_ratio, common_rate_user * self.config.rate_scale, private_rate_user * self.config.rate_scale, cbr * self.config.cbr_scale], dim=-1)
        
        done = True if time == self.config.max_ep_len - 1 else False
        
        return state, reward, done

    def rate(self, power, channel_gain, commen_power_ratio, private_power_ratio, bandwidth):
        
        commen_signal_power = power * channel_gain * commen_power_ratio.unsqueeze(1)
        commen_noise_power = channel_gain * torch.sum(power * private_power_ratio, dim=1).unsqueeze(1) + torch.ones_like(channel_gain) * self.noise_power
        commen_sinr = commen_signal_power / commen_noise_power
        commen_rate = bandwidth * torch.log2(commen_sinr + 1)
        commen_rate_min_value, commen_rate_min_index = torch.min(commen_rate, dim=-1)

        private_signal_power = power * channel_gain * private_power_ratio
        private_noise_power = torch.sum(power * private_power_ratio, dim=1).unsqueeze(1)  * torch.ones_like(channel_gain)
        for i in range(self.num_users):
            private_noise_power[:, i] = private_noise_power[:, i] - power * private_power_ratio[:, i]
        private_noise_power = private_noise_power * channel_gain + torch.ones_like(channel_gain) * self.noise_power
        private_sinr = private_signal_power / private_noise_power
        private_rate_user = bandwidth * torch.log2(private_sinr + 1)
        
        return commen_rate_min_value, private_rate_user
