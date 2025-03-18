import torch

class RSMA_config:
    latency_weight = 1
    immerssive_quality_weight = 2 - latency_weight

    frame_dims = (3, 960, 1920)
    bandwidth = 200e6
    bs_transmist_power_density = 5e-9
    bs_transmist_power = bandwidth * bs_transmist_power_density
    noise_power_density = 5e-18
    noise_power = bandwidth * noise_power_density
    num_users = 6
    channel_gain_scale = 1e4
    rate_scale = 1e-7
    cbr_scale = 10
    cbr_scale_normalization = 0.02
    
    latency_max = 10
    latency_socre_scale = 0.1
    wspsnr_min = 20
    wspsnr_max = 35
    wsssim_min = 5
    wsssim_max = 11

    max_norm = 80
    state_dim = num_users * 6 + 1
    action_dim = num_users * 3 + 1
    hidden_dim = 96

    actor_layers = 1
    actor_dropout = 0.0
    actor_bidirectional = False
    critic_layers = 1
    critic_dropout = 0.0
    critic_bidirectional = False

    batch_size = 128
    train_epoch = 150
    device = torch.device("cuda:0")
    K_epochs = 2
    eps_clip = 0.1
    gamma = 0.4
    lr_actor = 0.0001
    lr_critic = 0.0001
    random_seed = 1024
    action_std_init = 0.5

    max_ep_len = 200                       
    action_std = 0.5
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(5000)
    update_timestep = int(max_ep_len * 5)
    commen_weight = 10

    default_latency_weight = 1
    default_bandwidth = 200e6
    default_wspsnr_min = 20
    default_wsssim_min = 5
    default_latency_max = 10