import torch
import torch.nn as nn

class config:
    image_dims = (3, 960, 1920)
    eta = 0.2
    device = torch.device("cuda:0")
    
    channel = {'type': 'awgn', 'chan_param': 10}
    
    v_pl = [0, 2, 4, 6, 8, 10, 16, 20, 26, 32, 40, 48, 56, 64, 80, 96]
    v_ml = [0, 2, 4, 8, 16, 32, 48]
    out_channel_mv = 128
    out_channel_M = 96
    out_channel_N = 64

    ga_kwargs = dict(out_channel_M = out_channel_M, out_channel_N = out_channel_N)
    ga_mv_kwargs = dict(out_channel_mv = out_channel_mv)

    gs_kwargs = dict(out_channel_M = out_channel_M, out_channel_N = out_channel_N)
    gs_mv_kwargs = dict(out_channel_mv = out_channel_mv)

    fe_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim=96, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=v_pl, out_channel_M = 96,
    )
    fe_mv_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim_mv=128, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice_mv=v_ml, out_channel_mv = 128
    )

    fd_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim=96, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=v_pl, out_channel_M = 96,
    )
    fd_mv_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim_mv=128, depths=[4], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice_mv=v_ml, out_channel_mv = 128
    )

    entropy_kwargs = dict(out_channel_M = out_channel_M,  out_channel_N = out_channel_N)
    entropy_mv_kwargs = dict(out_channel_mv = out_channel_mv)
