import torch
import torch.nn as nn
import torch.nn.functional as F
from params import T

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, noise, t, device="cpu"):
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

# Define beta schedule

betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)




# train_loader = load_data(dataset_path, "train", batch_size)
# # Simulate forward diffusion
# data, noise = next(iter(train_loader))
# stepsize = int(1)
# sgs = []
# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     noisy_data = forward_diffusion_sample(data, noise, t)
#     mel_spectrogram = MelSpectrogram(n_mels=64, n_fft=512)
#     spectogram = mel_spectrogram(noisy_data.clone().detach()).squeeze(0).numpy()
#     sgs.append(spectogram)

# visualize_spectogram(spectograms=sgs)
# display_spectrograms(spectograms=sgs)