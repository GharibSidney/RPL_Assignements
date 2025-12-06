import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def extract(a, t, x_shape):
    t = t.to(a.device)

    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(x_start: torch.tensor, t:torch.tensor, coefficients:tuple, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    if len(coefficients) == 2:
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = coefficients
    else:
        betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = coefficients
        alphas_cumprod = 1.0 - (sqrt_one_minus_alphas_cumprod ** 2)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    sqrt_clear_alpha_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_noise_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_noisy = (sqrt_clear_alpha_t  * x_start + sqrt_noise_alpha_t * noise)
    return x_noisy


def p_sample(model, x, t, t_index, coefficients, noise=None):
    with torch.no_grad():
        betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = coefficients

        predicted_noise = model(x, t)
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        p_mean = sqrt_recip_alphas_t * (x - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)

        if t_index == 0:
            sample = p_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            if noise is None:
                noise = torch.randn_like(x)
            sample = p_mean + torch.sqrt(posterior_variance_t) * noise
        return sample


def p_sample_loop(model, shape, timesteps, T, coefficients, noise=None):
    with torch.no_grad():
        b = shape[0]
        img = torch.randn(shape, device=model.device) if noise is None else noise[0].to(model.device)
        imgs = []

        for i in tqdm(
            reversed(range(0, timesteps)), desc="Sampling", total=T, leave=False
        ):
            t = torch.full((b,), i, device=model.device, dtype=torch.long)
            if noise is None:
                img = p_sample(model, img, t, i, coefficients)
            else:
                img = p_sample(model, img, t, i, coefficients, noise=noise[i].to(model.device))
            imgs.append(img.cpu())

        return torch.stack(imgs)


def p_losses(denoise_model, x_start, t, coefficients, noise=None):
    noise = torch.randn_like(x_start) if noise is None else noise
    x_noisy = q_sample(x_start, t, coefficients, noise)
    predicted_noise = denoise_model(x_noisy, t)
    loss = F.smooth_l1_loss(noise, predicted_noise)
    return loss


def t_sample(timesteps, batch_size, device):
    ts = torch.randint(low=0, high=timesteps, size=(batch_size,), device=device)
    return ts
