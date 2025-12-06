import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def extract(a, t, x_shape):
    """
    Safe device-aware indexing: returns a[t[i]] for each i, reshaped for broadcasting
    into x_shape (i.e. shape (B, 1, 1, 1, ...)).
    a: 1-D tensor of length T on some device
    t: LongTensor of shape (B,) on any device
    x_shape: tuple e.g. (B, C, H, W)
    """
    t = t.to(a.device)
    out = a[t]
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(x_start, t, coefficients, noise=None):
    """
    q_sample assumes coefficients is the 2-tuple:
       (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    where each is a 1-D tensor of length T on some device.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    if len(coefficients) != 2:
        raise ValueError("q_sample expects coefficients=(sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)")

    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = coefficients

    sqrt_clear_alpha_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_noise_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_noisy = sqrt_clear_alpha_t * x_start + sqrt_noise_alpha_t * noise
    return x_noisy


@torch.no_grad()
def p_sample(model, x, t, t_index, coefficients, y=None, guidance_scale=1.0, noise=None):
    """
    Reverse step with Classifier-Free Guidance (CFG).
    model(x, t, y=None) -> predicted noise epsilon (shape same as x)
    coefficients: (betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
    """

    betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = coefficients

    eps_uncond = model(x, t, y=None)

    if y is None or guidance_scale == 0.0:
        eps_theta = eps_uncond
    else:
        eps_cond = model(x, t, y)
        eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    p_mean = sqrt_recip_alphas_t * (x - (betas_t / sqrt_one_minus_alphas_cumprod_t) * eps_theta)

    if t_index == 0:
        return p_mean

    if noise is None:
        noise = torch.randn_like(x)
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    sample = p_mean + torch.sqrt(posterior_variance_t) * noise
    return sample


@torch.no_grad()
def p_sample_loop(model, shape, timesteps, T, coefficients, y=None, guidance_scale=1.0, noise=None):
    """
    Full reverse loop with CFG.
    - noise: optional tensor of shape (timesteps+1, B, C, H, W). If provided, we use noise[0] as x_T,
      and noise[i] as the step-noise at step i.
    - Returns stacked images of shape (T, B, C, H, W) where index 0 corresponds to the first output
      (x_{T-1}) and last corresponds to x_0.
    """
    b = shape[0]

    # initialize x_T
    if noise is None:
        img = torch.randn(shape, device=model.device)
    else:
        # ensure noise on model device
        img = noise[0].to(model.device)

    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling", total=T, leave=False):
        t = torch.full((b,), i, device=model.device, dtype=torch.long)

        if noise is None:
            img = p_sample(model, img, t, i, coefficients, y=y, guidance_scale=guidance_scale, noise=None)
        else:
            # use deterministic provided noise for this step
            step_noise = noise[i].to(model.device)
            img = p_sample(model, img, t, i, coefficients, y=y, guidance_scale=guidance_scale, noise=step_noise)

        imgs.append(img.cpu())

    return torch.stack(imgs)  # shape (T, B, C, H, W)


def p_losses(denoise_model, x_start, t, coefficients, y=None, p_uncond=0.1, noise=None, loss_type="l2"):
    """
    Training loss for Classifier-Free Guidance (CFG).
    - With probability p_uncond per sample, drop the label (use y=None).
    - Compute both unconditional and conditional predictions and pick per-sample.
    """
    if t.dtype != torch.long:
        t = t.long()

    device = x_start.device
    noise = torch.randn_like(x_start) if noise is None else noise.to(device)

    if len(coefficients) == 4:
        # coefficients may be (betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
        betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = coefficients
        alphas_cumprod = 1.0 - (sqrt_one_minus_alphas_cumprod ** 2)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        q_coeffs = (sqrt_alphas_cumprod.to(device), sqrt_one_minus_alphas_cumprod.to(device))
    elif len(coefficients) == 2:
        q_coeffs = (coefficients[0].to(device), coefficients[1].to(device))
    else:
        raise ValueError("coefficients must be either 2- or 4-tuple")

    # produce x_t from x_0
    x_noisy = q_sample(x_start, t, q_coeffs, noise=noise)

    B = x_start.shape[0]

    if y is None:
        eps_theta = denoise_model(x_noisy, t, y=None)
    else:
        mask = (torch.rand(B, device=device) < p_uncond)
        eps_uncond = denoise_model(x_noisy, t, y=None)
        eps_cond = denoise_model(x_noisy, t, y) 

        mask_broadcast = mask.view(B, *((1,) * (len(x_start.shape) - 1)))
        eps_theta = torch.where(mask_broadcast, eps_uncond, eps_cond)

    if loss_type == "l1":
        loss = F.l1_loss(eps_theta, noise)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(eps_theta, noise)
    else:
        loss = F.mse_loss(eps_theta, noise)

    return loss


def t_sample(timesteps, batch_size, device):
    ts = torch.randint(low=0, high=timesteps, size=(batch_size,), device=device, dtype=torch.long)
    return ts
