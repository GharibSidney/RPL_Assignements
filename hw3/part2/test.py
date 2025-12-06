import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------
# Diffusion helper functions (filled in)
# -------------------------
def extract(a, t, x_shape):
    t = t.to(a.device)

    out = a.gather(0, t)   # gather along correct dimension

    # reshape to (batch, 1, 1, 1, ...)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(x_start: torch.tensor, t:torch.tensor, coefficients:tuple, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    # coefficients might be either:
    # - (sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    # - or a 4-tuple (betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
    if len(coefficients) == 2:
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = coefficients
    else:
        # recover sqrt_alphas_cumprod from sqrt_one_minus_alphas_cumprod
        betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance = coefficients
        alphas_cumprod = 1.0 - (sqrt_one_minus_alphas_cumprod ** 2)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    sqrt_clear_alpha_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_noise_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_noisy = (sqrt_clear_alpha_t  * x_start + sqrt_noise_alpha_t * noise)
    return x_noisy


def p_sample(model, x, t, t_index, coefficients, noise=None):
    # coefficients: (betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
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
        # Start from pure noise (x_T)
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


# -------------------------
# Dummy model (small conv), deterministic
# -------------------------
class DummyDenoiseUNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # minimal net: map channels->channels with small conv
        # note: shape channels=1 in your requested parameters
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        # t is expected but unused in dummy model
        return self.net(x)


# -------------------------
# Parameters from your message (coefficients)
# -------------------------
def make_coefficients_from_user():
    # First element = betas (len 20)
    betas = torch.tensor([1.0000e-04, 1.1474e-03, 2.1947e-03, 3.2421e-03, 4.2895e-03,
                          5.3368e-03, 6.3842e-03, 7.4316e-03, 8.4789e-03, 9.5263e-03,
                          1.0574e-02, 1.1621e-02, 1.2668e-02, 1.3716e-02, 1.4763e-02,
                          1.5811e-02, 1.6858e-02, 1.7905e-02, 1.8953e-02, 2.0000e-02], dtype=torch.float32)

    sqrt_one_minus_alphas_cumprod = torch.tensor(
        [0.0100, 0.0353, 0.0586, 0.0817, 0.1046, 0.1273, 0.1500, 0.1725, 0.1949,
         0.2171, 0.2392, 0.2611, 0.2828, 0.3043, 0.3256, 0.3466, 0.3674, 0.3879,
         0.4081, 0.4280], dtype=torch.float32
    )

    sqrt_recip_alphas = torch.tensor(
        [1.0000, 1.0006, 1.0011, 1.0016, 1.0022, 1.0027, 1.0032, 1.0037, 1.0043,
         1.0048, 1.0053, 1.0059, 1.0064, 1.0069, 1.0075, 1.0080, 1.0085, 1.0091,
         1.0096, 1.0102], dtype=torch.float32
    )

    posterior_variance = torch.tensor(
        [0.0000e+00, 9.2004e-05, 7.9594e-04, 1.6717e-03, 2.6175e-03,
         3.5990e-03, 4.6013e-03, 5.6172e-03, 6.6424e-03, 7.6745e-03,
         8.7119e-03, 9.7535e-03, 1.0799e-02, 1.1847e-02, 1.2897e-02,
         1.3950e-02, 1.5005e-02, 1.6062e-02, 1.7120e-02, 1.8180e-02], dtype=torch.float32
    )

    # Return as tuple (same ordering your p_sample expects)
    return (betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)


# -------------------------
# Test driver using your requested parameters
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # As requested:
    shape = (2, 1, 32, 32)   # batch=2, channels=1
    timesteps = 5
    T = 5

    # Build coefficients (we use the 4-tuple you provided)
    coeffs = make_coefficients_from_user()
    # Move coefficients to device
    coeffs = tuple(c.to(device) for c in coeffs)

    # Deterministic noise generator (repeatable)
    torch.manual_seed(0)
    # noise shape expected by p_sample_loop doc: (timesteps + 1, batch_size, C, H, W)
    noise = torch.randn((timesteps + 1, shape[0], shape[1], shape[2], shape[3]), device=device)

    # Build model
    model = DummyDenoiseUNet(device).to(device)

    # Run sampling loop
    print("\nRunning p_sample_loop...")
    imgs = p_sample_loop(
        model=model,
        shape=shape,
        timesteps=timesteps,
        T=T,
        coefficients=coeffs,
        noise=noise
    )

    print("\nReturned imgs shape:", imgs.shape)
    # imgs shape -> (T, batch_size, channels, H, W)
    # Quick asserts:
    assert imgs.shape == (T, shape[0], shape[1], shape[2], shape[3])
    print("Shape check passed.")

    # Run a quick loss check (p_losses) for shape verification
    x_start = torch.randn(shape, device=device)
    t_batch = t_sample(timesteps, shape[0], device)
    loss = p_losses(model, x_start, t_batch, coeffs)
    print("p_losses output (scalar):", float(loss.item()))
    print("All tests completed successfully.")
