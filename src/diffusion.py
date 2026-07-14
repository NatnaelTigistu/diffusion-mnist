import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class Diffusion:
    """
    Implements the DDPM forward and reverse processes (Ho et al., 2020).

    Forward:  q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) x_0, (1 - alpha_cumprod_t) I)
              -> closed form, so we can jump straight to any timestep t.

    Reverse:  p_theta(x_{t-1} | x_t) is parameterized by a neural net that
              predicts the noise epsilon added at step t. Training reduces to
              a simple MSE between true noise and predicted noise.
    """

    def __init__(self, timesteps=300, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        """Pull out the coefficients for a batch of timesteps t and reshape
        so they broadcast against an image batch (B, C, H, W)."""
        out = a.gather(-1, t.cpu()).to(t.device)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Sample x_t directly from x_0 using the closed-form forward process."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    def p_losses(self, model, x0, t):
        """Training objective: predict the noise that was added (simplified ELBO)."""
        x_noisy, noise = self.q_sample(x0, t)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """One reverse step: x_t -> x_{t-1}."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_omac_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_omac_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size, channels=1):
        """Full reverse process: pure noise -> image."""
        device = next(model.parameters()).device
        img = torch.randn(batch_size, channels, image_size, image_size, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)

        return img
