import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Same idea as transformer positional embeddings, applied to the
    diffusion timestep t so the model knows how noisy its input is."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = time[:, None].float() * freqs[None, :]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class Block(nn.Module):
    """Conv block with a time-embedding injected additively, like Ho et al.'s UNet."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.act(self.time_mlp(t_emb))[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """Deliberately small UNet: 2 down / 2 up + bottleneck. Enough capacity
    for MNIST, small enough to train in a few minutes on a free Colab GPU."""

    def __init__(self, time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        self.down1 = Block(1, 32, time_emb_dim)
        self.down2 = Block(32, 64, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = Block(64, 128, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block1 = Block(128, 64, time_emb_dim)  # 128 = 64 (up) + 64 (skip)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_block2 = Block(64, 32, time_emb_dim)   # 64 = 32 (up) + 32 (skip)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        d1 = self.down1(x, t)          # 28x28x32
        p1 = self.pool(d1)             # 14x14x32
        d2 = self.down2(p1, t)         # 14x14x64
        p2 = self.pool(d2)             # 7x7x64

        b = self.bottleneck(p2, t)     # 7x7x128

        u1 = self.up1(b)               # 14x14x64
        u1 = self.up_block1(torch.cat([u1, d2], dim=1), t)
        u2 = self.up2(u1)              # 28x28x32
        u2 = self.up_block2(torch.cat([u2, d1], dim=1), t)

        return self.out(u2)            # predicted noise, same shape as input
