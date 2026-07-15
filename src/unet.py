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


class ResidualBlock(nn.Module):
    """Conv block with a time-embedding injected additively, PLUS an actual
    residual (skip) connection around the two convolutions, as in the
    original DDPM UNet. If in_ch != out_ch, the residual path is projected
    with a 1x1 conv so the shapes match for the addition.
    """

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        # Project the residual when channel counts differ, otherwise pass through.
        self.residual_proj = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        residual = self.residual_proj(x)

        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.act(self.time_mlp(t_emb))[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))

        return h + residual


class Downsample(nn.Module):
    """Learned downsampling via a stride-2 convolution, replacing MaxPool2d
    so the network can learn how to downsample instead of using a fixed rule.
    """

    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class SelfAttention2d(nn.Module):
    """Standard multi-head self-attention over spatial positions, used at
    the bottleneck resolution to improve global consistency. Implemented as
    a residual block: output = input + attention(input).
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)  # each: (B, heads, C//heads, H*W)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


class SimpleUNet(nn.Module):
    """Same overall down/bottleneck/up shape as before (2 down / 2 up), but
    with the five improvements applied:
      1. True residual connections (ResidualBlock instead of Block).
      2. Wider channels: 64 -> 128 -> 256 instead of 32 -> 64 -> 128.
      3. Learned downsampling (strided conv) instead of MaxPool2d.
      4. Self-attention at the 7x7 bottleneck.
      5. Larger time embedding (256-d instead of 32-d).

    Public interface is unchanged: __init__(time_emb_dim=...) and
    forward(x, timestep) -> predicted noise tensor of the same shape as x.
    Existing call sites do not need to change.
    """

    def __init__(self, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Down path
        self.down1 = ResidualBlock(1, 64, time_emb_dim)          # 28x28x64
        self.downsample1 = Downsample(64)                        # -> 14x14x64
        self.down2 = ResidualBlock(64, 128, time_emb_dim)        # 14x14x128
        self.downsample2 = Downsample(128)                       # -> 7x7x128

        # Bottleneck (with self-attention)
        self.bottleneck1 = ResidualBlock(128, 256, time_emb_dim)  # 7x7x256
        self.bottleneck_attn = SelfAttention2d(256)
        self.bottleneck2 = ResidualBlock(256, 256, time_emb_dim)  # 7x7x256

        # Up path
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # -> 14x14x128
        self.up_block1 = ResidualBlock(256, 128, time_emb_dim)  # 256 = 128 (up) + 128 (skip)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # -> 28x28x64
        self.up_block2 = ResidualBlock(128, 64, time_emb_dim)   # 128 = 64 (up) + 64 (skip)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        d1 = self.down1(x, t)              # 28x28x64
        p1 = self.downsample1(d1)          # 14x14x64
        d2 = self.down2(p1, t)             # 14x14x128
        p2 = self.downsample2(d2)          # 7x7x128

        b = self.bottleneck1(p2, t)        # 7x7x256
        b = self.bottleneck_attn(b)        # 7x7x256
        b = self.bottleneck2(b, t)         # 7x7x256

        u1 = self.up1(b)                   # 14x14x128
        u1 = self.up_block1(torch.cat([u1, d2], dim=1), t)
        u2 = self.up2(u1)                  # 28x28x64
        u2 = self.up_block2(torch.cat([u2, d1], dim=1), t)

        return self.out(u2)                # predicted noise, same shape as input