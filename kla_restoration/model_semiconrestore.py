"""
SemiconRestore-DA: a degradation-aware extension of the NAFNet-style model.

Custom architecture built on the NAFNet backbone (Chen et al., ECCV 2022),
extended with two non-destructive additions:

1. MULTI-CHANNEL INFORMATIONAL STEM: raw intensity, asinh-compressed
   intensity (handles negative values unlike log-domain transforms),
   morphological gradient (dilate-erode via 3x3 max/min pooling), and
   local variance -- all deterministic, near-zero cost, computed inside
   forward() from the raw 1-channel input.

2. LIGHTWEIGHT DEGRADATION ENCODER + FiLM CONDITIONING: a small encoder
   compresses the input into a continuous "degradation code" that modulates
   every NAFBlock via FiLM (feature-wise scale/shift), inspired by the
   all-in-one restoration literature (AirNet, PromptIR) but much lighter.

Both additions preserve the validated NAFNet backbone (activation-free
blocks, simplified channel attention, U-Net skip connections, residual
learning with zero-initialized output).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        attn = self.conv(self.pool(x))
        return x * attn


class FiLMGenerator(nn.Module):
    def __init__(self, z_dim, channels):
        super().__init__()
        self.net = nn.Linear(z_dim, channels * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        self.channels = channels

    def forward(self, z):
        out = self.net(z)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return gamma, beta


class FiLMNAFBlock(nn.Module):
    def __init__(self, channels, z_dim=None, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.z_dim = z_dim

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden // 2)
        self.conv1_out = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, hidden, 1)
        self.gate2 = SimpleGate()
        self.conv2_out = nn.Conv2d(hidden // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        if z_dim is not None:
            self.film1 = FiLMGenerator(z_dim, channels)
            self.film2 = FiLMGenerator(z_dim, channels)

    def _apply_film(self, x, film, z):
        if z is None or self.z_dim is None:
            return x
        gamma, beta = film(z)
        return (1 + gamma) * x + beta

    def forward(self, x, z=None):
        residual = x
        y = self.norm1(x)
        y = self._apply_film(y, self.film1, z) if self.z_dim is not None else y
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv1_out(y)
        x = residual + y * self.beta

        residual = x
        y = self.norm2(x)
        y = self._apply_film(y, self.film2, z) if self.z_dim is not None else y
        y = self.conv2(y)
        y = self.gate2(y)
        y = self.conv2_out(y)
        x = residual + y * self.gamma

        return x


class DegradationEncoder(nn.Module):
    def __init__(self, in_ch=4, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 96, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, z_dim),
        )

    def forward(self, x):
        feat = self.net(x)
        z = self.mlp(feat)
        return z


def build_informational_channels(x, asinh_scale=0.2):
    raw = x
    compressed = torch.asinh(x / asinh_scale)
    compressed = compressed * asinh_scale

    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    morph_grad = dilated - eroded

    kernel_size = 5
    pad = kernel_size // 2
    x_mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    x_sq_mean = F.avg_pool2d(x * x, kernel_size, stride=1, padding=pad)
    local_var = (x_sq_mean - x_mean ** 2).clamp(min=0)

    return torch.cat([raw, compressed, morph_grad, local_var], dim=1)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 2, 2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 2, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class SemiconRestoreDA(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, enc_blocks=(2, 2, 4), middle_blocks=4,
                 dec_blocks=(2, 2, 2), upscale=2, z_dim=64, asinh_scale=0.2):
        super().__init__()
        self.upscale = upscale
        self.asinh_scale = asinh_scale

        self.intro = nn.Conv2d(4, base_ch, 3, padding=1)
        self.degradation_encoder = DegradationEncoder(in_ch=4, z_dim=z_dim)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        for n in enc_blocks:
            self.encoders.append(nn.ModuleList(
                [FiLMNAFBlock(ch, z_dim=z_dim) for _ in range(n)]
            ))
            self.downs.append(Downsample(ch))
            ch *= 2

        self.middle = nn.ModuleList(
            [FiLMNAFBlock(ch, z_dim=z_dim) for _ in range(middle_blocks)]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(Upsample(ch))
            ch //= 2
            self.fuse_convs.append(nn.Conv2d(ch * 2, ch, 1))
            self.decoders.append(nn.ModuleList(
                [FiLMNAFBlock(ch, z_dim=z_dim) for _ in range(n)]
            ))

        final_up = []
        c = ch
        n_up = int(math.log2(upscale))
        for _ in range(max(n_up, 1)):
            final_up += [
                nn.Conv2d(c, c * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.final_upsample = nn.Sequential(*final_up)
        self.tail = nn.Conv2d(c, in_ch, 3, padding=1)

        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic",
                              align_corners=False)
        base = torch.clamp(base, 0, 1)

        info = build_informational_channels(x, asinh_scale=self.asinh_scale)
        z = self.degradation_encoder(info)

        feat = self.intro(info)

        skips = []
        for enc_blocks, down in zip(self.encoders, self.downs):
            for block in enc_blocks:
                feat = block(feat, z)
            skips.append(feat)
            feat = down(feat)

        for block in self.middle:
            feat = block(feat, z)

        for up, fuse, dec_blocks, skip in zip(self.ups, self.fuse_convs,
                                               self.decoders, reversed(skips)):
            feat = up(feat)
            feat = torch.cat([feat, skip], dim=1)
            feat = fuse(feat)
            for block in dec_blocks:
                feat = block(feat, z)

        feat = self.final_upsample(feat)
        residual = self.tail(feat)

        out = torch.clamp(base + residual, 0, 1)
        return out


if __name__ == "__main__":
    model = SemiconRestoreDA()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    x1 = torch.randn(2, 1, 64, 64)
    y1 = model(x1)
    assert y1.shape == (2, 1, 128, 128)

    x2 = torch.randn(1, 1, 128, 128)
    y2 = model(x2)
    assert y2.shape == (1, 1, 256, 256)

    print("SEMICONRESTORE-DA MODEL OK")
