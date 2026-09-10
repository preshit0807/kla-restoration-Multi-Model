"""
CNN + Transformer hybrid restoration model.

CNN encoder for local features + skip connection, transformer bottleneck
(self-attention on a 4x-downsampled feature map, keeping attention cost
manageable) for global context, CNN decoder fused with the skip connection,
then pixel-shuffle 2x super-resolution tail. 2D sinusoidal positional
encoding is computed at runtime from actual H/W, so the model stays
resolution-agnostic.

NOTE (from later ablation): this scored slightly lower on PSNR/SSIM than
the plain CNN and NAFNet models at full-dataset scale, while being slower.
Kept in the repo as a documented ablation, not the final submission.
"""

import math
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return residual + x


def sinusoidal_2d_pos_encoding(h, w, dim, device):
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sinusoidal encoding"
    quarter = dim // 4

    div_term = torch.exp(
        torch.arange(0, quarter, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / quarter)
    )

    y_pos = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)
    x_pos = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(1)

    pe_y = torch.zeros(h, quarter * 2, device=device)
    pe_y[:, 0::2] = torch.sin(y_pos * div_term)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term)

    pe_x = torch.zeros(w, quarter * 2, device=device)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term)

    pe_y = pe_y.unsqueeze(1).expand(h, w, quarter * 2)
    pe_x = pe_x.unsqueeze(0).expand(h, w, quarter * 2)
    pe = torch.cat([pe_y, pe_x], dim=-1)
    return pe.reshape(h * w, dim)


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, num_layers=4, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pos = sinusoidal_2d_pos_encoding(h, w, c, x.device).unsqueeze(0)
        tokens = tokens + pos
        tokens = self.encoder(tokens)
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return out


class HybridRestorationNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=48, enc_blocks=4, downsample_factor=4,
                 transformer_dim=192, transformer_layers=4, transformer_heads=4,
                 upscale=2):
        super().__init__()
        assert downsample_factor in (2, 4, 8)
        n_down = int(math.log2(downsample_factor))

        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.stem_blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(enc_blocks)])

        down_layers = []
        ch = base_ch
        for _ in range(n_down):
            down_layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        self.downsample = nn.Sequential(*down_layers)
        self.to_transformer_dim = nn.Conv2d(ch, transformer_dim, 1)
        self.from_transformer_dim = nn.Conv2d(transformer_dim, ch, 1)

        self.bottleneck = TransformerBottleneck(
            dim=transformer_dim, num_layers=transformer_layers, num_heads=transformer_heads,
        )

        self.upsample_from_bottleneck = self._build_upsampler(ch, n_down)

        self.fuse = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.decoder_blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(enc_blocks)])

        final_up = []
        c = base_ch
        n_up = int(math.log2(upscale))
        for _ in range(max(n_up, 1)):
            final_up += [
                nn.Conv2d(c, c * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.final_upsample = nn.Sequential(*final_up)
        self.tail = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    @staticmethod
    def _build_upsampler(start_ch, n_down):
        layers = []
        ch = start_ch
        for _ in range(n_down):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch // 2
        return nn.Sequential(*layers)

    def forward(self, x):
        stem_feat = self.stem_blocks(self.stem(x))
        down_feat = self.downsample(stem_feat)

        t = self.to_transformer_dim(down_feat)
        t = self.bottleneck(t)
        bottleneck_out = self.from_transformer_dim(t)

        up_feat = self.upsample_from_bottleneck(bottleneck_out)

        fused = torch.cat([up_feat, stem_feat], dim=1)
        fused = self.fuse(fused)
        fused = self.decoder_blocks(fused)

        out = self.final_upsample(fused)
        out = self.tail(out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = HybridRestorationNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    x1 = torch.randn(2, 1, 64, 64)
    y1 = model(x1)
    assert y1.shape == (2, 1, 128, 128)

    x2 = torch.randn(1, 1, 128, 128)
    y2 = model(x2)
    assert y2.shape == (1, 1, 256, 256)

    print("HYBRID MODEL OK -- resolution-agnostic confirmed")
