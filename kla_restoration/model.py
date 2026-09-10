"""
Baseline restoration model: denoising + 2x super-resolution in one pass.

Design choices:
- Fully convolutional -> resolution-agnostic, works on any input size.
- Residual blocks in a U-Net-ish encoder/bottleneck, then PixelShuffle
  upsampling (sub-pixel conv) instead of transposed conv, avoiding
  checkerboard artifacts and staying fast at inference.
- Sigmoid output activation: GT is guaranteed to be in [0,1].
- Kept deliberately lightweight since inference speed is scored.
"""

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


class RestorationNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=48, num_blocks=8, upscale=2):
        super().__init__()
        self.upscale = upscale

        self.head = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base_ch) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        upsample_layers = []
        ch = base_ch
        n_upsamples = int(torch.log2(torch.tensor(float(upscale))).item())
        for _ in range(max(n_upsamples, 1)):
            upsample_layers += [
                nn.Conv2d(ch, ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.tail = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_tail(self.body(feat))
        feat = feat + body_out
        up = self.upsample(feat)
        out = self.tail(up)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = RestorationNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input shape: {tuple(x.shape)} -> Output shape: {tuple(y.shape)}")
    assert y.shape == (2, 1, 256, 256), "Output should be 2x upsampled"
    print("Output range:", y.min().item(), y.max().item())
