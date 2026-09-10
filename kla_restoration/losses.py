"""
Combined loss for restoration training: L1 + SSIM + Sobel edge loss +
VGG16 perceptual loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def gaussian_window(window_size, sigma, channels=1):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", gaussian_window(window_size, sigma=1.5, channels=channels))

    def forward(self, pred, target):
        window = self.window.to(pred.device)
        pad = self.window_size // 2

        mu_p = F.conv2d(pred, window, padding=pad, groups=self.channels)
        mu_t = F.conv2d(target, window, padding=pad, groups=self.channels)

        mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

        sigma_p_sq = F.conv2d(pred * pred, window, padding=pad, groups=self.channels) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, window, padding=pad, groups=self.channels) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=self.channels) - mu_pt

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
            (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
        )
        return 1 - ssim_map.mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        gx_p = F.conv2d(pred, self.sobel_x.to(pred.device), padding=1)
        gy_p = F.conv2d(pred, self.sobel_y.to(pred.device), padding=1)
        gx_t = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
        gy_t = F.conv2d(target, self.sobel_y.to(target.device), padding=1)

        grad_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-6)
        grad_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-6)
        return F.l1_loss(grad_p, grad_t)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_idx=16, resize_for_small_inputs=True):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:layer_idx]).eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize_for_small_inputs = resize_for_small_inputs

    def forward(self, pred, target):
        def prep(x):
            x = x.repeat(1, 3, 1, 1)
            if self.resize_for_small_inputs and x.shape[-1] < 64:
                x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

        pred_feat = self.features(prep(pred))
        target_feat = self.features(prep(target))
        return F.l1_loss(pred_feat, target_feat)


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, use_perceptual=True):
        super().__init__()
        self.weights = weights or {"l1": 1.0, "ssim": 0.1, "edge": 0.4, "perceptual": 0.1}
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = VGGPerceptualLoss()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_l = self.ssim(pred, target)
        edge_l = self.edge(pred, target)

        total = (
            self.weights["l1"] * l1
            + self.weights["ssim"] * ssim_l
            + self.weights["edge"] * edge_l
        )
        parts = {"l1": l1.item(), "ssim_loss": ssim_l.item(), "edge": edge_l.item()}

        if self.use_perceptual:
            perc_l = self.perceptual(pred, target)
            total = total + self.weights["perceptual"] * perc_l
            parts["perceptual"] = perc_l.item()

        return total, parts


if __name__ == "__main__":
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    loss_fn = CombinedLoss()
    total, parts = loss_fn(pred, target)
    print("Total loss:", total.item())
    print("Breakdown:", parts)
