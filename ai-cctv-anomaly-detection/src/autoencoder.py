"""Convolutional autoencoder (CNN feature learning + reconstruction-based anomaly scoring).

Encoder : 4 x [Conv(4x4, stride 2) -> BatchNorm -> ReLU]  => CNN feature map (8x8 for a 128px input)
          -> Linear -> latent vector (the learned "normal activity" representation)
Decoder : mirror image with ConvTranspose layers, final Sigmoid so output is in [0,1] like the input.

Trained on NORMAL frames only, so frames unlike the training distribution reconstruct worse. The
per-frame reconstruction error is therefore used as the anomaly score.
Limitation: a low-capacity AE can also reconstruct unseen content well, and a lighting/camera change can
raise the error without any real "event". Always judge it against labelled validation/test data.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.utils import ProjectError, resolve_path


def _down(cin, cout):
    return [nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


def _up(cin, cout):
    return [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, image_size: int = 128, base_channels: int = 32, latent_dim: int = 128):
        super().__init__()
        if image_size % 16 != 0 or image_size < 32:
            raise ValueError("image_size must be a multiple of 16 and >= 32.")
        self.cfg = dict(in_channels=in_channels, image_size=image_size, base_channels=base_channels,
                        latent_dim=latent_dim)
        c = base_channels
        chans = [c, c * 2, c * 4, c * 8]
        enc, prev = [], in_channels
        for ch in chans:
            enc += _down(prev, ch)
            prev = ch
        self.encoder = nn.Sequential(*enc)
        self.feat_hw = image_size // 16
        self.flat = chans[-1] * self.feat_hw * self.feat_hw
        self.to_latent = nn.Linear(self.flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat)
        dec = _up(chans[3], chans[2]) + _up(chans[2], chans[1]) + _up(chans[1], chans[0])
        dec += [nn.ConvTranspose2d(chans[0], in_channels, 4, 2, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)
        self._last_ch = chans[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """CNN features -> latent vector (N, latent_dim)."""
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(-1, self._last_ch, self.feat_hw, self.feat_hw)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


def reconstruction_scores(x: torch.Tensor, x_hat: torch.Tensor, method: str = "mse", topk_fraction: float = 0.02):
    """Per-sample anomaly score from squared reconstruction error.

    mse  : mean over all pixels (global change)
    topk : mean of the worst `topk_fraction` pixels (more sensitive to small local changes, noisier)
    """
    err = (x - x_hat) ** 2
    flat = err.flatten(1)
    if method == "mse":
        return flat.mean(1)
    if method == "topk":
        k = max(1, int(flat.shape[1] * topk_fraction))
        return flat.topk(k, dim=1).values.mean(1)
    raise ValueError(f"Unknown score_method '{method}' (use 'mse' or 'topk').")


def error_map(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """(N,H,W) pixel-wise squared error averaged over channels - used for evidence heatmaps."""
    return ((x - x_hat) ** 2).mean(1)


def save_checkpoint(model: ConvAutoencoder, path: str | Path, meta: dict | None = None) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": model.cfg, "meta": meta or {}}, path)
    return path


def load_checkpoint(path: str | Path, device="cpu") -> tuple[ConvAutoencoder, dict]:
    """Rebuild the exact architecture stored in the checkpoint (so inference can never mismatch training)."""
    path = resolve_path(path)
    if not path.exists():
        raise ProjectError(f"Model checkpoint not found: {path}\nTrain first: python -m src.train "
                           "(or `python -m src.train --synthetic` for a software smoke test).")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # very old torch without weights_only
        ckpt = torch.load(path, map_location=device)
    model = ConvAutoencoder(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})
