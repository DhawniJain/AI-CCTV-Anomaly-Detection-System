"""Inference engine: frames -> anomaly scores, plus shared score helpers used by calibration/evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.autoencoder import error_map, load_checkpoint, reconstruction_scores
from src.dataset import make_loader
from src.preprocessing import preprocess_frame
from src.signal_utils import causal_moving_average, heatmap_overlay, smooth_by_source  # noqa: F401 (re-exported)
from src.utils import ProjectError, get_device, load_json, resolve_path


@torch.no_grad()
def score_dataframe(df: pd.DataFrame, model, device, cfg: dict, score_method: str, topk_fraction: float,
                    batch_size: int = 64) -> np.ndarray:
    """Score every row of a manifest slice (batched, constant memory). Returns scores aligned with df rows."""
    in_ch = model.cfg["in_channels"]
    loader = make_loader(df, cfg, shuffle=False, batch_size=batch_size, image_size=model.cfg["image_size"],
                         grayscale=(in_ch == 1))
    scores = np.zeros(len(df), dtype=np.float64)
    model.eval()
    for x, _, idx in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        scores[idx.numpy()] = reconstruction_scores(x, x_hat, score_method, topk_fraction).cpu().numpy()
    return scores


class AnomalyDetector:
    """Loads models/autoencoder.pt + models/threshold.json once and scores BGR frames."""

    def __init__(self, models_dir: str | Path = "models", device=None, checkpoint: str = "autoencoder.pt",
                 threshold_file: str = "threshold.json"):
        models_dir = resolve_path(models_dir)
        thr_path = models_dir / threshold_file
        if not thr_path.exists():
            raise ProjectError(f"Threshold file not found: {thr_path}\nRun: python -m src.calibrate_threshold")
        self.device = device or get_device("auto")
        self.model, self.ckpt_meta = load_checkpoint(models_dir / checkpoint, self.device)
        self.thr_info = load_json(thr_path)
        self.threshold = float(self.thr_info["threshold"])
        self.score_method = self.thr_info.get("score_method", "mse")
        self.topk_fraction = float(self.thr_info.get("topk_fraction", 0.02))
        self.smoothing_window = int(self.thr_info.get("smoothing_window", 1))
        self.image_size = self.model.cfg["image_size"]
        self.grayscale = self.model.cfg["in_channels"] == 1  # taken from the checkpoint, not from config

    @torch.no_grad()
    def score_frames(self, frames_bgr: list[np.ndarray], want_maps: bool = False):
        """Returns raw scores (N,) and optionally error maps (N,H,W) as numpy arrays."""
        batch = np.stack([preprocess_frame(f, self.image_size, self.grayscale) for f in frames_bgr])
        x = torch.from_numpy(batch).to(self.device)
        x_hat, _ = self.model(x)
        s = reconstruction_scores(x, x_hat, self.score_method, self.topk_fraction).cpu().numpy()
        if want_maps:
            return s, error_map(x, x_hat).cpu().numpy()
        return s
