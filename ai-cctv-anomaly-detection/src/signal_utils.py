"""Torch-free numeric/visual helpers (kept separate so they are importable and testable without PyTorch)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing (causal) moving average - usable in real time because it never looks at future frames."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def smooth_by_source(df: pd.DataFrame, scores: np.ndarray, window: int) -> np.ndarray:
    """Smooth scores separately inside each video/folder, in frame order (rows align with `scores`)."""
    out = np.asarray(scores, dtype=np.float64).copy()
    src, fidx = df["source_id"].to_numpy(), df["frame_idx"].to_numpy()
    for s in np.unique(src):
        idx = np.where(src == s)[0]
        idx = idx[np.argsort(fidx[idx], kind="stable")]
        out[idx] = causal_moving_average(out[idx], window)
    return out


def heatmap_overlay(frame_bgr: np.ndarray, emap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a (relative) reconstruction-error heatmap on a frame - shows WHERE the model was surprised."""
    h, w = frame_bgr.shape[:2]
    m = cv2.resize(emap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    color = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, color, alpha, 0)
