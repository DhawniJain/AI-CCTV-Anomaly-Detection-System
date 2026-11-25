"""Shared helpers: config loading, device selection, seeding, JSON, logging, paths."""
from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ProjectError(RuntimeError):
    """Base class for errors that should be shown to the user as a clear message."""


def get_logger(name: str = "cctv") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_path(p: str | Path) -> Path:
    """Relative paths are interpreted relative to the project root, not the current directory."""
    path = Path(p).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(p: str | Path) -> Path:
    path = resolve_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path = "config.yaml", smoke: bool = False, synthetic: bool = False) -> dict:
    """Load config.yaml.

    smoke      -> tiny/fast settings, writes to models/smoke, outputs/smoke (never mixes with real results)
    synthetic  -> additionally swaps in the generated SYNTHETIC test fixture instead of the real dataset
    """
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise ProjectError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["mode"] = {"smoke": bool(smoke or synthetic), "synthetic": bool(synthetic)}
    if smoke or synthetic:
        cfg["paths"].update(models_dir="models/smoke", outputs_dir="outputs/smoke", cache_dir="data/cache_smoke")
        cfg["data"]["max_frames_per_video"] = min(cfg["data"]["max_frames_per_video"], 40)
        cfg["data"]["max_images"] = min(cfg["data"].get("max_images") or 200, 200)
        cfg["training"].update(epochs=2, batch_size=16, num_workers=0)
    if synthetic:
        cfg["paths"]["dataset_root"] = "data/synthetic_fixture"
        cfg["data"].update(label_mode="folders", normal_class_names=["normal"],
                           abnormal_class_names=["abnormal"], labels_csv=None, frame_stride=2, image_size=64)
        cfg["inference"]["frame_stride"] = 2
        cfg["calibration"]["method"] = "percentile"
        from src.synthetic import make_synthetic_dataset  # lazy import
        make_synthetic_dataset(resolve_path(cfg["paths"]["dataset_root"]))
    if cfg["data"]["image_size"] % 16 != 0:
        raise ProjectError("data.image_size must be divisible by 16 (the autoencoder downsamples 4 times).")
    return cfg


def get_device(preference: str = "auto"):
    """Pick CUDA (Colab), Apple MPS (M-series Mac) or CPU. torch is imported lazily."""
    import torch

    pref = (preference or "auto").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device_arg(device) -> str:
    """Translate a torch device into the string Ultralytics expects."""
    t = str(getattr(device, "type", device))
    return {"cuda": "0", "mps": "mps", "cpu": "cpu"}.get(t, "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def format_timestamp(seconds: float) -> str:
    """Seconds -> HH:MM:SS.s"""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"
