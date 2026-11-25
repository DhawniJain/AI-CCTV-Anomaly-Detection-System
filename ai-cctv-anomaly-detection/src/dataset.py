"""Manifest building, label mapping, leakage-aware splitting and the PyTorch Dataset.

Design (memory safe): the whole dataset is NEVER loaded into RAM.
  1. build_manifest() walks the dataset, samples frames from videos (stride + cap), caches them as small
     PNGs on disk and records one row per frame in a pandas DataFrame (path, source, label, split ...).
  2. FrameDataset reads one cached image at a time inside a DataLoader.

Labels: 1 = abnormal, 0 = normal, -1 = unknown. Unknown is never silently turned into abnormal.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocessing import VideoError, extract_frames_to_cache, preprocess_frame, probe_video
from src.utils import (IMAGE_EXTS, VIDEO_EXTS, ProjectError, ensure_dir, get_logger, load_json, resolve_path,
                       save_json)

try:  # torch is optional for manifest utilities / tests that do not need tensors
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch, DataLoader, Dataset = None, None, object

log = get_logger()


# ----------------------------------------------------------------------------- discovery & labels
def discover_media(root: Path) -> tuple[list[Path], list[Path]]:
    videos, images = [], []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            ext = Path(fn).suffix.lower()
            if ext in VIDEO_EXTS:
                videos.append(Path(dirpath) / fn)
            elif ext in IMAGE_EXTS:
                images.append(Path(dirpath) / fn)
    return sorted(videos), sorted(images)


def _norm(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def label_from_folders(rel_path: str, normal_names: list[str], abnormal_names: list[str]) -> int:
    """Look at directory names from the deepest upward; EXACT (normalised) match only.

    Exact matching matters: 'abnormal' contains the substring 'normal', so substring matching would
    silently mislabel the whole dataset.
    """
    nn, an = {_norm(n) for n in normal_names}, {_norm(n) for n in abnormal_names}
    for part in reversed(Path(rel_path).parts[:-1]):
        p = _norm(part)
        if p in an:
            return 1
        if p in nn:
            return 0
    return -1


def _to_binary(value, normal_names, abnormal_names) -> int:
    s = _norm(str(value))
    if s in {"1", "abnormal", "anomaly", "anomalous"} or s in {_norm(n) for n in abnormal_names}:
        return 1
    if s in {"0", "normal"} or s in {_norm(n) for n in normal_names}:
        return 0
    raise ProjectError(f"Cannot map label value '{value}' to normal/abnormal. Add it to "
                       "data.normal_class_names / data.abnormal_class_names in config.yaml.")


def load_label_csv(csv_path: Path, root: Path, normal_names, abnormal_names) -> dict[str, list[tuple]]:
    """CSV columns: path,label[,start_frame,end_frame]  ->  {rel_posix_path: [(label,start,end), ...]}"""
    csv_path = resolve_path(csv_path)
    if not csv_path.exists():
        raise ProjectError(f"labels_csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"path", "label"} <= set(df.columns):
        raise ProjectError(f"labels_csv must have columns 'path' and 'label' (found {list(df.columns)}). "
                           "Optional: start_frame,end_frame for temporal segments.")
    out: dict[str, list[tuple]] = {}
    for _, r in df.iterrows():
        key = str(r["path"]).replace("\\", "/").lstrip("./")
        s = int(r["start_frame"]) if "start_frame" in df.columns and pd.notna(r.get("start_frame")) else None
        e = int(r["end_frame"]) if "end_frame" in df.columns and pd.notna(r.get("end_frame")) else None
        out.setdefault(key, []).append((_to_binary(r["label"], normal_names, abnormal_names), s, e))
    return out


def _csv_frame_label(entries: list[tuple], frame_idx: int) -> int:
    """Whole-file label if no segments; else 1 inside an abnormal segment and 0 elsewhere."""
    if all(s is None and e is None for _, s, e in entries):
        return entries[-1][0]
    for lab, s, e in entries:
        if lab == 1 and (s is None or frame_idx >= s) and (e is None or frame_idx <= e):
            return 1
    return 0


# ----------------------------------------------------------------------------- manifest
def _signature(cfg: dict) -> str:
    d = cfg["data"]
    keys = {k: d.get(k) for k in ["label_mode", "labels_csv", "normal_class_names", "abnormal_class_names",
                                   "frame_stride", "max_frames_per_video", "max_videos", "max_images",
                                   "image_size", "split", "seed"]}
    keys["root"] = str(cfg["paths"]["dataset_root"])
    keys["smoke"] = cfg["mode"]["smoke"]
    return hashlib.md5(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()


def build_manifest(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    root = resolve_path(cfg["paths"]["dataset_root"])
    if not root.exists():
        raise ProjectError(f"Dataset folder not found: {root}\nSee README 'Getting the dataset', then set "
                           "paths.dataset_root in config.yaml.")
    videos, images = discover_media(root)
    if not videos and not images:
        raise ProjectError(f"No videos or images found under {root}. Run `python -m src.dataset_inspector` "
                           "to see what the folder contains.")
    mode = d["label_mode"]
    normal_names, abnormal_names = d["normal_class_names"], d["abnormal_class_names"]
    csv_map = None
    if mode == "auto":
        mode = "csv" if d.get("labels_csv") else None
    if mode == "csv":
        if not d.get("labels_csv"):
            raise ProjectError("label_mode is 'csv' but data.labels_csv is not set.")
        csv_map = load_label_csv(d["labels_csv"], root, normal_names, abnormal_names)
    elif mode is None:  # auto without csv: try folders; fall back to none if nothing matches
        probe = [label_from_folders(str(p.relative_to(root)), normal_names, abnormal_names) for p in (videos + images)]
        mode = "folders" if any(v != -1 for v in probe) else "none"
    log.info("Label mode in use: %s", mode)

    if d.get("max_videos"):
        videos = videos[: int(d["max_videos"])]

    rng = np.random.default_rng(d["seed"])
    rows, skipped = [], []
    cache_root = ensure_dir(cfg["paths"]["cache_dir"]) / f"frames_{d['image_size']}_s{d['frame_stride']}"

    def label_for(rel: str, frame_idx: int) -> int:
        if mode == "folders":
            return label_from_folders(rel, normal_names, abnormal_names)
        if mode == "csv":
            ent = csv_map.get(rel)
            return _csv_frame_label(ent, frame_idx) if ent else -1
        return -1

    for i, v in enumerate(videos):
        rel = v.relative_to(root).as_posix()
        try:
            probe_video(v)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
            recs = extract_frames_to_cache(v, cache_root / safe, d["frame_stride"], d["max_frames_per_video"],
                                           d["image_size"])
        except VideoError as e:
            log.warning("Skipping video %s: %s", rel, e)
            skipped.append({"file": rel, "reason": str(e)})
            continue
        for idx, t, p in recs:
            rows.append(dict(frame_path=p, source_id=rel, kind="video", frame_idx=idx, time_s=t,
                             label=label_for(rel, idx)))
        if (i + 1) % 10 == 0 or i + 1 == len(videos):
            log.info("Extracted %d/%d videos", i + 1, len(videos))

    if images:
        cap = d.get("max_images")
        if cap and len(images) > cap:  # uniform subsample, keeps order
            keep = np.sort(rng.choice(len(images), size=int(cap), replace=False))
            images = [images[k] for k in keep]
        for k, p in enumerate(images):
            rel = p.relative_to(root).as_posix()
            rows.append(dict(frame_path=str(p), source_id=str(Path(rel).parent.as_posix()), kind="image",
                             frame_idx=k, time_s=float(k), label=label_for(rel, k)))

    if not rows:
        raise ProjectError("No usable frames were produced (all videos unreadable?). See warnings above.")
    df = pd.DataFrame(rows)
    df, split_info = assign_splits(df, d["split"], d["seed"])
    df.attrs["split_info"] = split_info
    df.attrs["skipped"] = skipped
    return df


def get_manifest(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Load the cached manifest if it matches the current config, otherwise (re)build it."""
    cache_dir = ensure_dir(cfg["paths"]["cache_dir"])
    csv_p, meta_p = cache_dir / "manifest.csv", cache_dir / "manifest_meta.json"
    sig = _signature(cfg)
    if not force and csv_p.exists() and meta_p.exists() and load_json(meta_p).get("signature") == sig:
        df = pd.read_csv(csv_p)
        log.info("Loaded cached manifest (%d frames).", len(df))
        return df
    df = build_manifest(cfg)
    df.to_csv(csv_p, index=False)
    save_json({"signature": sig, "split_info": df.attrs.get("split_info"), "skipped": df.attrs.get("skipped"),
               "n_frames": len(df), "synthetic": cfg["mode"]["synthetic"]}, meta_p)
    save_json({"split_info": df.attrs.get("split_info"), "skipped_files": df.attrs.get("skipped"),
               "counts": summarize_manifest(df)}, Path(cfg["paths"]["outputs_dir"]) / "split_info.json")
    log.info("Manifest built: %d frames.", len(df))
    return df


# ----------------------------------------------------------------------------- splitting
def _split_list(items: list, ratios: dict, rng: np.random.Generator) -> dict[str, list]:
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": items, "val": [], "test": []}
    if n == 2:
        return {"train": [items[0]], "val": [items[1]], "test": []}
    n_val = max(1, round(n * ratios["val"]))
    n_test = max(1, round(n * ratios["test"]))
    while n - n_val - n_test < 1:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    n_train = n - n_val - n_test
    return {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}


def assign_splits(df: pd.DataFrame, split_cfg: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    """Assign train/val/test WITHOUT leaking near-identical neighbouring frames across splits.

    group    : whole videos/folders go to one split (stratified by label so each split has both classes
               when possible). Preferred, needs >= 3 sources.
    temporal : inside each source, first part of the timeline = train, then val, then test. Used when there
               are too few sources (e.g. a single long video).
    random   : frame-level shuffle (WARNING: neighbouring frames leak between splits; optimistic results).
    """
    ratios = {"train": split_cfg["train"], "val": split_cfg["val"], "test": split_cfg["test"]}
    total = sum(ratios.values())
    ratios = {k: v / total for k, v in ratios.items()}
    strategy = split_cfg.get("strategy", "auto")
    n_sources = df["source_id"].nunique()
    if strategy == "auto":
        strategy = "group" if n_sources >= 3 else "temporal"
    if strategy == "group" and n_sources < 3:
        log.warning("Only %d source(s): falling back from 'group' to 'temporal' split.", n_sources)
        strategy = "temporal"
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "train"

    if strategy == "group":
        g_label = df.groupby("source_id")["label"].apply(lambda s: 1 if (s == 1).any() else (0 if (s == 0).any() else -1))
        assign = {}
        for lab in sorted(g_label.unique()):
            groups = list(g_label[g_label == lab].index)
            parts = _split_list(groups, ratios, rng)
            for sp, gs in parts.items():
                for g in gs:
                    assign[g] = sp
        df["split"] = df["source_id"].map(assign)
    elif strategy == "temporal":
        df = df.sort_values(["source_id", "frame_idx"]).reset_index(drop=True)
        for _, idx in df.groupby("source_id").groups.items():
            idx = list(idx)
            n = len(idx)
            a, b = int(n * ratios["train"]), int(n * (ratios["train"] + ratios["val"]))
            df.loc[idx[:a], "split"] = "train"
            df.loc[idx[a:b], "split"] = "val"
            df.loc[idx[b:], "split"] = "test"
    elif strategy == "random":
        log.warning("Random frame-level split selected: adjacent frames leak across splits.")
        r = rng.random(len(df))
        df.loc[r < ratios["train"], "split"] = "train"
        df.loc[(r >= ratios["train"]) & (r < ratios["train"] + ratios["val"]), "split"] = "val"
        df.loc[r >= ratios["train"] + ratios["val"], "split"] = "test"
    else:
        raise ProjectError(f"Unknown split strategy '{strategy}' (use auto|group|temporal|random).")
    df = df.sort_values(["source_id", "frame_idx"]).reset_index(drop=True)
    return df, {"strategy": strategy, "n_sources": int(n_sources), "ratios": ratios}


def summarize_manifest(df: pd.DataFrame) -> dict:
    out = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        out[sp] = {"frames": int(len(sub)), "sources": int(sub["source_id"].nunique()),
                   "normal": int((sub["label"] == 0).sum()), "abnormal": int((sub["label"] == 1).sum()),
                   "unlabeled": int((sub["label"] == -1).sum())}
    return out


def select_normal_frames(df: pd.DataFrame, split: str, cfg: dict) -> pd.DataFrame:
    """Frames allowed for normal-only learning in a split (abnormal frames are NEVER used)."""
    sub = df[df["split"] == split]
    keep = sub["label"] == 0
    if cfg["data"].get("assume_unlabeled_normal", False):
        keep = keep | (sub["label"] == -1)
    return sub[keep]


# ----------------------------------------------------------------------------- torch dataset
class FrameDataset(Dataset):
    """Reads one frame at a time from disk -> float32 tensor (C,H,W) in [0,1]."""

    def __init__(self, frame_paths: list[str], labels: list[int], image_size: int, grayscale: bool = False):
        self.paths, self.labels = list(frame_paths), list(labels)
        self.image_size, self.grayscale = image_size, grayscale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        for attempt in range(5):  # tolerate a few unreadable images instead of crashing a long run
            j = (i + attempt) % len(self.paths)
            img = cv2.imread(self.paths[j], cv2.IMREAD_COLOR)
            if img is not None:
                arr = preprocess_frame(img, self.image_size, self.grayscale)
                return torch.from_numpy(arr), int(self.labels[j]), j
            log.warning("Unreadable image skipped: %s", self.paths[j])
        raise ProjectError(f"Too many unreadable images near index {i}.")


def make_loader(df: pd.DataFrame, cfg: dict, shuffle: bool, batch_size: int | None = None, image_size: int | None = None,
                grayscale: bool | None = None):
    if torch is None:
        raise ProjectError("PyTorch is not installed. Run: pip install -r requirements.txt")
    if len(df) == 0:
        raise ProjectError("No frames available for this split. Check labels/split settings (see README).")
    ds = FrameDataset(df["frame_path"].tolist(), df["label"].tolist(),
                      image_size or cfg["data"]["image_size"],
                      cfg["data"]["grayscale"] if grayscale is None else grayscale)
    nw = int(cfg["training"]["num_workers"])
    return DataLoader(ds, batch_size=batch_size or cfg["training"]["batch_size"], shuffle=shuffle, num_workers=nw,
                      persistent_workers=nw > 0, drop_last=False)
