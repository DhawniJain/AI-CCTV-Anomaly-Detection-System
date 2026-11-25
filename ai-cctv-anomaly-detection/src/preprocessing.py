"""Frame extraction (OpenCV) and preprocessing shared by training AND inference.

Keeping ONE preprocessing function for both phases is what guarantees consistent tensor
shapes/values: BGR frame -> resize (squash, INTER_AREA) -> RGB/gray -> float32 in [0,1] -> CHW.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.utils import ProjectError, resolve_path


class VideoError(ProjectError):
    """Raised for unreadable / corrupt / empty videos."""


def preprocess_frame(frame_bgr: np.ndarray, image_size: int, grayscale: bool = False) -> np.ndarray:
    """BGR (or gray) uint8 frame -> float32 array shaped (C, image_size, image_size) in [0, 1]."""
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        raise ValueError("Empty frame passed to preprocess_frame.")
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if grayscale:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[..., None]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def tensor_to_bgr(chw: np.ndarray) -> np.ndarray:
    """Inverse of preprocess_frame (for visualisation): (C,H,W) float [0,1] -> BGR uint8."""
    arr = np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def probe_video(path: str | Path) -> dict:
    """Open a video and return fps / frame_count / size / duration, or raise VideoError."""
    path = resolve_path(path)
    if not path.exists():
        raise VideoError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"OpenCV cannot open '{path.name}' (corrupt file or unsupported codec).")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0 or fps > 240:
            fps = 25.0  # sensible fallback when container metadata is missing
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n <= 0:  # some containers do not report a count: verify at least one frame decodes
            ok, _ = cap.read()
            if not ok:
                raise VideoError(f"'{path.name}' contains no readable frames.")
        return {"fps": float(fps), "frame_count": n, "width": w, "height": h,
                "duration_s": (n / fps) if n > 0 else None}
    finally:
        cap.release()


def iter_video_frames(path: str | Path, stride: int = 1, max_frames: int | None = None,
                      max_seconds: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_seconds, frame_bgr) for every `stride`-th frame, one at a time.

    Frames are decoded lazily so memory stays flat regardless of video length. Non-selected frames
    are only grabbed (not decoded) which is considerably faster.
    """
    info = probe_video(path)
    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(resolve_path(path)))
    idx, yielded = 0, 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    t = idx / info["fps"]
                    if max_seconds is not None and t > max_seconds:
                        break
                    yield idx, t, frame
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        break
            idx += 1
    finally:
        cap.release()


def extract_frames_to_cache(video_path: str | Path, out_dir: str | Path, stride: int,
                            max_frames: int | None, image_size: int) -> list[tuple[int, float, str]]:
    """Extract sampled frames, resize to image_size and store as lossless PNG.

    PNG (not JPEG) is used so cached training frames carry no extra compression artefacts that
    live inference frames would not have. Returns [(frame_idx, time_s, png_path), ...].
    Re-running is cheap: a `_done.json` marker lets finished videos be reused.
    """
    import json

    out_dir = resolve_path(out_dir)
    marker = out_dir / "_done.json"
    if marker.exists():
        with open(marker) as f:
            return [tuple(r) for r in json.load(f)]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, t, frame in iter_video_frames(video_path, stride, max_frames):
        small = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        p = out_dir / f"f{idx:08d}.png"
        cv2.imwrite(str(p), small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        records.append((idx, float(t), str(p)))
    if not records:
        raise VideoError(f"No frames could be extracted from {Path(video_path).name}.")
    with open(marker, "w") as f:
        json.dump(records, f)
    return records
