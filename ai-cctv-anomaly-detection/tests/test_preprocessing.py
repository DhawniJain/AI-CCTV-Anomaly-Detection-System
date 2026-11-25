"""Tests for preprocessing, labelling, splitting and signal helpers (no PyTorch needed).
All video data here is SYNTHETIC and used for software testing only."""
import cv2
import numpy as np
import pandas as pd
import pytest

from src.dataset import assign_splits, label_from_folders
from src.preprocessing import VideoError, iter_video_frames, preprocess_frame, probe_video, tensor_to_bgr
from src.signal_utils import causal_moving_average, smooth_by_source
from src.synthetic import make_synthetic_dataset


def test_preprocess_shape_range_rgb_order():
    frame = np.zeros((90, 160, 3), np.uint8)
    frame[..., 2] = 255  # pure RED in BGR
    out = preprocess_frame(frame, 64)
    assert out.shape == (3, 64, 64) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out[0].mean() > 0.99 and out[2].mean() < 0.01  # channel 0 is R after BGR->RGB


def test_preprocess_grayscale_and_roundtrip():
    frame = np.random.default_rng(0).integers(0, 255, (50, 80, 3), dtype=np.uint8)
    g = preprocess_frame(frame, 32, grayscale=True)
    assert g.shape == (1, 32, 32)
    assert tensor_to_bgr(g).shape == (32, 32, 3)
    c = preprocess_frame(frame, 32)
    assert tensor_to_bgr(c).shape == (32, 32, 3)


def test_preprocess_rejects_empty():
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((0, 0, 3), np.uint8), 64)


def test_video_reading_stride_and_errors(tmp_path):
    root = make_synthetic_dataset(tmp_path / "fx")
    v = next((root / "normal").glob("*.mp4"))
    info = probe_video(v)
    assert info["frame_count"] == 50 and info["fps"] == 10.0
    idxs = [i for i, _, _ in iter_video_frames(v, stride=5)]
    assert idxs == list(range(0, 50, 5))
    assert len(list(iter_video_frames(v, stride=1, max_frames=7))) == 7
    assert all(t <= 2.0 for _, t, _ in iter_video_frames(v, stride=1, max_seconds=2.0))
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        probe_video(bad)
    with pytest.raises(VideoError):
        probe_video(tmp_path / "missing.mp4")


def test_folder_label_exact_match_not_substring():
    n, a = ["normal"], ["abnormal", "anomaly"]
    assert label_from_folders("abnormal/v1.mp4", n, a) == 1  # 'abnormal' must NOT be read as 'normal'
    assert label_from_folders("train/Normal/v1.mp4", n, a) == 0
    assert label_from_folders("misc/v1.mp4", n, a) == -1


def _fake_df(n_sources=6, frames=20):
    rows = [dict(source_id=f"s{s}", frame_idx=i, label=int(s % 2), frame_path="x") for s in range(n_sources) for i in range(frames)]
    return pd.DataFrame(rows)


def test_group_split_has_no_leakage():
    df, info = assign_splits(_fake_df(8), {"strategy": "auto", "train": .7, "val": .15, "test": .15}, 0)
    assert info["strategy"] == "group"
    assert (df.groupby("source_id")["split"].nunique() == 1).all()
    assert set(df["split"]) == {"train", "val", "test"}


def test_temporal_split_single_source_is_ordered():
    df, info = assign_splits(_fake_df(1, 100), {"strategy": "auto", "train": .7, "val": .15, "test": .15}, 0)
    assert info["strategy"] == "temporal"
    order = {"train": 0, "val": 1, "test": 2}
    assert df.sort_values("frame_idx")["split"].map(order).is_monotonic_increasing


def test_moving_average_is_causal():
    x = np.array([0, 0, 0, 10, 0, 0], float)
    y = causal_moving_average(x, 3)
    assert np.allclose(y[:3], 0) and y[3] == pytest.approx(10 / 3)  # no leakage from the future into index 2
    df = pd.DataFrame({"source_id": ["a"] * 3 + ["b"] * 3, "frame_idx": [2, 1, 0, 0, 1, 2]})
    out = smooth_by_source(df, np.array([3.0, 2.0, 1.0, 5.0, 5.0, 5.0]), 2)
    assert out[2] == 1.0 and out[1] == pytest.approx(1.5) and out[0] == pytest.approx(2.5)
