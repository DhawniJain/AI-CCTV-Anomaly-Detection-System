"""Video-pipeline tests with a STAND-IN scorer (no PyTorch/YOLO needed).

The fake scorer measures the fraction of saturated-red pixels - it is NOT the autoencoder; it only lets us verify
the plumbing (batching, smoothing, event merging, alert evidence, annotated video, CSV logs).
Everything uses the SYNTHETIC fixture; nothing here is a dataset result."""
import numpy as np
import pytest

from src.signal_utils import causal_moving_average
from src.synthetic import make_synthetic_dataset
from src.video_processor import ProcessingOptions, process_video
from src.yolo_detector import YoloDetector


class FakeScorer:
    threshold, smoothing_window = 0.01, 2

    def score_frames(self, frames, want_maps=False):
        s = np.array([float(((f[..., 2] > 200) & (f[..., 0] < 100)).mean()) for f in frames])
        maps = np.stack([(f[..., 2] > 200).astype(np.float32) for f in frames]) if want_maps else None
        return (s, maps) if want_maps else s


class FakeYolo:
    summarize, draw = staticmethod(YoloDetector.summarize), staticmethod(YoloDetector.draw)

    def detect(self, frame):
        return [{"label": "person", "conf": 0.9, "box": (10, 40, 40, 100)}]


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("fx") / "data")


def test_abnormal_video_creates_event_and_outputs(fixture_root, tmp_path):
    opts = ProcessingOptions(frame_stride=2, batch_size=4, use_yolo=True, min_event_frames=2)
    res = process_video(next((fixture_root / "abnormal").glob("*.mp4")), tmp_path / "o", FakeScorer(), FakeYolo(), opts)
    assert res.summary["n_events"] >= 1 and len(res.events) >= 1
    assert res.frame_csv.exists() and res.event_csv.exists() and res.video_path.exists()
    assert any(res.evidence_dir.glob("event_*.jpg"))
    assert {"timestamp", "smoothed_score", "is_anomaly", "person_count"} <= set(res.frame_log.columns)
    assert res.frame_log["person_count"].eq(1).all()
    ev = res.events.iloc[0]
    assert ev["start_time_s"] <= ev["end_time_s"] and ev["peak_score"] > res.summary["threshold_effective"]


def test_normal_video_has_no_events(fixture_root, tmp_path):
    opts = ProcessingOptions(frame_stride=2, batch_size=8, use_yolo=False)
    res = process_video(next((fixture_root / "normal").glob("*.mp4")), tmp_path / "o", FakeScorer(), None, opts)
    assert res.summary["n_events"] == 0 and res.frame_log["is_anomaly"].sum() == 0


def test_threshold_multiplier_suppresses_alerts(fixture_root, tmp_path):
    v = next((fixture_root / "abnormal").glob("*.mp4"))
    hi = process_video(v, tmp_path / "o", FakeScorer(), None, ProcessingOptions(frame_stride=2, use_yolo=False, threshold_multiplier=1000.0))
    assert hi.summary["n_events"] == 0


def test_yolo_only_mode_and_max_seconds(fixture_root, tmp_path):
    v = next((fixture_root / "normal").glob("*.mp4"))
    res = process_video(v, tmp_path / "o", None, FakeYolo(), ProcessingOptions(frame_stride=1, use_yolo=True, max_seconds=2.0, save_video=False))
    assert res.summary["anomaly_scoring"] is False and res.frame_log["time_s"].max() <= 2.0 and res.video_path is None


def test_nothing_to_run_is_an_error(fixture_root, tmp_path):
    from src.utils import ProjectError
    with pytest.raises(ProjectError):
        process_video(next((fixture_root / "normal").glob("*.mp4")), tmp_path / "o", None, None, ProcessingOptions(use_yolo=False))


def test_corrupt_upload_is_rejected(tmp_path):
    from src.preprocessing import VideoError
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(VideoError):
        process_video(bad, tmp_path / "o", FakeScorer(), None, ProcessingOptions(use_yolo=False))
