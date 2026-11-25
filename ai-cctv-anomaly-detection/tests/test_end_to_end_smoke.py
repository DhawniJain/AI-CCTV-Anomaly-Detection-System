"""End-to-end smoke test on the SYNTHETIC fixture: train -> calibrate -> evaluate -> infer.
Requires PyTorch (skipped otherwise). Results are meaningless for the project; this only checks that the
pieces fit together (shapes, checkpoints, threshold file, outputs)."""
import json

import pytest

pytest.importorskip("torch")

from src.utils import PROJECT_ROOT  # noqa: E402


def test_train_calibrate_evaluate_infer_synthetic():
    from src import calibrate_threshold, evaluate, train, video_processor
    train.main(["--synthetic", "--epochs", "2", "--rebuild-cache"])
    calibrate_threshold.main(["--synthetic"])
    evaluate.main(["--synthetic"])
    assert (PROJECT_ROOT / "models/smoke/autoencoder.pt").exists()
    thr = json.loads((PROJECT_ROOT / "models/smoke/threshold.json").read_text())
    assert thr["threshold"] > 0 and thr["synthetic_fixture"] is True
    metrics = json.loads((PROJECT_ROOT / "outputs/smoke/evaluation/metrics.json").read_text())
    assert metrics["synthetic_fixture"] is True and "roc_auc" in metrics
    vid = next((PROJECT_ROOT / "data/synthetic_fixture/abnormal").glob("*.mp4"))
    video_processor.main(["--video", str(vid), "--synthetic", "--no-yolo", "--out-dir", str(PROJECT_ROOT / "outputs/smoke/inference_test")])
    assert (PROJECT_ROOT / "outputs/smoke/inference_test/event_log.csv").exists()
