# outputs/

| Path | Content |
|---|---|
| `dataset_report.json` | Output of the dataset inspector |
| `split_info.json` | Split strategy + per-split frame/label counts |
| `training_history.csv`, `training_curve.png`, `reconstruction_samples.png`, `train_info.json` | Training run |
| `threshold_calibration.png` | Validation histogram + threshold |
| `evaluation/` | `metrics.json`, confusion matrix, ROC/PR, score distributions, per-video table, highest/lowest-scoring frames, `test_scores.csv` |
| `inference/<video>/` and `dashboard/<timestamp>_<video>/` | `annotated_video.mp4`, `frame_log.csv`, `event_log.csv`, `summary.json`, `evidence/event_*.jpg` |
| `smoke/` | Software-test outputs (SYNTHETIC data - never report these) |
