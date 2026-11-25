# models/

Created by the pipeline (not shipped with the project):

| File | Created by | Purpose |
|---|---|---|
| `autoencoder.pt` | `python -m src.train` | Best autoencoder weights + architecture + preprocessing metadata |
| `threshold.json` | `python -m src.calibrate_threshold` | Calibrated threshold, score method, smoothing window, validation statistics |
| `smoke/` | `--smoke` / `--synthetic` runs | Throw-away models from software tests. Never use for results. |

YOLO weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on first use (internet needed once).
