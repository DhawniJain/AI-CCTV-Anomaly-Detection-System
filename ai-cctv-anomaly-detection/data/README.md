# data/

Put the extracted Kaggle dataset here:

```
data/raw/cctv-exam-monitor-dataset/      <- default paths.dataset_root in config.yaml
```
Any sub-folder layout works (files are discovered recursively). Run `python -m src.dataset_inspector` to see what you have.

Auto-generated (safe to delete): `data/cache/` (extracted frames + manifest), `data/cache_smoke/`, `data/synthetic_fixture/` (SYNTHETIC test videos, software testing only).
