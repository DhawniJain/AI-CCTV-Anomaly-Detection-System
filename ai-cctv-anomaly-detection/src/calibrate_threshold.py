"""Choose the anomaly threshold using the VALIDATION split only (the test split is never touched).

Methods (config calibration.method):
  percentile : threshold = Pth percentile of validation NORMAL scores (default 99 -> ~1% false alarms)
  mean_std   : mean + k * std of validation normal scores
  f1         : maximise F1 on labelled validation data (needs both classes in val; else falls back)
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from src.anomaly_detector import score_dataframe, smooth_by_source
from src.autoencoder import load_checkpoint
from src.dataset import get_manifest
from src.utils import ProjectError, ensure_dir, get_device, get_logger, load_config, resolve_path, save_json

log = get_logger()


def calibrate(cfg: dict, device) -> dict:
    df = get_manifest(cfg)
    val = df[df["split"] == "val"].reset_index(drop=True)
    if len(val) == 0:
        raise ProjectError("Validation split is empty - cannot calibrate. Check data.split settings.")
    model, meta = load_checkpoint(resolve_path(cfg["paths"]["models_dir"]) / "autoencoder.pt", device)
    a, c = cfg["anomaly"], cfg["calibration"]
    scores = score_dataframe(val, model, device, cfg, a["score_method"], a["topk_fraction"])
    scores = smooth_by_source(val, scores, a["smoothing_window"])
    labels = val["label"].to_numpy()
    normal_mask = labels == 0
    if cfg["data"].get("assume_unlabeled_normal", False):
        normal_mask |= labels == -1
    if normal_mask.sum() == 0:
        raise ProjectError("No normal validation frames to calibrate on.")
    ns = scores[normal_mask]

    method = c["method"]
    if method == "f1":
        lab_mask = np.isin(labels, [0, 1])
        if lab_mask.sum() and len(np.unique(labels[lab_mask])) == 2:
            from sklearn.metrics import precision_recall_curve
            p, r, th = precision_recall_curve(labels[lab_mask], scores[lab_mask])
            f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
            thr = float(th[int(np.argmax(f1))])
            log.info("F1-optimal validation threshold %.6g (val F1=%.3f)", thr, f1.max())
        else:
            log.warning("method 'f1' needs labelled normal AND abnormal validation frames; using percentile instead.")
            method = "percentile"
    if method == "percentile":
        thr = float(np.percentile(ns, c["percentile"]))
    elif method == "mean_std":
        thr = float(ns.mean() + c["std_k"] * ns.std())
    elif method != "f1":
        raise ProjectError(f"Unknown calibration.method '{method}'.")

    info = {"threshold": thr, "method": method, "percentile": c["percentile"], "std_k": c["std_k"],
            "score_method": a["score_method"], "topk_fraction": a["topk_fraction"],
            "smoothing_window": a["smoothing_window"], "calibrated_on": "validation split only",
            "n_val_frames_scored": int(len(val)), "n_val_normal_used": int(normal_mask.sum()),
            "val_normal_score_stats": {"mean": float(ns.mean()), "std": float(ns.std()), "median": float(np.median(ns)),
                                       "p95": float(np.percentile(ns, 95)), "p99": float(np.percentile(ns, 99)),
                                       "max": float(ns.max())},
            "synthetic_fixture": cfg["mode"]["synthetic"], "created": datetime.now().isoformat(timespec="seconds")}
    models_dir, out_dir = ensure_dir(cfg["paths"]["models_dir"]), ensure_dir(cfg["paths"]["outputs_dir"])
    save_json(info, models_dir / "threshold.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.5, 4))
        plt.hist(ns, bins=40, alpha=.75, label="validation normal")
        if (labels == 1).any():
            plt.hist(scores[labels == 1], bins=40, alpha=.6, label="validation abnormal")
        plt.axvline(thr, color="r", ls="--", label=f"threshold = {thr:.4g}")
        plt.xlabel("anomaly score (reconstruction error)"); plt.ylabel("frames"); plt.legend()
        plt.title("Threshold calibration (validation split)"); plt.tight_layout()
        plt.savefig(out_dir / "threshold_calibration.png", dpi=150); plt.close()
    except Exception as e:
        log.warning("Could not save calibration plot: %s", e)
    log.info("Threshold %.6g (%s) saved to %s", thr, method, models_dir / "threshold.json")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, synthetic=args.synthetic)
    calibrate(cfg, get_device(args.device or cfg["training"]["device"]))


if __name__ == "__main__":
    main()
