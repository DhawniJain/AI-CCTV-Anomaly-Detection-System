"""Evaluate on the TEST split using the threshold calibrated on the validation split.

If test frames carry normal/abnormal labels (both classes present): accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, confusion matrix. Otherwise: score distribution + qualitative montage ONLY - supervised
metrics are not computed because without ground truth they would be meaningless.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from src.anomaly_detector import score_dataframe, smooth_by_source
from src.autoencoder import load_checkpoint
from src.dataset import get_manifest
from src.utils import ProjectError, ensure_dir, get_device, get_logger, load_config, load_json, resolve_path, save_json

log = get_logger()


def _montage(paths, scores, size=160, cols=4):
    tiles = []
    for p, s in zip(paths, scores):
        img = cv2.imread(p)
        img = np.zeros((size, size, 3), np.uint8) if img is None else cv2.resize(img, (size, size))
        cv2.putText(img, f"{s:.4g}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def evaluate(cfg: dict, device) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve,
                                 precision_recall_fscore_support, roc_auc_score, roc_curve)

    df = get_manifest(cfg)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if len(test) == 0:
        raise ProjectError("Test split is empty. Check data.split (need >= 3 sources for a group split).")
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    thr_path = models_dir / "threshold.json"
    if not thr_path.exists():
        raise ProjectError("threshold.json missing. Run: python -m src.calibrate_threshold")
    thr_info = load_json(thr_path)
    thr = float(thr_info["threshold"])
    model, _ = load_checkpoint(models_dir / "autoencoder.pt", device)

    scores = score_dataframe(test, model, device, cfg, thr_info["score_method"], thr_info["topk_fraction"])
    scores = smooth_by_source(test, scores, int(thr_info["smoothing_window"]))
    test["score"], test["pred"] = scores, (scores > thr).astype(int)

    out = ensure_dir(cfg["paths"]["outputs_dir"]) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    test[["source_id", "frame_idx", "time_s", "label", "score", "pred", "frame_path"]].to_csv(out / "test_scores.csv", index=False)

    lab = test[test["label"].isin([0, 1])]
    res = {"threshold": thr, "threshold_method": thr_info["method"], "n_test_frames": int(len(test)),
           "n_labeled": int(len(lab)), "n_unlabeled": int((test["label"] == -1).sum()),
           "synthetic_fixture": cfg["mode"]["synthetic"],
           "overall_flag_rate": float(test["pred"].mean())}
    has_both = len(lab) > 0 and lab["label"].nunique() == 2

    if has_both:
        y, p, s = lab["label"].to_numpy(), lab["pred"].to_numpy(), lab["score"].to_numpy()
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
        cm = confusion_matrix(y, p, labels=[0, 1])
        res.update(accuracy=float(accuracy_score(y, p)), precision=float(pr), recall=float(rc), f1=float(f1),
                   roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                   confusion_matrix={"labels": ["normal", "abnormal"], "matrix": cm.tolist()},
                   abnormal_prevalence=float((y == 1).mean()),
                   score_stats={"normal": pd.Series(s[y == 0]).describe().to_dict(),
                                "abnormal": pd.Series(s[y == 1]).describe().to_dict()},
                   note="Frame-level metrics on the test split. Check abnormal_prevalence before trusting accuracy.")
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
        ax.set_xticks([0, 1], ["normal", "abnormal"]); ax.set_yticks([0, 1], ["normal", "abnormal"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix (test)")
        fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

        fpr, tpr, _ = roc_curve(y, s)
        pp, rr, _ = precision_recall_curve(y, s)
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        axs[0].plot(fpr, tpr); axs[0].plot([0, 1], [0, 1], "k--", alpha=.4)
        axs[0].set(title=f"ROC (AUC={res['roc_auc']:.3f})", xlabel="false positive rate", ylabel="true positive rate")
        axs[1].plot(rr, pp); axs[1].set(title=f"Precision-Recall (AP={res['pr_auc']:.3f})", xlabel="recall", ylabel="precision")
        fig.tight_layout(); fig.savefig(out / "roc_pr_curves.png", dpi=150); plt.close(fig)

        plt.figure(figsize=(6.5, 4))
        plt.hist(s[y == 0], bins=40, alpha=.7, label="normal"); plt.hist(s[y == 1], bins=40, alpha=.6, label="abnormal")
        plt.axvline(thr, color="r", ls="--", label="threshold"); plt.xlabel("anomaly score"); plt.ylabel("frames")
        plt.legend(); plt.title("Score distribution by true class (test)"); plt.tight_layout()
        plt.savefig(out / "score_distribution.png", dpi=150); plt.close()
    else:
        res["note"] = ("Supervised metrics (accuracy/precision/recall/F1/confusion matrix) were NOT computed: the test "
                       "split has no frames of BOTH labelled classes, and fabricating ground truth is not valid. "
                       "Only score statistics and qualitative results are reported.")
        log.warning(res["note"])
        plt.figure(figsize=(6.5, 4))
        plt.hist(test["score"], bins=40); plt.axvline(thr, color="r", ls="--", label="threshold"); plt.legend()
        plt.xlabel("anomaly score"); plt.ylabel("frames"); plt.title("Test score distribution (no ground truth)")
        plt.tight_layout(); plt.savefig(out / "score_distribution.png", dpi=150); plt.close()
        res["score_stats"] = test["score"].describe().to_dict()

    per_src = test.groupby("source_id").agg(frames=("score", "size"), mean_score=("score", "mean"),
                                            max_score=("score", "max"), flagged_fraction=("pred", "mean"),
                                            label_min=("label", "min"), label_max=("label", "max")).reset_index()
    per_src.to_csv(out / "per_source_summary.csv", index=False)
    top = test.nlargest(8, "score")
    low = test.nsmallest(8, "score")
    cv2.imwrite(str(out / "highest_scoring_frames.png"), _montage(top["frame_path"], top["score"]))
    cv2.imwrite(str(out / "lowest_scoring_frames.png"), _montage(low["frame_path"], low["score"]))
    save_json(res, out / "metrics.json")
    log.info("Evaluation saved to %s", out)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if k in res:
            log.info("%-10s %.4f", k, res[k])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, synthetic=args.synthetic)
    evaluate(cfg, get_device(args.device or cfg["training"]["device"]))


if __name__ == "__main__":
    main()
