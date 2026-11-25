"""Dataset inspector: tells you what is ACTUALLY inside the downloaded dataset.

Run it BEFORE configuring labels. It never guesses: it reports file types, folder layout, candidate
label files and suggests (but does not force) a `data.label_mode`.

    python -m src.dataset_inspector                       # uses paths.dataset_root from config.yaml
    python -m src.dataset_inspector --root /path/to/data  # inspect any folder
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from src.preprocessing import probe_video
from src.utils import (IMAGE_EXTS, VIDEO_EXTS, ProjectError, ensure_dir, load_config, resolve_path, save_json)

LABEL_FILE_EXTS = {".csv", ".json", ".xml", ".yaml", ".yml", ".txt", ".names", ".tsv"}


def inspect_dataset(root: str | Path, max_depth: int = 3, n_sample_videos: int = 5, cfg: dict | None = None) -> dict:
    root = resolve_path(root)
    if not root.exists():
        raise ProjectError(
            f"Dataset folder not found: {root}\n"
            "Download it from Kaggle (see README 'Getting the dataset'), extract it, then set "
            "paths.dataset_root in config.yaml or pass --root.")
    if not root.is_dir():
        raise ProjectError(f"{root} is not a directory (did you forget to unzip the dataset?).")

    ext_counts: Counter = Counter()
    dir_media: dict[str, Counter] = defaultdict(Counter)
    videos, images, label_files = [], 0, []
    total_files = 0
    for dirpath, _, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "." if rel_dir == "." else rel_dir.replace(os.sep, "/")
        for fn in files:
            if fn.startswith("."):
                continue
            total_files += 1
            ext = Path(fn).suffix.lower()
            ext_counts[ext or "(none)"] += 1
            full = Path(dirpath) / fn
            if ext in VIDEO_EXTS:
                videos.append(full)
                dir_media[rel_dir]["videos"] += 1
            elif ext in IMAGE_EXTS:
                images += 1
                dir_media[rel_dir]["images"] += 1
            elif ext in LABEL_FILE_EXTS:
                label_files.append(full)

    report: dict = {"root": str(root), "total_files": total_files, "extensions": dict(ext_counts.most_common()),
                    "n_videos": len(videos), "n_images": images, "warnings": [], "label_file_previews": []}

    if total_files == 0:
        raise ProjectError(f"{root} is empty. Check that the Kaggle zip was extracted into this folder.")
    if not videos and not images:
        report["warnings"].append("No supported video/image files found. Supported video: "
                                  f"{sorted(VIDEO_EXTS)}; images: {sorted(IMAGE_EXTS)}. "
                                  f"Extensions present: {dict(ext_counts)}")

    # directory summary (only directories that directly contain media)
    shown = {d: dict(c) for d, c in sorted(dir_media.items()) if d == "." or d.count("/") < max_depth}
    report["media_directories"] = shown
    report["top_level_entries"] = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))[:60]

    # sample video properties
    props = []
    for v in videos[:n_sample_videos]:
        try:
            info = probe_video(v)
            info["file"] = str(v.relative_to(root))
            props.append(info)
        except ProjectError as e:
            report["warnings"].append(f"Unreadable video: {v.relative_to(root)} ({e})")
    report["sample_video_properties"] = props

    # preview candidate label / annotation files (never assumed to be normal/abnormal labels)
    import pandas as pd
    for lf in label_files[:200]:
        rel = str(lf.relative_to(root))
        entry: dict = {"file": rel, "size_kb": round(lf.stat().st_size / 1024, 1)}
        try:
            if lf.suffix.lower() in {".csv", ".tsv"}:
                df = pd.read_csv(lf, nrows=5, sep=None, engine="python")
                entry["columns"] = list(df.columns)
                entry["first_rows"] = df.head(3).astype(str).to_dict("records")
            elif lf.suffix.lower() == ".json" and lf.stat().st_size < 20 * 1024 * 1024:
                with open(lf) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    entry["top_level_keys"] = list(data.keys())[:20]
                    if {"images", "annotations", "categories"} <= set(data.keys()):
                        entry["format_guess"] = "COCO-style OBJECT-DETECTION annotations"
                        entry["categories"] = [c.get("name") for c in data["categories"]][:40]
                elif isinstance(data, list):
                    entry["list_length"] = len(data)
            elif lf.suffix.lower() in {".txt", ".names", ".yaml", ".yml"}:
                with open(lf, errors="replace") as f:
                    entry["first_lines"] = [next(f, "").strip()[:120] for _ in range(3)]
        except Exception as e:  # inspection must never crash on odd files
            entry["read_error"] = str(e)[:150]
        report["label_file_previews"].append(entry)
        if len(report["label_file_previews"]) >= 25:
            break

    txt_in_label_dirs = [p for p in label_files if p.suffix.lower() == ".txt" and "label" in p.parent.name.lower()]
    if txt_in_label_dirs:
        report["warnings"].append(
            f"{len(txt_in_label_dirs)} .txt files in a 'labels' folder: this usually means YOLO-format "
            "OBJECT annotations (class + bounding box). These are NOT normal/abnormal labels.")
    if any(e.get("format_guess") for e in report["label_file_previews"]):
        report["warnings"].append("COCO-style annotation JSON found. It describes objects, not normal/abnormal events.")

    # suggest a label mode (suggestion only)
    normal_names = {n.lower() for n in (cfg or {}).get("data", {}).get("normal_class_names", ["normal"])}
    abn_names = {n.lower() for n in (cfg or {}).get("data", {}).get("abnormal_class_names", ["abnormal", "anomaly"])}
    leaf_names = {Path(d).name.lower() for d in dir_media if d != "."}
    folder_hits_n, folder_hits_a = leaf_names & normal_names, leaf_names & abn_names
    if folder_hits_n and folder_hits_a:
        suggestion = "folders"
        why = f"Found folders matching normal={sorted(folder_hits_n)} and abnormal={sorted(folder_hits_a)}."
    elif any(e.get("columns") for e in report["label_file_previews"]):
        suggestion = "csv (review the CSV columns above; you may need to reshape it to path,label)"
        why = "A CSV file exists, but it must be reviewed manually before being used as labels."
    else:
        suggestion = "none"
        why = ("No folder names match your configured normal/abnormal names and no usable label CSV was "
               "recognised. Use normal-only training + qualitative analysis, or create a labels CSV.")
    report["suggested_label_mode"] = {"mode": suggestion, "reason": why,
                                      "all_media_folder_names": sorted(leaf_names)[:60]}
    return report


def print_report(r: dict) -> None:
    line = "=" * 78
    print(line, "\nDATASET INSPECTION REPORT\n", line, sep="")
    print(f"Root           : {r['root']}")
    print(f"Total files    : {r['total_files']}   videos: {r['n_videos']}   images: {r['n_images']}")
    print(f"Extensions     : {r['extensions']}")
    print(f"Top-level items: {r['top_level_entries']}")
    print("\nFolders containing media (direct counts):")
    for d, c in list(r["media_directories"].items())[:60]:
        print(f"  {d:<55} {c}")
    if r["sample_video_properties"]:
        print("\nSample video properties:")
        for p in r["sample_video_properties"]:
            print("  ", p)
    if r["label_file_previews"]:
        print("\nCandidate label/annotation files (NOT automatically treated as labels):")
        for e in r["label_file_previews"]:
            print("  ", e)
    for w in r["warnings"]:
        print(f"\n[WARNING] {w}")
    s = r["suggested_label_mode"]
    print(f"\nSuggested data.label_mode: {s['mode']}\n  reason: {s['reason']}")
    print("\nNEXT: edit config.yaml (label_mode, class names, labels_csv) to match what you see above.")
    print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", default=None, help="Dataset folder (default: paths.dataset_root)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true", help="Inspect the SYNTHETIC test fixture")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, synthetic=args.synthetic)
    root = args.root or cfg["paths"]["dataset_root"]
    try:
        report = inspect_dataset(root, args.depth, cfg=cfg)
    except ProjectError as e:
        raise SystemExit(f"\n[ERROR] {e}")
    print_report(report)
    out = ensure_dir(cfg["paths"]["outputs_dir"]) / "dataset_report.json"
    save_json(report, out)
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
