"""Video inference: frames -> YOLO boxes + anomaly score -> alerts, annotated video, CSV logs, evidence frames.

    python -m src.video_processor --video path/to/cctv.mp4
    python -m src.video_processor --video clip.mp4 --no-yolo --threshold-multiplier 1.5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from src.signal_utils import causal_moving_average, heatmap_overlay
from src.preprocessing import VideoError, iter_video_frames, probe_video
from src.utils import ProjectError, ensure_dir, format_timestamp, get_logger, load_config, resolve_path, save_json

log = get_logger()


@dataclass
class ProcessingOptions:
    frame_stride: int = 5
    batch_size: int = 16
    smoothing_window: int | None = None      # None -> value stored with the calibrated threshold
    threshold_multiplier: float = 1.0
    min_event_frames: int = 2
    event_gap_seconds: float = 2.0
    use_yolo: bool = True
    yolo_every: int = 1
    max_seconds: float | None = None
    max_output_width: int = 960
    save_video: bool = True

    @classmethod
    def from_config(cls, cfg: dict, **over):
        i = cfg["inference"]
        base = dict(frame_stride=i["frame_stride"], batch_size=i["batch_size"], threshold_multiplier=i["threshold_multiplier"],
                    min_event_frames=i["min_event_frames"], event_gap_seconds=i["event_gap_seconds"],
                    max_output_width=i["max_output_width"], save_video=i["save_annotated_video"],
                    use_yolo=cfg["yolo"]["enabled"], yolo_every=cfg["yolo"]["every_n_frames"])
        base.update(over)
        return cls(**base)


@dataclass
class ProcessingResult:
    frame_log: pd.DataFrame
    events: pd.DataFrame
    summary: dict
    video_path: Path | None
    frame_csv: Path
    event_csv: Path
    evidence_dir: Path
    out_dir: Path = field(default=None)


def _overlay(frame, t, score, thr, flagged, ratio, n_people, score_available):
    h, w = frame.shape[:2]
    bar_h = 34
    cv2.rectangle(frame, (0, 0), (w, bar_h), (25, 25, 25), -1)
    status, color = ("ANOMALY" if flagged else "normal"), ((0, 0, 255) if flagged else (0, 200, 0))
    if not score_available:
        status, color = "YOLO only", (200, 200, 200)
    txt = f"{format_timestamp(t)}  |  {status}"
    if score_available:
        txt += f"  |  score {score:.4g} ({ratio:.2f}x thr)"
    if n_people >= 0:
        txt += f"  |  people {n_people}"
    cv2.putText(frame, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    if flagged:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    return frame


def _make_browser_playable(path: Path) -> None:
    """OpenCV's mp4v is not playable in most browsers; re-encode to H.264 if ffmpeg is installed."""
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found: annotated video may not play inside the browser (still downloadable). "
                    "Install: `brew install ffmpeg` (Mac) / already present on Colab.")
        return
    tmp = path.with_suffix(".h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vcodec", "libx264", "-pix_fmt",
                        "yuv420p", "-movflags", "+faststart", "-an", str(tmp)], capture_output=True)
    if r.returncode == 0 and tmp.exists():
        tmp.replace(path)
    else:
        log.warning("ffmpeg re-encode failed; keeping original file. %s", r.stderr.decode()[:200])


def process_video(video_path: str | Path, out_dir: str | Path, detector, yolo, opts: ProcessingOptions,
                  progress_cb: Callable[[float], None] | None = None) -> ProcessingResult:
    """Run the full pipeline on one video. `detector` (AnomalyDetector) may be None => YOLO-only mode."""
    video_path = resolve_path(video_path)
    info = probe_video(video_path)
    out_dir = ensure_dir(out_dir)
    ev_dir = ensure_dir(out_dir / "evidence")
    for old in ev_dir.glob("event_*.jpg"):
        old.unlink()
    if detector is None and (yolo is None or not opts.use_yolo):
        raise ProjectError("Nothing to run: no anomaly model and YOLO is disabled.")

    fps = info["fps"]
    total_est = info["frame_count"] if info["frame_count"] > 0 else None
    if total_est and opts.max_seconds:
        total_est = min(total_est, int(opts.max_seconds * fps))
    n_expected = max(1, (total_est or 1) // max(1, opts.frame_stride))
    scale = min(1.0, opts.max_output_width / max(1, info["width"]))
    out_w, out_h = int(info["width"] * scale) // 2 * 2, int(info["height"] * scale) // 2 * 2
    window = opts.smoothing_window if opts.smoothing_window is not None else (detector.smoothing_window if detector else 1)
    thr = detector.threshold * opts.threshold_multiplier if detector else float("nan")

    writer, vid_path = None, None
    if opts.save_video:
        vid_path = out_dir / "annotated_video.mp4"
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / opts.frame_stride), (out_w, out_h))
        if not writer.isOpened():
            log.warning("Could not open video writer; continuing without annotated video.")
            writer, vid_path = None, None

    records, events = [], []
    raw_hist: list[float] = []
    cur = None  # open event

    def finalize(ev):
        if ev["n_flag"] >= opts.min_event_frames:
            k = len(events) + 1
            p = ev_dir / f"event_{k:03d}_t{ev['peak_t']:.1f}s.jpg"
            cv2.imwrite(str(p), ev["img"], [cv2.IMWRITE_JPEG_QUALITY, 92])
            events.append({"event_id": k, "start_time_s": round(ev["start_t"], 2), "end_time_s": round(ev["end_t"], 2),
                           "start_ts": format_timestamp(ev["start_t"]), "end_ts": format_timestamp(ev["end_t"]),
                           "duration_s": round(ev["end_t"] - ev["start_t"], 2), "peak_score": ev["peak"],
                           "peak_ratio_to_threshold": ev["peak"] / thr, "peak_frame_idx": ev["peak_idx"],
                           "flagged_frames": ev["n_flag"], "max_people_detected": ev["max_people"],
                           "evidence_image": str(p)})

    def handle_batch(batch):
        nonlocal cur
        frames = [b[2] for b in batch]
        if detector is not None:
            raw, maps = detector.score_frames(frames, want_maps=True)
        else:
            raw, maps = np.zeros(len(frames)), None
        for j, (idx, t, frame) in enumerate(batch):
            raw_hist.append(float(raw[j]))
            sm = float(causal_moving_average(np.array(raw_hist[-window:]), window)[-1]) if detector is not None else 0.0
            flagged = bool(detector is not None and sm > thr)
            disp = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()
            dets, n_people, counts = [], -1, {}
            if yolo is not None and opts.use_yolo and (len(records) % max(1, opts.yolo_every) == 0):
                dets = yolo.detect(disp)
                n_people, counts = yolo.summarize(dets)
                yolo.draw(disp, dets)
            ratio = sm / thr if detector is not None else 0.0
            _overlay(disp, t, sm, thr, flagged, ratio, n_people, detector is not None)
            if writer is not None:
                writer.write(disp)
            records.append({"frame_idx": idx, "time_s": round(t, 3), "timestamp": format_timestamp(t), "raw_score": raw[j],
                            "smoothed_score": sm, "threshold": thr, "score_ratio": ratio, "is_anomaly": int(flagged),
                            "person_count": n_people, "objects": json.dumps(counts)})
            if flagged:
                if cur is None:
                    cur = {"start_t": t, "end_t": t, "peak": sm, "peak_t": t, "peak_idx": idx, "n_flag": 0, "img": None, "max_people": 0}
                cur["end_t"] = t
                cur["n_flag"] += 1
                cur["max_people"] = max(cur["max_people"], n_people)
                if sm >= cur["peak"] or cur["img"] is None:
                    heat = heatmap_overlay(disp, maps[j])
                    cv2.putText(heat, "reconstruction-error heatmap (relative)", (8, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cur.update(peak=sm, peak_t=t, peak_idx=idx, img=np.hstack([disp, heat]))
            elif cur is not None and t - cur["end_t"] > opts.event_gap_seconds:
                finalize(cur)
                cur = None

    batch, n_done = [], 0
    try:
        for item in iter_video_frames(video_path, opts.frame_stride, None, opts.max_seconds):
            batch.append(item)
            if len(batch) >= opts.batch_size:
                handle_batch(batch)
                n_done += len(batch)
                batch = []
                if progress_cb:
                    progress_cb(min(1.0, n_done / n_expected))
        if batch:
            handle_batch(batch)
            n_done += len(batch)
        if cur is not None:
            finalize(cur)
    finally:
        if writer is not None:
            writer.release()
    if not records:
        raise VideoError("No frames were processed (video empty, corrupt, or shorter than the frame stride).")
    if vid_path is not None:
        _make_browser_playable(vid_path)
    if progress_cb:
        progress_cb(1.0)

    flog = pd.DataFrame(records)
    ev_cols = ["event_id", "start_time_s", "end_time_s", "start_ts", "end_ts", "duration_s", "peak_score",
               "peak_ratio_to_threshold", "peak_frame_idx", "flagged_frames", "max_people_detected", "evidence_image"]
    ev_df = pd.DataFrame(events, columns=ev_cols)
    frame_csv, event_csv = out_dir / "frame_log.csv", out_dir / "event_log.csv"
    flog.to_csv(frame_csv, index=False)
    ev_df.to_csv(event_csv, index=False)
    summary = {"video": video_path.name, "processed_frames": len(flog), "frame_stride": opts.frame_stride, "fps": fps,
               "duration_processed_s": float(flog["time_s"].iloc[-1]), "anomaly_scoring": detector is not None,
               "threshold_effective": thr, "threshold_multiplier": opts.threshold_multiplier, "n_events": len(ev_df),
               "flagged_frame_fraction": float(flog["is_anomaly"].mean()), "max_score": float(flog["smoothed_score"].max()),
               "max_people": int(flog["person_count"].max()), "yolo_used": bool(yolo is not None and opts.use_yolo),
               "processed_at": datetime.now().isoformat(timespec="seconds"),
               "caveat": "Alerts indicate frames that reconstruct poorly vs. the learned normal appearance - "
                         "they are candidates for human review, not confirmed incidents."}
    save_json(summary, out_dir / "summary.json")
    return ProcessingResult(flog, ev_df, summary, vid_path, frame_csv, event_csv, ev_dir, out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default=None, help="default: <outputs_dir>/inference/<video name>")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-anomaly", action="store_true", help="YOLO-only (no autoencoder needed)")
    ap.add_argument("--threshold-multiplier", type=float)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, synthetic=args.synthetic)
    from src.utils import get_device
    device = get_device(args.device or cfg["training"]["device"])
    over = {}
    if args.no_yolo:
        over["use_yolo"] = False
    if args.threshold_multiplier:
        over["threshold_multiplier"] = args.threshold_multiplier
    if args.stride:
        over["frame_stride"] = args.stride
    over["max_seconds"] = args.max_seconds or cfg["app"]["max_video_seconds"]
    opts = ProcessingOptions.from_config(cfg, **over)
    detector = None
    if not args.no_anomaly:
        from src.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(cfg["paths"]["models_dir"], device)
    yolo = None
    if opts.use_yolo:
        from src.yolo_detector import YoloDetector
        y = cfg["yolo"]
        yolo = YoloDetector(y["weights"], y["conf"], y["imgsz"], y["classes"], device)
    out = args.out_dir or (Path(cfg["paths"]["outputs_dir"]) / "inference" / Path(args.video).stem)
    res = process_video(args.video, out, detector, yolo, opts, progress_cb=lambda f: print(f"\rprogress {f*100:5.1f}%", end=""))
    print()
    print(json.dumps(res.summary, indent=2))
    print(f"Outputs in: {res.out_dir}\n  annotated video: {res.video_path}\n  frame log: {res.frame_csv}\n"
          f"  event log: {res.event_csv}\n  evidence: {res.evidence_dir}")


if __name__ == "__main__":
    main()
