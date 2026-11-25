"""Streamlit dashboard - AI CCTV Anomaly Detection (YOLO + CNN autoencoder).

Run:  streamlit run app.py
The trained autoencoder and threshold are loaded from disk (models/); nothing is retrained on startup.
"""
from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.preprocessing import VideoError, probe_video
from src.utils import ProjectError, ensure_dir, format_timestamp, get_device, load_config, load_json, resolve_path
from src.video_processor import ProcessingOptions, process_video

st.set_page_config(page_title="AI CCTV Anomaly Detection", page_icon="🎥", layout="wide")
CFG = load_config("config.yaml")
MODELS_DIR = resolve_path(CFG["paths"]["models_dir"])
OUT_DIR = resolve_path(CFG["paths"]["outputs_dir"])


# ----------------------------------------------------------------------------- cached model loading
@st.cache_resource(show_spinner="Loading anomaly model...")
def load_detector(models_dir: str):
    from src.anomaly_detector import AnomalyDetector
    return AnomalyDetector(models_dir, get_device("auto"))


@st.cache_resource(show_spinner="Loading YOLO (downloads weights on first use - internet needed once)...")
def load_yolo(weights: str, conf: float, imgsz: int, classes: tuple | None):
    from src.yolo_detector import YoloDetector
    return YoloDetector(weights, conf, imgsz, list(classes) if classes else None, get_device("auto"))


def model_status():
    ok = (MODELS_DIR / "autoencoder.pt").exists() and (MODELS_DIR / "threshold.json").exists()
    info = load_json(MODELS_DIR / "threshold.json") if (MODELS_DIR / "threshold.json").exists() else None
    return ok, info


# ----------------------------------------------------------------------------- sidebar
ok_model, thr_info = model_status()
st.sidebar.title("⚙️ Controls")
st.sidebar.caption(f"Device: **{get_device('auto')}**")
if ok_model:
    st.sidebar.success(f"Autoencoder ready · calibrated threshold = {thr_info['threshold']:.4g} ({thr_info['method']})")
    if thr_info.get("synthetic_fixture"):
        st.sidebar.warning("This model was trained on the SYNTHETIC test fixture - not a real result.")
else:
    st.sidebar.error("No trained model found in `models/`. Train + calibrate first (see README). "
                     "You can still run YOLO-only mode.")

use_ae = st.sidebar.checkbox("Anomaly detection (autoencoder)", value=ok_model, disabled=not ok_model)
use_yolo = st.sidebar.checkbox("YOLO object detection", value=CFG["yolo"]["enabled"])
st.sidebar.subheader("Anomaly alerting")
mult = st.sidebar.slider("Threshold multiplier", 0.5, 3.0, float(CFG["inference"]["threshold_multiplier"]), 0.05,
                         help="Effective threshold = calibrated threshold x this. Higher = fewer alerts.")
if ok_model:
    st.sidebar.caption(f"Effective threshold: **{thr_info['threshold'] * mult:.4g}**")
stride = st.sidebar.slider("Frame stride (analyse every Nth frame)", 1, 30, int(CFG["inference"]["frame_stride"]),
                           help="Keep equal to training stride for calibrated behaviour.")
smooth = st.sidebar.slider("Smoothing window (frames)", 1, 15, int(thr_info["smoothing_window"]) if ok_model else 5)
min_ev = st.sidebar.slider("Min flagged frames per event", 1, 20, int(CFG["inference"]["min_event_frames"]))
gap = st.sidebar.slider("Merge events closer than (s)", 0.0, 10.0, float(CFG["inference"]["event_gap_seconds"]), 0.5)
st.sidebar.subheader("YOLO")
yconf = st.sidebar.slider("Confidence", 0.05, 0.95, float(CFG["yolo"]["conf"]), 0.05)
people_only = st.sidebar.checkbox("People only", value=(CFG["yolo"]["classes"] == [0]))
st.sidebar.subheader("Limits")
max_s = st.sidebar.slider("Max seconds to process", 5, int(CFG["app"]["max_video_seconds"]), int(CFG["app"]["max_video_seconds"]))
save_video = st.sidebar.checkbox("Save annotated video", value=True)

# ----------------------------------------------------------------------------- header
st.title("🎥 AI CCTV Anomaly Detection")
st.caption("YOLO scene context + CNN autoencoder reconstruction error. Alerts are **candidates for human review**, "
           "not confirmed incidents.")
tab_run, tab_alerts, tab_log, tab_eval, tab_about = st.tabs(["▶ Analyse video", "🚨 Alerts", "📄 Frame log", "📊 Evaluation", "ℹ️ About & limits"])


def save_upload(upload) -> Path:
    tmp_dir = ensure_dir(CFG["paths"].get("temp_dir") or Path(tempfile.gettempdir()) / "cctv_uploads")
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as f:
        f.write(upload.getbuffer())
        return Path(f.name)


# ----------------------------------------------------------------------------- analyse
with tab_run:
    up = st.file_uploader("Upload a CCTV / surveillance video", type=CFG["app"]["allowed_extensions"])
    if up is not None:
        size_mb = up.size / 1024 / 1024
        if size_mb > CFG["app"]["max_upload_mb"]:
            st.error(f"File is {size_mb:.0f} MB; the limit is {CFG['app']['max_upload_mb']} MB (config.yaml: app.max_upload_mb).")
            up = None
    if up is not None:
        tmp_path = save_upload(up)
        try:
            info = probe_video(tmp_path)
            c = st.columns(4)
            c[0].metric("Resolution", f"{info['width']}×{info['height']}")
            c[1].metric("FPS", f"{info['fps']:.1f}")
            c[2].metric("Duration", f"{info['duration_s']:.1f}s" if info["duration_s"] else "unknown")
            c[3].metric("Will process", f"{min(max_s, info['duration_s'] or max_s):.0f}s")
            if info["duration_s"] and info["duration_s"] > max_s:
                st.warning(f"Only the first {max_s}s will be analysed.")
            st.video(up.getvalue())
            if st.button("🔍 Run analysis", type="primary", disabled=not (use_ae or use_yolo)):
                try:
                    detector = load_detector(str(MODELS_DIR)) if use_ae else None
                    yolo = load_yolo(CFG["yolo"]["weights"], yconf, CFG["yolo"]["imgsz"], (0,) if people_only else None) if use_yolo else None
                    opts = ProcessingOptions.from_config(CFG, frame_stride=stride, smoothing_window=smooth, threshold_multiplier=mult,
                                                         min_event_frames=min_ev, event_gap_seconds=gap, use_yolo=use_yolo,
                                                         max_seconds=float(max_s), save_video=save_video)
                    out = OUT_DIR / "dashboard" / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(up.name).stem}"
                    bar, t0 = st.progress(0.0, text="Processing..."), time.time()
                    res = process_video(tmp_path, out, detector, yolo, opts,
                                        progress_cb=lambda f: bar.progress(f, text=f"Processing... {f*100:.0f}%"))
                    bar.empty()
                    st.session_state["result"] = res
                    st.session_state["elapsed"] = time.time() - t0
                except (ProjectError, VideoError, ImportError) as e:
                    st.error(f"{e}")
        except (ProjectError, VideoError) as e:
            st.error(f"Cannot read this video: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)  # uploaded temp file never lingers

    res = st.session_state.get("result")
    if res is not None:
        s = res.summary
        st.divider()
        if s["anomaly_scoring"]:
            if s["n_events"]:
                st.error(f"🚨 {s['n_events']} potential anomaly event(s) detected - review the Alerts tab.")
            else:
                st.success("✅ No anomaly events at the current threshold.")
        else:
            st.info("YOLO-only mode: no anomaly scoring was performed.")
        m = st.columns(5)
        m[0].metric("Frames analysed", s["processed_frames"])
        m[1].metric("Events", s["n_events"])
        m[2].metric("Flagged frames", f"{s['flagged_frame_fraction']*100:.1f}%")
        m[3].metric("Max people (YOLO)", s["max_people"] if s["yolo_used"] else "off")
        m[4].metric("Processing time", f"{st.session_state.get('elapsed', 0):.0f}s")

        fl = res.frame_log
        if s["anomaly_scoring"]:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(fl["time_s"], fl["smoothed_score"], lw=1.4, label="anomaly score (smoothed)")
            ax.axhline(s["threshold_effective"], color="r", ls="--", label="threshold")
            ax.fill_between(fl["time_s"], 0, fl["smoothed_score"].max(), where=fl["is_anomaly"] == 1, color="red", alpha=.15, label="flagged")
            ax.set_xlabel("time (s)"); ax.set_ylabel("reconstruction error"); ax.legend(loc="upper right"); ax.grid(alpha=.3)
            st.pyplot(fig); plt.close(fig)
        if res.video_path and Path(res.video_path).exists():
            st.subheader("Annotated video")
            st.video(Path(res.video_path).read_bytes())
            st.download_button("⬇ Download annotated video", Path(res.video_path).read_bytes(), "annotated_video.mp4", "video/mp4")
        d = st.columns(2)
        d[0].download_button("⬇ Event log (CSV)", res.event_csv.read_bytes(), "event_log.csv", "text/csv")
        d[1].download_button("⬇ Frame log (CSV)", res.frame_csv.read_bytes(), "frame_log.csv", "text/csv")
        st.caption(f"All outputs saved in `{res.out_dir}`")

with tab_alerts:
    res = st.session_state.get("result")
    if res is None:
        st.info("Run an analysis first.")
    elif res.events.empty:
        st.success("No alerts for this run.")
    else:
        st.dataframe(res.events.drop(columns=["evidence_image"]), use_container_width=True, hide_index=True)
        for _, ev in res.events.iterrows():
            with st.expander(f"🚨 Event {int(ev.event_id)}  ·  {ev.start_ts} → {ev.end_ts}  ·  peak {ev.peak_score:.4g} ({ev.peak_ratio_to_threshold:.2f}× threshold)", expanded=int(ev.event_id) == 1):
                st.image(ev.evidence_image, caption="Left: peak frame with YOLO boxes · Right: reconstruction-error heatmap (where the model was most surprised)")
                st.caption(f"Flagged frames: {int(ev.flagged_frames)} · max people: {int(ev.max_people_detected)} · frame index {int(ev.peak_frame_idx)}")

with tab_log:
    res = st.session_state.get("result")
    if res is None:
        st.info("Run an analysis first.")
    else:
        st.dataframe(res.frame_log, use_container_width=True, height=420)

with tab_eval:
    ev_dir = OUT_DIR / "evaluation"
    st.write("Results of `python -m src.evaluate` on the held-out **test split** (threshold calibrated on validation only).")
    if (ev_dir / "metrics.json").exists():
        mj = load_json(ev_dir / "metrics.json")
        if mj.get("synthetic_fixture"):
            st.warning("These numbers come from the SYNTHETIC fixture - software test only.")
        if "f1" in mj:
            c = st.columns(6)
            for col, k in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
                col.metric(k.upper().replace("_", "-"), f"{mj[k]:.3f}")
        st.info(mj.get("note", ""))
        for img in ["confusion_matrix.png", "roc_pr_curves.png", "score_distribution.png", "highest_scoring_frames.png"]:
            if (ev_dir / img).exists():
                st.image(str(ev_dir / img), caption=img)
        with st.expander("Raw metrics.json"):
            st.json(mj)
    else:
        st.info("No evaluation results yet. Run `python -m src.evaluate`.")
    if (OUT_DIR / "training_curve.png").exists():
        st.image(str(OUT_DIR / "training_curve.png"), caption="Training curve")

with tab_about:
    st.markdown("""
**Pipeline:** video → OpenCV frame extraction → resize/normalise → YOLO (people/objects) + CNN autoencoder → reconstruction-error score → smoothing → threshold → alert + evidence.

**What an alert means:** the frame looked *unlike the normal footage the autoencoder was trained on*. That can be a genuine unusual event, or just a lighting change, camera shake or a rarely-seen but harmless activity.

**What YOLO does here:** counts people and detects COCO objects as context. Stock YOLO does **not** recognise fighting, falling, cheating or violence.

**Limits:** frame-level model (no motion modelling beyond score smoothing); threshold depends on calibration data; results on other cameras/scenes may differ. Treat output as decision support for a human operator.
""")
