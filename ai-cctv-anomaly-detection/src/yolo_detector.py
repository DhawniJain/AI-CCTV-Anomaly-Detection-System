"""Pretrained YOLO (Ultralytics) person/object detector - a COMPLEMENTARY component.

IMPORTANT: stock COCO-pretrained YOLO detects objects (person, cell phone, laptop, ...). It does NOT
recognise fighting, falling, cheating or any "abnormal behaviour". Here it only provides scene context
(people count, objects present) next to the autoencoder's anomaly score.
Weights are downloaded automatically on first use (internet needed ONCE; afterwards cached).
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from src.utils import ProjectError, get_device, get_logger, yolo_device_arg

log = get_logger()


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640, classes=None, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ProjectError("Ultralytics is not installed. Run: pip install ultralytics") from e
        try:
            self.model = YOLO(weights)  # auto-downloads known weights such as yolov8n.pt
        except Exception as e:
            raise ProjectError(f"Could not load YOLO weights '{weights}'. If this is the first run, an internet "
                               f"connection is required to download them. Original error: {e}") from e
        self.conf, self.imgsz, self.classes = conf, imgsz, classes
        self.device = yolo_device_arg(device or get_device("auto"))
        self.names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """-> [{label, conf, box:(x1,y1,x2,y2)}]"""
        try:
            res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, classes=self.classes,
                                     device=self.device, verbose=False)[0]
        except Exception as e:
            if self.device == "cpu":
                raise
            log.warning("YOLO failed on device '%s' (%s). Falling back to CPU.", self.device, str(e)[:120])
            self.device = "cpu"
            res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, classes=self.classes,
                                     device="cpu", verbose=False)[0]
        out = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                out.append({"label": str(self.names[int(k)]), "conf": float(c), "box": tuple(int(v) for v in b)})
        return out

    @staticmethod
    def draw(frame_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            color = (0, 200, 0) if d["label"] == "person" else (255, 160, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            txt = f"{d['label']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame_bgr, txt, (x1 + 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame_bgr

    @staticmethod
    def summarize(dets: list[dict]) -> tuple[int, dict]:
        counts = Counter(d["label"] for d in dets)
        return counts.get("person", 0), dict(counts)
