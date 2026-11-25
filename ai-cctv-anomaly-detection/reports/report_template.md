# AI CCTV Anomaly Detection System Using YOLO, CNN Autoencoder and Streamlit
**Project report - aligned with Milestone 2 (Project Design)**
Name: ________  Roll No: ________  Guide: ________  Date: ________

> Rule: every number below must come from files produced by *your own* run
> (`outputs/...`). Replace each `[FILL: ...]`. Do not report anything from `outputs/smoke/` (synthetic fixture).

## 1. Introduction
Problem (manual CCTV monitoring is tiring and error-prone), proposed solution (Milestone 2 §1), scope. [FILL]

## 2. Objectives (Milestone 2 §2)
| Objective | How it is implemented | Module |
|---|---|---|
| Video monitoring | Uploaded/recorded video read with OpenCV | `preprocessing.py`, `video_processor.py` |
| Person/object detection | Pretrained YOLO via Ultralytics | `yolo_detector.py` |
| Normal activity learning | CNN autoencoder trained on normal frames only | `autoencoder.py`, `train.py` |
| Anomaly detection | Reconstruction-error score + smoothing | `anomaly_detector.py` |
| Alert generation | Threshold → events with timestamps + evidence frames | `video_processor.py` |
| Performance evaluation | Metrics when labels exist, error analysis always | `evaluate.py` |

## 3. Methodology (Milestone 2 §3)
Insert the workflow figure (see README Mermaid diagram) and describe: frame extraction (stride [FILL]), preprocessing (size [FILL], RGB, [0,1]), YOLO, autoencoder (layers, latent size, parameters [FILL from train log]), score (`mse`/`topk`), smoothing window, threshold method.

## 4. Relevance of algorithms and techniques (Milestone 2 §4)
Computer vision · YOLO · CNN features · Autoencoder reconstruction · Anomaly score · Thresholding. State honestly what each contributes; YOLO is contextual and does not itself identify abnormal behaviour.

## 5. Dataset and training setup (Milestone 2 §5)
- Dataset: CCTV Exam Monitor Dataset (Kaggle: cctvdataset/cctv-exam-monitor-dataset); licence/citation: [FILL from the Kaggle page]
- Contents found by the inspector: [FILL from `outputs/dataset_report.json` - files, formats, label structure]
- Label situation (labelled normal/abnormal? normal-only? none?): [FILL]
- Split strategy and counts: [FILL from `outputs/split_info.json`]
- Hardware (Colab GPU / M2 MPS), epochs, batch size, learning rate: [FILL]

## 6. Technology stack (Milestone 2 §6)
Python, PyTorch, Ultralytics YOLO, OpenCV, NumPy/Pandas, scikit-learn, Matplotlib, Streamlit. Versions: [FILL from `pip freeze`]

## 7. Results
### 7.1 Training - insert `training_curve.png`, `reconstruction_samples.png`. Best val MSE: [FILL]
### 7.2 Threshold - insert `threshold_calibration.png`. Method: [FILL] Value: [FILL]
### 7.3 Test evaluation
*If labelled test data exists:*
| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

Insert `confusion_matrix.png`, `roc_pr_curves.png`. Report abnormal prevalence: [FILL] (so accuracy can be interpreted).
*If no labels:* state that supervised metrics were not computed and why; include `score_distribution.png` and `highest_scoring_frames.png` and discuss qualitatively.
### 7.4 Demonstration - dashboard screenshots, an annotated video, evidence frames, `event_log.csv` excerpt. [FILL]
### 7.5 Error analysis - false alarms: [FILL]; missed events: [FILL]; patterns (lighting, crowding, camera shake)? [FILL]

## 8. Limitations
Reconstruction error ≠ semantic abnormality · frame-level (no true temporal model) · dependence on the normal training data and the calibration percentile · YOLO is not a behaviour classifier · dataset-specific (generalisation to other cameras untested) · [FILL anything you observed].

## 9. Deviations from Milestone 2
List from README "Deviations" and justify. [FILL]

## 10. Conclusion and future scope
Milestone 2 future scope: real-time cameras, CNN-LSTM/Transformer temporal models, multi-camera, false-alarm reduction, edge deployment. [FILL]

## References
Dataset citation · Ultralytics YOLO · PyTorch · OpenCV · Streamlit · papers on reconstruction-based video anomaly detection you actually read. [FILL]
