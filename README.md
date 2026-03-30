# 🚗 Automatic Number Plate Recognition System — Jetson Nano Deployment

**Real-Time Automatic Number Plate Recognition on Edge Hardware**  
Embedded Machine Vision

---

## Overview

An end-to-end Automatic Number Plate Recognition (ANPR) system deployed on an NVIDIA Jetson Nano (4GB). The system detects vehicle license plates from a live camera feed, extracts plate text using OCR, and cross-references results against a local SQLite database to make real-time access control decisions.

Each detected plate is classified as one of three states: **AUTHORIZED**, **FLAGGED**, or **UNKNOWN**.

---

## Pipeline

```
Camera / Video Input
       │
       ▼
YOLOv8n Plate Detection  ←── best.onnx (ONNX Runtime, CPUExecutionProvider)
       │
       ▼
Plate Region Cropping
       │
       ▼
OCR Text Extraction  ←── Tesseract (fast) or EasyOCR (accurate)
       │
       ▼
Text Cleaning  ←── O→0, strip symbols, uppercase
       │
       ▼
Multi-Frame Majority Voting  ←── deque(maxlen=10), Counter, resets after 15 miss frames
       │
       ▼
Database Lookup  ←── Exact match → rapidfuzz Levenshtein ≥90%
       │
       ▼
Access Decision  ──→  Visual Overlay (colour-coded boxes)
                  └──→  SQLite Logging (access_log table)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| YOLOv8n mAP@0.5 | **92.85%** |
| YOLOv8n mAP@0.5:0.95 | 56.20% |
| Precision | 93.45% |
| Recall | 84.14% |
| FPS — Tesseract | ~0.7 FPS |
| FPS — EasyOCR | ~0.015 FPS |
| Multi-frame voting stability | 80% (4/5 plates) |
| Total identification rate | 66% (with fuzzy matching) |

### OCR Engine Comparison

| Metric | Tesseract | EasyOCR |
|--------|-----------|---------|
| OCR Latency | 523.9 ms | 65,618.8 ms |
| FPS (Jetson Nano) | ~0.7 | ~0.015 |
| Accuracy (5 test plates) | 60% | 80% |
| Engine type | Traditional (LSTM) | Deep Learning (CRAFT+CRNN) |

Tesseract is ~125× faster at the cost of ~20% lower accuracy on this hardware.

---

## Repository Structure

```
anpr-jetson/
├── README.md
├── .gitignore
├── LICENSE
├── requirements_training.txt      # PC / Colab (Python 3.8+)
├── requirements_jetson.txt        # Jetson Nano (Python 3.6.9)
│
├── src/
│   ├── anpr_tesseract.py          # Main pipeline — Tesseract OCR (~0.7 FPS)
│   └── anpr_easyocr.py            # Main pipeline — EasyOCR (~0.015 FPS, more accurate)
│
├── models/
│   ├── best.onnx                  # ONNX export for Jetson inference
│   ├── best_pt.zip                # Original PyTorch weights
│   └── README.md
│
├── notebooks/
│   └── phase1_training.ipynb      # YOLOv8n training + OCR prototype (Phase 1)
│
├── data/
│   ├── anpr_database.db           # SQLite DB — registered plates + access log
│   └── README.md                  # Dataset download instructions
│
├── scripts/
│   └── show_db.sh                 # Print database contents to terminal
│
└── docs/
    └── jetson_setup.md            # Full Jetson Nano dependency installation guide
```

---

## Quick Start

### Run on Jetson Nano

```bash
git clone https://github.com/YOUR_USERNAME/anpr-jetson.git
cd anpr-jetson

# Tesseract (fast, ~0.7 FPS)
python3 src/anpr_tesseract.py --video your_video.mp4

# EasyOCR (accurate, ~0.015 FPS)
python3 src/anpr_easyocr.py --video your_video.mp4
```

**All run modes:**

```bash
python3 src/anpr_tesseract.py                   # live USB camera
python3 src/anpr_tesseract.py --gstreamer        # Pi Camera (CSI via GStreamer)
python3 src/anpr_tesseract.py --video demo.mp4   # video file
python3 src/anpr_tesseract.py --test             # static test images in test_images/
python3 src/anpr_tesseract.py --benchmark        # speed benchmark
```

**Live mode keys:**  `q` quit &nbsp;|&nbsp; `s` screenshot &nbsp;|&nbsp; `r` reset vote buffer

**View database:**
```bash
bash scripts/show_db.sh
```

---

## Hardware & Software Stack

### Hardware
- **Inference:** NVIDIA Jetson Nano Developer Kit (4GB RAM), JetPack 4.6.1 (L4T R32.7.1), CUDA 10.2
- **Camera:** Raspberry Pi Camera v2 (CSI ribbon cable) or USB webcam
- **Training:** Intel i5-1235U (CPU only)

### Software

| Component | Details |
|-----------|---------|
| Plate Detection | YOLOv8n fine-tuned on Kaggle ANPR dataset, ONNX opset 11 |
| Inference Engine | ONNX Runtime 1.10.0 — CPUExecutionProvider |
| OCR (fast) | Tesseract OCR — LSTM, `--psm 7`, alphanumeric whitelist |
| OCR (accurate) | EasyOCR 1.7.2 — CRAFT detection + CRNN recognition |
| Fuzzy Matching | rapidfuzz — Levenshtein distance, threshold ≥ 90% |
| Database | SQLite3 — `plates` + `access_log` tables |
| Display | OpenCV 4.1.1 — colour-coded bounding boxes + HUD |

---

## Setup

### Phase 1 — Training (PC / Colab)

```bash
pip install -r requirements_training.txt
# Open notebooks/phase1_training.ipynb and run all cells
```

**Export to ONNX** (run on PC before transferring to Jetson):

```python
from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="onnx", opset=11, imgsz=640)
```

### Phase 2 — Jetson Nano

See [`docs/jetson_setup.md`](docs/jetson_setup.md) for the complete installation guide covering PyTorch (pre-built wheel), torchvision (source build), NumPy (source build), EasyOCR patching, and GStreamer camera setup.

> **Why ONNX?** Ultralytics requires Python 3.8+, incompatible with JetPack 4.6.1's Python 3.6.9. ONNX export decouples training from inference.

---

## Database Schema

**`plates`** — registered vehicles (admin-configured):

| Column | Type | Notes |
|--------|------|-------|
| plate_text | TEXT PK | e.g. `KL01CA2555` |
| status | TEXT | `AUTHORIZED` / `FLAGGED` |
| owner_name | TEXT | e.g. `Blue Subaru` |

**`access_log`** — auto-written at runtime:

| Column | Type | Notes |
|--------|------|-------|
| plate_text | TEXT | OCR + voted result |
| status | TEXT | Access decision |
| match_type | TEXT | `exact` / `fuzzy` / `none` |
| confidence | REAL | YOLO detection confidence |
| timestamp | DATETIME | Auto-set |

---

## Notable Engineering Challenges

1. **Python 3.6 on JetPack 4.6.1** — manual version pinning, source builds, and monkey-patching across multiple libraries
2. **ONNX export pipeline** — bridges Python version gap between training (3.8+) and Jetson inference (3.6)
3. **Custom YOLO ONNX inference** — full pre/post-processing + NMS implemented without ultralytics
4. **Multi-frame voting** — ring buffer + Counter stabilized plate reads from 40% → 80% stability
5. **Fuzzy matching** — improved total identification from 53% → 66%
6. **GStreamer camera** — `cap.set()` disrupts the CSI pipeline; conditionally skipped

---

## Project Division

| Phase | Owner | Description |
|-------|-------|-------------|
| Phase 1: Model Training | Jescintha | Dataset prep, YOLOv8n training, ONNX export |
| Phase 2: Jetson Deployment | Justin Issac | Full pipeline, dependency resolution, OCR comparison, testing |

---

## License

MIT — see [LICENSE](LICENSE)

---

## Contact

**Justin Issac** — MSc Intelligent Robotics, University of Galway  
[GitHub](https://github.com/YOUR_USERNAME)
