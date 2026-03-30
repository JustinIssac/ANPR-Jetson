# Jetson Nano Setup Guide

**Platform:** NVIDIA Jetson Nano Developer Kit (4GB)  
**JetPack:** 4.6.1 (L4T R32.7.1) | **Python:** 3.6.9 | **CUDA:** 10.2

---

## 1. System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    libopenblas-base libopenblas-dev \
    libhdf5-serial-dev hdf5-tools \
    python3-pip python3-dev \
    tesseract-ocr libtesseract-dev
```

---

## 2. PyTorch — NVIDIA Pre-Built Wheel

```bash
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
     -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

---

## 3. torchvision — Build from Source (~20 min)

```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev \
    libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

---

## 4. NumPy — Build from Source

Pre-built wheels cause `Illegal instruction` on the Tegra CPU.

```bash
pip3 install numpy --no-binary numpy
```

---

## 5. Core Dependencies

```bash
pip3 install \
    Pillow==8.4.0 \
    onnxruntime==1.10.0 \
    pytesseract \
    rapidfuzz \
    python-Levenshtein
```

---

## 6. EasyOCR — With Patches

```bash
pip3 install easyocr==1.7.2 --no-deps
```

**Patch 1 — `weights_only` kwarg** (not supported in Python 3.6's `torch.load`):

Find the EasyOCR `utils.py` and remove `weights_only=True` from any `torch.load()` call.

```bash
# Find the file:
python3 -c "import easyocr; import os; print(os.path.dirname(easyocr.__file__))"
# Edit utils.py — change torch.load(..., weights_only=True) to torch.load(...)
```

**Patch 2 — `PIL.Image.Resampling`** (added in Pillow 9.1.0, unavailable in 8.4.0):

In the same EasyOCR directory, replace `Image.Resampling.LANCZOS` with `Image.LANCZOS`.

```bash
grep -rn "Resampling" ~/.local/lib/python3.6/site-packages/easyocr/
# Edit each occurrence
```

---

## 7. Camera Setup — Pi Camera v2 (CSI)

```bash
# Verify camera detected:
ls /dev/video*
nvgstcapture-1.0  # test capture

# GStreamer pipeline (used in both scripts):
# nvarguscamerasrc → NVMM → nvvidconv → BGR → appsink
```

> **Critical:** Do NOT call `cap.set()` after opening a GStreamer pipeline — it disrupts the stream. Both scripts guard against this with `if not args.gstreamer`.

---

## 8. Verify Installation

```bash
python3 -c "import torch; print(torch.__version__)"       # 1.10.0
python3 -c "import cv2; print(cv2.__version__)"            # 4.1.1
python3 -c "import onnxruntime; print(onnxruntime.__version__)"  # 1.10.0
python3 -c "import pytesseract; print('Tesseract OK')"
python3 -c "import easyocr; print('EasyOCR OK')"
```

---

## 9. Transfer Files from PC

```bash
scp models/best.onnx jetson@<IP>:~/anpr/
scp data/anpr_database.db jetson@<IP>:~/anpr/data/
```

---

## 10. Run

```bash
cd ~/anpr
python3 src/anpr_tesseract.py --gstreamer    # Pi Camera, fast
python3 src/anpr_easyocr.py --video demo.mp4 # Video file, accurate
```
