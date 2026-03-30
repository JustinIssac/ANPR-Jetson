# Models

## best.onnx — Jetson Inference Model
ONNX export of the trained YOLOv8n (opset 11, IR version 6).  
Used by both `anpr_tesseract.py` and `anpr_easyocr.py` via ONNX Runtime.

## best_pt.zip — Original PyTorch Weights
PyTorch sharded checkpoint (~6MB). Required if re-exporting to ONNX.

**Re-export to ONNX (run on PC, Python 3.8+):**
```python
from ultralytics import YOLO
import zipfile, os
with zipfile.ZipFile("models/best_pt.zip", "r") as z:
    z.extractall("models/")
model = YOLO("models/best/data.pkl")  # or best.pt if you have it
model.export(format="onnx", opset=11, imgsz=640)
```

## Training Results
| Metric | Value |
|--------|-------|
| mAP@0.5 | 92.85% |
| mAP@0.5:0.95 | 56.20% |
| Precision | 93.45% |
| Recall | 84.14% |
| Epochs | 20 @ 640×640, batch 8 |
| Base model | YOLOv8n (COCO pre-trained, ~3.2M params) |
