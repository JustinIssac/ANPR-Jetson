"""
anpr_jetson.py — Main ANPR deployment script for Jetson Nano

TODO: Copy this file from the Jetson Nano and remove this placeholder.

Architecture:
    - YOLOv8n plate detection via ONNX Runtime (CPUExecutionProvider)
    - EasyOCR or Tesseract OCR (switchable via --ocr flag)
    - Text cleaning: O→0, strip non-alphanumeric, uppercase
    - Multi-frame majority voting: deque(maxlen=10), Counter-based, resets after 15 frames no detection
    - SQLite database lookup: exact match first, then rapidfuzz Levenshtein ≥90%
    - SQLite tables:
        plates(plate_text PK, status, owner_name)
        access_log(plate_text, status, match_type, confidence, timestamp)
    - Visual overlay: green=AUTHORIZED, red=FLAGGED, yellow=UNKNOWN
    - Info bar: FPS, detection count, timestamp
    - Input: Pi Camera (GStreamer nvarguscamerasrc) or video file

Usage:
    python3 anpr_jetson.py --source 0            # Pi Camera
    python3 anpr_jetson.py --source video.mp4    # Video file
    python3 anpr_jetson.py --ocr tesseract       # Use Tesseract (faster, default)
    python3 anpr_jetson.py --ocr easyocr         # Use EasyOCR (more accurate)
"""

raise NotImplementedError(
    "Jetson deployment script not yet uploaded. "
    "Copy anpr_jetson.py from the Jetson Nano to this directory."
)
