"""
ANPR Real-Time Pipeline — Jetson Nano (FULL DEMO VERSION)
==========================================================
Camera -> YOLOv8n (ONNX) -> Crop -> EasyOCR -> Clean -> Vote -> Fuzzy Match -> SQLite -> Display

Usage:
    python3 anpr_full.py                  # live camera
    python3 anpr_full.py --test           # test on static images
    python3 anpr_full.py --benchmark      # benchmark speed
    python3 anpr_full.py --gstreamer      # use CSI Pi Camera via GStreamer

Keys (live mode):
    q  - quit
    s  - screenshot
    r  - reset voting buffer
"""

import cv2
import pytesseract
import numpy as np
import onnxruntime as ort
import os
import sys
import time
import sqlite3
import argparse
import datetime
from collections import Counter, deque

try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    print("[WARN] rapidfuzz not installed. Fuzzy matching disabled.")
    FUZZY_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "best.onnx"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640
VOTING_BUFFER_SIZE = 10
FUZZY_THRESHOLD = 90
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
DB_PATH = "anpr_database.db"

GSTREAMER_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

STATUS_COLORS = {
    "AUTHORIZED": (0, 200, 0),
    "FLAGGED":    (0, 0, 200),
    "UNKNOWN":    (0, 200, 200),
}


# =============================================================================
# YOLOv8 ONNX INFERENCE (replaces ultralytics)
# =============================================================================
class YOLODetector:
    """YOLOv8 detector using ONNX Runtime — no ultralytics needed."""

    def __init__(self, model_path, conf_threshold=0.5, img_size=640):
        self.conf_threshold = conf_threshold
        self.img_size = img_size

        print("[YOLO] Loading ONNX model: {}".format(model_path))
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        active = self.session.get_providers()
        print("[YOLO] Running on: {}".format(active[0]))

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        """Resize, pad, normalize image for YOLOv8 input."""
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        return blob, scale, pad_w, pad_h

    def postprocess(self, output, scale, pad_w, pad_h, orig_h, orig_w):
        """Parse YOLOv8 output: shape (1, 5, 8400) -> list of (x1,y1,x2,y2,conf)."""
        predictions = output[0].T  # (8400, 5)

        scores = predictions[:, 4]
        mask = scores > self.conf_threshold
        filtered = predictions[mask]

        if len(filtered) == 0:
            return []

        boxes = []
        for det in filtered:
            cx, cy, w, h, conf = det[:5]
            x1 = (cx - w / 2 - pad_w) / scale
            y1 = (cy - h / 2 - pad_h) / scale
            x2 = (cx + w / 2 - pad_w) / scale
            y2 = (cy + h / 2 - pad_h) / scale

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(orig_w, int(x2))
            y2 = min(orig_h, int(y2))

            boxes.append((x1, y1, x2, y2, float(conf)))

        if len(boxes) > 1:
            boxes = self._nms(boxes, 0.5)

        return boxes

    def _nms(self, boxes, iou_threshold=0.5):
        """Simple non-maximum suppression."""
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < iou_threshold]
        return keep

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def detect(self, img):
        """Run detection on an image. Returns list of (x1,y1,x2,y2,conf)."""
        h, w = img.shape[:2]
        blob, scale, pad_w, pad_h = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs[0], scale, pad_w, pad_h, h, w)


# =============================================================================
# DATABASE — SQLite whitelist/blacklist + logging
# =============================================================================
class PlateDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._seed_data()

    def _create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                plate_text TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                owner_name TEXT DEFAULT 'Unknown'
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                status TEXT NOT NULL,
                match_type TEXT DEFAULT 'exact',
                confidence REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _seed_data(self):
        self.cursor.execute("SELECT COUNT(*) FROM plates")
        if self.cursor.fetchone()[0] == 0:
            sample_plates = [
                ("KL01CA2555", "AUTHORIZED", "Test Vehicle 1"),
                ("MH12AB1234", "FLAGGED",    "Flagged Vehicle"),
                ("PGMN112",    "AUTHORIZED", "Montenegro Car"),
                ("PRENUP",     "AUTHORIZED", "Camaro"),
                ("DZ17YXR",    "AUTHORIZED", "Blue Subaru"),
                ("PU18BES",    "AUTHORIZED", "Peugeot"),
            ]
            self.cursor.executemany(
                "INSERT OR IGNORE INTO plates (plate_text, status, owner_name) VALUES (?, ?, ?)",
                sample_plates
            )
            self.conn.commit()
            print("[DB] Seeded {} sample plates.".format(len(sample_plates)))

    def lookup(self, plate_text):
        self.cursor.execute(
            "SELECT status FROM plates WHERE plate_text = ?", (plate_text,)
        )
        row = self.cursor.fetchone()
        if row:
            return row[0], "exact", plate_text, 100.0

        if FUZZY_AVAILABLE:
            self.cursor.execute("SELECT plate_text, status FROM plates")
            all_plates = self.cursor.fetchall()
            if all_plates:
                plate_texts = [p[0] for p in all_plates]
                result = process.extractOne(
                    plate_text, plate_texts,
                    scorer=fuzz.ratio,
                    score_cutoff=FUZZY_THRESHOLD
                )
                if result:
                    matched_text, score, _ = result
                    for pt, st in all_plates:
                        if pt == matched_text:
                            return st, "fuzzy", matched_text, score

        return "UNKNOWN", "none", plate_text, 0.0

    def log_access(self, plate_text, status, match_type="exact", confidence=0.0):
        self.cursor.execute(
            "INSERT INTO access_log (plate_text, status, match_type, confidence) VALUES (?, ?, ?, ?)",
            (plate_text, status, match_type, confidence)
        )
        self.conn.commit()

    def get_recent_logs(self, limit=20):
        self.cursor.execute(
            "SELECT plate_text, status, match_type, confidence, timestamp "
            "FROM access_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()


# =============================================================================
# TEXT CLEANING
# =============================================================================
def clean_plate_text(raw_text):
    text = raw_text.strip().upper()
    text = text.replace("O", "0")
    cleaned = ""
    for ch in text:
        if ch.isalnum() or ch == " ":
            cleaned += ch
    return cleaned


# =============================================================================
# MULTI-FRAME VOTING
# =============================================================================
class PlateVoter:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, plate_text):
        if plate_text:
            self.buffer.append(plate_text)

    def get_voted_plate(self):
        if not self.buffer:
            return ""
        return Counter(self.buffer).most_common(1)[0][0]

    def reset(self):
        self.buffer.clear()


# =============================================================================
# OCR — Tesseract (fast, lightweight)
# =============================================================================
def ocr_plate(plate_img, config):
    """
    Run Tesseract OCR on a cropped plate image.
    Returns raw text string, or empty string if nothing detected.
    """
    # Preprocess for better OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Resize to standard height for better recognition
    h, w = gray.shape
    if h < 40:
        scale = 40.0 / h
        gray = cv2.resize(gray, (int(w * scale), 40))
    # Threshold to get clean black text on white background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run Tesseract
    text = pytesseract.image_to_string(thresh, config=config).strip()
    return text


# =============================================================================
# OVERLAY
# =============================================================================
def draw_overlay(frame, x1, y1, x2, y2, plate_text, status, match_type="", conf=0.0):
    color = STATUS_COLORS.get(status, (0, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = plate_text
    if match_type == "fuzzy":
        label += " (fuzzy)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
    label_y = max(y1 - 10, th + 10)
    cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4), color, -1)
    cv2.putText(frame, label, (x1 + 4, label_y - 2), font, 0.7, (255, 255, 255), 2)

    (sw, sh), _ = cv2.getTextSize(status, font, 0.6, 2)
    cv2.rectangle(frame, (x1, y2 + 2), (x1 + sw + 8, y2 + sh + 14), color, -1)
    cv2.putText(frame, status, (x1 + 4, y2 + sh + 8), font, 0.6, (255, 255, 255), 2)


def draw_info_bar(frame, fps, total_detections):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Detections: {}".format(total_detections), (150, 25), font, 0.6, (255, 255, 255), 2)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w - 100, 25), font, 0.6, (255, 255, 255), 2)


# =============================================================================
# FPS COUNTER
# =============================================================================
class FPSCounter:
    def __init__(self, avg_over=30):
        self.times = deque(maxlen=avg_over)

    def tick(self):
        self.times.append(time.time())

    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


# =============================================================================
# TEST MODE
# =============================================================================
def run_test_mode(detector, reader, db):
    test_dirs = ["test_images", "."]
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    test_images = []

    for d in test_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(image_exts):
                    test_images.append(os.path.join(d, f))

    if not test_images:
        print("[ERROR] No test images found. Put images in test_images/ folder.")
        return

    print("[TEST] Found {} images.".format(len(test_images)))

    for img_path in test_images:
        print("\n--- {} ---".format(img_path))
        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = detector.detect(img)
        for (x1, y1, x2, y2, conf) in boxes:
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.shape[0] < 15 or plate_crop.shape[1] < 40:
                continue

            raw = ocr_plate(plate_crop, reader)
            if raw:
                cleaned = clean_plate_text(raw)
                status, match_type, matched, similarity = db.lookup(cleaned)

                print("  Raw OCR:    {}".format(raw))
                print("  Cleaned:    {}".format(cleaned))
                print("  Status:     {} ({}, matched: {}, sim: {:.0f}%)".format(
                    status, match_type, matched, similarity))
                print("  YOLO conf:  {:.2f}".format(conf))

                draw_overlay(img, x1, y1, x2, y2, cleaned, status, match_type, conf)
                db.log_access(cleaned, status, match_type, conf)

        cv2.imshow("ANPR Test", img)
        print("  Press any key for next...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("ACCESS LOG:")
    for log in db.get_recent_logs(50):
        print("  {} | {} | {} | conf={:.2f} | {}".format(*log))


# =============================================================================
# BENCHMARK MODE
# =============================================================================
def run_benchmark(detector, reader, db):
    test_dirs = ["test_images", "."]
    image_exts = (".jpg", ".jpeg", ".png", ".bmp")
    test_images = []

    for d in test_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(image_exts):
                    test_images.append(os.path.join(d, f))

    if not test_images:
        print("[ERROR] No test images found.")
        return

    print("[BENCHMARK] {} images".format(len(test_images)))
    yolo_times, ocr_times, total_times = [], [], []
    detections, ocr_ok = 0, 0

    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.time()
        t_y = time.time()
        boxes = detector.detect(img)
        yolo_times.append((time.time() - t_y) * 1000)

        for (x1, y1, x2, y2, conf) in boxes:
            detections += 1
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.shape[0] < 15 or plate_crop.shape[1] < 40:
                continue
            t_o = time.time()
            raw = ocr_plate(plate_crop, reader)
            ocr_times.append((time.time() - t_o) * 1000)
            if raw:
                ocr_ok += 1

        total_times.append((time.time() - t0) * 1000)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print("Images:           {}".format(len(test_images)))
    print("Plates detected:  {}".format(detections))
    print("OCR successes:    {}/{}".format(ocr_ok, detections))
    if yolo_times:
        print("YOLO avg:         {:.1f} ms".format(sum(yolo_times) / len(yolo_times)))
    if ocr_times:
        print("OCR avg:          {:.1f} ms".format(sum(ocr_times) / len(ocr_times)))
    if total_times:
        avg = sum(total_times) / len(total_times)
        print("Total avg:        {:.1f} ms".format(avg))
        print("Estimated FPS:    {:.1f}".format(1000.0 / avg if avg > 0 else 0))
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ANPR Jetson Nano")
    parser.add_argument("--test", action="store_true", help="Test on static images")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark mode")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--gstreamer", action="store_true", help="Use GStreamer CSI camera")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Inference size")
    args = parser.parse_args()

    # Load model
    print("[INFO] Loading YOLO ONNX: {}".format(MODEL_PATH))
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Not found: {}".format(MODEL_PATH))
        sys.exit(1)

    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD, args.imgsz)

    # OCR setup (Tesseract — much faster than EasyOCR on Jetson)
    print("[INFO] Initializing Tesseract OCR...")
    # Config: treat image as single line of text, whitelist alphanumeric chars
    ocr_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    reader = ocr_config  # pass config string as "reader"
    print("[INFO] Tesseract ready.")

    # Database
    db = PlateDatabase(DB_PATH)

    if args.test:
        run_test_mode(detector, reader, db)
        db.close()
        return

    if args.benchmark:
        run_benchmark(detector, reader, db)
        db.close()
        return

    # Camera or Video
    print("[INFO] Opening video source...")
    if args.video:
        if not os.path.exists(args.video):
            print("[ERROR] Video file not found: {}".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        print("[INFO] Playing video: {}".format(args.video))
    elif args.gstreamer:
        cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source!")
        print("  Try: python3 anpr_full.py --video plate_video.mp4")
        print("  Or:  python3 anpr_full.py --gstreamer")
        print("  Or:  python3 anpr_full.py --camera 1")
        sys.exit(1)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    voter = PlateVoter(buffer_size=VOTING_BUFFER_SIZE)
    fps_counter = FPSCounter()
    os.makedirs("screenshots", exist_ok=True)

    no_detection_count = 0
    total_detections = 0
    last_logged_plate = ""

    print("[INFO] Running! q=quit  s=screenshot  r=reset")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                continue
            else:
                continue

        fps_counter.tick()
        boxes = detector.detect(frame)
        plate_detected = False

        for (x1, y1, x2, y2, conf) in boxes:
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.shape[0] < 15 or plate_crop.shape[1] < 40:
                continue

            raw_text = ocr_plate(plate_crop, reader)
            if raw_text:
                cleaned = clean_plate_text(raw_text)
                voter.add(cleaned)
                voted_plate = voter.get_voted_plate()

                status, match_type, matched, similarity = db.lookup(voted_plate)

                if voted_plate != last_logged_plate:
                    db.log_access(voted_plate, status, match_type, conf)
                    last_logged_plate = voted_plate
                    total_detections += 1

                draw_overlay(frame, x1, y1, x2, y2, voted_plate, status, match_type, conf)

                plate_detected = True
                no_detection_count = 0

                fps = fps_counter.get_fps()
                print("  Raw: {:20s} | Clean: {:15s} | Vote: {:15s} | "
                      "{:12s} | {} ({:.0f}%) | {:.1f} FPS".format(
                          raw_text, cleaned, voted_plate,
                          status, match_type, similarity, fps))

        if not plate_detected:
            no_detection_count += 1
            if no_detection_count > 15:
                voter.reset()
                no_detection_count = 0

        fps = fps_counter.get_fps()
        draw_info_bar(frame, fps, total_detections)

        cv2.imshow("ANPR - Jetson Nano", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = "screenshots/capture_{}.jpg".format(int(time.time()))
            cv2.imwrite(fname, frame)
            print("  [SCREENSHOT] {}".format(fname))
        elif key == ord("r"):
            voter.reset()
            print("  [RESET] Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()

    # =================================================================
    # SUMMARY DASHBOARD
    # =================================================================
    logs = db.get_recent_logs(100)
    total = len(logs)
    authorized = sum(1 for l in logs if l[1] == "AUTHORIZED")
    flagged = sum(1 for l in logs if l[1] == "FLAGGED")
    unknown = sum(1 for l in logs if l[1] == "UNKNOWN")
    exact_matches = sum(1 for l in logs if l[2] == "exact")
    fuzzy_matches = sum(1 for l in logs if l[2] == "fuzzy")
    no_matches = sum(1 for l in logs if l[2] == "none")

    unique_plates = list(set(l[0] for l in logs))

    print("\n")
    print("=" * 60)
    print("       ANPR SESSION SUMMARY DASHBOARD")
    print("=" * 60)
    print("")
    print("  DETECTION STATISTICS")
    print("  --------------------")
    print("  Total plate reads:      {}".format(total))
    print("  Unique plates seen:     {}".format(len(unique_plates)))
    print("  Avg YOLO confidence:    {:.2f}".format(
        sum(l[3] for l in logs) / total if total > 0 else 0))
    print("")
    print("  ACCESS DECISIONS")
    print("  ----------------")
    print("  AUTHORIZED:  {:3d}  ({:.0f}%)".format(
        authorized, 100.0 * authorized / total if total > 0 else 0))
    print("  FLAGGED:     {:3d}  ({:.0f}%)".format(
        flagged, 100.0 * flagged / total if total > 0 else 0))
    print("  UNKNOWN:     {:3d}  ({:.0f}%)".format(
        unknown, 100.0 * unknown / total if total > 0 else 0))
    print("")
    print("  MATCHING ANALYSIS")
    print("  -----------------")
    print("  Exact matches:   {:3d}  ({:.0f}%)".format(
        exact_matches, 100.0 * exact_matches / total if total > 0 else 0))
    print("  Fuzzy matches:   {:3d}  ({:.0f}%)  [threshold >= {}%]".format(
        fuzzy_matches, 100.0 * fuzzy_matches / total if total > 0 else 0, FUZZY_THRESHOLD))
    print("  No match:        {:3d}  ({:.0f}%)".format(
        no_matches, 100.0 * no_matches / total if total > 0 else 0))
    print("")
    print("  UNIQUE PLATES DETECTED")
    print("  ----------------------")
    for plate in unique_plates:
        for l in logs:
            if l[0] == plate:
                print("    {:20s} -> {:12s} ({})".format(plate, l[1], l[2]))
                break
    print("")
    print("  ACCESS LOG (last 15)")
    print("  --------------------")
    print("  {:<20s} {:<12s} {:<8s} {:<8s} {}".format(
        "PLATE", "STATUS", "MATCH", "CONF", "TIMESTAMP"))
    print("  " + "-" * 70)
    for log in logs[:15]:
        print("  {:<20s} {:<12s} {:<8s} {:.2f}     {}".format(
            log[0], log[1], log[2], log[3], log[4]))
    print("")
    print("=" * 60)
    print("  Database saved: {}".format(DB_PATH))
    print("  Screenshots:    screenshots/")
    print("=" * 60)

    db.close()


if __name__ == "__main__":
    main()
