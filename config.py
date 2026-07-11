"""
ANPR Traffic Camera System — Central Configuration
All tunable parameters, paths, and constants in one place.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VEHICLE_MODEL_PATH = "yolov8n.pt"  # Auto-downloaded by Ultralytics
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license_plate_detector.pt")

# ─────────────────────────────────────────────
# Detection Thresholds
# ─────────────────────────────────────────────
VEHICLE_CONFIDENCE = 0.25
PLATE_CONFIDENCE = 0.3   # Lowering to 0.25 measured no gain in confirmed
                         # plates (28 vs 34-vehicle baseline) — it only adds
                         # junk candidates for the OCR stage to reject

# COCO class IDs for vehicles
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# ─────────────────────────────────────────────
# OCR Settings
# ─────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.25
# Nepali Devanagari + English. The "ne" recognizer roughly doubles
# per-read latency, but OCR runs on a background thread, so this costs
# plate-reads-per-second — not frame rate.
OCR_LANGUAGES = ["ne", "en"]
OCR_CHAR_WHITELIST = None  # Must stay None to allow Devanagari characters
OCR_MIN_PLATE_LENGTH = 3   # Reject readings shorter than this
OCR_MAX_PLATE_LENGTH = 15  # Reject readings longer than this
OCR_MIN_CROP_HEIGHT = 16   # Skip OCR below this crop height (px) — smaller crops
                           # measured consistently <0.1 raw confidence (pure noise)

# ─────────────────────────────────────────────
# Plate Consensus
# ─────────────────────────────────────────────
PLATE_MIN_VOTES_TO_CONFIRM = 1  # OCR votes required before a reading is logged

# ─────────────────────────────────────────────
# Performance / Frame Skipping
# ─────────────────────────────────────────────
OCR_EVERY_N_FRAMES = 6     # Per-vehicle OCR cadence (offset by track_id)

# Vehicle detection runs on its own thread — this no longer gates
# detection itself, but still sets the tracker's default coast window
# (N-1 frames of Kalman-predicted output between real detector results).
# Sized for a ~12Hz detector under a 40-60 FPS frame loop: up to ~7
# frames can pass between detector results.
DETECT_EVERY_N_FRAMES = 8

# Cap the async detector thread's rate. Uncapped, it runs YOLO
# back-to-back and starves the frame loop of CPU/GIL time; ~12 detector
# updates/sec is plenty for smooth Kalman-interpolated tracking.
DETECT_MAX_RATE_HZ = 12.0

# Hard per-frame budget on EasyOCR reads. OCR is by far the slowest stage
# (~50-100ms/read with the Devanagari model); in dense traffic the per-vehicle
# cadence alone still fires many reads per frame. The track_id offset in the
# cadence spreads vehicles across frames, so every vehicle still gets read —
# just never more than this many in any single frame.
OCR_MAX_READS_PER_FRAME = 2

# OCR engine: "fast" = recognition-only EasyOCR (CRAFT detection
# bypassed) + Nepali plate-grammar snapping, ~5-8x faster per read.
# "easyocr" = legacy full readtext path.
OCR_ENGINE = "fast"

# Downscale frames wider than this before they enter the pipeline.
# 4K sources are processed (and rendered/written) at 1080p — detection
# models resize to 640px internally anyway, and this makes every CPU-side
# stage (copy, annotate, encode, preview) ~4x cheaper. Costs some OCR
# accuracy on small/distant plates. Set to None to process at native res.
PROCESS_MAX_WIDTH = 1920

# Run YOLO inference in FP16 on CUDA. Halves activation/weight memory and
# is noticeably faster on VRAM-constrained cards (e.g. 4GB laptop GPUs)
# with negligible accuracy impact. Ignored on CPU (unsupported there).
USE_FP16_ON_GPU = True


# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────
# When True, the OCR process saves every plate crop it OCRs (accepted
# AND rejected, with outcome/confidence in the filename) plus a capped
# sample of vehicle ROIs where plate detection found nothing, to
# DIAG_CROPS_DIR for manual inspection. Combine with -v for per-job
# debug logs from the OCR process.
DIAGNOSTIC_MODE = False
DIAG_CROPS_DIR = os.path.join(BASE_DIR, "debug_crops")
DIAG_MAX_EMPTY_ROIS_PER_TRACK = 3   # Cap "no candidates" ROI dumps per vehicle

# The OCR process logs a stage-funnel summary (jobs → candidates →
# OCR attempts → accepted) every N jobs, so failures are attributable
# to a stage even without -v.
OCR_STATS_EVERY_N_JOBS = 200

# ─────────────────────────────────────────────
# Tracker (SORT) Settings
# ─────────────────────────────────────────────
TRACKER_MAX_AGE = 30       # Frames to keep a track without detection
TRACKER_MIN_HITS = 2       # Min detections before a track is output — filters
                           # one-frame YOLO flicker (validated 2026-07-09: cuts
                           # spurious track IDs ~24% on dense-traffic footage)
TRACKER_IOU_THRESHOLD = 0.3

# ─────────────────────────────────────────────
# Tripwire / Trigger Line
# ─────────────────────────────────────────────
TRIPWIRE_Y_RATIO = 0.75   # Position as fraction of frame height

# ─────────────────────────────────────────────
# UI / Visual Settings
# ─────────────────────────────────────────────
OUTPUT_SCALE = 1.0                      # Upscale multiplier for rendering annotations & logs.
                                        # Keep at 1.0 for HD/4K sources — upscaling 4K to 8K
                                        # makes CPU-side MJPG encoding the pipeline bottleneck.
                                        # Only raise above 1.0 for low-res (e.g. 480p) inputs.
LOG_PANEL_WIDTH = 320
LOG_PANEL_MAX_ENTRIES = 20
LOG_PANEL_BG_COLOR = (15, 15, 15)       # Cyberpunk dark background
LOG_PANEL_BG_ALPHA = 0.90               # Less transparent for readability

# Colors (BGR for OpenCV)
COLOR_VEHICLE_BOX = (255, 180, 0)       # Cyan/electric blue (target tracking)
COLOR_VEHICLE_TEXT = (255, 255, 255)    # Clean white
COLOR_PLATE_BOX = (0, 255, 100)         # Neon green (success/active lock)
COLOR_PLATE_TEXT_BG = (20, 20, 20)      # Sleek dark background for labels
COLOR_PLATE_TEXT_FG = (0, 255, 100)     # Neon green text for labels
COLOR_TRIPWIRE = (0, 255, 255)          # Bright Yellow/Amber for default state
COLOR_TRIPWIRE_ALERT = (0, 0, 255)      # Red alert when vehicle crosses
COLOR_LOG_TITLE = (255, 180, 0)         # Cyan title
COLOR_LOG_TEXT = (220, 220, 220)        # Bright silver
COLOR_LOG_SEPARATOR = (50, 50, 50)      # Dark gray borders

# Font
FONT_SCALE_ID = 0.5
FONT_SCALE_PLATE = 0.6
FONT_SCALE_LOG = 0.40
FONT_THICKNESS = 1
FONT_THICKNESS_LOG = 1

# ─────────────────────────────────────────────
# Video Output
# ─────────────────────────────────────────────
OUTPUT_CODEC = "MJPG"
DEFAULT_OUTPUT_PATH = "output.avi"
DEFAULT_FPS = 30.0

# ─────────────────────────────────────────────
# Camera Identifier (mock)
# ─────────────────────────────────────────────
CAMERA_ID = "CAM-01"

