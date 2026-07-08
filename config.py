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
PLATE_CONFIDENCE = 0.3

# COCO class IDs for vehicles
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# ─────────────────────────────────────────────
# OCR Settings
# ─────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.1
OCR_LANGUAGES = ["ne", "en"]  # Supports both Devanagari (ne) and English (en) for Nepali plates
OCR_CHAR_WHITELIST = None      # Disable strict whitelist to allow Devanagari characters
OCR_MIN_PLATE_LENGTH = 3   # Reject readings shorter than this
OCR_MAX_PLATE_LENGTH = 15  # Reject readings longer than this

# ─────────────────────────────────────────────
# Tracker (SORT) Settings
# ─────────────────────────────────────────────
TRACKER_MAX_AGE = 30       # Frames to keep a track without detection
TRACKER_MIN_HITS = 1       # Min detections before track is confirmed
TRACKER_IOU_THRESHOLD = 0.3

# ─────────────────────────────────────────────
# Tripwire / Trigger Line
# ─────────────────────────────────────────────
TRIPWIRE_Y_RATIO = 0.75   # Position as fraction of frame height

# ─────────────────────────────────────────────
# UI / Visual Settings
# ─────────────────────────────────────────────
LOG_PANEL_WIDTH = 320
LOG_PANEL_MAX_ENTRIES = 20
LOG_PANEL_BG_COLOR = (30, 30, 30)       # Dark gray
LOG_PANEL_BG_ALPHA = 0.85               # Semi-transparent

# Colors (BGR for OpenCV)
COLOR_VEHICLE_BOX = (255, 150, 50)      # Blue-ish
COLOR_VEHICLE_TEXT = (255, 150, 50)     # Same blue
COLOR_PLATE_BOX = (0, 220, 0)          # Green
COLOR_PLATE_TEXT_BG = (0, 180, 0)      # Darker green for text bg
COLOR_PLATE_TEXT_FG = (255, 255, 255)  # White text
COLOR_TRIPWIRE = (0, 255, 0)           # Bright green
COLOR_LOG_TITLE = (0, 220, 0)          # Green
COLOR_LOG_TEXT = (200, 200, 200)       # Light gray
COLOR_LOG_SEPARATOR = (80, 80, 80)     # Dim gray

# Font
FONT_SCALE_ID = 0.6
FONT_SCALE_PLATE = 0.7
FONT_SCALE_LOG = 0.45
FONT_THICKNESS = 2
FONT_THICKNESS_LOG = 1

# ─────────────────────────────────────────────
# Video Output
# ─────────────────────────────────────────────
OUTPUT_CODEC = "mp4v"
DEFAULT_OUTPUT_PATH = "output.mp4"
DEFAULT_FPS = 30.0

# ─────────────────────────────────────────────
# Camera Identifier (mock)
# ─────────────────────────────────────────────
CAMERA_ID = "CAM-01"
