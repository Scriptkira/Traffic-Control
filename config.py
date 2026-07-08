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
OUTPUT_SCALE = 2.0                      # HD Upscale multiplier for rendering annotations & logs
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
OUTPUT_CODEC = "mp4v"
DEFAULT_OUTPUT_PATH = "output.mp4"
DEFAULT_FPS = 30.0

# ─────────────────────────────────────────────
# Camera Identifier (mock)
# ─────────────────────────────────────────────
CAMERA_ID = "CAM-01"

