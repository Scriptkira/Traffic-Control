# ANPR Traffic Camera System

> **Automatic Number Plate Recognition** — Vehicle Detection, Tracking & License Plate Recognition

A production-ready Python application for traffic camera monitoring that detects vehicles, tracks them with persistent IDs, detects license plates, and reads plate text using OCR.

---

## Features

- **Vehicle Detection** — YOLOv8 (COCO-pretrained) detects cars, trucks, buses, motorcycles
- **Multi-Object Tracking** — SORT algorithm assigns persistent integer IDs across frames
- **License Plate Detection** — YOLO-based (custom model) or contour-based fallback
- **OCR** — EasyOCR reads plate text with preprocessing pipeline
- **Annotated Output** — Blue vehicle boxes, green plate boxes, white-on-green plate text
- **Log Panel** — Right-side panel logs every detected plate with ID, text, and timestamp
- **Tripwire Line** — Green trigger line for zone-based detection

---

## Requirements

- Python 3.9 or higher
- CUDA-capable GPU (optional, for faster inference)

---

## Setup

### 1. Clone / Navigate to the project

```bash
cd "(Automatic Number Plate Recognition"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls in `torch`/`torchvision` transitively via `ultralytics`/`easyocr`, which by default installs the **CPU-only** build. If you have a CUDA-capable GPU, install the matching CUDA wheels afterward (check your driver's supported CUDA version with `nvidia-smi`, then pick a matching index, e.g. for CUDA 12.6):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify it worked with `python -c "import torch; print(torch.cuda.is_available())"` — pipeline startup also logs GPU status clearly (`GPU ACCELERATION: ENABLED/DISABLED`).

### 5. Download YOLO Weights

The **vehicle detection** model (`yolov8n.pt`) is **automatically downloaded** by Ultralytics on first run. No manual download needed.

For **license plate detection**, you have two options:

#### Option A: Use the contour-based fallback (no extra download)
The system works out of the box using OpenCV contour detection for plates. This is less accurate but requires zero setup.

#### Option B: Use a trained YOLO plate model (recommended)
1. Train a YOLOv8 model on a license plate dataset, or download a pretrained one
2. Place the weights file at: `models/license_plate_detector.pt`
3. The system will automatically detect and use it

---

## Usage

### Basic usage (process the included sample video):
```bash
python main.py --input sample.mp4
```

### Save output and show preview:
```bash
python main.py --input sample.mp4 --output result.mp4 --show
```

### Process from webcam:
```bash
python main.py --input 0 --output webcam_output.mp4
```

### Headless mode (no preview window):
```bash
python main.py --input sample.mp4 --output result.mp4 --no-show
```

### Verbose logging:
```bash
python main.py --input sample.mp4 -v
```

### Keyboard Controls (when preview is enabled):
| Key       | Action        |
|-----------|---------------|
| `Q` / `ESC` | Quit         |
| `SPACE`   | Pause/Resume  |

---

## Project Structure

```
├── main.py                    # CLI entry point
├── config.py                  # All tunable parameters
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── detectors/
│   ├── vehicle_detector.py    # YOLOv8 vehicle detection
│   └── plate_detector.py      # Plate detection (YOLO + fallback)
│
├── tracker/
│   └── sort_tracker.py        # SORT multi-object tracker
│
├── ocr/
│   └── plate_reader.py        # EasyOCR wrapper
│
├── ui/
│   ├── annotator.py           # Frame annotation (boxes, text)
│   └── log_panel.py           # Right-side logging panel
│
├── core/
│   ├── pipeline.py            # Main processing pipeline
│   └── vehicle_record.py      # Vehicle state tracking
│
├── utils/
│   └── video_io.py            # Video I/O helpers
│
└── models/
    └── license_plate_detector.pt  # (Optional) trained plate model
```

---

## Configuration

All tunable parameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VEHICLE_CONFIDENCE` | 0.4 | Min confidence for vehicle detection |
| `PLATE_CONFIDENCE` | 0.3 | Min confidence for plate detection |
| `OCR_CONFIDENCE_THRESHOLD` | 0.3 | Min confidence for OCR readings |
| `TRACKER_MAX_AGE` | 30 | Frames to keep a lost track alive |
| `TRACKER_MIN_HITS` | 3 | Detections before a track is confirmed |
| `TRIPWIRE_Y_RATIO` | 0.75 | Trigger line position (0=top, 1=bottom) |
| `LOG_PANEL_WIDTH` | 320 | Width of the side panel in pixels |

---

## Output

The system produces:
1. **Annotated video** (`.mp4`) with all visual overlays
2. **Console log** with real-time detection events and performance metrics

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check the file path exists |
| `CUDA out of memory` | Use a smaller model or set `gpu=False` in plate_reader.py |
| `EasyOCR download hangs` | First run downloads ~100MB of OCR models — wait for it |
| Poor plate detection | Use a trained YOLO plate model instead of contour fallback |
| Low FPS | Use `yolov8n.pt` (nano), reduce resolution, or skip frames |

---

## License

MIT License
