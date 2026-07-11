# ANPR Traffic Camera System

> **Automatic Number Plate Recognition** — Vehicle Detection, Tracking & License Plate Recognition

A production-ready Python application for traffic camera monitoring that detects vehicles, tracks them with persistent IDs, detects license plates, and reads plate text using OCR.

---

## Features

- **Vehicle Detection** — YOLOv8 (COCO-pretrained) detects cars, trucks, buses, motorcycles
- **Multi-Object Tracking** — SORT algorithm assigns persistent integer IDs across frames
- **License Plate Detection** — YOLO-based (custom model) or contour-based fallback
- **Fast Plate OCR** — recognition-only EasyOCR (CRAFT detection bypassed, ~5-8x faster per
  read) with Nepali plate-grammar snapping and Devanagari→Latin transliteration; legacy full
  `readtext` path available via `OCR_ENGINE = "easyocr"`
- **Fully Asynchronous Pipeline** — video decode, vehicle detection, and plate OCR each run
  off the frame loop (see [Architecture](#architecture)); the loop itself only tracks and draws
- **Consensus Plate Voting** — per-vehicle weighted character voting across OCR reads, so a
  few close-up reads outvote many noisy distant ones
- **Annotated Output** — vehicle boxes with persistent IDs, plate boxes, plate text overlays
- **Log Panel** — Right-side panel logs every confirmed plate with ID, text, and timestamp
- **Tripwire Line** — trigger line for zone-based detection

---

## Architecture

The frame loop never blocks on inference:

| Stage | Where it runs | Notes |
|-------|---------------|-------|
| Video decode + downscale | Background thread (`FramePrefetcher`, `utils/video_io.py`) | Keeps the loop fed; retains the original hi-res frame for OCR crops |
| Vehicle detection | Background thread (`core/pipeline.py::_detect_worker`) | Always processes the newest frame; rate-capped by `DETECT_MAX_RATE_HZ`; tracker coasts on Kalman prediction between results |
| Plate detection + OCR | **Separate child process** (`core/ocr_process.py`) | Own GIL and CUDA context, so EasyOCR's CPU-heavy work can't stall the frame loop; fed vehicle-ROI crops via a bounded queue (drops when full — the vehicle retries on its next cadence slot) |
| Tracking, annotation, display | Main frame loop | Per-frame cost is per-track Python work only |

OCR crops are always cut from the **original full-resolution frame**, so downscaled
processing (`PROCESS_MAX_WIDTH`) doesn't cost plate readability.

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

### Basic usage (process the included demo video):
```bash
python main.py --input Trafficnepalvideodemo_1440p.mp4
```

### Save output as a compact .mp4 (default MJPG .avi is much larger):
```bash
python main.py --input Trafficnepalvideodemo_1440p.mp4 --output result.mp4 --codec mp4v
```

### Live preview only, skip writing the output video (frees encoding CPU):
```bash
python main.py --input Trafficnepalvideodemo_1440p.mp4 --no-output
```

### Process from webcam:
```bash
python main.py --input 0 --output webcam_output.mp4
```

### Headless mode (no preview window):
```bash
python main.py --input Trafficnepalvideodemo_1440p.mp4 --output result.mp4 --no-show
```

### Verbose logging:
```bash
python main.py --input Trafficnepalvideodemo_1440p.mp4 -v
```

> `sample.mp4` is a static CCTV clip in which no readable plate is ever presented to the
> camera — 0 plate reads on it is expected. Use the Nepali traffic demo video above.

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
│   ├── plate_reader.py        # Legacy EasyOCR readtext wrapper
│   ├── fast_plate_ocr.py      # Recognition-only EasyOCR + plate-grammar snapping
│   └── text_normalize.py      # Cleanup + Devanagari→Latin transliteration
│
├── ui/
│   ├── annotator.py           # Frame annotation (boxes, text)
│   └── log_panel.py           # Right-side logging panel
│
├── core/
│   ├── pipeline.py            # Main processing pipeline (async orchestration)
│   ├── ocr_process.py         # Plate-detection + OCR child process
│   └── vehicle_record.py      # Vehicle state tracking + consensus voting
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
| `VEHICLE_CONFIDENCE` | 0.25 | Min confidence for vehicle detection |
| `PLATE_CONFIDENCE` | 0.3 | Min confidence for plate detection |
| `OCR_ENGINE` | `"fast"` | `"fast"` = recognition-only EasyOCR + plate-grammar snapping (~5-8x faster per read); `"easyocr"` = legacy full readtext |
| `OCR_LANGUAGES` | `["ne", "en"]` | The Devanagari `"ne"` recognizer ~doubles per-read latency, but OCR runs off the frame loop, so it costs reads/sec — not FPS |
| `OCR_CONFIDENCE_THRESHOLD` | 0.25 | Min confidence for OCR readings (format-valid reads bypass this) |
| `OCR_MIN_CROP_HEIGHT` | 16 | Skip OCR on plate crops shorter than this (px); smaller crops are pure noise |
| `OCR_EVERY_N_FRAMES` | 6 | Per-vehicle OCR cadence (staggered by track ID) |
| `OCR_MAX_READS_PER_FRAME` | 2 | Hard per-frame budget on enqueued OCR jobs |
| `PLATE_MIN_VOTES_TO_CONFIRM` | 1 | OCR votes required before a reading is logged (see consensus voting below) |
| `DETECT_MAX_RATE_HZ` | 12.0 | Rate cap for the async vehicle-detector thread |
| `DETECT_EVERY_N_FRAMES` | 8 | Tracker's Kalman coast window between detector results (detection itself is async) |
| `PROCESS_MAX_WIDTH` | 1920 | Downscale wider sources to this width for processing; OCR still crops from the original hi-res frame |
| `USE_FP16_ON_GPU` | True | Run YOLO inference in half precision on CUDA (faster, lower VRAM; ignored on CPU) |
| `TRACKER_MAX_AGE` | 30 | Frames to keep a lost track alive |
| `TRACKER_MIN_HITS` | 2 | Detections before a track is confirmed (filters one-frame YOLO flicker) |
| `TRIPWIRE_Y_RATIO` | 0.75 | Trigger line position (0=top, 1=bottom) |
| `OUTPUT_SCALE` | 1.0 | Upscale multiplier for rendered output — keep 1.0 for HD/4K sources; only raise for low-res inputs |
| `LOG_PANEL_WIDTH` | 320 | Width of the side panel in pixels |

### Plate reading consensus

Each vehicle accumulates every OCR reading it gets while tracked. Characters are combined
across readings via a confidence × crop-area weighted vote (`core/vehicle_record.py`), so a
few high-confidence, close-up reads outweigh many noisy, distant ones. The first reading that
clears `PLATE_MIN_VOTES_TO_CONFIRM` votes is logged immediately; after that, a new consensus
value must agree on **two consecutive votes** before it's allowed to overwrite the logged
reading, so a single noisy OCR pass doesn't re-trigger a log entry for the same vehicle.

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
