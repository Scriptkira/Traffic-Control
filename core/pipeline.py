"""
ANPR Pipeline — Main processing pipeline for the traffic camera system.

Orchestrates the per-frame flow:
    1. Detect vehicles
    2. Track vehicles (assign persistent IDs)
    3. Detect plates within each vehicle ROI
    4. OCR the plate crops
    5. Annotate the frame
    6. Update the log panel
"""

import cv2
import logging
import multiprocessing as mp
import queue
import threading
from typing import Dict

import numpy as np
import torch

import config
from core.ocr_process import plate_ocr_process_main
from core.vehicle_record import VehicleRecord
from detectors.vehicle_detector import VehicleDetector
from tracker.sort_tracker import SORTTracker
from ui.annotator import FrameAnnotator
from ui.log_panel import LogPanel

logger = logging.getLogger(__name__)


class ANPRPipeline:
    """
    Main ANPR processing pipeline.

    Manages the complete flow from raw frame to annotated output,
    including vehicle detection, tracking, plate detection, OCR,
    visual annotation, and logging.
    """

    def __init__(self):
        """Initialize all pipeline components."""
        logger.info("=" * 60)
        logger.info("Initializing ANPR Pipeline...")
        logger.info("=" * 60)

        if torch.cuda.is_available():
            # Autotune cuDNN kernels for the fixed input sizes used here —
            # worthwhile since detector input shapes repeat every frame.
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU ACCELERATION: ENABLED — {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU ACCELERATION: DISABLED — running on CPU (expect slow inference)")
        logger.info("=" * 60)

        # Detection
        self.vehicle_detector = VehicleDetector()

        # Tracking
        self.tracker = SORTTracker()

        # UI
        self.annotator = FrameAnnotator()
        self.log_panel = LogPanel()

        # Vehicle state tracking
        self.vehicle_records: Dict[int, VehicleRecord] = {}

        # Plate detection + OCR run in a separate PROCESS (own GIL, own
        # CUDA context). As a thread, EasyOCR's CPU-side work — doubled
        # by the Nepali Devanagari recognizer — and the frame loop starve
        # each other; as a process they genuinely run in parallel.
        # Bounded input queue: when OCR falls behind, new crops are
        # dropped and the vehicle retries on its next cadence slot.
        # Results come back on _ocr_out and are applied each frame by
        # _apply_ocr_results().
        self._ocr_in: mp.Queue = mp.Queue(maxsize=8)
        self._ocr_out: mp.Queue = mp.Queue()
        self._ocr_proc = mp.Process(
            target=plate_ocr_process_main,
            args=(
                self._ocr_in,
                self._ocr_out,
                logging.getLogger().getEffectiveLevel(),
            ),
            daemon=True,
        )
        self._ocr_proc.start()
        logger.info("Waiting for OCR process to load its models...")
        kind, _ = self._ocr_out.get(timeout=300)
        if kind != "ready":
            raise RuntimeError(f"OCR process sent unexpected message: {kind}")
        logger.info("OCR process ready.")

        # Async vehicle detection: the detector runs continuously on its
        # own thread, always on the newest frame (frames it can't keep up
        # with are simply skipped). The frame loop consumes whichever
        # result is ready and lets the tracker coast on Kalman prediction
        # in between — so YOLO never blocks a frame.
        self._det_lock = threading.Lock()
        self._det_pending = None       # newest frame awaiting detection
        self._det_event = threading.Event()
        self._det_result = None        # latest finished detection list
        self._det_result_fresh = False
        self._det_thread = threading.Thread(target=self._detect_worker, daemon=True)
        self._det_thread.start()

        # Frame counter
        self.frame_count = 0

        logger.info("ANPR Pipeline initialized successfully.")
        logger.info("=" * 60)

    def process_frame(
        self,
        frame: np.ndarray,
        hires_frame: np.ndarray = None,
    ) -> np.ndarray:
        """
        Process a single video frame through the complete pipeline.

        Args:
            frame: Input BGR frame at processing resolution.
            hires_frame: Optional original full-resolution frame. When
                provided, OCR crops are cut from it instead of `frame`,
                so downscaled processing doesn't cost plate readability.

        Returns:
            Annotated frame with log panel appended.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        scale = getattr(config, "OUTPUT_SCALE", 1.0)

        # ── Step 0: Apply plate readings the OCR process finished ──
        self._apply_ocr_results()

        # ── Step 1: Async vehicle detection ──
        # Hand the newest frame to the detector thread and consume a
        # finished result if one is ready. When no fresh result exists,
        # pass no detections so the tracker coasts on velocity-based
        # Kalman prediction (see SORTTracker's coast_frames) instead of
        # being "corrected" toward a stale bbox.
        with self._det_lock:
            self._det_pending = frame
            if self._det_result_fresh:
                detections = self._det_result
                self._det_result_fresh = False
            else:
                detections = []
        self._det_event.set()

        if detections:
            logger.debug(
                f"Frame {self.frame_count}: {len(detections)} vehicle(s) detected"
            )

        # ── Step 2: Track vehicles (assign persistent IDs) ──
        tracked_objects = self.tracker.update(detections)

        if tracked_objects:
            logger.debug(
                f"Frame {self.frame_count}: {len(tracked_objects)} tracked object(s)"
            )

        # ── Step 3: Create upscaled frame for drawing ──
        if scale == 1.0:
            annotated_frame = frame.copy()
        else:
            annotated_frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )

        # Calculate tripwire Y level on native resolution
        tripwire_y = int(h * config.TRIPWIRE_Y_RATIO)
        any_crossing = False

        # ── Step 4: Update records & hand plate work to the async worker ──
        # Plate detection AND OCR both run on the background thread; the
        # frame loop only crops the vehicle ROI and enqueues it. Plate
        # boxes are drawn from each record's cached plate position,
        # projected onto the vehicle's current bbox.
        plate_confirm_conf = getattr(config, "PLATE_CONFIRM_CONFIDENCE", 0.70)

        # Per-vehicle OCR cadence, offset by track_id so vehicles spread
        # across frames instead of all landing on the same one, plus a
        # hard per-frame budget on enqueued plate jobs.
        ocr_every = getattr(config, "OCR_EVERY_N_FRAMES", 1)
        ocr_budget = getattr(config, "OCR_MAX_READS_PER_FRAME", 2)
        ocr_reads_done = 0

        hires_scale = hires_frame.shape[1] / w if hires_frame is not None else 1.0
        projected_by_idx = {}  # idx-in-tracked_objects -> cached plate bbox or None

        for idx, tracked in enumerate(tracked_objects):
            track_id = tracked.track_id
            bbox = tracked.bbox

            # Check if this vehicle is currently crossing the tripwire
            if bbox[1] <= tripwire_y <= bbox[3]:
                any_crossing = True

            # Get or create vehicle record
            if track_id not in self.vehicle_records:
                self.vehicle_records[track_id] = VehicleRecord(
                    track_id=track_id
                )

            record = self.vehicle_records[track_id]
            record.update_position(bbox)

            confirmed = record.has_plate and record.best_confidence >= plate_confirm_conf
            run_ocr = (
                not confirmed
                and ocr_reads_done < ocr_budget
                and (self.frame_count + track_id) % ocr_every == 0
            )

            if run_ocr and self._enqueue_plate_work(
                track_id, bbox, frame, hires_frame, hires_scale, tracked_objects
            ):
                ocr_reads_done += 1

            projected_by_idx[idx] = record.project_plate_bbox(bbox)

        # ── Step 4b: Per-vehicle annotation ──

        for idx, tracked in enumerate(tracked_objects):
            track_id = tracked.track_id
            bbox = tracked.bbox
            record = self.vehicle_records[track_id]

            # Scale vehicle coordinates for drawing
            scaled_bbox = (
                bbox[0] * scale,
                bbox[1] * scale,
                bbox[2] * scale,
                bbox[3] * scale
            )

            # Draw vehicle bounding box and ID on the upscaled frame
            self.annotator.draw_vehicle_box(annotated_frame, scaled_bbox, track_id, scale=scale)

            # ── Plate box/text from the cached position ──
            # Detection/OCR results arrive asynchronously, so the record's
            # current best reading and cached plate position are always
            # what gets drawn.
            fallback_plate_bbox = projected_by_idx.get(idx)

            if fallback_plate_bbox is not None:
                scaled_fb = (
                    fallback_plate_bbox[0] * scale,
                    fallback_plate_bbox[1] * scale,
                    fallback_plate_bbox[2] * scale,
                    fallback_plate_bbox[3] * scale
                )
                self.annotator.draw_plate_box(annotated_frame, scaled_fb, scale=scale)

                if record.has_plate:
                    self.annotator.draw_plate_text(
                        annotated_frame,
                        record.best_plate_text,
                        scaled_bbox,
                        scaled_fb,
                        scale=scale,
                    )
            elif record.has_plate:
                # No candidates found but we have history — draw text at bottom of vehicle
                self.annotator.draw_plate_text(
                    annotated_frame,
                    record.best_plate_text,
                    scaled_bbox,
                    scale=scale,
                )

            # ── Step 5: Check tripwire crossing (native height) ──
            self._check_tripwire(record, h)

        # ── Step 6: Draw tripwire line on upscaled frame ──
        self.annotator.draw_tripwire(annotated_frame, config.TRIPWIRE_Y_RATIO, is_alert=any_crossing, scale=scale)

        # ── Step 7: Render log panel ──
        annotated = self.log_panel.render(annotated_frame, scale=scale)

        return annotated

    def _detect_worker(self):
        """
        Background thread: detect vehicles on the newest frame.

        Throttled to DETECT_MAX_RATE_HZ — running YOLO back-to-back
        starves the frame loop of CPU/GIL time for little tracking
        benefit, since the Kalman tracker interpolates smoothly between
        detector results anyway.
        """
        import time as _time
        min_interval = 1.0 / getattr(config, "DETECT_MAX_RATE_HZ", 12.0)

        while True:
            self._det_event.wait()
            with self._det_lock:
                work_frame = self._det_pending
                self._det_pending = None
                self._det_event.clear()
            if work_frame is None:
                continue

            started = _time.perf_counter()
            detections = self.vehicle_detector.detect(work_frame)

            with self._det_lock:
                self._det_result = detections
                self._det_result_fresh = True

            leftover = min_interval - (_time.perf_counter() - started)
            if leftover > 0:
                _time.sleep(leftover)

    def _enqueue_plate_work(
        self,
        track_id: int,
        vehicle_bbox,
        frame: np.ndarray,
        hires_frame,
        hires_scale: float,
        tracked_objects,
    ) -> bool:
        """
        Queue a vehicle ROI for asynchronous plate detection + OCR.

        Returns True if enqueued, False if the ROI is unusable or the
        queue is full (the vehicle retries on its next cadence slot).
        """
        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
        h, w = frame.shape[:2]
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w, vx2), min(h, vy2)
        if vx2 - vx1 < 20 or vy2 - vy1 < 20:
            return False

        # Cheap full-check before paying for the crop copies (put_nowait
        # below still guards the race — mp.Queue's full() is approximate).
        if self._ocr_in.full():
            return False

        # Copies — these are views into frame buffers the main loop
        # immediately moves on from.
        proc_crop = frame[vy1:vy2, vx1:vx2].copy()
        hires_crop = None
        if hires_frame is not None:
            s = hires_scale
            hires_crop = hires_frame[
                int(vy1 * s):int(vy2 * s), int(vx1 * s):int(vx2 * s)
            ].copy()

        # Neighbor bboxes so the worker can reject plates that physically
        # sit on an overlapping vehicle.
        other_bboxes = [t.bbox for t in tracked_objects if t.track_id != track_id]

        try:
            self._ocr_in.put_nowait(
                (track_id, proc_crop, hires_crop, (vx1, vy1, vx2, vy2), other_bboxes)
            )
            return True
        except queue.Full:
            return False

    def _apply_ocr_results(self):
        """
        Drain finished plate readings from the OCR process and apply
        them to vehicle records (runs on the main thread, so record and
        log-panel updates need no locking).
        """
        while True:
            try:
                kind, payload = self._ocr_out.get_nowait()
            except queue.Empty:
                return

            if kind != "plate":
                continue

            track_id, plate_text, confidence, plate_bbox, vehicle_bbox = payload
            record = self.vehicle_records.get(track_id)
            if record is None:
                continue

            updated = record.update_plate(plate_text, confidence, plate_bbox)

            # Cache relative plate position so the frame loop can draw
            # a projected box without any detector call.
            record.cache_relative_plate_pos(plate_bbox, vehicle_bbox)

            if updated and record.best_plate_text:
                self.log_panel.add_entry(
                    vehicle_id=track_id,
                    plate_text=record.best_plate_text,
                    confidence=record.best_confidence,
                )
                logger.info(
                    f"Plate detected — ID:{track_id} "
                    f"Plate:{record.best_plate_text} "
                    f"Conf:{confidence:.2f}"
                )
            else:
                # The read reached consensus voting but didn't (yet)
                # change the logged text — make that visible so reads
                # "disappearing" here aren't mistaken for OCR failures.
                logger.debug(
                    f"OCR read ID:{track_id} '{plate_text}' "
                    f"conf={confidence:.2f} — vote {len(record.plate_history)} "
                    f"recorded, consensus unchanged "
                    f"('{record.best_plate_text}')"
                )

    def _check_tripwire(self, record: VehicleRecord, frame_height: int):
        """
        Check if a vehicle has crossed the tripwire line.

        Updates the vehicle record's crossed_tripwire flag.
        """
        if record.last_bbox is None:
            return

        tripwire_y = int(frame_height * config.TRIPWIRE_Y_RATIO)

        # Check if the bottom of the vehicle box is near the tripwire
        _, _, _, y2 = record.last_bbox
        if y2 >= tripwire_y and not record.crossed_tripwire:
            record.crossed_tripwire = True
            logger.debug(
                f"Vehicle ID:{record.track_id} crossed tripwire"
            )

    def shutdown(self):
        """
        Stop the OCR child process cleanly: send the None sentinel, join,
        and escalate to terminate() if it doesn't exit in time. Without
        this the daemon process is killed mid-read at interpreter exit,
        which can leave the parent's queue feeder threads blocked.
        """
        try:
            self._ocr_in.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._ocr_proc.join(timeout=5.0)
        if self._ocr_proc.is_alive():
            logger.warning("OCR process didn't exit in time — terminating.")
            self._ocr_proc.terminate()
            self._ocr_proc.join(timeout=2.0)
        self._ocr_in.cancel_join_thread()
        self._ocr_out.cancel_join_thread()

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        total_plates = sum(
            1 for r in self.vehicle_records.values() if r.has_plate
        )
        return {
            "frames_processed": self.frame_count,
            "total_vehicles_tracked": len(self.vehicle_records),
            "plates_detected": total_plates,
            "active_tracks": len(self.tracker.tracks),
            "log_entries": len(self.log_panel.entries),
        }
