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
from typing import Dict

import numpy as np

import config
from core.vehicle_record import VehicleRecord
from detectors.plate_detector import PlateDetector
from detectors.vehicle_detector import VehicleDetector
from ocr.plate_reader import PlateReader
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

        # Detection
        self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()

        # Tracking
        self.tracker = SORTTracker()

        # OCR
        self.plate_reader = PlateReader()

        # UI
        self.annotator = FrameAnnotator()
        self.log_panel = LogPanel()

        # Vehicle state tracking
        self.vehicle_records: Dict[int, VehicleRecord] = {}

        # Frame counter
        self.frame_count = 0

        logger.info("ANPR Pipeline initialized successfully.")
        logger.info("=" * 60)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame through the complete pipeline.

        Args:
            frame: Input BGR frame.

        Returns:
            Annotated frame with log panel appended.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        scale = getattr(config, "OUTPUT_SCALE", 1.0)
        detect_every = getattr(config, "DETECT_EVERY_N_FRAMES", 1)

        # ── Step 1: Detect vehicles (skip frames for speed) ──
        if self.frame_count % detect_every == 0:
            detections = self.vehicle_detector.detect(frame)
            self._last_detections = detections
        else:
            # Reuse last detections — tracker will predict positions
            detections = getattr(self, "_last_detections", [])

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
        annotated_frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

        # Calculate tripwire Y level on native resolution
        tripwire_y = int(h * config.TRIPWIRE_Y_RATIO)
        any_crossing = False

        # ── Step 4: Process each tracked vehicle ──
        for tracked in tracked_objects:
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

            # Scale vehicle coordinates for drawing
            scaled_bbox = (
                bbox[0] * scale,
                bbox[1] * scale,
                bbox[2] * scale,
                bbox[3] * scale
            )

            # Draw vehicle bounding box and ID on the upscaled frame
            self.annotator.draw_vehicle_box(annotated_frame, scaled_bbox, track_id, scale=scale)

            # ── Step 4a: Detect plate within vehicle ROI (native resolution) ──
            plate_candidates = self.plate_detector.detect_in_roi(frame, bbox)

            best_candidate = None
            ocr_success = False

            # Only run OCR on key frames, and skip if we already have a high-confidence reading
            ocr_every = getattr(config, "OCR_EVERY_N_FRAMES", 1)
            run_ocr = (self.frame_count % ocr_every == 0) and (not record.has_plate or record.best_confidence < 0.70)

            if plate_candidates:
                # Save the top candidate (which has highest contour score)
                best_candidate = plate_candidates[0]

                # ── Step 4b: Evaluate candidates with OCR (only on key frames) ──
                if run_ocr:
                    for candidate in plate_candidates:
                        ocr_result = self.plate_reader.read(candidate.crop)

                        if ocr_result is not None:
                            plate_text, confidence = ocr_result

                            # Update vehicle record with best reading
                            updated = record.update_plate(
                                plate_text, confidence, candidate.bbox
                            )

                            # Scale plate coordinates
                            scaled_plate_bbox = (
                                candidate.bbox[0] * scale,
                                candidate.bbox[1] * scale,
                                candidate.bbox[2] * scale,
                                candidate.bbox[3] * scale
                            )

                            # Draw green plate box for the successful candidate
                            self.annotator.draw_plate_box(annotated_frame, scaled_plate_bbox, scale=scale)

                            # Draw plate text on frame
                            self.annotator.draw_plate_text(
                                annotated_frame,
                                record.best_plate_text,
                                scaled_bbox,
                                scaled_plate_bbox,
                                scale=scale,
                            )

                            # Log to panel if new or updated
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
                            
                            ocr_success = True
                            break  # Found a valid plate via OCR!

            # If no OCR success this frame
            if not ocr_success:
                if best_candidate is not None:
                    # Scale best candidate coordinates
                    scaled_best_candidate_bbox = (
                        best_candidate.bbox[0] * scale,
                        best_candidate.bbox[1] * scale,
                        best_candidate.bbox[2] * scale,
                        best_candidate.bbox[3] * scale
                    )
                    # Draw the green box of the top candidate so we still see a green box
                    self.annotator.draw_plate_box(annotated_frame, scaled_best_candidate_bbox, scale=scale)
                    
                    if record.has_plate:
                        # Draw historical plate text at the top candidate box
                        self.annotator.draw_plate_text(
                            annotated_frame,
                            record.best_plate_text,
                            scaled_bbox,
                            scaled_best_candidate_bbox,
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
