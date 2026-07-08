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

        # ── Step 1: Detect vehicles ──
        detections = self.vehicle_detector.detect(frame)

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

        # ── Step 3: Process each tracked vehicle ──
        for tracked in tracked_objects:
            track_id = tracked.track_id
            bbox = tracked.bbox

            # Get or create vehicle record
            if track_id not in self.vehicle_records:
                self.vehicle_records[track_id] = VehicleRecord(
                    track_id=track_id
                )

            record = self.vehicle_records[track_id]
            record.update_position(bbox)

            # Draw vehicle bounding box and ID
            self.annotator.draw_vehicle_box(frame, bbox, track_id)

            # ── Step 3a: Detect plate within vehicle ROI ──
            plate_candidates = self.plate_detector.detect_in_roi(frame, bbox)

            best_candidate = None
            ocr_success = False

            if plate_candidates:
                # Save the top candidate (which has highest contour score)
                best_candidate = plate_candidates[0]

                # ── Step 3b: Evaluate candidates with OCR ──
                for candidate in plate_candidates:
                    ocr_result = self.plate_reader.read(candidate.crop)

                    if ocr_result is not None:
                        plate_text, confidence = ocr_result

                        # Update vehicle record with best reading
                        updated = record.update_plate(
                            plate_text, confidence, candidate.bbox
                        )

                        # Draw green plate box for the successful candidate
                        self.annotator.draw_plate_box(frame, candidate.bbox)

                        # Draw plate text on frame
                        self.annotator.draw_plate_text(
                            frame,
                            record.best_plate_text,
                            bbox,
                            candidate.bbox,
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
                    # Draw the green box of the top candidate so we still see a green box
                    self.annotator.draw_plate_box(frame, best_candidate.bbox)
                    
                    if record.has_plate:
                        # Draw historical plate text at the top candidate box
                        self.annotator.draw_plate_text(
                            frame,
                            record.best_plate_text,
                            bbox,
                            best_candidate.bbox,
                        )
                elif record.has_plate:
                    # No candidates found but we have history — draw text at bottom of vehicle
                    self.annotator.draw_plate_text(
                        frame,
                        record.best_plate_text,
                        bbox,
                    )

            # ── Step 4: Check tripwire crossing ──
            self._check_tripwire(record, frame.shape[0])

        # ── Step 5: Draw tripwire line ──
        self.annotator.draw_tripwire(frame)

        # ── Step 6: Render log panel ──
        annotated = self.log_panel.render(frame)

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
