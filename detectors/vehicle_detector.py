"""
Vehicle Detector — YOLOv8-based vehicle detection wrapper.

Detects vehicles (cars, motorcycles, buses, trucks) using a pretrained
YOLOv8 model on COCO classes.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

import config
from utils.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single vehicle detection result."""
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Return the area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class VehicleDetector:
    """
    Wraps a YOLOv8 model for detecting vehicles in video frames.

    Filters predictions to only return vehicle classes (car, motorcycle,
    bus, truck) above the configured confidence threshold.
    """

    def __init__(
        self,
        model_path: str = config.VEHICLE_MODEL_PATH,
        confidence: float = config.VEHICLE_CONFIDENCE,
        target_classes: list = None,
    ):
        """
        Initialize the vehicle detector.

        Args:
            model_path: Path to YOLOv8 weights. 'yolov8n.pt' is
                        auto-downloaded by Ultralytics if not present.
            confidence: Minimum confidence threshold for detections.
            target_classes: List of COCO class IDs to keep.
        """
        self.confidence = confidence
        self.target_classes = target_classes or config.VEHICLE_CLASSES

        logger.info(f"Loading vehicle detection model: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.device = resolve_device("VehicleDetector")
            self.model.to(self.device)
            logger.info("Vehicle detection model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vehicle model '{model_path}': {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects for vehicles found.
        """
        detections = []

        try:
            results = self.model(
                frame,
                conf=self.confidence,
                classes=self.target_classes,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Vehicle detection failed on frame: {e}")
            return detections

        # YOLOv8 returns a list; we process the first (and only) result
        if not results or len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            try:
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append(Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                ))
            except Exception as e:
                logger.debug(f"Skipping malformed detection {i}: {e}")
                continue

        logger.debug(f"Detected {len(detections)} vehicles in frame.")
        return detections
