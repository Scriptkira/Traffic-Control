"""
License Plate Detector — YOLO-based with contour fallback.

Primary: Uses a custom-trained YOLO model for plate detection.
Fallback: If no YOLO plate model is available, uses OpenCV contour-based
          heuristics to locate rectangular plate-like regions.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config
from utils.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class PlateDetection:
    """A detected license plate region."""
    bbox: tuple          # (x1, y1, x2, y2) in FRAME coordinates
    confidence: float
    crop: np.ndarray     # Cropped plate image for OCR


class PlateDetector:
    """
    Detects license plates within a vehicle's bounding box region.

    Attempts YOLO-based detection first. Falls back to contour-based
    detection if no YOLO plate model is available.
    """

    def __init__(
        self,
        model_path: str = config.PLATE_MODEL_PATH,
        confidence: float = config.PLATE_CONFIDENCE,
    ):
        """
        Initialize the plate detector.

        Args:
            model_path: Path to trained YOLO plate detection model.
            confidence: Minimum confidence for plate detections.
        """
        self.confidence = confidence
        self.model = None
        self.use_yolo = False
        self.device = "cpu"

        # Try to load YOLO plate model
        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.device = resolve_device("PlateDetector")
                self.model.to(self.device)
                self.use_yolo = True
                logger.info(f"Plate detection YOLO model loaded: {model_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load plate YOLO model: {e}. "
                    "Falling back to contour-based detection."
                )
        else:
            logger.info(
                f"No plate model found at '{model_path}'. "
                "Using contour-based fallback for plate detection."
            )

    def detect_in_roi(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple,
    ) -> list[PlateDetection]:
        """
        Detect candidate license plates within a vehicle's bounding box.

        Args:
            frame: Full BGR frame.
            vehicle_bbox: (x1, y1, x2, y2) of the vehicle in frame coords.

        Returns:
            List of PlateDetection candidates, sorted by score.
        """
        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
        h, w = frame.shape[:2]

        # Clamp to frame boundaries
        vx1 = max(0, vx1)
        vy1 = max(0, vy1)
        vx2 = min(w, vx2)
        vy2 = min(h, vy2)

        if vx2 - vx1 < 20 or vy2 - vy1 < 20:
            return []

        vehicle_crop = frame[vy1:vy2, vx1:vx2]

        if self.use_yolo:
            return self._detect_yolo(frame, vehicle_crop, vx1, vy1)
        else:
            return self._detect_contour(frame, vehicle_crop, vx1, vy1)

    def _detect_yolo(
        self,
        frame: np.ndarray,
        vehicle_crop: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[PlateDetection]:
        """Detect plate using YOLO model within vehicle crop."""
        detections = []
        try:
            results = self.model(
                vehicle_crop,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )

            if not results or len(results) == 0:
                return []

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                # Convert crop-relative coords to frame coords
                px1 = bbox[0] + offset_x
                py1 = bbox[1] + offset_y
                px2 = bbox[2] + offset_x
                py2 = bbox[3] + offset_y

                plate_crop = frame[py1:py2, px1:px2]

                if plate_crop.size > 0:
                    detections.append(PlateDetection(
                        bbox=(px1, py1, px2, py2),
                        confidence=conf,
                        crop=plate_crop,
                    ))

            # Sort by confidence descending
            detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
            return detections

        except Exception as e:
            logger.debug(f"YOLO plate detection failed: {e}")
            return []

    def _detect_contour(
        self,
        frame: np.ndarray,
        vehicle_crop: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[PlateDetection]:
        """
        Fallback: detect plate-like rectangles using contour analysis.

        Uses multiple strategies (edge-based + morphological) and scans
        both the lower portion and full vehicle crop to maximize recall.
        Works on low-resolution video by upscaling small crops.
        """
        detections = []
        seen_bboxes = set()

        try:
            orig_crop_h, orig_crop_w = vehicle_crop.shape[:2]
            vehicle_area = orig_crop_h * orig_crop_w

            # Always upscale the vehicle crop 3.0x for detailed contour analysis
            scale = 3.0
            work_crop = cv2.resize(
                vehicle_crop, None,
                fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            crop_h, crop_w = work_crop.shape[:2]

            # Max plate area: 35% of vehicle crop (plates are small)
            max_plate_area = vehicle_area * 0.35

            # Try two search regions: lower 50%, then full crop
            search_regions = [
                (int(crop_h * 0.3), crop_h, "lower"),
                (0, crop_h, "full"),
            ]

            all_candidates = []

            for region_start, region_end, region_name in search_regions:
                region = work_crop[region_start:region_end, :]
                if region.size == 0:
                    continue

                # Try multiple detection strategies
                all_candidates.extend(
                    self._edge_contours(region, region_start, scale)
                )
                all_candidates.extend(
                    self._morph_contours(region, region_start, scale)
                )

            # Sort candidates by score descending
            all_candidates = sorted(all_candidates, key=lambda x: x[4], reverse=True)

            for (x, y, w, h, score) in all_candidates:
                # Convert back to original scale + frame coords
                ox = int(x / scale) + offset_x
                oy = int(y / scale) + offset_y
                ow = max(int(w / scale), 1)
                oh = max(int(h / scale), 1)

                bbox_key = (ox, oy, ow, oh)
                if bbox_key in seen_bboxes:
                    continue
                seen_bboxes.add(bbox_key)

                # Reject if plate is too large relative to vehicle
                plate_area = ow * oh
                if plate_area > max_plate_area:
                    continue

                px1 = max(0, ox)
                py1 = max(0, oy)
                px2 = min(frame.shape[1], ox + ow)
                py2 = min(frame.shape[0], oy + oh)

                plate_crop = frame[py1:py2, px1:px2]

                if plate_crop.size > 0:
                    detections.append(PlateDetection(
                        bbox=(px1, py1, px2, py2),
                        confidence=score,
                        crop=plate_crop,
                    ))

            # Limit to top 5 candidates
            return detections[:5]

        except Exception as e:
            logger.debug(f"Contour plate detection failed: {e}")
            return []

    def _edge_contours(
        self, region: np.ndarray, y_offset: int, scale: float
    ) -> list:
        """Edge-based contour detection (Canny)."""
        candidates = []
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(gray, 20, 150)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                result = self._evaluate_contour(contour, y_offset)
                if result:
                    candidates.append(result)
        except Exception:
            pass
        return candidates

    def _morph_contours(
        self, region: np.ndarray, y_offset: int, scale: float
    ) -> list:
        """Morphological contour detection (threshold + close)."""
        candidates = []
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 5,
            )

            # Morphological close to merge plate characters into a blob
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                result = self._evaluate_contour(contour, y_offset)
                if result:
                    candidates.append(result)
        except Exception:
            pass
        return candidates

    def _evaluate_contour(
        self, contour, y_offset: int
    ) -> Optional[tuple]:
        """
        Evaluate if a contour looks like a license plate.

        Returns (x, y, w, h, score) or None.
        """
        area = cv2.contourArea(contour)
        if area < 100:  # Very permissive for low-res
            return None

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

        if len(approx) < 4 or len(approx) > 12:
            return None

        x, y, w, h = cv2.boundingRect(approx)

        if h == 0 or w == 0:
            return None

        aspect_ratio = w / h

        # Plate aspect ratio: 1.5 to 7.0
        # Minimum size: w >= 20px, h >= 8px (after potential upscaling)
        # Maximum area: 8000px^2 (in upscaled space) to avoid grabbing vehicle body
        if 1.5 <= aspect_ratio <= 7.0 and w >= 20 and h >= 8 and area < 8000:
            # Score: prefer medium-sized, rectangular shapes with plate-like AR
            rect_score = area / (w * h)  # How rectangular (0 to 1)
            # Penalize very large detections (likely not plates)
            size_score = min(area / 1500.0, 1.0) * (1.0 - min(area / 8000.0, 0.5))
            # Prefer aspect ratios close to typical plates (3.0 to 5.0)
            ar_score = 1.0 - min(abs(aspect_ratio - 4.0) / 3.0, 1.0)
            score = rect_score * 0.3 + size_score * 0.3 + ar_score * 0.4

            return (x, y + y_offset, w, h, score)

        return None
