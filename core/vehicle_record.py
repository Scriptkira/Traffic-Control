"""
Vehicle Record — Data class for tracked vehicle state.

Maintains the best OCR reading and tracking state for each
vehicle across multiple frames.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VehicleRecord:
    """
    Persistent record for a tracked vehicle.

    Stores the best OCR reading seen so far and tracking metadata.
    Only updates plate text when a higher-confidence reading arrives.
    """

    track_id: int
    best_plate_text: Optional[str] = None
    best_confidence: float = 0.0
    plate_bbox: Optional[tuple] = None
    crossed_tripwire: bool = False
    logged: bool = False
    frames_seen: int = 0
    last_bbox: Optional[tuple] = None

    def update_plate(
        self,
        plate_text: str,
        confidence: float,
        plate_bbox: tuple = None,
    ) -> bool:
        """
        Update the plate reading if the new one is better.

        A reading is considered "better" if:
        1. No existing reading exists, OR
        2. The new confidence is higher, OR
        3. The new text is longer (more characters recognized)

        Args:
            plate_text: New OCR reading.
            confidence: OCR confidence score.
            plate_bbox: Bounding box of the plate.

        Returns:
            True if the reading was updated, False otherwise.
        """
        is_better = (
            self.best_plate_text is None
            or confidence > self.best_confidence
            or (
                confidence >= self.best_confidence * 0.9
                and len(plate_text) > len(self.best_plate_text or "")
            )
        )

        if is_better:
            self.best_plate_text = plate_text
            self.best_confidence = confidence
            self.plate_bbox = plate_bbox
            return True

        return False

    def update_position(self, bbox: tuple):
        """Update the vehicle's last known position."""
        self.last_bbox = bbox
        self.frames_seen += 1

    @property
    def has_plate(self) -> bool:
        """Whether a plate has been successfully read."""
        return self.best_plate_text is not None
