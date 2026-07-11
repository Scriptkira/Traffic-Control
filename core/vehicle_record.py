"""
Vehicle Record — Data class for tracked vehicle state.

Maintains the best OCR reading and tracking state for each
vehicle across multiple frames.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import config
from ocr.text_normalize import clean_consensus_reading


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

    # History of raw readings: list of (plate_text, confidence, area)
    plate_history: list = field(default_factory=list)

    # A candidate replacement consensus text seen on the immediately
    # preceding vote, awaiting a second, agreeing vote before it's
    # allowed to overwrite an already-logged reading (see update_plate).
    pending_consensus: Optional[str] = None

    def update_plate(
        self,
        plate_text: str,
        confidence: float,
        plate_bbox: tuple = None,
    ) -> bool:
        """
        Update the plate reading using area-weighted confidence character consensus.
        """
        if not plate_text:
            return False

        # Compute crop area as spatial confidence weight
        if plate_bbox:
            x1, y1, x2, y2 = plate_bbox
            area = max(1.0, (x2 - x1) * (y2 - y1))
        else:
            area = 1.0

        # Append new reading to history: (plate_text, confidence, area)
        self.plate_history.append((plate_text, confidence, area))

        # Find max length of cleaned readings in history
        max_len = max(len(r) for r, _, _ in self.plate_history)

        consensus_chars = []
        for i in range(1, max_len + 1):
            char_weights = defaultdict(float)
            for r, conf, a in self.plate_history:
                if len(r) >= i:
                    char = r[-i]
                    # Weight vote by confidence * area
                    char_weights[char] += conf * a
            if char_weights:
                best_char = max(char_weights.items(), key=lambda x: x[1])[0]
                consensus_chars.append(best_char)

        consensus = "".join(reversed(consensus_chars))

        # Normalize Devanagari numerals and strip punctuation/whitespace
        cleaned_consensus = clean_consensus_reading(consensus)

        self.best_confidence = max(self.best_confidence, confidence)
        if plate_bbox:
            self.plate_bbox = plate_bbox

        # Consensus is noisy on the first few votes — wait until it has
        # enough evidence before treating it as worth reporting. Without
        # this, every wobble in the running consensus gets logged as a
        # "new" plate reading for the same vehicle.
        min_votes = getattr(config, "PLATE_MIN_VOTES_TO_CONFIRM", 3)
        if len(self.plate_history) < min_votes:
            self.best_plate_text = cleaned_consensus
            self.pending_consensus = None
            return False

        if not self.logged:
            self.logged = True
            self.best_plate_text = cleaned_consensus
            self.pending_consensus = None
            return True

        # Already logged once. A single new vote that disagrees with the
        # stored reading is often just noise from a fresh OCR angle, not
        # a genuine correction — only accept the change once the same
        # new value shows up on two consecutive votes.
        if cleaned_consensus == self.best_plate_text:
            self.pending_consensus = None
            return False

        if cleaned_consensus == self.pending_consensus:
            self.best_plate_text = cleaned_consensus
            self.pending_consensus = None
            return True

        self.pending_consensus = cleaned_consensus
        return False

    def update_position(self, bbox: tuple):
        """Update the vehicle's last known position."""
        self.last_bbox = bbox
        self.frames_seen += 1

    @property
    def has_plate(self) -> bool:
        """Whether a plate has been successfully read."""
        return self.best_plate_text is not None

    # ── Relative plate position caching ──────────────────────────
    # Once a vehicle's plate has been successfully read at high confidence,
    # we store where the plate sits relative to the vehicle bbox so we can
    # project it onto future frames without running the plate detector.

    def cache_relative_plate_pos(self, plate_bbox: tuple, vehicle_bbox: tuple):
        """Store plate position as fractions of the vehicle bbox dimensions."""
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        px1, py1, px2, py2 = plate_bbox
        self._rel_plate = (
            (px1 - vx1) / vw,
            (py1 - vy1) / vh,
            (px2 - vx1) / vw,
            (py2 - vy1) / vh,
        )

    def project_plate_bbox(self, vehicle_bbox: tuple):
        """Return the cached plate bbox projected onto the current vehicle bbox, or None."""
        rel = getattr(self, "_rel_plate", None)
        if rel is None:
            return None
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vw = vx2 - vx1
        vh = vy2 - vy1
        return (
            int(vx1 + rel[0] * vw),
            int(vy1 + rel[1] * vh),
            int(vx1 + rel[2] * vw),
            int(vy1 + rel[3] * vh),
        )

