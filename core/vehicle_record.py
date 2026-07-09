"""
Vehicle Record — Data class for tracked vehicle state.

Maintains the best OCR reading and tracking state for each
vehicle across multiple frames.
"""

from dataclasses import dataclass, field
from typing import Optional

import config


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
        
        # Compute consensus over all historical readings
        from collections import defaultdict
        
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
        
        # Clean and translate consensus using shape-based mapping
        cleaned_consensus = self.clean_consensus(consensus)

        text_changed = cleaned_consensus != self.best_plate_text
        self.best_plate_text = cleaned_consensus
        self.best_confidence = max(self.best_confidence, confidence)
        if plate_bbox:
            self.plate_bbox = plate_bbox

        # Consensus is noisy on the first few votes — wait until it has
        # enough evidence before treating it as worth reporting. Without
        # this, every wobble in the running consensus gets logged as a
        # "new" plate reading for the same vehicle.
        min_votes = getattr(config, "PLATE_MIN_VOTES_TO_CONFIRM", 3)
        if len(self.plate_history) < min_votes:
            return False

        if not self.logged:
            self.logged = True
            return True

        return text_changed

    def clean_consensus(self, text: str) -> str:
        """
        Translates character mappings (English letters visually similar to Devanagari numerals)
        and isolates the registration digits.
        """
        import re
        
        # Convert Devanagari numbers to English numbers
        devanagari_nums = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        
        text_upper = text.upper()
        normalized = ""
        for char in text_upper:
            if char in devanagari_nums:
                normalized += devanagari_nums[char]
            else:
                normalized += char
                
        # Common English OCR misreads of Devanagari plates
        normalized = re.sub(r'PHI', '88', normalized)
        normalized = re.sub(r'PH', '88', normalized)
        normalized = re.sub(r'DN', '39', normalized)
        normalized = re.sub(r'DI', '39', normalized)
        normalized = re.sub(r'PI', '87', normalized)
        
        # Character mapping based on visual shape similarity to Devanagari numerals
        char_map = {
            'O': '0', 'Q': '0', 'U': '0',
            'I': '1', 'L': '1', 'T': '6',
            'Z': '2',
            'D': '3', 'J': '3', 'A': '3',
            'Y': '4',
            'S': '5',
            'B': '8', 'P': '8', 'H': '8', 'R': '8',
            'N': '9', 'G': '9', 'V': '6'
        }
        
        mapped = ""
        for char in normalized:
            if char in char_map:
                mapped += char_map[char]
            else:
                mapped += char
                
        # Keep only digits
        digits = re.sub(r'[^0-9]', '', mapped)
        
        if len(digits) >= 4:
            return digits[-4:]
        return digits if digits else text

    def update_position(self, bbox: tuple):
        """Update the vehicle's last known position."""
        self.last_bbox = bbox
        self.frames_seen += 1

    @property
    def has_plate(self) -> bool:
        """Whether a plate has been successfully read."""
        return self.best_plate_text is not None
