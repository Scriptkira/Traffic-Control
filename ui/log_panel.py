"""
Log Panel — Right-side on-screen logging panel for plate detections.

Creates a semi-transparent dark panel on the right side of the video
frame that displays detected license plates with vehicle IDs and
timestamps.
"""

import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


class LogEntry:
    """A single log entry for a detected plate."""

    def __init__(
        self,
        vehicle_id: int,
        plate_text: str,
        timestamp: str = None,
        camera_id: str = config.CAMERA_ID,
    ):
        self.vehicle_id = vehicle_id
        self.plate_text = plate_text
        self.timestamp = timestamp or datetime.datetime.now().strftime("%H:%M:%S")
        self.camera_id = camera_id

    def __repr__(self):
        return (
            f"ID:{self.vehicle_id} | {self.plate_text} | "
            f"{self.timestamp} | {self.camera_id}"
        )


class LogPanel:
    """
    Renders a right-side logging panel showing detected plates.

    Features:
    - Semi-transparent dark background
    - Title header ("PLATE LOG")
    - Scrolling entries with vehicle ID, plate text, and timestamp
    - Deduplication: only logs each vehicle ID once (keeps best reading)
    """

    def __init__(
        self,
        panel_width: int = config.LOG_PANEL_WIDTH,
        max_entries: int = config.LOG_PANEL_MAX_ENTRIES,
    ):
        """
        Initialize the log panel.

        Args:
            panel_width: Width of the panel in pixels.
            max_entries: Maximum visible entries before scrolling.
        """
        self.panel_width = panel_width
        self.max_entries = max_entries
        self.entries: List[LogEntry] = []
        self._logged_ids: dict = {}  # vehicle_id -> LogEntry

        self.bg_color = config.LOG_PANEL_BG_COLOR
        self.bg_alpha = config.LOG_PANEL_BG_ALPHA
        self.title_color = config.COLOR_LOG_TITLE
        self.text_color = config.COLOR_LOG_TEXT
        self.separator_color = config.COLOR_LOG_SEPARATOR

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.FONT_SCALE_LOG
        self.font_thickness = config.FONT_THICKNESS_LOG

    def add_entry(
        self,
        vehicle_id: int,
        plate_text: str,
        confidence: float = 0.0,
        timestamp: str = None,
    ) -> bool:
        """
        Add or update a log entry.

        If the vehicle ID already has an entry, only updates if the
        new reading has higher confidence.

        Args:
            vehicle_id: Tracked vehicle ID.
            plate_text: Recognized plate text.
            confidence: OCR confidence score.
            timestamp: Optional timestamp string.

        Returns:
            True if entry was added/updated, False if skipped.
        """
        if vehicle_id in self._logged_ids:
            existing = self._logged_ids[vehicle_id]
            # Only update if this reading seems better (longer text)
            if len(plate_text) <= len(existing.plate_text):
                return False

        entry = LogEntry(
            vehicle_id=vehicle_id,
            plate_text=plate_text,
            timestamp=timestamp,
        )

        self._logged_ids[vehicle_id] = entry

        # Rebuild sorted entry list
        self.entries = list(self._logged_ids.values())

        # Trim to max entries (keep newest)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        return True

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Render the log panel onto the right side of the frame.

        Creates a new wider frame with the panel appended on the right.

        Args:
            frame: Original BGR frame.

        Returns:
            New frame with the log panel appended on the right.
        """
        h, w = frame.shape[:2]

        # Create the panel as a separate image
        panel = np.zeros((h, self.panel_width, 3), dtype=np.uint8)
        panel[:] = self.bg_color

        # ── Header ──
        header_h = 50
        cv2.rectangle(
            panel, (0, 0), (self.panel_width, header_h),
            (20, 20, 20), -1
        )

        # Title
        cv2.putText(
            panel, "PLATE DETECTION LOG",
            (10, 32),
            self.font, 0.55, self.title_color, 1, cv2.LINE_AA,
        )

        # Separator line under header
        cv2.line(
            panel,
            (5, header_h), (self.panel_width - 5, header_h),
            self.title_color, 1,
        )

        # ── Column Headers ──
        col_y = header_h + 25
        cv2.putText(
            panel, "ID    PLATE         TIME     CAM",
            (10, col_y),
            self.font, 0.38, self.title_color, 1, cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (5, col_y + 8), (self.panel_width - 5, col_y + 8),
            self.separator_color, 1,
        )

        # ── Entries ──
        entry_start_y = col_y + 30
        line_height = 28

        for i, entry in enumerate(reversed(self.entries)):
            y = entry_start_y + i * line_height

            if y > h - 20:  # Stop if we'd overflow the panel
                break

            # Alternate row shading
            if i % 2 == 0:
                cv2.rectangle(
                    panel,
                    (2, y - 16), (self.panel_width - 2, y + 8),
                    (40, 40, 40), -1,
                )

            # Format: "ID:XX  PLATE_TEXT  HH:MM:SS  CAM"
            id_text = f"{entry.vehicle_id:>3}"
            plate_text = f"{entry.plate_text:<13}"
            time_text = entry.timestamp
            cam_text = entry.camera_id

            # Vehicle ID (green)
            cv2.putText(
                panel, id_text,
                (10, y),
                self.font, self.font_scale, self.title_color,
                self.font_thickness, cv2.LINE_AA,
            )

            # Plate text (white)
            cv2.putText(
                panel, plate_text,
                (55, y),
                self.font, self.font_scale, (255, 255, 255),
                self.font_thickness, cv2.LINE_AA,
            )

            # Timestamp (gray)
            cv2.putText(
                panel, time_text,
                (190, y),
                self.font, self.font_scale - 0.05, (160, 160, 160),
                self.font_thickness, cv2.LINE_AA,
            )

            # Camera (dim)
            cv2.putText(
                panel, cam_text,
                (255, y),
                self.font, self.font_scale - 0.05, (120, 120, 120),
                self.font_thickness, cv2.LINE_AA,
            )

        # ── Footer / Stats ──
        footer_y = h - 30
        cv2.line(
            panel,
            (5, footer_y - 10), (self.panel_width - 5, footer_y - 10),
            self.separator_color, 1,
        )
        total_text = f"Total Plates: {len(self.entries)}"
        cv2.putText(
            panel, total_text,
            (10, footer_y + 5),
            self.font, 0.45, self.title_color, 1, cv2.LINE_AA,
        )

        # ── Combine frame + panel ──
        combined = np.hstack([frame, panel])

        return combined

    def get_entries(self) -> List[LogEntry]:
        """Return all current log entries."""
        return list(self.entries)

    def clear(self):
        """Clear all log entries."""
        self.entries.clear()
        self._logged_ids.clear()
