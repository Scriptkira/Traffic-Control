"""
Log Panel — Right-side on-screen logging panel for plate detections.

Creates a semi-transparent dark panel on the right side of the video
frame that displays detected license plates with vehicle IDs,
confidence scores, and timestamps.
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
        confidence: float = 0.0,
        timestamp: str = None,
        camera_id: str = config.CAMERA_ID,
    ):
        self.vehicle_id = vehicle_id
        self.plate_text = plate_text
        self.confidence = confidence
        self.timestamp = timestamp or datetime.datetime.now().strftime("%H:%M:%S")
        self.camera_id = camera_id

    def __repr__(self):
        return (
            f"ID:{self.vehicle_id} | {self.plate_text} | "
            f"Conf:{self.confidence:.2f} | {self.timestamp} | {self.camera_id}"
        )


class LogPanel:
    """
    Renders an advanced right-side logging panel showing detected plates.

    Features:
    - High-definition scaling support
    - Blinking telemetry active indicator
    - Real-time clock and date display
    - Scrolling entries with vehicle ID, plate text, confidence, and timestamp
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

        Callers (VehicleRecord.update_plate via the pipeline) already
        decide whether a reading is worth (re-)logging — this just
        records it, so the panel never silently disagrees with the
        console log about whether an update happened.

        Args:
            vehicle_id: Tracked vehicle ID.
            plate_text: Recognized plate text.
            confidence: OCR confidence score.
            timestamp: Optional timestamp string.

        Returns:
            True if entry was added/updated.
        """
        entry = LogEntry(
            vehicle_id=vehicle_id,
            plate_text=plate_text,
            confidence=confidence,
            timestamp=timestamp,
        )

        self._logged_ids[vehicle_id] = entry

        # Rebuild entry list sorted by timestamp
        self.entries = list(self._logged_ids.values())

        # Trim to max entries (keep newest)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        return True

    def render(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Render the log panel onto the right side of the frame.

        Creates a new wider frame with the panel appended on the right.

        Args:
            frame: Original BGR frame (already upscaled).
            scale: Scale multiplier used for fonts and margins.

        Returns:
            New frame with the log panel appended on the right.
        """
        h, w = frame.shape[:2]
        scaled_width = int(self.panel_width * scale)

        # Create the panel canvas
        panel = np.zeros((h, scaled_width, 3), dtype=np.uint8)
        panel[:] = self.bg_color

        # Draw left border separator
        border_thickness = max(1, int(1 * scale))
        cv2.line(panel, (0, 0), (0, h), self.separator_color, border_thickness)

        # ── Title Header Block ──
        header_h = int(55 * scale)
        cv2.rectangle(panel, (0, 0), (scaled_width, header_h), (22, 22, 22), -1)

        # Blinking LED state (based on current time seconds)
        now = datetime.datetime.now()
        led_color = (0, 0, 255) if now.second % 2 == 0 else (20, 20, 100) # Blinking Red
        led_radius = int(4 * scale)
        led_center = (scaled_width - int(95 * scale), int(26 * scale))

        # LED circle
        cv2.circle(panel, led_center, led_radius, led_color, -1, cv2.LINE_AA)
        
        # Telemetry title
        cv2.putText(
            panel,
            "▲ DEEP ANPR TELEMETRY",
            (int(15 * scale), int(31 * scale)),
            self.font,
            0.45 * scale,
            self.title_color,
            max(1, int(1 * scale)),
            cv2.LINE_AA
        )

        # "ACTIVE" label next to LED
        cv2.putText(
            panel,
            "LIVE FEED",
            (scaled_width - int(85 * scale), int(30 * scale)),
            self.font,
            0.35 * scale,
            (150, 150, 150),
            max(1, int(1 * scale)),
            cv2.LINE_AA
        )

        # Header line divider
        cv2.line(
            panel,
            (0, header_h),
            (scaled_width, header_h),
            self.separator_color,
            border_thickness
        )

        # ── Telemetry Stats Panel ──
        telemetry_h = int(30 * scale)
        telemetry_y = header_h + telemetry_h
        cv2.rectangle(panel, (0, header_h + 1), (scaled_width, telemetry_y), (18, 18, 18), -1)

        # Date & Time string
        time_str = now.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            panel,
            f"CAM: {config.CAMERA_ID}  |  {time_str}",
            (int(15 * scale), header_h + int(19 * scale)),
            self.font,
            0.35 * scale,
            (120, 120, 120),
            max(1, int(1 * scale)),
            cv2.LINE_AA
        )

        cv2.line(
            panel,
            (0, telemetry_y),
            (scaled_width, telemetry_y),
            self.separator_color,
            border_thickness
        )

        # ── Column Headers ──
        col_y = telemetry_y + int(24 * scale)
        cv2.putText(
            panel,
            "ID    PLATE NUMBER       CONF     TIME",
            (int(15 * scale), col_y),
            self.font,
            0.35 * scale,
            self.title_color,
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (int(10 * scale), col_y + int(8 * scale)),
            (scaled_width - int(10 * scale), col_y + int(8 * scale)),
            self.separator_color,
            border_thickness,
        )

        # ── Logs Entries ──
        entry_start_y = col_y + int(32 * scale)
        line_height = int(32 * scale)

        for i, entry in enumerate(reversed(self.entries)):
            y = entry_start_y + i * line_height

            if y > h - int(50 * scale):  # Avoid overflow
                break

            # Styled container card background (alternating rows)
            card_bg = (25, 25, 25) if i % 2 == 0 else (18, 18, 18)
            cv2.rectangle(
                panel,
                (int(10 * scale), y - int(18 * scale)),
                (scaled_width - int(10 * scale), y + int(10 * scale)),
                card_bg,
                -1,
            )
            # Subtle card border
            cv2.rectangle(
                panel,
                (int(10 * scale), y - int(18 * scale)),
                (scaled_width - int(10 * scale), y + int(10 * scale)),
                (40, 40, 40),
                1,
                cv2.LINE_AA
            )

            # Columns formatting
            id_txt = f"{entry.vehicle_id:02d}"
            plate_txt = entry.plate_text
            conf_val = int(entry.confidence * 100)
            conf_txt = f"{conf_val}%"
            time_txt = entry.timestamp

            # Vehicle ID (Cyan)
            cv2.putText(
                panel,
                id_txt,
                (int(20 * scale), y),
                self.font,
                self.font_scale * scale,
                self.title_color,
                max(1, int(self.font_thickness * scale)),
                cv2.LINE_AA,
            )

            # Plate Text (White)
            cv2.putText(
                panel,
                plate_txt,
                (int(60 * scale), y),
                self.font,
                self.font_scale * scale,
                (255, 255, 255),
                max(1, int(self.font_thickness * scale)),
                cv2.LINE_AA,
            )

            # Confidence Badge (Green/Yellow based on score)
            conf_color = (0, 255, 100) if conf_val >= 50 else (0, 220, 220)
            cv2.putText(
                panel,
                conf_txt,
                (int(200 * scale), y),
                self.font,
                self.font_scale * scale,
                conf_color,
                max(1, int(self.font_thickness * scale)),
                cv2.LINE_AA,
            )

            # Timestamp (Gray)
            cv2.putText(
                panel,
                time_txt,
                (int(255 * scale), y),
                self.font,
                (self.font_scale - 0.05) * scale,
                (140, 140, 140),
                max(1, int(self.font_thickness * scale)),
                cv2.LINE_AA,
            )

        # ── Footer / Stats ──
        footer_y = h - int(30 * scale)
        cv2.line(
            panel,
            (int(10 * scale), footer_y - int(10 * scale)),
            (scaled_width - int(10 * scale), footer_y - int(10 * scale)),
            self.separator_color,
            border_thickness,
        )

        total_text = f"Total Unique Plates Logged: {len(self._logged_ids)}"
        cv2.putText(
            panel,
            total_text,
            (int(15 * scale), footer_y + int(5 * scale)),
            self.font,
            0.40 * scale,
            self.title_color,
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )

        # Combine frame + panel side-by-side
        combined = np.hstack([frame, panel])

        return combined

    def get_entries(self) -> List[LogEntry]:
        """Return all current log entries."""
        return list(self.entries)

    def clear(self):
        """Clear all log entries."""
        self.entries.clear()
        self._logged_ids.clear()
