"""
Frame Annotator — Draws all visual overlays on video frames.

Handles drawing of:
- Blue bounding boxes + ID labels for vehicles
- Green bounding boxes for license plates
- White-on-green text for recognized plate numbers
- Green tripwire/trigger line
"""

import cv2
import numpy as np

import config


class FrameAnnotator:
    """
    Draws visual annotations on video frames for the ANPR system.

    All drawing methods modify the frame in-place and also return it
    for chaining convenience.
    """

    def __init__(self):
        """Initialize with color and font settings from config."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.vehicle_color = config.COLOR_VEHICLE_BOX
        self.plate_color = config.COLOR_PLATE_BOX
        self.plate_text_bg = config.COLOR_PLATE_TEXT_BG
        self.plate_text_fg = config.COLOR_PLATE_TEXT_FG
        self.tripwire_color = config.COLOR_TRIPWIRE

    def draw_vehicle_box(
        self,
        frame: np.ndarray,
        bbox: tuple,
        track_id: int,
    ) -> np.ndarray:
        """
        Draw a blue bounding box around a vehicle with its ID label.

        Args:
            frame: BGR frame to annotate.
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            track_id: Persistent tracking ID number.

        Returns:
            Annotated frame.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1), (x2, y2),
            self.vehicle_color,
            thickness=2,
        )

        # Draw ID label above the box
        label = f"ID: {track_id}"
        font_scale = config.FONT_SCALE_ID
        thickness = config.FONT_THICKNESS

        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, font_scale, thickness
        )

        # Background rectangle for text readability
        label_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 5),
            (x1 + text_w + 8, label_y + 3),
            self.vehicle_color,
            -1,  # Filled
        )

        # Text in white on the colored background
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 2),
            self.font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def draw_plate_box(
        self,
        frame: np.ndarray,
        plate_bbox: tuple,
    ) -> np.ndarray:
        """
        Draw a green bounding box around a detected license plate.

        Args:
            frame: BGR frame to annotate.
            plate_bbox: (x1, y1, x2, y2) plate bounding box.

        Returns:
            Annotated frame.
        """
        x1, y1, x2, y2 = [int(v) for v in plate_bbox]

        cv2.rectangle(
            frame,
            (x1, y1), (x2, y2),
            self.plate_color,
            thickness=2,
        )

        return frame

    def draw_plate_text(
        self,
        frame: np.ndarray,
        text: str,
        vehicle_bbox: tuple,
        plate_bbox: tuple = None,
    ) -> np.ndarray:
        """
        Draw the recognized plate text with a filled green background.

        Places the text either at the plate location or at the bottom
        center of the vehicle bounding box.

        Args:
            frame: BGR frame to annotate.
            text: Recognized plate text string.
            vehicle_bbox: (x1, y1, x2, y2) of the vehicle.
            plate_bbox: Optional (x1, y1, x2, y2) of the plate.

        Returns:
            Annotated frame.
        """
        font_scale = config.FONT_SCALE_PLATE
        thickness = config.FONT_THICKNESS

        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.font, font_scale, thickness
        )

        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]

        if plate_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox]
            # Position text just below the plate box
            text_x = px1
            text_y = py2 + text_h + 8
        else:
            # Center at bottom of vehicle box
            text_x = vx1 + (vx2 - vx1 - text_w) // 2
            text_y = vy2 - 8

        # Ensure text stays within frame
        h, w = frame.shape[:2]
        text_x = max(0, min(text_x, w - text_w - 10))
        text_y = max(text_h + 5, min(text_y, h - 5))

        # Green filled background
        pad_x, pad_y = 6, 4
        cv2.rectangle(
            frame,
            (text_x - pad_x, text_y - text_h - pad_y),
            (text_x + text_w + pad_x, text_y + pad_y + baseline),
            self.plate_text_bg,
            -1,
        )

        # White text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            self.font,
            font_scale,
            self.plate_text_fg,
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def draw_tripwire(
        self,
        frame: np.ndarray,
        y_ratio: float = config.TRIPWIRE_Y_RATIO,
    ) -> np.ndarray:
        """
        Draw a green horizontal trigger line across the frame.

        Args:
            frame: BGR frame to annotate.
            y_ratio: Vertical position as fraction of frame height.

        Returns:
            Annotated frame.
        """
        h, w = frame.shape[:2]
        y_pos = int(h * y_ratio)

        cv2.line(
            frame,
            (0, y_pos), (w, y_pos),
            self.tripwire_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Label
        cv2.putText(
            frame,
            "TRIGGER LINE",
            (10, y_pos - 8),
            self.font,
            0.5,
            self.tripwire_color,
            1,
            cv2.LINE_AA,
        )

        return frame
