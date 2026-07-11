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
        self.tripwire_alert_color = getattr(config, "COLOR_TRIPWIRE_ALERT", (0, 0, 255))

    def _draw_hud_corners(self, img, pt1, pt2, color, thickness, scale=1.0):
        """Draw tech HUD corner brackets around a box."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = int(14 * scale)  # length of corner lines

        # Top left
        cv2.line(img, (x1, y1), (x1 + r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + r), color, thickness, cv2.LINE_AA)
        # Top right
        cv2.line(img, (x2, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + r), color, thickness, cv2.LINE_AA)
        # Bottom left
        cv2.line(img, (x1, y2), (x1 + r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        # Bottom right
        cv2.line(img, (x2, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - r), color, thickness, cv2.LINE_AA)

    def draw_vehicle_box(
        self,
        frame: np.ndarray,
        bbox: tuple,
        track_id: int,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Draw a HUD-style target tracker around a vehicle.

        Args:
            frame: BGR frame to annotate.
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            track_id: Persistent tracking ID number.
            scale: Current scaling factor.

        Returns:
            Annotated frame.
            """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 1. Subtle semi-transparent fill — blend only inside the box ROI.
        # (A full-frame copy + addWeighted per vehicle made this the
        # pipeline's hottest function in dense traffic: ~38ms/frame.)
        fh, fw = frame.shape[:2]
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(fw, x2), min(fh, y2)
        if rx2 > rx1 and ry2 > ry1:
            roi = frame[ry1:ry2, rx1:rx2]
            tint = np.empty_like(roi)
            tint[:] = self.vehicle_color
            cv2.addWeighted(tint, 0.06, roi, 0.94, 0, dst=roi)

        # 2. Draw thin bounding box border
        border_thickness = max(1, int(1 * scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.vehicle_color, border_thickness, cv2.LINE_AA)

        # 3. Draw high-tech HUD corner brackets
        bracket_thickness = max(2, int(2 * scale))
        self._draw_hud_corners(frame, (x1, y1), (x2, y2), self.vehicle_color, bracket_thickness, scale)

        # 4. Draw ID label block
        label = f"SYS-TRK: {track_id:02d}"
        font_scale = config.FONT_SCALE_ID * scale
        text_thickness = max(1, int(config.FONT_THICKNESS * scale))

        (text_w, text_h), baseline = cv2.getTextSize(label, self.font, font_scale, text_thickness)
        label_y = max(y1 - int(6 * scale), text_h + int(5 * scale))

        # Filled background rectangle for text
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - int(6 * scale)),
            (x1 + text_w + int(8 * scale), label_y + int(4 * scale)),
            self.vehicle_color,
            -1
        )

        # Text in black on the cyan background for contrast
        cv2.putText(
            frame,
            label,
            (x1 + int(4 * scale), label_y - int(1 * scale)),
            self.font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

        return frame

    def draw_plate_box(
        self,
        frame: np.ndarray,
        plate_bbox: tuple,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Draw a green bounding box around a detected license plate.

        Args:
            frame: BGR frame to annotate.
            plate_bbox: (x1, y1, x2, y2) plate bounding box.
            scale: Current scaling factor.

        Returns:
            Annotated frame.
        """
        x1, y1, x2, y2 = [int(v) for v in plate_bbox]

        # Draw a clean neon green box border
        thickness = max(2, int(2 * scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.plate_color, thickness, cv2.LINE_AA)

        return frame

    def draw_plate_text(
        self,
        frame: np.ndarray,
        text: str,
        vehicle_bbox: tuple,
        plate_bbox: tuple = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Draw the recognized plate text inside a high-tech HUD card.

        Args:
            frame: BGR frame to annotate.
            text: Recognized plate text string.
            vehicle_bbox: (x1, y1, x2, y2) of the vehicle.
            plate_bbox: Optional (x1, y1, x2, y2) of the plate.
            scale: Current scaling factor.

        Returns:
            Annotated frame.
        """
        font_scale = config.FONT_SCALE_PLATE * scale
        thickness = max(1, int(config.FONT_THICKNESS * scale))

        # Format display text
        display_text = f" {text} "

        (text_w, text_h), baseline = cv2.getTextSize(display_text, self.font, font_scale, thickness)

        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]

        if plate_bbox is not None:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox]
            # Position text just below the plate box
            text_x = px1
            text_y = py2 + text_h + int(8 * scale)
        else:
            # Center at bottom of vehicle box
            text_x = vx1 + (vx2 - vx1 - text_w) // 2
            text_y = vy2 - int(8 * scale)

        # Ensure text stays within frame
        h, w = frame.shape[:2]
        text_x = max(0, min(text_x, w - text_w - int(10 * scale)))
        text_y = max(text_h + int(5 * scale), min(text_y, h - int(5 * scale)))

        # 1. Dark card background
        pad_x, pad_y = int(6 * scale), int(4 * scale)
        cv2.rectangle(
            frame,
            (text_x - pad_x, text_y - text_h - pad_y),
            (text_x + text_w + pad_x, text_y + pad_y + baseline),
            self.plate_text_bg,
            -1,
        )

        # 2. Glowing neon green border for the card
        border_thickness = max(1, int(1 * scale))
        cv2.rectangle(
            frame,
            (text_x - pad_x, text_y - text_h - pad_y),
            (text_x + text_w + pad_x, text_y + pad_y + baseline),
            self.plate_color,
            border_thickness,
            cv2.LINE_AA,
        )

        # 3. Draw text in glowing neon green
        cv2.putText(
            frame,
            display_text,
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
        is_alert: bool = False,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Draw a glowing horizontal trigger line across the frame.

        Args:
            frame: BGR frame to annotate.
            y_ratio: Vertical position as fraction of frame height.
            is_alert: If True, draws the alert state (Red glow).
            scale: Current scaling factor.

        Returns:
            Annotated frame.
        """
        h, w = frame.shape[:2]
        y_pos = int(h * y_ratio)

        color = self.tripwire_alert_color if is_alert else self.tripwire_color
        thickness = max(2, int(2 * scale))

        # Draw main tripwire line
        cv2.line(frame, (0, y_pos), (w, y_pos), color, thickness, cv2.LINE_AA)

        # Semi-transparent glow strip — blend only the band of rows the
        # glow touches instead of copying the whole frame.
        glow_t = int(8 * scale)
        by1 = max(0, y_pos - glow_t)
        by2 = min(h, y_pos + glow_t + 1)
        if by2 > by1:
            band = frame[by1:by2]
            glow = band.copy()
            cv2.line(glow, (0, y_pos - by1), (w, y_pos - by1), color, glow_t, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.15, band, 0.85, 0, dst=band)

        # Text label
        label = "▲ TRIPWIRE ALERT ACTIVE" if is_alert else "▲ VEHICLE TRIPWIRE"
        font_scale = 0.45 * scale
        text_thickness = max(1, int(1 * scale))

        cv2.putText(
            frame,
            label,
            (int(15 * scale), y_pos - int(8 * scale)),
            self.font,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

        return frame

