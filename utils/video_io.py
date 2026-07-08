"""
Video I/O — Helpers for video capture and writing.

Wraps OpenCV VideoCapture and VideoWriter with error handling,
progress reporting, and automatic codec/FPS configuration.
"""

import logging
from typing import Optional, Tuple

import cv2

import config

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Wraps cv2.VideoCapture with metadata access and error handling.

    Supports video files and webcam indices.
    """

    def __init__(self, source):
        """
        Initialize video capture.

        Args:
            source: File path (str) or webcam index (int).
        """
        self.source = source

        if isinstance(source, str):
            logger.info(f"Opening video file: {source}")
        else:
            logger.info(f"Opening webcam: {source}")

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise IOError(
                f"Cannot open video source: {source}. "
                "Check the file path or webcam index."
            )

        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video properties: {self._width}x{self._height} @ "
            f"{self._fps:.1f} FPS, {self._total_frames} total frames"
        )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def read(self) -> Tuple[bool, Optional['numpy.ndarray']]:
        """Read the next frame. Returns (success, frame)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video capture released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


class VideoWriter:
    """
    Wraps cv2.VideoWriter with automatic codec and FPS setup.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = None,
        codec: str = None,
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path for the output video file.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Frames per second (defaults to config).
            codec: FourCC codec string (defaults to config).
        """
        self.output_path = output_path
        fps = fps or config.DEFAULT_FPS
        codec = codec or config.OUTPUT_CODEC

        fourcc = cv2.VideoWriter_fourcc(*codec)

        logger.info(
            f"Creating video writer: {output_path} "
            f"({width}x{height} @ {fps:.1f} FPS, codec={codec})"
        )

        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise IOError(
                f"Cannot create video writer for: {output_path}. "
                "Check the output path and codec."
            )

    def write(self, frame):
        """Write a single frame to the video file."""
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """Release the video writer resource."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"Video saved to: {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()
