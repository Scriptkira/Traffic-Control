"""
Video I/O — Helpers for video capture and writing.

Wraps OpenCV VideoCapture and VideoWriter with error handling,
progress reporting, and automatic codec/FPS configuration.
"""

import logging
import queue
import threading
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

        # Request hardware-accelerated decode (GPU/D3D11 on Windows) —
        # 4K H.264/HEVC decode on CPU costs 15-30ms per frame otherwise.
        # Falls back to plain software decode if unsupported.
        try:
            self.cap = cv2.VideoCapture(
                source,
                cv2.CAP_ANY,
                [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
            )
            if not self.cap.isOpened():
                raise IOError("HW-accelerated open failed")
            logger.info("Video decode: hardware acceleration requested.")
        except Exception:
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


class FramePrefetcher:
    """
    Reads (and optionally downscales) frames on a background thread so
    decode never blocks the processing loop.

    read() returns (ret, proc_frame, hires_frame):
      - proc_frame: frame at processing resolution
      - hires_frame: the original full-res frame when downscaling is
        active (for OCR-quality crops), else None
    """

    def __init__(
        self,
        capture: "VideoCapture",
        proc_size: Optional[Tuple[int, int]] = None,
        keep_hires: bool = False,
        depth: int = 4,
    ):
        self.capture = capture
        self.proc_size = proc_size
        self.keep_hires = keep_hires
        # Two-stage pipeline: decode and resize each get their own
        # thread, so per-frame cost is max(decode, resize) instead of
        # their sum. Modest depths: each hires slot can be ~25MB.
        self._raw_queue: queue.Queue = queue.Queue(maxsize=depth)
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stopped = False
        self._decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self._resize_thread = threading.Thread(target=self._resize_worker, daemon=True)
        self._decode_thread.start()
        self._resize_thread.start()

    def _decode_worker(self):
        while not self._stopped:
            ret, frame = self.capture.read()
            self._raw_queue.put((ret, frame))
            if not ret:
                break

    def _resize_worker(self):
        while not self._stopped:
            ret, frame = self._raw_queue.get()
            if not ret:
                self._queue.put((False, None, None))
                break

            hires = None
            proc = frame
            if self.proc_size is not None:
                if self.keep_hires:
                    hires = frame
                proc = cv2.resize(
                    frame, self.proc_size, interpolation=cv2.INTER_LINEAR
                )
            self._queue.put((True, proc, hires))

    def read(self):
        """Blocking read of the next prefetched frame."""
        return self._queue.get()

    def stop(self):
        """Stop the reader threads (call before releasing the capture)."""
        self._stopped = True
        while self._decode_thread.is_alive() or self._resize_thread.is_alive():
            # Drain so blocked put() calls can complete and workers exit.
            for q in (self._raw_queue, self._queue):
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            self._decode_thread.join(timeout=0.05)
            self._resize_thread.join(timeout=0.05)


class VideoWriter:
    """
    Wraps cv2.VideoWriter with automatic codec and FPS setup.

    Frames are written asynchronously via a background thread so the
    main processing loop isn't blocked by MJPG compression / disk I/O.
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

        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            raise IOError(
                f"Cannot create video writer for: {output_path}. "
                "Check the output path and codec."
            )

        # Async write machinery: bounded queue + daemon worker thread.
        # Cap at 120 frames (~4s @ 30 fps) to keep memory bounded.
        self._queue: queue.Queue = queue.Queue(maxsize=120)
        self._stopped = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread: pull frames from the queue and write them."""
        while True:
            frame = self._queue.get()
            if frame is None:  # Poison pill → shut down
                break
            self._writer.write(frame)

    def write(self, frame):
        """Enqueue a frame for async writing."""
        if not self._stopped:
            self._queue.put(frame)

    def release(self):
        """Drain the queue, stop the worker, and release the writer."""
        if self._stopped:
            return
        self._stopped = True

        # Poison pill tells the worker to exit after draining.
        self._queue.put(None)
        self._thread.join()

        self._writer.release()
        logger.info(f"Video saved to: {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()

