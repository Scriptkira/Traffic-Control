"""
ANPR Traffic Camera System — Main Entry Point

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --output result.mp4
    python main.py --input video.mp4 --output result.mp4 --show
    python main.py --input 0              # Webcam
    python main.py --input video.mp4 --no-ocr  # Skip OCR (faster)

Controls (when --show is enabled):
    Q or ESC  : Quit
    SPACE     : Pause/Resume
"""

import argparse
import logging
import sys
import time

import cv2

import config
from core.pipeline import ANPRPipeline
from utils.video_io import FramePrefetcher, VideoCapture, VideoWriter


def setup_logging(verbose: bool = False):
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Suppress noisy libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ANPR Traffic Camera System — "
                    "Vehicle Detection, Tracking & License Plate Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file, or camera index (e.g., 0 for webcam).",
    )

    parser.add_argument(
        "--output", "-o",
        default=config.DEFAULT_OUTPUT_PATH,
        help=f"Path for annotated output video (default: {config.DEFAULT_OUTPUT_PATH}).",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display live preview window (default: True).",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live preview window.",
    )

    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip writing the annotated output video (live preview only; "
             "frees CPU that MJPG encoding would otherwise use).",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    parser.add_argument(
        "--output-scale",
        type=float,
        default=None,
        help=f"Override the output upscale multiplier "
             f"(default: {config.OUTPUT_SCALE} from config.py). "
             f"Use 1.0 to avoid upscaling already-HD/4K sources.",
    )

    parser.add_argument(
        "--codec",
        default=None,
        help=f"Override the output FourCC codec "
             f"(default: {config.OUTPUT_CODEC} from config.py). "
             f"e.g. 'mp4v' for a much smaller .mp4 than MJPG .avi.",
    )

    return parser.parse_args()


def main():
    """Main entry point — video processing loop."""
    args = parse_args()
    setup_logging(args.verbose)

    # Shorter GIL switch interval: with detection/OCR/decode all on
    # background threads, the default 5ms lets a CPU-bound worker hold
    # the GIL long enough to visibly stall the frame loop.
    sys.setswitchinterval(0.001)

    logger = logging.getLogger("main")

    if args.output_scale is not None:
        config.OUTPUT_SCALE = args.output_scale
    if args.codec is not None:
        config.OUTPUT_CODEC = args.codec

    # Determine show mode
    show_preview = args.show and not args.no_show

    # Determine input source (file or webcam)
    try:
        source = int(args.input)  # Webcam index
    except ValueError:
        source = args.input  # File path

    logger.info("=" * 60)
    logger.info("  ANPR Traffic Camera System")
    logger.info("=" * 60)
    logger.info(f"  Input  : {source}")
    logger.info(f"  Output : {args.output}")
    logger.info(f"  Preview: {'ON' if show_preview else 'OFF'}")
    logger.info("=" * 60)

    # ── Initialize video capture ──
    try:
        video_cap = VideoCapture(source)
    except IOError as e:
        logger.error(f"Failed to open video: {e}")
        sys.exit(1)

    # ── Initialize pipeline ──
    pipeline = ANPRPipeline()

    # ── Determine processing resolution ──
    # Frames wider than PROCESS_MAX_WIDTH are downscaled before entering
    # the pipeline (see config.py for the speed/accuracy trade-off).
    proc_w, proc_h = video_cap.width, video_cap.height
    proc_scale = 1.0
    max_proc_w = getattr(config, "PROCESS_MAX_WIDTH", None)
    if max_proc_w and video_cap.width > max_proc_w:
        proc_scale = max_proc_w / video_cap.width
        proc_w = max_proc_w
        proc_h = int(round(video_cap.height * proc_scale))
        logger.info(
            f"Processing at {proc_w}x{proc_h} "
            f"(downscaled from {video_cap.width}x{video_cap.height} for speed)"
        )

    # ── Initialize video writer ──
    # Output width includes the log panel, upscaled for HD output
    scale = getattr(config, "OUTPUT_SCALE", 1.0)
    output_width = int(proc_w * scale) + int(config.LOG_PANEL_WIDTH * scale)
    output_height = int(proc_h * scale)

    video_writer = None
    if args.no_output:
        logger.info("Output video disabled (--no-output).")
    else:
        try:
            video_writer = VideoWriter(
                output_path=args.output,
                width=output_width,
                height=output_height,
                fps=video_cap.fps,
            )
        except IOError as e:
            logger.error(f"Failed to create output video: {e}")
            video_cap.release()
            sys.exit(1)

    # ── Frame prefetcher: decode + downscale on a background thread ──
    prefetcher = FramePrefetcher(
        video_cap,
        proc_size=(proc_w, proc_h) if proc_scale != 1.0 else None,
        keep_hires=proc_scale != 1.0,
    )

    # ── Processing loop ──
    frame_num = 0
    paused = False
    start_time = time.time()

    logger.info("Processing started. Press 'Q' or ESC to quit.")

    try:
        while True:
            if not paused:
                ret, frame, hires_frame = prefetcher.read()

                if not ret:
                    logger.info("End of video reached.")
                    break

                frame_num += 1

                # Process through pipeline (hires_frame carries the
                # original full-res frame for OCR-quality crops)
                annotated = pipeline.process_frame(frame, hires_frame=hires_frame)

                # Write output
                if video_writer is not None:
                    video_writer.write(annotated)

                # Progress logging (every 100 frames)
                if frame_num % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_num / elapsed if elapsed > 0 else 0
                    progress = ""
                    if video_cap.total_frames > 0:
                        pct = (frame_num / video_cap.total_frames) * 100
                        progress = f" ({pct:.1f}%)"

                    stats = pipeline.get_stats()
                    logger.info(
                        f"Frame {frame_num}{progress} | "
                        f"{fps:.1f} FPS | "
                        f"Vehicles: {stats['total_vehicles_tracked']} | "
                        f"Plates: {stats['plates_detected']}"
                    )

                # Show preview (every 2nd frame — halves display overhead
                # without visibly changing the preview)
                if show_preview and frame_num % 2 == 0:
                    # Resize for display if too large
                    display = annotated
                    disp_h, disp_w = display.shape[:2]
                    max_width = 1400
                    if disp_w > max_width:
                        scale = max_width / disp_w
                        display = cv2.resize(
                            display,
                            (int(disp_w * scale), int(disp_h * scale)),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    cv2.imshow("ANPR Traffic Camera System", display)

            # Handle keyboard input (only on frames we actually display —
            # waitKey costs several ms on Windows)
            if show_preview and (paused or frame_num % 2 == 0):
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    logger.info("User quit.")
                    break
                elif key == ord(' '):  # Space = pause
                    paused = not paused
                    state = "PAUSED" if paused else "RESUMED"
                    logger.info(f"Playback {state}")
            else:
                # Small delay to prevent CPU thrashing
                pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        # ── Cleanup ──
        elapsed = time.time() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0
        stats = pipeline.get_stats()

        logger.info("")
        logger.info("=" * 60)
        logger.info("  Processing Complete — Summary")
        logger.info("=" * 60)
        logger.info(f"  Frames processed : {frame_num}")
        logger.info(f"  Total time       : {elapsed:.1f}s")
        logger.info(f"  Average FPS      : {avg_fps:.1f}")
        logger.info(f"  Vehicles tracked : {stats['total_vehicles_tracked']}")
        logger.info(f"  Plates detected  : {stats['plates_detected']}")
        if video_writer is not None:
            logger.info(f"  Output saved to  : {args.output}")
        logger.info("=" * 60)

        pipeline.shutdown()
        prefetcher.stop()
        video_cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
