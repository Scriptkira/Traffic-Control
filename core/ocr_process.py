"""
Plate-detection + OCR child process.

Runs in its own interpreter (own GIL, own CUDA context), so EasyOCR's
heavy CPU-side work — doubled by the Nepali Devanagari recognizer —
can't steal cycles from the frame loop. The parent sends vehicle-ROI
crops over an input queue and drains recognized plates from an output
queue (see ANPRPipeline._apply_ocr_results).

Protocol:
    in_q   : (track_id, proc_crop, hires_crop, vehicle_bbox, other_bboxes)
             or None to shut down.
    out_q  : ("ready", None) once models are loaded, then
             ("plate", (track_id, text, conf, plate_bbox, vehicle_bbox))
"""

import logging
import sys


def _bbox_contains(bbox, x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def plate_ocr_process_main(in_q, out_q):
    """Entry point for the OCR child process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)
    logger = logging.getLogger("ocr_process")

    import config
    from detectors.plate_detector import PlateDetector

    plate_detector = PlateDetector()

    # "fast" = recognition-only EasyOCR + plate-grammar snapping
    # (~5-8x faster per read); "easyocr" = legacy full readtext path.
    engine = getattr(config, "OCR_ENGINE", "fast")
    if engine == "fast":
        from ocr.fast_plate_ocr import FastPlateReader
        plate_reader = FastPlateReader()
    else:
        from ocr.plate_reader import PlateReader
        plate_reader = PlateReader()

    out_q.put(("ready", None))
    logger.info("OCR process ready.")

    while True:
        item = in_q.get()
        if item is None:
            break

        track_id, proc_crop, hires_crop, vehicle_bbox, other_bboxes = item
        try:
            ch, cw = proc_crop.shape[:2]
            candidates = plate_detector.detect_in_roi(proc_crop, (0, 0, cw, ch))
            if not candidates:
                continue

            vx1, vy1 = vehicle_bbox[0], vehicle_bbox[1]

            # Pick the best candidate that plausibly belongs to this
            # vehicle (candidates are already sorted by confidence).
            picked = None
            for c in candidates:
                fb = (
                    c.bbox[0] + vx1, c.bbox[1] + vy1,
                    c.bbox[2] + vx1, c.bbox[3] + vy1,
                )
                cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                if not _bbox_contains(vehicle_bbox, cx, cy) and any(
                    _bbox_contains(ob, cx, cy) for ob in other_bboxes
                ):
                    continue  # plate belongs to a neighboring vehicle
                picked = (c, fb)
                break

            if picked is None:
                continue
            cand, plate_bbox = picked

            # OCR from the hi-res crop when available — downscaled
            # processing shouldn't cost plate readability.
            ocr_crop = cand.crop
            if hires_crop is not None:
                sx = hires_crop.shape[1] / max(1, cw)
                sy = hires_crop.shape[0] / max(1, ch)
                hx1, hy1 = int(cand.bbox[0] * sx), int(cand.bbox[1] * sy)
                hx2, hy2 = int(cand.bbox[2] * sx), int(cand.bbox[3] * sy)
                hi = hires_crop[hy1:hy2, hx1:hx2]
                if hi.size > 0:
                    ocr_crop = hi

            ocr_result = plate_reader.read(ocr_crop)
            if ocr_result is None:
                continue

            plate_text, confidence = ocr_result
            out_q.put(
                ("plate", (track_id, plate_text, confidence, plate_bbox, vehicle_bbox))
            )
        except Exception as e:
            logger.debug(f"Plate work failed for track {track_id}: {e}")
