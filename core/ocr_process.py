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

Diagnostics: a stage-funnel counter summary is logged every
config.OCR_STATS_EVERY_N_JOBS jobs, so a dead pipeline is attributable
to a stage (no candidates vs. OCR rejection) from the log alone. With
config.DIAGNOSTIC_MODE, every OCR'd plate crop (accepted and rejected)
is saved to config.DIAG_CROPS_DIR, plus a capped sample of vehicle ROIs
where plate detection found nothing.
"""

import logging
import os
import sys
from collections import defaultdict


def _bbox_contains(bbox, x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


class _StageStats:
    """Counters for the plate-work funnel, logged periodically."""

    FIELDS = (
        "jobs", "roi_too_small", "no_candidates", "all_neighbor_owned",
        "ocr_attempted", "ocr_rejected", "accepted",
    )

    def __init__(self, logger, every_n: int):
        self.logger = logger
        self.every_n = max(1, every_n)
        self.counts = dict.fromkeys(self.FIELDS, 0)

    def bump(self, field: str):
        self.counts[field] += 1
        if field == "jobs" and self.counts["jobs"] % self.every_n == 0:
            self.log_summary()

    def log_summary(self):
        c = self.counts
        self.logger.info(
            "Plate funnel: %d jobs → %d no-candidates, %d roi-too-small, "
            "%d neighbor-owned → %d OCR attempts → %d rejected, %d ACCEPTED",
            c["jobs"], c["no_candidates"], c["roi_too_small"],
            c["all_neighbor_owned"], c["ocr_attempted"],
            c["ocr_rejected"], c["accepted"],
        )


def plate_ocr_process_main(in_q, out_q, log_level: int = logging.INFO):
    """Entry point for the OCR child process."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)
    logger = logging.getLogger("ocr_process")

    import cv2

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

    stats = _StageStats(logger, getattr(config, "OCR_STATS_EVERY_N_JOBS", 200))

    diag = getattr(config, "DIAGNOSTIC_MODE", False)
    diag_dir = getattr(config, "DIAG_CROPS_DIR", "debug_crops")
    empty_roi_cap = getattr(config, "DIAG_MAX_EMPTY_ROIS_PER_TRACK", 3)
    empty_roi_dumps = defaultdict(int)
    if diag:
        os.makedirs(diag_dir, exist_ok=True)
        logger.info(f"DIAGNOSTIC MODE: saving plate crops to {diag_dir}")

    def _dump(img, name):
        try:
            cv2.imwrite(os.path.join(diag_dir, name), img)
        except Exception as e:
            logger.warning(f"Failed to save diagnostic crop {name}: {e}")

    out_q.put(("ready", None))
    logger.info("OCR process ready.")

    while True:
        item = in_q.get()
        if item is None:
            break

        track_id, proc_crop, hires_crop, vehicle_bbox, other_bboxes = item
        stats.bump("jobs")
        job_no = stats.counts["jobs"]
        try:
            ch, cw = proc_crop.shape[:2]
            if cw < 20 or ch < 20:
                stats.bump("roi_too_small")
                continue

            candidates = plate_detector.detect_in_roi(proc_crop, (0, 0, cw, ch))
            if not candidates:
                stats.bump("no_candidates")
                logger.debug(
                    f"track {track_id}: no plate candidates in "
                    f"{cw}x{ch} vehicle ROI"
                )
                if diag and empty_roi_dumps[track_id] < empty_roi_cap:
                    empty_roi_dumps[track_id] += 1
                    _dump(proc_crop, f"t{track_id}_j{job_no}_roi_nocand.png")
                continue

            logger.debug(
                f"track {track_id}: {len(candidates)} plate candidate(s), "
                f"conf={[f'{c.confidence:.2f}' for c in candidates]}"
            )

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
                stats.bump("all_neighbor_owned")
                logger.debug(
                    f"track {track_id}: all candidates rejected as "
                    "neighbor-owned"
                )
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

            stats.bump("ocr_attempted")
            ocr_result = plate_reader.read(ocr_crop)
            if ocr_result is None:
                stats.bump("ocr_rejected")
                logger.debug(
                    f"track {track_id}: OCR rejected "
                    f"{ocr_crop.shape[1]}x{ocr_crop.shape[0]} plate crop "
                    f"(det conf {cand.confidence:.2f}) — see fast_plate_ocr "
                    "debug logs for the reason"
                )
                if diag:
                    _dump(ocr_crop, f"t{track_id}_j{job_no}_rej.png")
                continue

            plate_text, confidence = ocr_result
            stats.bump("accepted")
            logger.debug(
                f"track {track_id}: OCR accepted '{plate_text}' "
                f"conf={confidence:.2f}"
            )
            if diag:
                safe = "".join(ch for ch in plate_text if ch.isalnum())
                _dump(ocr_crop, f"t{track_id}_j{job_no}_ok_{safe}_{confidence:.2f}.png")
            out_q.put(
                ("plate", (track_id, plate_text, confidence, plate_bbox, vehicle_bbox))
            )
        except Exception as e:
            # Never silent: a broken stage inside this process is
            # invisible from the parent unless it reaches the log.
            logger.warning(f"Plate work failed for track {track_id}: {e}")

    stats.log_summary()
    logger.info("OCR process shutting down.")
