"""
Plate Reader — EasyOCR-based license plate text recognition.

Preprocesses cropped plate images and applies OCR to extract
alphanumeric characters. Includes confidence filtering and
text post-processing.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from ocr.text_normalize import clean_raw_reading
from utils.device import resolve_device

logger = logging.getLogger(__name__)


class PlateReader:
    """
    Reads license plate text from cropped plate images using EasyOCR.

    Applies a preprocessing pipeline (resize, grayscale, CLAHE, threshold)
    before OCR to improve accuracy. Results are filtered by confidence
    and post-processed to remove noise.
    """

    def __init__(
        self,
        languages: list = None,
        gpu: bool = True,
    ):
        """
        Initialize the OCR reader.

        Args:
            languages: List of language codes for EasyOCR.
            gpu: Whether to use GPU acceleration.
        """
        import easyocr

        langs = languages or config.OCR_LANGUAGES

        # EasyOCR silently falls back to CPU if gpu=True but CUDA isn't
        # actually available — resolve it ourselves first so the choice
        # is explicit and logged, not silent.
        use_gpu = gpu and resolve_device("PlateReader (EasyOCR)") == "cuda"

        logger.info(f"Initializing EasyOCR (languages={langs}, gpu={use_gpu})...")
        try:
            self.reader = easyocr.Reader(
                langs,
                gpu=use_gpu,
                verbose=False,
            )
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR GPU init failed, trying CPU: {e}")
            self.reader = easyocr.Reader(langs, gpu=False, verbose=False)

        self.min_length = config.OCR_MIN_PLATE_LENGTH
        self.max_length = config.OCR_MAX_PLATE_LENGTH
        self.conf_threshold = config.OCR_CONFIDENCE_THRESHOLD

    def read(self, plate_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Read text from plate crop.

        Args:
            plate_crop: Crop of the plate.

        Returns:
            Tuple of (plate_text, confidence) if successful, None otherwise.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        if plate_crop.shape[0] < config.OCR_MIN_CROP_HEIGHT:
            return None

        try:
            # Preprocess the plate image
            processed = self._preprocess(plate_crop)

            # Run OCR
            ocr_args = {
                "detail": 1,
                "paragraph": False,
            }
            if config.OCR_CHAR_WHITELIST is not None:
                ocr_args["allowlist"] = config.OCR_CHAR_WHITELIST

            # 1. OCR on the full processed crop
            full_results = self.reader.readtext(processed, **ocr_args)

            # 2. Bottom-half fallback pass — sometimes catches the
            # registration-digit line when the full-crop pass misses it
            # (small/blurry text). Each readtext call is the single most
            # expensive operation in the pipeline, so only pay for the
            # second pass when the first came back empty.
            bottom_results = []
            if not full_results:
                h, w = processed.shape[:2]
                bottom_half = processed[int(h * 0.45):h, :]
                bottom_results = self.reader.readtext(bottom_half, **ocr_args)

            # Nepali plates are commonly two lines (province/category on
            # top, registration digits below). EasyOCR returns each line
            # as a separate (bbox, text, conf) detection — merge same-pass
            # detections top-to-bottom into one string instead of picking
            # only the single highest-confidence fragment, so the full
            # plate is preserved rather than just whichever line won.
            full_text, full_conf = self._merge_segments(full_results)
            bottom_text, bottom_conf = self._merge_segments(bottom_results)

            candidates = []
            if full_text:
                candidates.append((full_text, full_conf))
            if bottom_text:
                candidates.append((bottom_text, bottom_conf))

            if not candidates:
                return None

            # Prefer the more complete cleaned reading (more of the plate
            # captured); break ties on confidence.
            best_text = None
            best_conf = 0.0

            for text, conf in candidates:
                cleaned = self._postprocess(text)
                if not cleaned:
                    continue

                is_better = best_text is None or (
                    (len(cleaned), conf) > (len(best_text), best_conf)
                )
                if is_better:
                    best_text = cleaned
                    best_conf = conf

            if (
                best_text
                and len(best_text) >= self.min_length
                and best_conf >= self.conf_threshold
            ):
                return (best_text, min(1.0, best_conf))

            return None

        except Exception as e:
            logger.debug(f"OCR failed on plate crop: {e}")
            return None

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR accuracy.

        Upscales small crops by a clean integer factor (6x, 4x, or 2x) to preserve
        sharp pixel boundaries and applies a sharpening filter to make character edges crisp.
        """
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        # Choose clean scaling factor based on height
        if h < 20:
            scale = 6.0
        elif h < 40:
            scale = 4.0
        elif h < 80:
            scale = 2.0
        else:
            scale = 1.0

        if scale > 1.0:
            resized = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            # Apply sharpening to crisp up the edges of upscaled text
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed = cv2.filter2D(resized, -1, kernel)
        else:
            processed = crop.copy()

        return processed

    def _merge_segments(self, results: list) -> Tuple[Optional[str], float]:
        """
        Merge multiple text detections from one readtext() pass into a
        single top-to-bottom string (see call site for why), averaging
        confidence across the merged segments.

        EasyOCR sometimes returns two overlapping boxes for the same line
        (e.g. a sub-word box nested inside a full-line box) — segments
        whose vertical extent overlaps heavily are treated as duplicates
        of the same line and only the higher-confidence one is kept,
        rather than concatenating both into a garbled repeat.
        """
        if not results:
            return None, 0.0

        def y_range(bbox):
            ys = [pt[1] for pt in bbox]
            return min(ys), max(ys)

        def y_overlap_ratio(r1, r2):
            lo = max(r1[0], r2[0])
            hi = min(r1[1], r2[1])
            overlap = max(0.0, hi - lo)
            shorter = min(r1[1] - r1[0], r2[1] - r2[0])
            return overlap / shorter if shorter > 0 else 0.0

        ordered = sorted(results, key=lambda r: y_range(r[0])[0])

        deduped = []
        for bbox, text, conf in ordered:
            yr = y_range(bbox)
            dup_idx = None
            for i, (kept_yr, _, kept_conf) in enumerate(deduped):
                if y_overlap_ratio(yr, kept_yr) > 0.5:
                    dup_idx = i
                    break
            if dup_idx is None:
                deduped.append((yr, text, conf))
            elif conf > deduped[dup_idx][2]:
                deduped[dup_idx] = (yr, text, conf)

        merged_text = "".join(t for _, t, _ in deduped)
        merged_conf = sum(c for _, _, c in deduped) / len(deduped)
        return merged_text, merged_conf

    def _postprocess(self, text: str) -> Optional[str]:
        """
        Clean and validate OCR output using Devanagari-to-English translation mapping.
        """
        return clean_raw_reading(text, self.min_length, self.max_length)
