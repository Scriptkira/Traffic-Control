"""
Plate Reader — EasyOCR-based license plate text recognition.

Preprocesses cropped plate images and applies OCR to extract
alphanumeric characters. Includes confidence filtering and
text post-processing.
"""

import logging
import re
from typing import Optional, Tuple

import cv2
import numpy as np

import config

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

        logger.info(f"Initializing EasyOCR (languages={langs}, gpu={gpu})...")
        try:
            self.reader = easyocr.Reader(
                langs,
                gpu=gpu,
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
        Read text from a cropped license plate image.

        Args:
            plate_crop: BGR image of the license plate region.

        Returns:
            Tuple of (plate_text, confidence) if successful, None otherwise.
        """
        if plate_crop is None or plate_crop.size == 0:
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

            results = self.reader.readtext(processed, **ocr_args)

            if not results:
                return None

            # Combine all detected text segments
            texts = []
            total_conf = 0.0
            count = 0

            for (bbox, text, conf) in results:
                if conf >= self.conf_threshold:
                    texts.append(text)
                    total_conf += conf
                    count += 1

            if not texts or count == 0:
                return None

            combined = " ".join(texts)
            avg_conf = total_conf / count

            # Post-process
            cleaned = self._postprocess(combined)

            if cleaned is None:
                return None

            return (cleaned, avg_conf)

        except Exception as e:
            logger.debug(f"OCR failed on plate crop: {e}")
            return None

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR accuracy.

        Upscales small crops by a clean integer factor (4x or 2x) to preserve
        sharp pixel boundaries, keeping the BGR color channels intact.
        """
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        # Choose clean scaling factor to avoid fractional resizing artifacts
        if h < 30:
            scale = 4.0
        elif h < 60:
            scale = 2.0
        else:
            scale = 1.0

        if scale > 1.0:
            resized = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            resized = crop.copy()

        return resized

    def _postprocess(self, text: str) -> Optional[str]:
        """
        Clean and validate OCR output.

        Removes non-alphanumeric characters (including Devanagari),
        checks length constraints, and normalizes output.
        """
        # Keep English A-Z, 0-9, and Devanagari letters/numbers (\u0900-\u097F)
        cleaned = re.sub(r'[^A-Z0-9\u0900-\u097F]', '', text.upper())
        
        # Length validation
        if len(cleaned) < self.min_length:
            return None

        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]

        if len(cleaned) == 0:
            return None

        return cleaned
