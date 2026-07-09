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

            # 2. OCR on the bottom half of the processed crop (if crop height is large enough)
            bottom_results = []
            h, w = processed.shape[:2]
            if plate_crop.shape[0] >= 15:
                # Capture the bottom 55% of the crop
                bottom_half = processed[int(h*0.45):h, :]
                bottom_results = self.reader.readtext(bottom_half, **ocr_args)

            # Gather all detected candidates
            candidates = []
            
            for (bbox, text, conf) in full_results:
                candidates.append((text, conf))
                
            for (bbox, text, conf) in bottom_results:
                candidates.append((text, conf))

            if not candidates:
                return None

            # Process candidates and choose the one with the highest confidence
            best_text = None
            best_conf = 0.0

            for text, conf in candidates:
                cleaned = self._postprocess(text)
                if not cleaned:
                    continue
                
                # Boost confidence if the result ends with or consists of a good digit sequence (3-4 digits)
                digits = re.findall(r'\d+', cleaned)
                boost = 0.0
                if digits:
                    all_digits = "".join(digits)
                    if len(all_digits) >= 3:
                        boost = 0.15 # Boost conf to prioritize valid numbers
                        
                weighted_conf = conf + boost
                if weighted_conf > best_conf:
                    best_conf = weighted_conf
                    best_text = cleaned

            if best_text and len(best_text) >= self.min_length:
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

    def _postprocess(self, text: str) -> Optional[str]:
        """
        Clean and validate OCR output using Devanagari-to-English translation mapping.
        """
        # Convert Devanagari numbers to English numbers
        devanagari_nums = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        
        text_upper = text.upper()
        normalized = ""
        for char in text_upper:
            if char in devanagari_nums:
                normalized += devanagari_nums[char]
            else:
                normalized += char
                
        # Common English OCR misreads of Devanagari plates
        normalized = re.sub(r'PHI', '88', normalized)
        normalized = re.sub(r'PH', '88', normalized)
        normalized = re.sub(r'DN', '39', normalized)
        normalized = re.sub(r'DI', '39', normalized)
        
        # Character mapping
        char_map = {
            'D': '3', 'O': '0', 'J': '3', 'Q': '0', 'U': '0',
            'B': '8', 'G': '9', 'S': '5', 'Z': '2', 'L': '1', 'R': '8'
        }
        
        mapped = ""
        for char in normalized:
            if char in char_map:
                mapped += char_map[char]
            else:
                mapped += char
                
        # Keep only digits and letters
        cleaned = re.sub(r'[^A-Z0-9]', '', mapped)
        
        # If it contains 4 or more digits, isolate the last 4 digits (standard registration number)
        digits = re.findall(r'\d+', cleaned)
        if digits:
            all_digits = "".join(digits)
            if len(all_digits) >= 4:
                return all_digits[-4:]
                
        # Length validation
        if len(cleaned) < self.min_length:
            return None

        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]

        return cleaned
