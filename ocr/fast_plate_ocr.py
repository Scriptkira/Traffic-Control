"""
Fast Nepali plate OCR — recognition-only EasyOCR with grammar snapping.

Why this exists: EasyOCR's readtext() spends ~80% of its 200-400ms per
plate running CRAFT neural *text detection* — pure waste on a crop that
the YOLO plate detector already localized. This reader:

  1. Splits the two-line Nepali plate deterministically using a
     horizontal ink-projection profile (no neural detection),
  2. Feeds the line strips straight to EasyOCR's *recognizer*
     (reader.recognize) — same Devanagari+English model, ~15-25ms/line,
  3. Transliterates Devanagari (बा२ह४१५१ -> BA2HA4151) and snaps the
     result onto the Nepali plate grammar
     [province letters][digit][category letters][4 digits],
     with zone-aware confusion correction (O->0 in digit zones,
     0->O in letter zones, etc.).

Net effect: ~5-8x faster per read than readtext(), with higher accuracy
on format-valid plates. Same interface as PlateReader: read(crop) ->
(text, confidence) or None.
"""

import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from ocr.text_normalize import clean_raw_reading
from utils.device import resolve_device

logger = logging.getLogger(__name__)

# Nepali plate structure: province letters, province number,
# category letters, 4-digit registration number.
PLATE_RE = re.compile(r'^([A-Z]{1,4})(\d{1,2})([A-Z]{1,4})(\d{4})$')

# Zone-aware confusion pairs (aggressive but one-directional per zone).
_TO_DIGIT = str.maketrans({
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1',
    'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7',
})
# Conservative on purpose: only the unambiguous digit->letter pairs.
# Anything broader manufactures "format-valid" garbage (e.g. reading
# GASHAN27708 as GA-5-HANZ-7708), which then outvotes honest reads.
_TO_LETTER = str.maketrans({
    '0': 'O', '1': 'I', '8': 'B',
})


def snap_to_plate_grammar(text: str) -> Tuple[str, bool]:
    """
    Try to conform a cleaned reading to the Nepali plate format.

    Returns (possibly corrected text, format_valid). Correction is only
    attempted when the string is close to the format — a trailing
    4-char block is treated as the digit zone, the middle single char
    as the province digit, the rest as letter zones.
    """
    if PLATE_RE.match(text):
        return text, True

    if len(text) < 6 or len(text) > 12:
        return text, False

    # Assume last 4 chars are the registration digits.
    head, tail = text[:-4], text[-4:]
    tail = tail.translate(_TO_DIGIT)
    if not tail.isdigit():
        return text, False

    # Head should be letters-digit(s)-letters. Fix obvious confusions:
    # a digit sandwiched between letters is the province number; other
    # positions are letters. Only unambiguous look-alikes qualify.
    m = re.match(r'^([A-Z018]{1,4}?)([\dOIL]{1,2})([A-Z018]{1,4})$', head)
    if m:
        p1 = m.group(1).translate(_TO_LETTER)
        num = m.group(2).translate(_TO_DIGIT)
        p2 = m.group(3).translate(_TO_LETTER)
        candidate = f"{p1}{num}{p2}{tail}"
        if PLATE_RE.match(candidate):
            return candidate, True

    return head + tail, False


class FastPlateReader:
    """
    Recognition-only Nepali plate reader.

    Drop-in replacement for PlateReader — exposes read(plate_crop) ->
    (plate_text, confidence) | None.
    """

    def __init__(self, languages: list = None, gpu: bool = True):
        import easyocr

        langs = languages or config.OCR_LANGUAGES
        use_gpu = gpu and resolve_device("FastPlateReader (EasyOCR rec-only)") == "cuda"

        logger.info(
            f"Initializing FastPlateReader (languages={langs}, gpu={use_gpu}, "
            f"recognition-only — CRAFT detection bypassed)..."
        )
        try:
            self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        except Exception as e:
            logger.warning(f"EasyOCR GPU init failed, trying CPU: {e}")
            self.reader = easyocr.Reader(langs, gpu=False, verbose=False)

        self.min_length = config.OCR_MIN_PLATE_LENGTH
        self.max_length = config.OCR_MAX_PLATE_LENGTH
        self.conf_threshold = config.OCR_CONFIDENCE_THRESHOLD
        logger.info("FastPlateReader initialized.")

    # ── Public API ──────────────────────────────────────────────────

    def read(self, plate_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Read text from a plate crop.

        Returns (plate_text, confidence) or None.
        """
        if plate_crop is None or plate_crop.size == 0:
            logger.debug("reject: empty crop")
            return None
        if plate_crop.shape[0] < config.OCR_MIN_CROP_HEIGHT:
            logger.debug(
                f"reject: crop height {plate_crop.shape[0]}px < "
                f"OCR_MIN_CROP_HEIGHT ({config.OCR_MIN_CROP_HEIGHT}px)"
            )
            return None

        try:
            gray = self._preprocess(plate_crop)
            strips = self._split_lines(gray)

            # One batched recognizer forward for all line strips.
            line_results = self._recognize_batched(strips)

            texts = [t for t, _ in line_results if t]
            confs = [c for t, c in line_results if t]

            # Salvage pass: if the bottom (digits) line came back empty
            # or garbled, retry just that strip with a digits-only
            # decoder — the registration number alone is still useful.
            if len(strips) == 2 and (len(texts) < 2 or not any(
                ch.isdigit() for t in texts for ch in t
            )):
                res = self._recognize_strip(strips[-1], digits_only=True)
                if res is not None:
                    texts.append(res[0])
                    confs.append(res[1])

            if not texts:
                logger.debug(
                    f"reject: recognizer returned no text "
                    f"({len(strips)} line strip(s))"
                )
                return None

            # Guard against reading the same text twice via a bad line
            # split: if one line's text contains the other's, keep only
            # the longer one.
            if len(texts) == 2:
                a, b = texts
                if a in b:
                    texts = [b]
                elif b in a:
                    texts = [a]

            raw = "".join(texts)
            cleaned = clean_raw_reading(raw, self.min_length, self.max_length)
            if not cleaned:
                logger.debug(
                    f"reject: raw read '{raw}' emptied by normalization "
                    f"(length bounds {self.min_length}-{self.max_length})"
                )
                return None

            confidence = float(np.mean(confs))

            snapped, format_valid = snap_to_plate_grammar(cleaned)
            if format_valid:
                # A format-valid plate is stronger evidence than raw
                # decoder confidence (which runs low on small crops
                # without CRAFT's tight boxes) — accept it outright and
                # let it win consensus votes accordingly.
                confidence = min(1.0, max(confidence + 0.15, 0.40))
                return (snapped, confidence)

            if len(cleaned) >= self.min_length and confidence >= self.conf_threshold:
                return (cleaned, min(1.0, confidence))
            logger.debug(
                f"reject: '{cleaned}' conf={confidence:.2f} below "
                f"threshold {self.conf_threshold} (not format-valid)"
            )
            return None

        except Exception as e:
            logger.warning(f"Fast OCR failed on plate crop: {e}")
            return None

    # ── Internals ───────────────────────────────────────────────────

    _SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Grayscale, aggressive integer upscale + sharpen (the combination
        validated on this footage by the legacy reader), then CLAHE.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

        h = gray.shape[0]
        # Two lines of ~64px each — recognition model input height.
        if h < 32:
            s = 6.0
        elif h < 64:
            s = 4.0
        elif h < 128:
            s = 2.0
        else:
            s = 1.0
        if s > 1.0:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            gray = cv2.filter2D(gray, -1, self._SHARPEN)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _tighten(self, strip: np.ndarray) -> np.ndarray:
        """
        Trim a line strip to its ink extents (plus a small margin).

        This substitutes for what CRAFT detection gave the recognizer:
        a tight text box without the plate border, rivets, and shadow
        that otherwise wreck recognition confidence.
        """
        h, w = strip.shape[:2]
        _, binary = cv2.threshold(
            strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Ignore a thin frame border in the ink search
        bh, bw = max(2, h // 12), max(2, w // 24)
        inner = binary[bh:h - bh, bw:w - bw]
        if inner.size == 0:
            return strip

        rows = inner.sum(axis=1)
        cols = inner.sum(axis=0)
        r_ink = np.nonzero(rows > 0.05 * (rows.max() + 1))[0]
        c_ink = np.nonzero(cols > 0.05 * (cols.max() + 1))[0]
        if len(r_ink) < 8 or len(c_ink) < 8:
            return strip

        m = 4  # margin
        y1 = max(0, bh + r_ink[0] - m)
        y2 = min(h, bh + r_ink[-1] + m)
        x1 = max(0, bw + c_ink[0] - m)
        x2 = min(w, bw + c_ink[-1] + m)
        if y2 - y1 < 12 or x2 - x1 < 16:
            return strip
        return strip[y1:y2, x1:x2]

    def _split_lines(self, gray: np.ndarray) -> List[np.ndarray]:
        """
        Split a (possibly) two-line plate into line strips using the
        horizontal ink-projection profile — the row band with the least
        ink near the vertical middle is the line gap.
        """
        h, w = gray.shape[:2]
        if h < 40:  # too short to hold two readable lines
            return [self._tighten(gray)]

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        profile = binary.sum(axis=1).astype(np.float64)
        # Smooth to ignore single-row noise
        kernel = np.ones(5) / 5.0
        profile = np.convolve(profile, kernel, mode="same")

        lo, hi = int(h * 0.30), int(h * 0.70)
        band = profile[lo:hi]
        if band.size == 0:
            return [self._tighten(gray)]
        split = lo + int(np.argmin(band))

        # A real line gap is nearly ink-free relative to text rows, and
        # both halves must contain substantial text of their own —
        # otherwise this is a single-line plate whose characters merely
        # thin out mid-height (e.g. 0/8 digit strokes), and splitting it
        # reads the same text twice.
        if profile[split] > 0.15 * profile.max():
            return [self._tighten(gray)]
        top_peak = profile[:split].max() if split > 0 else 0
        bot_peak = profile[split:].max() if split < h else 0
        if min(top_peak, bot_peak) < 0.40 * profile.max():
            return [self._tighten(gray)]

        top, bottom = gray[:split], gray[split:]
        if top.shape[0] < 12 or bottom.shape[0] < 12:
            return [self._tighten(gray)]
        return [self._tighten(top), self._tighten(bottom)]

    def _recognize_batched(self, strips: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Recognize all line strips in ONE recognizer call: strips are
        stacked into a single canvas (padded to common width, separated
        by blank gaps) and passed as a horizontal_list, so the model
        runs one batched forward instead of one call per line.
        """
        if len(strips) == 1:
            res = self._recognize_strip(strips[0])
            return [res] if res else []

        gap = 16
        w = max(s.shape[1] for s in strips)
        total_h = sum(s.shape[0] for s in strips) + gap * (len(strips) - 1)
        canvas = np.full((total_h, w), 255, dtype=np.uint8)

        boxes = []
        y = 0
        for s in strips:
            sh, sw = s.shape[:2]
            canvas[y:y + sh, :sw] = s
            boxes.append([0, w, y, y + sh])  # [x_min, x_max, y_min, y_max]
            y += sh + gap

        results = self.reader.recognize(
            canvas,
            horizontal_list=boxes,
            free_list=[],
            detail=1,
            paragraph=False,
            batch_size=len(boxes),
            contrast_ths=0.0,
        )

        # Map results back to strips by vertical position, top-to-bottom.
        out: List[Tuple[str, float]] = []
        ordered = sorted(results or [], key=lambda r: min(p[1] for p in r[0]))
        for r in ordered:
            text, conf = r[1], float(r[2])
            if text.strip():
                out.append((text, conf))
        return out

    def _recognize_strip(
        self, strip: np.ndarray, digits_only: bool = False
    ) -> Optional[Tuple[str, float]]:
        """
        Run EasyOCR's recognizer (no detection stage) on one line strip.

        digits_only constrains the decoder to digits for the bottom
        (registration-number) line — a large accuracy win there.
        """
        # contrast_ths=0 disables EasyOCR's low-contrast retry, which
        # silently re-runs the recognition model (doubling latency) on
        # exactly the kind of low-contrast crops plates produce — CLAHE
        # in _preprocess already handles contrast deterministically.
        kwargs = {
            "detail": 1,
            "paragraph": False,
            "batch_size": 2,
            "contrast_ths": 0.0,
        }
        if digits_only:
            kwargs["allowlist"] = "0123456789०१२३४५६७८९"

        results = self.reader.recognize(strip, **kwargs)
        if not results:
            return None

        # Whole strip is one line — concatenate left-to-right just in
        # case the recognizer returned fragments.
        results = sorted(results, key=lambda r: min(p[0] for p in r[0]))
        text = "".join(r[1] for r in results)
        conf = float(np.mean([r[2] for r in results]))
        if not text.strip():
            return None
        return text, conf
