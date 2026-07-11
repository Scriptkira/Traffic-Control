"""
Plate text normalization — shared Devanagari/Latin OCR cleanup.

Used by both the raw OCR postprocessing step (ocr/plate_reader.py) and
the cross-frame consensus step (core/vehicle_record.py) so the two
stay in sync instead of drifting independently.

Preserves the full plate string (province/category letters + registration
digits) rather than isolating just the trailing digits — Nepali plates
read cleanly in this pipeline as e.g. "BA1HA4151", and collapsing that
down to "4151" throws away real, correctly-read information.
"""

import re

# Devanagari numerals -> ASCII digits
DEVANAGARI_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}

# Devanagari consonants -> Latin romanization (with inherent 'A').
# Covers the letters used on Nepali plates: province codes
# (बा=BA, लु=LU, ना=NA, को=KO, गं=GAN, मे=ME, ...) and vehicle
# category letters (क=KA, ख=KHA, ग=GA, घ=GHA, च=CHA, ज=JA, झ=JHA,
# प=PA, ह=HA, ...).
DEVANAGARI_CONSONANTS = {
    'क': 'KA', 'ख': 'KHA', 'ग': 'GA', 'घ': 'GHA', 'ङ': 'NGA',
    'च': 'CHA', 'छ': 'CHHA', 'ज': 'JA', 'झ': 'JHA', 'ञ': 'NYA',
    'ट': 'TA', 'ठ': 'THA', 'ड': 'DA', 'ढ': 'DHA', 'ण': 'NA',
    'त': 'TA', 'थ': 'THA', 'द': 'DA', 'ध': 'DHA', 'न': 'NA',
    'प': 'PA', 'फ': 'PHA', 'ब': 'BA', 'भ': 'BHA', 'म': 'MA',
    'य': 'YA', 'र': 'RA', 'ल': 'LA', 'व': 'WA',
    'श': 'SHA', 'ष': 'SHA', 'स': 'SA', 'ह': 'HA',
}

# Devanagari standalone vowels
DEVANAGARI_VOWELS = {
    'अ': 'A', 'आ': 'AA', 'इ': 'I', 'ई': 'I', 'उ': 'U', 'ऊ': 'U',
    'ए': 'E', 'ऐ': 'AI', 'ओ': 'O', 'औ': 'AU',
}

# Vowel signs (matras): replace the consonant's inherent 'A'
DEVANAGARI_MATRAS = {
    'ा': 'A', 'ि': 'I', 'ी': 'I', 'ु': 'U', 'ू': 'U',
    'े': 'E', 'ै': 'AI', 'ो': 'O', 'ौ': 'AU',
}


def transliterate_devanagari(text: str) -> str:
    """
    Romanize Devanagari plate text: बा२ह४१५१ -> BA2HA4151.

    Consonants carry an inherent 'A' (ब -> BA); a following vowel sign
    replaces it (ल + ु -> LU); anusvara adds N (गं -> GAN); numerals map
    to ASCII digits. Non-Devanagari characters pass through unchanged.
    """
    out = []
    for ch in text:
        if ch in DEVANAGARI_DIGITS:
            out.append(DEVANAGARI_DIGITS[ch])
        elif ch in DEVANAGARI_CONSONANTS:
            out.append(DEVANAGARI_CONSONANTS[ch])
        elif ch in DEVANAGARI_VOWELS:
            out.append(DEVANAGARI_VOWELS[ch])
        elif ch in DEVANAGARI_MATRAS:
            # Replace the previous consonant's inherent 'A'
            if out and out[-1] and out[-1][-1] == 'A':
                out[-1] = out[-1][:-1] + DEVANAGARI_MATRAS[ch]
            else:
                out.append(DEVANAGARI_MATRAS[ch])
        elif ch == 'ं':  # anusvara
            out.append('N')
        elif ch == '्':  # virama (suppresses inherent vowel)
            if out and out[-1] and out[-1][-1] == 'A':
                out[-1] = out[-1][:-1]
        else:
            out.append(ch)
    return "".join(out)


def normalize_devanagari(text: str) -> str:
    """Romanize Devanagari (letters AND numerals), then upper-case."""
    return transliterate_devanagari(text).upper()


def _clean(text: str) -> str:
    normalized = normalize_devanagari(text)
    return re.sub(r'[^A-Z0-9]', '', normalized)


def clean_raw_reading(text: str, min_length: int, max_length: int) -> "str | None":
    """
    Clean a single raw OCR reading (used in ocr/plate_reader.py).

    Keeps the full alphanumeric string as read (letters and digits) —
    just normalizes Devanagari numerals and strips punctuation/whitespace.
    """
    cleaned = _clean(text)

    if len(cleaned) < min_length:
        return None
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def clean_consensus_reading(text: str) -> str:
    """
    Clean a cross-frame consensus reading (used in core/vehicle_record.py).

    Same normalization as clean_raw_reading, applied to the character-vote
    consensus string built across a vehicle's OCR history.
    """
    cleaned = _clean(text)
    return cleaned if cleaned else text
