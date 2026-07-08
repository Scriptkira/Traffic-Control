"""
Runs OCR on all saved candidate crops to identify which one contains
the actual license plate text "KL01BY1057" (or parts of it).
"""

import os
import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=True)
DEBUG_DIR = "debug_crops"

files = [f for f in os.listdir(DEBUG_DIR) if f.endswith(".jpg")]
print(f"Testing OCR on {len(files)} candidate crops...")

matches = []

for file in files:
    filepath = os.path.join(DEBUG_DIR, file)
    img = cv2.imread(filepath)
    if img is None:
        continue
    
    # Run OCR on the crop
    results = reader.readtext(img, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    if results:
        combined = "".join([r[1] for r in results]).upper()
        # Look for typical plate characters (KL, 01, BY, 1057)
        if any(token in combined for token in ["KL", "BY", "10", "57", "01", "1057"]):
            print(f"MATCH: {file} -> OCR: '{combined}' | Detail: {results}")
            matches.append((file, combined, results))
        elif len(combined) > 4:
            print(f"Possible: {file} -> OCR: '{combined}'")

print("="*60)
print(f"Found {len(matches)} matches.")
print("="*60)
