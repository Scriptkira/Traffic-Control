"""
Finds the exact pixel coordinates of the license plate "KL01BY1057"
in frame 250 by running OCR on sliding crops over the vehicle region.
"""

import cv2
import easyocr

VIDEO_PATH = r"D:\Traffic\Traffic-Control\input\1be628bc-c45e-4485-8bc8-296532215972.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame 250")
    import sys
    sys.exit(1)

# The vehicle bounding box on frame 250 is roughly (192, 79, 310, 201).
# Let's crop the entire lower half of the vehicle where the plate could be.
# Y range: 120 to 201, X range: 192 to 310.
y_start, y_end = 120, 201
x_start, x_end = 192, 310

vehicle_lower = frame[y_start:y_end, x_start:x_end]
cv2.imwrite("vehicle_lower.jpg", vehicle_lower)

# We will run EasyOCR on this whole region to see where the text "KL01BY1057" is found.
reader = easyocr.Reader(['en'], gpu=True)

print("Running OCR on the vehicle lower half...")
results = reader.readtext(vehicle_lower)

for bbox, text, conf in results:
    print(f"Detected: '{text}' (conf={conf:.2f}) at relative bbox: {bbox}")
    
    # Calculate absolute coordinates
    # bbox format: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    abs_x1 = int(min(xs)) + x_start
    abs_y1 = int(min(ys)) + y_start
    abs_x2 = int(max(xs)) + x_start
    abs_y2 = int(max(ys)) + y_start
    
    print(f"  Absolute coordinates: ({abs_x1}, {abs_y1}, {abs_x2}, {abs_y2})")
    
    # Save crop of this detection
    crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
    cv2.imwrite(f"detected_{text.replace(' ', '_')}.jpg", crop)
