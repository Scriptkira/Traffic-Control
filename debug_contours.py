"""
Debug script to inspect candidate plate crops from the grey car
in the test video. Saves all candidate crops to files.
"""

import os
import cv2
import numpy as np

VIDEO_PATH = r"D:\Traffic\Traffic-Control\input\1be628bc-c45e-4485-8bc8-296532215972.mp4"
DEBUG_DIR = "debug_crops"
os.makedirs(DEBUG_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

# Let's search frames 210 to 260 where the grey car (ID 5/6) passes the camera
for fnum in range(210, 260, 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, frame = cap.read()
    if not ret:
        break

    # Let's locate the grey car manually using a bounding box approximate to what YOLO found:
    # From diagnostic: Frame 250: bbox=(192, 79, 310, 201)
    # Let's search around that region in the frame
    # We will look at x: 170-330, y: 70-220
    h, w = frame.shape[:2]
    vx1, vy1, vx2, vy2 = 170, 70, 330, 220
    vehicle_crop = frame[vy1:vy2, vx1:vx2]

    # Run the edge-based and morph-based contour detection on this crop
    crop_h, crop_w = vehicle_crop.shape[:2]

    # Preprocess
    scale = 2.0
    work_crop = cv2.resize(vehicle_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(work_crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Try Canny Edge detection
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(bilateral, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours_edge, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Try Morphological closing detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    contours_morph, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Let's evaluate contours and save candidates
    all_contours = [("edge", contours_edge), ("morph", contours_morph)]
    count = 0
    for name, contours in all_contours:
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            x, y, w_box, h_box = cv2.boundingRect(approx)
            
            if h_box == 0 or w_box == 0:
                continue
            
            ar = w_box / h_box
            if 1.5 <= ar <= 7.0 and w_box >= 20 and h_box >= 8:
                # Save this crop
                candidate = work_crop[y:y+h_box, x:x+w_box]
                filename = os.path.join(DEBUG_DIR, f"frame_{fnum}_{name}_c{i}_ar_{ar:.1f}_area_{int(area)}.jpg")
                cv2.imwrite(filename, candidate)
                count += 1

print(f"Saved candidate crops to {DEBUG_DIR}")
cap.release()
