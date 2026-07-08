"""
SORT Tracker — Simple Online and Realtime Tracking.

Implements the SORT algorithm from scratch using:
- Kalman Filter for motion prediction (via filterpy)
- Hungarian Algorithm for detection-to-track assignment (via scipy)
- IoU-based cost matrix

References:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import config

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with a persistent ID and current state."""
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2) — current estimated bbox
    confidence: float
    hits: int = 0        # Total number of matched detections
    age: int = 0         # Frames since last matched detection
    is_confirmed: bool = False


def _iou(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        bb1: [x1, y1, x2, y2]
        bb2: [x1, y1, x2, y2]

    Returns:
        Intersection over Union score.
    """
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, bb1[2] - bb1[0]) * max(0, bb1[3] - bb1[1])
    area2 = max(0, bb2[2] - bb2[0]) * max(0, bb2[3] - bb2[1])

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _bbox_to_z(bbox):
    """
    Convert [x1, y1, x2, y2] to Kalman state [cx, cy, s, r].

    Where:
        cx, cy = center coordinates
        s = area (scale)
        r = aspect ratio (width / height)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h        # area
    r = w / max(h, 1e-6)  # aspect ratio
    return np.array([cx, cy, s, r]).reshape((4, 1))


def _z_to_bbox(z):
    """
    Convert Kalman state [cx, cy, s, r] back to [x1, y1, x2, y2].
    """
    cx, cy, s, r = z.flatten()
    s = max(s, 1.0)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ])


class _KalmanTrack:
    """
    Internal Kalman-filtered track for a single object.

    State vector: [cx, cy, s, r, vx, vy, vs]
    Measurement:  [cx, cy, s, r]
    """

    _next_id = 1  # Class-level ID counter

    def __init__(self, bbox: np.ndarray):
        """Initialize a new track from a detection bbox."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0   # High uncertainty for velocities
        self.kf.P *= 10.0

        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with the first measurement
        self.kf.x[:4] = _bbox_to_z(bbox)

        self.track_id = _KalmanTrack._next_id
        _KalmanTrack._next_id += 1

        self.hits = 1
        self.age = 0          # Frames since last update
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Advance state and return predicted bbox."""
        # Prevent area from going negative
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        return _z_to_bbox(self.kf.x[:4])

    def update(self, bbox: np.ndarray):
        """Update the track with a matched detection."""
        self.kf.update(_bbox_to_z(bbox))
        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> np.ndarray:
        """Get current estimated bbox [x1, y1, x2, y2]."""
        return _z_to_bbox(self.kf.x[:4])


class SORTTracker:
    """
    Multi-object tracker using the SORT algorithm.

    Assigns persistent integer IDs to detected objects across frames
    using Kalman filtering and IoU-based Hungarian matching.
    """

    def __init__(
        self,
        max_age: int = config.TRACKER_MAX_AGE,
        min_hits: int = config.TRACKER_MIN_HITS,
        iou_threshold: float = config.TRACKER_IOU_THRESHOLD,
    ):
        """
        Initialize the SORT tracker.

        Args:
            max_age: Max frames to keep a track alive without updates.
            min_hits: Min detections before a track is considered confirmed.
            iou_threshold: Minimum IoU for a valid match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[_KalmanTrack] = []

        logger.info(
            f"SORT tracker initialized (max_age={max_age}, "
            f"min_hits={min_hits}, iou_thresh={iou_threshold})"
        )

    def reset(self):
        """Reset all tracks and the ID counter."""
        self.tracks = []
        _KalmanTrack._next_id = 1

    def update(self, detections: list) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects (each has .bbox, .confidence).

        Returns:
            List of TrackedObject instances with persistent IDs.
        """
        # Step 1: Predict new locations for all existing tracks
        predicted_bboxes = []
        for track in self.tracks:
            pred = track.predict()
            predicted_bboxes.append(pred)

        # Step 2: Convert detections to numpy array
        det_bboxes = np.array(
            [list(d.bbox) for d in detections], dtype=float
        ) if detections else np.empty((0, 4))

        det_confs = [d.confidence for d in detections] if detections else []

        # Step 3: Match detections to tracks using IoU + Hungarian
        matched, unmatched_dets, unmatched_tracks = self._associate(
            det_bboxes, predicted_bboxes
        )

        # Step 4: Update matched tracks with assigned detections
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(det_bboxes[det_idx])

        # Step 5: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = _KalmanTrack(det_bboxes[det_idx])
            self.tracks.append(new_track)

        # Step 6: Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # Step 7: Build output — only return confirmed tracks
        results = []
        for track in self.tracks:
            if track.time_since_update > 0:
                continue  # Skip tracks not updated this frame

            is_confirmed = track.hits >= self.min_hits
            bbox = track.get_state()

            results.append(TrackedObject(
                track_id=track.track_id,
                bbox=tuple(bbox.astype(int)),
                confidence=1.0,
                hits=track.hits,
                age=track.age,
                is_confirmed=is_confirmed,
            ))

        logger.debug(
            f"Tracker: {len(self.tracks)} active tracks, "
            f"{len(results)} confirmed outputs"
        )
        return results

    def _associate(
        self,
        det_bboxes: np.ndarray,
        pred_bboxes: list,
    ) -> tuple:
        """
        Associate detections to existing tracks via IoU + Hungarian.

        Returns:
            matched: List of (track_idx, det_idx) pairs.
            unmatched_dets: List of unmatched detection indices.
            unmatched_tracks: List of unmatched track indices.
        """
        num_tracks = len(pred_bboxes)
        num_dets = len(det_bboxes)

        if num_tracks == 0:
            return [], list(range(num_dets)), []

        if num_dets == 0:
            return [], [], list(range(num_tracks))

        # Build IoU cost matrix
        cost_matrix = np.zeros((num_tracks, num_dets))
        for t_idx, pred_bb in enumerate(pred_bboxes):
            for d_idx in range(num_dets):
                cost_matrix[t_idx, d_idx] = _iou(pred_bb, det_bboxes[d_idx])

        # Hungarian algorithm (maximize IoU = minimize negative IoU)
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)

        matched = []
        unmatched_dets = list(range(num_dets))
        unmatched_tracks = list(range(num_tracks))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                if c in unmatched_dets:
                    unmatched_dets.remove(c)
                if r in unmatched_tracks:
                    unmatched_tracks.remove(r)

        return matched, unmatched_dets, unmatched_tracks
