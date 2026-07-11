"""
SORT Tracker — Simple Online and Realtime Tracking (vectorized).

Implements the SORT algorithm from scratch using:
- Batched Kalman filtering across ALL tracks as single numpy ops
- Hungarian Algorithm for detection-to-track assignment (via scipy)
- Broadcast IoU cost matrix

All tracks share the same motion model (F/H/Q/R), so their states are
stored as stacked arrays — X: (N, 7), P: (N, 7, 7) — and predicted /
updated together. The per-track filterpy objects this replaces cost
10-20ms/frame at ~80 live tracks; the batched version is <1ms.

References:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
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


def _iou_matrix(bb_a: np.ndarray, bb_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two bbox sets via broadcasting.

    Args:
        bb_a: (N, 4) boxes [x1, y1, x2, y2]
        bb_b: (M, 4) boxes [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix.
    """
    a = bb_a[:, None, :]  # (N, 1, 4)
    b = bb_b[None, :, :]  # (1, M, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    area_a = np.clip(bb_a[:, 2] - bb_a[:, 0], 0, None) * \
        np.clip(bb_a[:, 3] - bb_a[:, 1], 0, None)
    area_b = np.clip(bb_b[:, 2] - bb_b[:, 0], 0, None) * \
        np.clip(bb_b[:, 3] - bb_b[:, 1], 0, None)

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _bboxes_to_z(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert (M, 4) [x1, y1, x2, y2] to measurements (M, 4) [cx, cy, s, r]
    where s = area, r = aspect ratio (width / height).
    """
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    cx = bboxes[:, 0] + w / 2.0
    cy = bboxes[:, 1] + h / 2.0
    s = w * h
    r = w / np.maximum(h, 1e-6)
    return np.stack([cx, cy, s, r], axis=1)


def _states_to_bboxes(X: np.ndarray) -> np.ndarray:
    """
    Convert stacked states (N, 7) [cx, cy, s, r, ...] back to
    (N, 4) [x1, y1, x2, y2].
    """
    cx, cy = X[:, 0], X[:, 1]
    s = np.maximum(X[:, 2], 1.0)
    r = X[:, 3]
    w = np.sqrt(np.maximum(s * r, 1e-9))
    h = s / np.maximum(w, 1e-6)
    return np.stack([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ], axis=1)


class SORTTracker:
    """
    Multi-object tracker using the SORT algorithm.

    Assigns persistent integer IDs to detected objects across frames
    using batched Kalman filtering and IoU-based Hungarian matching.

    State vector per track: [cx, cy, s, r, vx, vy, vs]
    Measurement:            [cx, cy, s, r]
    """

    def __init__(
        self,
        max_age: int = config.TRACKER_MAX_AGE,
        min_hits: int = config.TRACKER_MIN_HITS,
        iou_threshold: float = config.TRACKER_IOU_THRESHOLD,
        coast_frames: int = None,
    ):
        """
        Initialize the SORT tracker.

        Args:
            max_age: Max frames to keep a track alive without updates.
            min_hits: Min detections before a track is considered confirmed.
            iou_threshold: Minimum IoU for a valid match.
            coast_frames: How many consecutive frames without a real
                detection match a track may still be output for, using
                pure Kalman prediction. Defaults to
                (DETECT_EVERY_N_FRAMES - 1) so callers that don't get a
                detector result every frame (see core/pipeline.py) get
                smooth interpolated boxes instead of the box freezing
                and snapping back on the next detection frame.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        if coast_frames is None:
            coast_frames = max(0, getattr(config, "DETECT_EVERY_N_FRAMES", 1) - 1)
        self.coast_frames = coast_frames

        # ── Shared motion model (identical for every track) ──
        # Constant-velocity transition matrix
        F = np.eye(7)
        F[0, 4] = F[1, 5] = F[2, 6] = 1.0
        self._F = F

        # Measurement matrix
        self._H = np.zeros((4, 7))
        self._H[:4, :4] = np.eye(4)

        # Measurement noise
        R = np.eye(4)
        R[2:, 2:] *= 10.0
        self._R = R

        # Process noise
        Q = np.eye(7)
        Q[-1, -1] *= 0.01
        Q[4:, 4:] *= 0.01
        self._Q = Q

        # Initial covariance for new tracks
        P0 = np.eye(7)
        P0[4:, 4:] *= 1000.0  # High uncertainty for velocities
        P0 *= 10.0
        self._P0 = P0

        # ── Stacked track state ──
        self._X = np.zeros((0, 7))       # states
        self._P = np.zeros((0, 7, 7))    # covariances
        self._ids = np.zeros(0, dtype=int)
        self._hits = np.zeros(0, dtype=int)
        self._age = np.zeros(0, dtype=int)
        self._tsu = np.zeros(0, dtype=int)  # time since update
        self._next_id = 1

        logger.info(
            f"SORT tracker initialized (max_age={max_age}, "
            f"min_hits={min_hits}, iou_thresh={iou_threshold}, "
            f"coast_frames={coast_frames}, vectorized)"
        )

    @property
    def tracks(self):
        """Active track IDs — kept for len(tracker.tracks) compatibility."""
        return self._ids

    def reset(self):
        """Reset all tracks and the ID counter."""
        self._X = np.zeros((0, 7))
        self._P = np.zeros((0, 7, 7))
        self._ids = np.zeros(0, dtype=int)
        self._hits = np.zeros(0, dtype=int)
        self._age = np.zeros(0, dtype=int)
        self._tsu = np.zeros(0, dtype=int)
        self._next_id = 1

    # ── Batched Kalman steps ────────────────────────────────────────

    def _predict_all(self):
        """Advance every track's state one step (single batched op)."""
        if len(self._X) == 0:
            return
        # Prevent area from going negative
        bad = (self._X[:, 6] + self._X[:, 2]) <= 0
        self._X[bad, 6] = 0.0

        self._X = self._X @ self._F.T
        # P' = F P F^T + Q, broadcast over the track axis
        self._P = self._F @ self._P @ self._F.T + self._Q

        self._age += 1
        self._tsu += 1

    def _update_matched(self, track_idx: np.ndarray, Z: np.ndarray):
        """Kalman-update the matched subset of tracks (batched)."""
        if len(track_idx) == 0:
            return
        H, R = self._H, self._R
        X = self._X[track_idx]              # (M, 7)
        P = self._P[track_idx]              # (M, 7, 7)

        y = Z - X @ H.T                     # (M, 4) innovation
        S = H @ P @ H.T + R                 # (M, 4, 4)
        K = P @ H.T @ np.linalg.inv(S)      # (M, 7, 4)

        self._X[track_idx] = X + np.einsum("mij,mj->mi", K, y)

        # Joseph-form covariance update (same as filterpy) for stability
        I_KH = np.eye(7) - K @ H            # (M, 7, 7)
        self._P[track_idx] = (
            I_KH @ P @ I_KH.transpose(0, 2, 1)
            + K @ R @ K.transpose(0, 2, 1)
        )

        self._hits[track_idx] += 1
        self._tsu[track_idx] = 0

    def _add_tracks(self, Z: np.ndarray):
        """Create new tracks from unmatched measurements (M, 4)."""
        m = len(Z)
        if m == 0:
            return
        X_new = np.zeros((m, 7))
        X_new[:, :4] = Z
        self._X = np.concatenate([self._X, X_new])
        self._P = np.concatenate([self._P, np.repeat(self._P0[None], m, axis=0)])
        self._ids = np.concatenate(
            [self._ids, np.arange(self._next_id, self._next_id + m)]
        )
        self._next_id += m
        self._hits = np.concatenate([self._hits, np.ones(m, dtype=int)])
        self._age = np.concatenate([self._age, np.zeros(m, dtype=int)])
        self._tsu = np.concatenate([self._tsu, np.zeros(m, dtype=int)])

    def _keep(self, mask: np.ndarray):
        """Drop tracks where mask is False."""
        self._X = self._X[mask]
        self._P = self._P[mask]
        self._ids = self._ids[mask]
        self._hits = self._hits[mask]
        self._age = self._age[mask]
        self._tsu = self._tsu[mask]

    # ── Public API ──────────────────────────────────────────────────

    def update(self, detections: list) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects (each has .bbox, .confidence).

        Returns:
            List of TrackedObject instances with persistent IDs.
        """
        # Step 1: Predict new locations for all existing tracks
        self._predict_all()

        # Step 2: Detections to arrays
        det_bboxes = np.array(
            [list(d.bbox) for d in detections], dtype=float
        ) if detections else np.empty((0, 4))

        # Step 3: Match detections to tracks using IoU + Hungarian
        matched, unmatched_dets = self._associate(det_bboxes)

        # Step 4: Update matched tracks with assigned detections (batched)
        if matched:
            t_idx = np.array([t for t, _ in matched], dtype=int)
            d_idx = np.array([d for _, d in matched], dtype=int)
            self._update_matched(t_idx, _bboxes_to_z(det_bboxes[d_idx]))

        # Step 5: Create new tracks for unmatched detections
        if len(unmatched_dets):
            self._add_tracks(_bboxes_to_z(det_bboxes[unmatched_dets]))

        # Step 6: Remove dead tracks
        self._keep(self._tsu <= self.max_age)

        # Step 7: Build output — tracks matched this frame, plus tracks
        # still coasting on prediction within coast_frames (see __init__
        # docstring for why). Tracks below min_hits are excluded
        # entirely: a single YOLO flicker (one-frame false positive)
        # shouldn't mint a persistent ID and start accumulating votes.
        out_mask = (self._tsu <= self.coast_frames) & (self._hits >= self.min_hits)
        out_idx = np.nonzero(out_mask)[0]

        results = []
        if len(out_idx):
            bboxes = _states_to_bboxes(self._X[out_idx]).astype(int)
            for row, i in enumerate(out_idx):
                results.append(TrackedObject(
                    track_id=int(self._ids[i]),
                    bbox=tuple(bboxes[row]),
                    confidence=1.0,
                    hits=int(self._hits[i]),
                    age=int(self._age[i]),
                    is_confirmed=True,
                ))

        logger.debug(
            f"Tracker: {len(self._ids)} active tracks, "
            f"{len(results)} confirmed outputs"
        )
        return results

    def _associate(self, det_bboxes: np.ndarray) -> tuple:
        """
        Associate detections to existing tracks via IoU + Hungarian.

        Returns:
            matched: List of (track_idx, det_idx) pairs.
            unmatched_dets: Array of unmatched detection indices.
        """
        num_tracks = len(self._X)
        num_dets = len(det_bboxes)

        if num_tracks == 0 or num_dets == 0:
            return [], np.arange(num_dets)

        pred_bboxes = _states_to_bboxes(self._X)
        iou = _iou_matrix(pred_bboxes, det_bboxes)

        # Hungarian algorithm (maximize IoU = minimize negative IoU)
        row_indices, col_indices = linear_sum_assignment(-iou)

        matched = []
        matched_dets = set()
        for r, c in zip(row_indices, col_indices):
            if iou[r, c] >= self.iou_threshold:
                matched.append((r, c))
                matched_dets.add(c)

        unmatched_dets = np.array(
            [d for d in range(num_dets) if d not in matched_dets], dtype=int
        )
        return matched, unmatched_dets
