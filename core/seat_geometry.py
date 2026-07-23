"""
Smart Seat Sync — core geometry
Bounding-box coordinate math + IoU overlap logic for seat-occupancy tracking.
================================================================================

Coordinate conventions (fixed for the entire codebase; do not mix)
------------------------------------------------------------------
We standardise on **pixel-space, top-left origin, x → right, y → down**, which
is what OpenCV frames and Ultralytics `.xyxy` results already use.

Two box formats appear in the pipeline:

  xyxy : (x1, y1, x2, y2)   two opposite corners; x2 >= x1, y2 >= y1
  xywh : (cx, cy, w, h)     centre point + width/height (YOLO's native label form)

A box is *degenerate* if w <= 0 or h <= 0. Degenerate boxes have zero area and
never overlap anything — we treat that as a hard invariant, not a special case
sprinkled through the callers.

Why IoU (Intersection over Union)
---------------------------------
Seat occupancy is "does a detected person box sufficiently cover a known seat
region?". Raw pixel distance between centres is scale-dependent (a person far
from the camera has a small box) and does not capture *coverage*. IoU is
scale-invariant and bounded in [0, 1], which makes a single threshold (e.g.
0.30) behave consistently across near and far seats.

Given two axis-aligned boxes A and B:

    inter_w = max(0, min(A.x2, B.x2) - max(A.x1, B.x1))
    inter_h = max(0, min(A.y2, B.y2) - max(A.y1, B.y1))
    I       = inter_w * inter_h
    U       = area(A) + area(B) - I
    IoU     = I / U            (0 if U == 0)

The two `max(0, ...)` clamps are the whole trick: when the projected intervals
on an axis do not overlap, the "intersection length" goes negative, and clamping
it to 0 makes the intersection area correctly vanish. Forgetting either clamp is
the classic IoU bug — you get spurious positive overlap for disjoint boxes whose
projections cross on only one axis.

This module is pure NumPy (no torch dependency) so it runs on the FastAPI worker
without importing the DL stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "box_area",
    "iou_pair",
    "iou_matrix",
    "match_detections_to_seats",
    "SeatAssignment",
]

_EPS = 1e-9


# ----------------------------------------------------------------------------
# Format conversions (vectorised; accept a single box or an (N, 4) array)
# ----------------------------------------------------------------------------
def _as_2d(boxes: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalise input to shape (N, 4). Returns (array, was_single_box)."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError(f"expected 4 values for a single box, got {arr.shape}")
        return arr.reshape(1, 4), True
    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr, False
    raise ValueError(f"expected shape (4,) or (N, 4), got {arr.shape}")


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) -> (x1, y1, x2, y2). Negative w/h are clamped to 0 width."""
    arr, single = _as_2d(boxes)
    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    w = np.clip(w, 0.0, None)
    h = np.clip(h, 0.0, None)
    out = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
    return out[0] if single else out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) -> (cx, cy, w, h). Corners are ordered defensively so a
    caller that passes x2 < x1 still yields a non-negative width."""
    arr, single = _as_2d(boxes)
    x1 = np.minimum(arr[:, 0], arr[:, 2])
    y1 = np.minimum(arr[:, 1], arr[:, 3])
    x2 = np.maximum(arr[:, 0], arr[:, 2])
    y2 = np.maximum(arr[:, 1], arr[:, 3])
    out = np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1)
    return out[0] if single else out


def box_area(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Area of xyxy boxes; degenerate boxes (x2<x1 or y2<y1) report 0."""
    arr, single = _as_2d(boxes_xyxy)
    w = np.clip(arr[:, 2] - arr[:, 0], 0.0, None)
    h = np.clip(arr[:, 3] - arr[:, 1], 0.0, None)
    area = w * h
    return float(area[0]) if single else area


# ----------------------------------------------------------------------------
# IoU
# ----------------------------------------------------------------------------
def iou_pair(box_a_xyxy: np.ndarray, box_b_xyxy: np.ndarray) -> float:
    """IoU of two single xyxy boxes. Returns 0.0 for any zero-area configuration."""
    a = np.asarray(box_a_xyxy, dtype=np.float64).reshape(4)
    b = np.asarray(box_b_xyxy, dtype=np.float64).reshape(4)

    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > _EPS else 0.0


def iou_matrix(boxes_a_xyxy: np.ndarray, boxes_b_xyxy: np.ndarray) -> np.ndarray:
    """Vectorised IoU between every A (rows) and every B (cols).

    Shapes: A is (N, 4), B is (M, 4) -> returns (N, M).

    Broadcasting layout: we lift A to (N, 1, 4) and B to (1, M, 4) so the
    pairwise corner comparisons produce (N, M) intersection extents in one shot,
    avoiding an N*M Python loop over frames.
    """
    A, _ = _as_2d(boxes_a_xyxy)
    B, _ = _as_2d(boxes_b_xyxy)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)

    # Top-left corner of intersection = elementwise max of the two top-lefts.
    inter_x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    inter_y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    # Bottom-right corner of intersection = elementwise min of the two bottom-rights.
    inter_x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    inter_y2 = np.minimum(A[:, None, 3], B[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = box_area(A)[:, None]           # (N, 1)
    area_b = box_area(B)[None, :]           # (1, M)
    union = area_a + area_b - inter
    # Guard the divide: union == 0 only when both boxes are degenerate -> IoU 0.
    return np.where(union > _EPS, inter / union, 0.0)


# ----------------------------------------------------------------------------
# Seat <-> detection assignment (greedy, deterministic)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SeatAssignment:
    seat_index: int          # index into the seat list
    detection_index: int     # index into the detection list, or -1 if empty
    iou: float               # overlap score of the winning pair (0.0 if empty)
    occupied: bool


def match_detections_to_seats(
    seat_boxes_xyxy: np.ndarray,
    person_boxes_xyxy: np.ndarray,
    iou_threshold: float = 0.30,
) -> list[SeatAssignment]:
    """Assign at most one person detection to each fixed seat region.

    Algorithm (greedy maximum-IoU, one-to-one):
      1. Compute the full seat x person IoU matrix.
      2. Repeatedly take the globally largest remaining IoU above threshold,
         lock that (seat, person) pair, and strike out that seat row and person
         column so neither is reused.
      3. Seats with no surviving pair above threshold are marked empty.

    Greedy (rather than Hungarian) is deliberate: seat counts per frame are tiny
    (tens), the score is a single scalar, and greedy maximum-weight matching is
    both O(S*P) to build + O(k) to resolve and fully deterministic, which keeps
    the occupancy signal stable frame-to-frame (no flip-flopping from an optimal
    solver picking an equal-cost alternative).

    Ties are broken by lower index (via a stable argmax on a flattened matrix),
    so identical scenes always yield identical assignments.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    seats, _ = _as_2d(seat_boxes_xyxy) if np.asarray(seat_boxes_xyxy).size else (
        np.empty((0, 4)), False)
    persons, _ = _as_2d(person_boxes_xyxy) if np.asarray(person_boxes_xyxy).size else (
        np.empty((0, 4)), False)

    n_seats = seats.shape[0]
    scores = iou_matrix(seats, persons)          # (S, P)
    seat_used = np.zeros(n_seats, dtype=bool)
    person_used = np.zeros(persons.shape[0], dtype=bool)

    assignments: dict[int, SeatAssignment] = {}

    # Flatten and sort candidate pairs by descending IoU for the greedy sweep.
    if scores.size:
        flat_order = np.argsort(scores, axis=None, kind="stable")[::-1]
        for flat_idx in flat_order:
            s, p = divmod(int(flat_idx), scores.shape[1])
            iou = float(scores[s, p])
            if iou < iou_threshold:
                break  # remaining pairs are all below threshold (sorted desc)
            if seat_used[s] or person_used[p]:
                continue
            seat_used[s] = True
            person_used[p] = True
            assignments[s] = SeatAssignment(s, p, iou, occupied=True)

    # Fill in the empty seats so the output is exactly one entry per seat, in order.
    result: list[SeatAssignment] = []
    for s in range(n_seats):
        result.append(assignments.get(s, SeatAssignment(s, -1, 0.0, occupied=False)))
    return result


# ----------------------------------------------------------------------------
# Self-test: run `python core/seat_geometry.py` to validate the math in isolation.
# ----------------------------------------------------------------------------
def _selftest() -> None:
    # 1. Identical boxes -> IoU 1.
    b = np.array([0, 0, 10, 10], dtype=float)
    assert abs(iou_pair(b, b) - 1.0) < 1e-9, "identical boxes must have IoU 1"

    # 2. Disjoint boxes touching only on one axis projection -> IoU 0 (the clamp).
    left = np.array([0, 0, 10, 10], dtype=float)
    right = np.array([20, 5, 30, 15], dtype=float)      # x-disjoint, y-overlapping
    assert iou_pair(left, right) == 0.0, "x-disjoint boxes must have IoU 0"

    # 3. Half-overlap sanity: two 10x10 boxes sharing a 5x10 strip.
    #    inter=50, union=100+100-50=150 -> IoU = 1/3.
    a = np.array([0, 0, 10, 10], dtype=float)
    c = np.array([5, 0, 15, 10], dtype=float)
    assert abs(iou_pair(a, c) - (1.0 / 3.0)) < 1e-9, "half-overlap IoU must be 1/3"

    # 4. Degenerate box -> IoU 0, area 0.
    degen = np.array([5, 5, 5, 5], dtype=float)
    assert box_area(degen) == 0.0 and iou_pair(a, degen) == 0.0

    # 5. Format round-trip.
    xywh = np.array([5.0, 5.0, 10.0, 10.0])
    assert np.allclose(xyxy_to_xywh(xywh_to_xyxy(xywh)), xywh), "xywh round-trip failed"

    # 6. Matrix agrees with pairwise.
    seats = np.array([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=float)
    people = np.array([[5, 0, 15, 10], [200, 200, 210, 210]], dtype=float)
    m = iou_matrix(seats, people)
    assert abs(m[0, 0] - iou_pair(seats[0], people[0])) < 1e-9
    assert m[1, 1] == 0.0

    # 7. Assignment: seat0 taken by person0, seat1 empty.
    out = match_detections_to_seats(seats, people, iou_threshold=0.3)
    assert out[0].occupied and out[0].detection_index == 0
    assert not out[1].occupied and out[1].detection_index == -1

    # 8. One-to-one: two people over one seat -> only the higher-IoU one wins.
    two_people = np.array([[5, 0, 15, 10], [1, 0, 11, 10]], dtype=float)
    out2 = match_detections_to_seats(seats[:1], two_people, iou_threshold=0.1)
    assert out2[0].detection_index == 1, "closer person (higher IoU) must win the seat"

    print("[ OK ] seat_geometry self-test passed (8 assertions).")


if __name__ == "__main__":
    _selftest()
