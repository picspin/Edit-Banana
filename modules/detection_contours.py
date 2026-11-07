"""
Detection-contour extractor (no SAM, no torch).

Replaces the SAM3 segmentation step in this branch with OpenCV
contour detection, as verified in the 2026-10-30/31 round-2 tests:

- Shape boxes: RETR_TREE child (hole) contours. On the synthetic
  4-box flowchart this returns exactly 4 boxes. Each box is the
  INNER contour of a box border, inset by about 4 px from the outer
  box edge. The inset is documented, not hidden, and the returned
  bbox is the measured inner bbox (not expanded by guessing).
- Arrows: a shaft is CONFIRMED only when a Hough line segment
  spans the gap between two shape boxes on (nearly) the same
  centre line. An arrow HEAD is reported as separated only when
  a square-like triangular contour is found; on the synthetic
  flowchart no such head is separated (head merges into the row
  outline), so head_separated=False and the arrow must NOT be
  described as a detected instance mask. This module never
  presents a layout-inferred arrow as a detected arrow.
- OCR prompt: the VLM OCR prompt verified in round 2. The old
  prompt (max_tokens=1500, no no-reasoning instruction) returned
  content=None on 11/20 dense-diagram calls. The new prompt below,
  with max_tokens=4000 and temperature=0, returned content on
  10/10 rerun calls (4 dense diagrams + synthetic control, both
  PaddleOCR-VL-1.5 and MinerU2.5-Pro). Boxes from VLM OCR remain
  ESTIMATES (e.g. PaddleOCR boxes on the synthetic control are
  narrow vertical strips and must be matched to OpenCV boxes by
  centre point, not used as measured geometry).

This module does not do YOLO detection, embedding classification,
or local OCR. Those are separate, separately-verified steps in
the round-2 scratch tests and are not claimed by this module.

Round-3 (detection-contours-v8n branch) adds OPTIONAL YOLO boxes
via modules/yolo_detection.py (ONNX Runtime only, no torch).
YOLO is off unless a model path is given: the default is the
round-2 yolov8n-seg ONNX file, never yolo11s. YOLO boxes are
merged only when they do not duplicate a contour box (IoU > 0.5
is dropped as a duplicate). YOLO has no arrow class, so a YOLO
box is never an arrow element.

Round-4 (detection-contours-v8n branch) adds OPTIONAL FastSAM
masks via modules/fastsam_detection.py (torch + ultralytics,
lazy import only). FastSAM is off when its model path is "" and
degrades (fastsam_available=False) when torch or the weights
are missing, same convention as YOLO. FastSAM arrow instances
are shaft+head as ONE mask, head_separated is always False,
and an instance is only reported when the mask spans a gap
between two contour shape boxes; dense-diagram long bars
without such a gap are not arrows. FastSAM shape masks are
reported for audit only and are NOT promoted to pipeline
elements, to avoid the round-3 dense-diagram overfire.
"""

import heapq
import itertools
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

OCR_PROMPT_NEW = (
    "OCR task. Output ONLY the final OCR result. Do NOT output any "
    "reasoning, thinking, chain-of-thought, explanation, or preamble. "
    "Extract all text visible in this image, verbatim, one text element "
    "per line. If you can give a bounding box, append it to the same "
    "line as [x1,y1,x2,y2] in this image's pixel coordinates. "
    "If there is no visible text, output exactly: NO_TEXT. "
    "Do not output anything else."
)
OCR_PARAMS = {"max_tokens": 4000, "temperature": 0}


def detect_shape_boxes(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Return hole-contour shape boxes (inner bboxes, inset ~4 px)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    boxes: List[Dict[str, Any]] = []
    seen = set()
    for i, cnt in enumerate(cnts):
        parent = int(hier[i][3])
        if parent == -1:
            continue  # top-level merged-row outline, not a shape box
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 30:
            continue
        key = (x, y, w, h)
        if key in seen:
            continue
        seen.add(key)
        boxes.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "area": int(area),
            "method": "hole_contour_inner",
            "inset_note": "inner contour, about 4 px inset from outer box edge",
        })
    boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return boxes


def _dist_to_box(bx: List[int], x: float, y: float) -> float:
    dx = max(0, bx[0] - x, x - bx[2])
    dy = max(0, bx[1] - y, y - bx[3])
    return math.hypot(dx, dy)


def bind_edge_label_to_arrow(text_bbox: List[int], arrow_start: Tuple[int, int],
                             arrow_end: Tuple[int, int],
                             vector_points: Optional[List[List[int]]] = None,
                             max_dist: float = 60.0) -> Optional[Tuple[int, int]]:
    """Calculate offset (offset_x, offset_y) from edge midpoint if text is near the arrow.

    Returns (offset_x, offset_y) if projected distance from text center to edge is <= max_dist,
    else None.
    """
    tcx = (text_bbox[0] + text_bbox[2]) / 2.0
    tcy = (text_bbox[1] + text_bbox[3]) / 2.0

    pts = [arrow_start] + (vector_points or []) + [arrow_end]
    seg_lens = [math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    total_len = sum(seg_lens)
    if total_len <= 0:
        mid_x, mid_y = arrow_start
    else:
        half_len = total_len / 2.0
        acc = 0.0
        mid_x, mid_y = pts[-1]
        for i, slen in enumerate(seg_lens):
            if acc + slen >= half_len:
                t = (half_len - acc) / max(1e-6, slen)
                mid_x = pts[i][0] + t * (pts[i+1][0] - pts[i][0])
                mid_y = pts[i][1] + t * (pts[i+1][1] - pts[i][1])
                break
            acc += slen

    min_dist = 1e9
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            d = math.hypot(tcx - x1, tcy - y1)
        else:
            t = max(0.0, min(1.0, ((tcx - x1) * dx + (tcy - y1) * dy) / (dx * dx + dy * dy)))
            d = math.hypot(tcx - (x1 + t * dx), tcy - (y1 + t * dy))
        if d < min_dist:
            min_dist = d

    if min_dist <= max_dist:
        return int(round(tcx - mid_x)), int(round(tcy - mid_y))
def segment_box_collision(p1: Tuple[float, float], p2: Tuple[float, float],
                          box: List[int], margin: float = 4.0) -> bool:
    """Check if line segment p1->p2 intersects or pierces into box interior with margin."""
    bx1, by1, bx2, by2 = box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin
    if (bx1 < p1[0] < bx2 and by1 < p1[1] < by2) or (bx1 < p2[0] < bx2 and by1 < p2[1] < by2):
        return True
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - bx1, bx2 - p1[0], p1[1] - by1, by2 - p1[1]]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return u1 < u2

def hanan_grid_route(start_pt: Tuple[int, int], end_pt: Tuple[int, int],
                     obstacles: List[List[int]], clearance: int = 25,
                     bend_penalty: int = 80,
                     canvas_width: Optional[int] = None,
                     canvas_height: Optional[int] = None,
                     margin: int = 10,
                     text_obstacles: Optional[List[List[int]]] = None,
                     text_clearance: int = 12,
                     text_penalty: int = 1000) -> Optional[List[List[int]]]:
    """Find shortest orthogonal collision-free path with minimal bends on Hanan grid.

    Hanan grid uses orthogonal lines through terminals and obstacle boundaries.
    Edge collisions with any obstacle box interior are pruned using Liang-Barsky.
    Supports canvas bounds clamping/margin protection and soft text obstacle penalty.
    A* cost = Manhattan distance + bend_penalty * (direction changes) + soft_penalties.
    """
    xs = {int(start_pt[0]), int(end_pt[0])}
    ys = {int(start_pt[1]), int(end_pt[1])}
    for o in obstacles:
        xs.add(int(o[0] - clearance))
        xs.add(int(o[2] + clearance))
        ys.add(int(o[1] - clearance))
        ys.add(int(o[3] + clearance))

    for t in (text_obstacles or []):
        xs.add(int(t[0] - text_clearance))
        xs.add(int(t[2] + text_clearance))
        ys.add(int(t[1] - text_clearance))
        ys.add(int(t[3] + text_clearance))

    if canvas_width is not None:
        xs = {max(margin, min(canvas_width - margin, x)) for x in xs}
    if canvas_height is not None:
        ys = {max(margin, min(canvas_height - margin, y)) for y in ys}

    sorted_x = sorted(list(xs))
    sorted_y = sorted(list(ys))

    x_map = {x: i for i, x in enumerate(sorted_x)}
    y_map = {y: i for i, y in enumerate(sorted_y)}

    sx = min(sorted_x, key=lambda x: abs(x - start_pt[0]))
    sy = min(sorted_y, key=lambda y: abs(y - start_pt[1]))
    ex = min(sorted_x, key=lambda x: abs(x - end_pt[0]))
    ey = min(sorted_y, key=lambda y: abs(y - end_pt[1]))

    start_node = (x_map[sx], y_map[sy])
    target_node = (x_map[ex], y_map[ey])

    counter = itertools.count()
    pq = [(0, 0, next(counter), start_node[0], start_node[1], 0, [[int(start_pt[0]), int(start_pt[1])]])]
    visited = {}

    while pq:
        f, cost, _, xi, yi, prev_dir, path = heapq.heappop(pq)
        state = (xi, yi, prev_dir)
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost

        curr_pt = (sorted_x[xi], sorted_y[yi])
        if (xi, yi) == target_node:
            simplified = [path[0]]
            for pt in path[1:]:
                if len(simplified) >= 2:
                    p_prev = simplified[-2]
                    p_last = simplified[-1]
                    if (p_prev[0] == p_last[0] == pt[0]) or (p_prev[1] == p_last[1] == pt[1]):
                        simplified[-1] = pt
                        continue
                simplified.append(pt)
            if simplified[-1] != [int(end_pt[0]), int(end_pt[1])]:
                simplified.append([int(end_pt[0]), int(end_pt[1])])
            return simplified

        neighbors = []
        if xi > 0: neighbors.append((xi - 1, yi, 2))
        if xi < len(sorted_x) - 1: neighbors.append((xi + 1, yi, 1))
        if yi > 0: neighbors.append((xi, yi - 1, 4))
        if yi < len(sorted_y) - 1: neighbors.append((xi, yi + 1, 3))

        for n_xi, n_yi, n_dir in neighbors:
            nxt_pt = (sorted_x[n_xi], sorted_y[n_yi])
            collides = any(segment_box_collision(curr_pt, nxt_pt, o, margin=2.0) for o in obstacles)
            if collides:
                continue
            seg_dist = abs(nxt_pt[0] - curr_pt[0]) + abs(nxt_pt[1] - curr_pt[1])
            turn_cost = bend_penalty if (prev_dir != 0 and prev_dir != n_dir) else 0

            t_penalty = 0
            if text_obstacles:
                t_collides = sum(1 for t in text_obstacles if segment_box_collision(curr_pt, nxt_pt, t, margin=1.0))
                t_penalty = t_collides * text_penalty

            n_cost = cost + seg_dist + turn_cost + t_penalty
            h = abs(nxt_pt[0] - end_pt[0]) + abs(nxt_pt[1] - end_pt[1])
            heapq.heappush(pq, (n_cost + h, n_cost, next(counter), n_xi, n_yi, n_dir, path + [[nxt_pt[0], nxt_pt[1]]]))

    return None


def compute_obstacle_avoidance_route(from_box: List[int], to_box: List[int],
                                     obstacles: List[List[int]],
                                     clearance: int = 25,
                                     lane_index: int = 0,
                                     lane_spacing: int = 15,
                                     canvas_width: Optional[int] = None,
                                     canvas_height: Optional[int] = None,
                                     margin: int = 10,
                                     text_obstacles: Optional[List[List[int]]] = None) -> Optional[Dict[str, Any]]:
    """Compute orthogonal detour waypoints routing around intermediate obstacle boxes.

    Supports lane offset allocation (channel separation), canvas bounds clamping/overflow protection,
    text/label soft obstacle avoidance, and falls back to Hanan-grid A* routing when needed.
    Returns dict with waypoints, bends count, obstacles_avoided list, routing_method, and lane_index.
    """
    effective_clearance = clearance + lane_index * lane_spacing
    fcx, fcy = (from_box[0] + from_box[2]) / 2.0, (from_box[1] + from_box[3]) / 2.0
    tcx, tcy = (to_box[0] + to_box[2]) / 2.0, (to_box[1] + to_box[3]) / 2.0

    other_obs = [o for o in obstacles if o != from_box and o != to_box]

    valid_text_obs = []
    for t in (text_obstacles or []):
        if (from_box[0] <= t[0] and t[2] <= from_box[2] and from_box[1] <= t[1] and t[3] <= from_box[3]):
            continue
        if (to_box[0] <= t[0] and t[2] <= to_box[2] and to_box[1] <= t[1] and t[3] <= to_box[3]):
            continue
        valid_text_obs.append(t)

    # 1. Horizontal span (left -> right or right -> left)
    if abs(fcy - tcy) <= 40 and (from_box[2] < to_box[0] or to_box[2] < from_box[0]):
        left_box = from_box if from_box[2] < to_box[0] else to_box
        right_box = to_box if from_box[2] < to_box[0] else from_box
        is_forward = (from_box is left_box)
        lx2 = left_box[2]
        rx1 = right_box[0]
        y_level = int((fcy + tcy) / 2)

        intervening = [o for o in other_obs if o[0] >= lx2 - 10 and o[2] <= rx1 + 10
                       and o[1] - effective_clearance <= y_level <= o[3] + effective_clearance]
        if intervening:
            min_ox1 = min(o[0] for o in intervening)
            max_ox2 = max(o[2] for o in intervening)
            min_oy1 = min(o[1] for o in intervening)
            max_oy2 = max(o[3] for o in intervening)

            turn1_x = int((lx2 + min_ox1) / 2)
            turn2_x = int((max_ox2 + rx1) / 2)

            raw_top_y = int(min_oy1 - effective_clearance)
            raw_bot_y = int(max_oy2 + effective_clearance)

            candidates = []
            for raw_y, is_top in [(raw_top_y, True), (raw_bot_y, False)]:
                detour_y = raw_y
                overflow = False
                if canvas_height is not None:
                    if detour_y < margin:
                        overflow = True
                        detour_y = margin
                    elif detour_y > canvas_height - margin:
                        overflow = True
                        detour_y = canvas_height - margin
                elif detour_y < margin:
                    overflow = True
                    detour_y = margin

                if is_forward:
                    w = [[lx2, y_level], [turn1_x, y_level], [turn1_x, detour_y],
                         [turn2_x, detour_y], [turn2_x, y_level], [rx1, y_level]]
                else:
                    w = [[rx1, y_level], [turn2_x, y_level], [turn2_x, detour_y],
                         [turn1_x, detour_y], [turn1_x, y_level], [lx2, y_level]]

                collides = any(segment_box_collision(w[i], w[i+1], o, margin=2.0)
                               for i in range(len(w) - 1) for o in other_obs)
                if not collides:
                    text_colls = sum(1 for t in valid_text_obs
                                     if any(segment_box_collision(w[i], w[i+1], t, margin=1.0)
                                            for i in range(len(w) - 1)))
                    overflow_pen = 1 if overflow else 0
                    candidates.append((overflow_pen, text_colls, abs(y_level - detour_y), w))

            if candidates:
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))
                best = candidates[0]
                if best[0] == 0 and best[1] == 0:
                    return {
                        "connector_type": "obstacle_avoidance",
                        "waypoints": best[3],
                        "bends": 4,
                        "obstacles_avoided": intervening,
                        "routing_method": "orthogonal_detour",
                        "lane_index": lane_index,
                    }
                elif not any(c[0] == 0 and c[1] == 0 for c in candidates):
                    start_pt = (lx2, y_level) if is_forward else (rx1, y_level)
                    end_pt = (rx1, y_level) if is_forward else (lx2, y_level)
                    hanan_w = hanan_grid_route(start_pt, end_pt, other_obs,
                                               clearance=effective_clearance,
                                               canvas_width=canvas_width,
                                               canvas_height=canvas_height,
                                               margin=margin,
                                               text_obstacles=valid_text_obs)
                    if hanan_w:
                        return {
                            "connector_type": "obstacle_avoidance",
                            "waypoints": hanan_w,
                            "bends": max(0, len(hanan_w) - 2),
                            "obstacles_avoided": intervening,
                            "routing_method": "hanan_grid_astar",
                            "lane_index": lane_index,
                        }
                return {
                    "connector_type": "obstacle_avoidance",
                    "waypoints": best[3],
                    "bends": 4,
                    "obstacles_avoided": intervening,
                    "routing_method": "orthogonal_detour",
                    "lane_index": lane_index,
                }

            start_pt = (lx2, y_level) if is_forward else (rx1, y_level)
            end_pt = (rx1, y_level) if is_forward else (lx2, y_level)
            hanan_w = hanan_grid_route(start_pt, end_pt, other_obs,
                                       clearance=effective_clearance,
                                       canvas_width=canvas_width,
                                       canvas_height=canvas_height,
                                       margin=margin,
                                       text_obstacles=valid_text_obs)
            if hanan_w:
                return {
                    "connector_type": "obstacle_avoidance",
                    "waypoints": hanan_w,
                    "bends": max(0, len(hanan_w) - 2),
                    "obstacles_avoided": intervening,
                    "routing_method": "hanan_grid_astar",
                    "lane_index": lane_index,
                }

    # 2. Vertical span (top -> bottom or bottom -> top)
    if abs(fcx - tcx) <= 40 and (from_box[3] < to_box[1] or to_box[3] < from_box[1]):
        top_box = from_box if from_box[3] < to_box[1] else to_box
        bot_box = to_box if from_box[3] < to_box[1] else from_box
        is_down = (from_box is top_box)
        ty2 = top_box[3]
        by1 = bot_box[1]
        x_level = int((fcx + tcx) / 2)

        intervening = [o for o in other_obs if o[1] >= ty2 - 10 and o[3] <= by1 + 10
                       and o[0] - effective_clearance <= x_level <= o[2] + effective_clearance]
        if intervening:
            min_ox1 = min(o[0] for o in intervening)
            max_ox2 = max(o[2] for o in intervening)
            min_oy1 = min(o[1] for o in intervening)
            max_oy2 = max(o[3] for o in intervening)

            turn1_y = int((ty2 + min_oy1) / 2)
            turn2_y = int((max_oy2 + by1) / 2)

            raw_left_x = int(min_ox1 - effective_clearance)
            raw_right_x = int(max_ox2 + effective_clearance)

            candidates = []
            for raw_x, is_left in [(raw_left_x, True), (raw_right_x, False)]:
                detour_x = raw_x
                overflow = False
                if canvas_width is not None:
                    if detour_x < margin:
                        overflow = True
                        detour_x = margin
                    elif detour_x > canvas_width - margin:
                        overflow = True
                        detour_x = canvas_width - margin
                elif detour_x < margin:
                    overflow = True
                    detour_x = margin

                if is_down:
                    w = [[x_level, ty2], [x_level, turn1_y], [detour_x, turn1_y],
                         [detour_x, turn2_y], [x_level, turn2_y], [x_level, by1]]
                else:
                    w = [[x_level, by1], [x_level, turn2_y], [detour_x, turn2_y],
                         [detour_x, turn1_y], [x_level, turn1_y], [x_level, ty2]]

                collides = any(segment_box_collision(w[i], w[i+1], o, margin=2.0)
                               for i in range(len(w) - 1) for o in other_obs)
                if not collides:
                    text_colls = sum(1 for t in valid_text_obs
                                     if any(segment_box_collision(w[i], w[i+1], t, margin=1.0)
                                            for i in range(len(w) - 1)))
                    overflow_pen = 1 if overflow else 0
                    candidates.append((overflow_pen, text_colls, abs(x_level - detour_x), w))

            if candidates:
                candidates.sort(key=lambda c: (c[0], c[1], c[2]))
                best = candidates[0]
                if best[0] == 0 and best[1] == 0:
                    return {
                        "connector_type": "obstacle_avoidance",
                        "waypoints": best[3],
                        "bends": 4,
                        "obstacles_avoided": intervening,
                        "routing_method": "orthogonal_detour",
                        "lane_index": lane_index,
                    }
                elif not any(c[0] == 0 and c[1] == 0 for c in candidates):
                    start_pt = (x_level, ty2) if is_down else (x_level, by1)
                    end_pt = (x_level, by1) if is_down else (x_level, ty2)
                    hanan_w = hanan_grid_route(start_pt, end_pt, other_obs,
                                               clearance=effective_clearance,
                                               canvas_width=canvas_width,
                                               canvas_height=canvas_height,
                                               margin=margin,
                                               text_obstacles=valid_text_obs)
                    if hanan_w:
                        return {
                            "connector_type": "obstacle_avoidance",
                            "waypoints": hanan_w,
                            "bends": max(0, len(hanan_w) - 2),
                            "obstacles_avoided": intervening,
                            "routing_method": "hanan_grid_astar",
                            "lane_index": lane_index,
                        }
                return {
                    "connector_type": "obstacle_avoidance",
                    "waypoints": best[3],
                    "bends": 4,
                    "obstacles_avoided": intervening,
                    "routing_method": "orthogonal_detour",
                    "lane_index": lane_index,
                }

            start_pt = (x_level, ty2) if is_down else (x_level, by1)
            end_pt = (x_level, by1) if is_down else (x_level, ty2)
            hanan_w = hanan_grid_route(start_pt, end_pt, other_obs,
                                       clearance=effective_clearance,
                                       canvas_width=canvas_width,
                                       canvas_height=canvas_height,
                                       margin=margin,
                                       text_obstacles=valid_text_obs)
            if hanan_w:
                return {
                    "connector_type": "obstacle_avoidance",
                    "waypoints": hanan_w,
                    "bends": max(0, len(hanan_w) - 2),
                    "obstacles_avoided": intervening,
                    "routing_method": "hanan_grid_astar",
                    "lane_index": lane_index,
                }

    # 3. Arbitrary / diagonal span with intervening collision
    ex_x = from_box[0] if tcx < fcx else from_box[2]
    ex_y = from_box[1] if tcy < fcy else from_box[3]
    en_x = to_box[2] if tcx < fcx else to_box[0]
    en_y = to_box[3] if tcy < fcy else to_box[1]
    direct_colliding = [o for o in other_obs if segment_box_collision((ex_x, ex_y), (en_x, en_y), o, margin=2.0)]
    direct_text_colliding = [t for t in valid_text_obs if segment_box_collision((ex_x, ex_y), (en_x, en_y), t, margin=1.0)]
    if direct_colliding or direct_text_colliding:
        hanan_w = hanan_grid_route((ex_x, ex_y), (en_x, en_y), other_obs,
                                   clearance=effective_clearance,
                                   canvas_width=canvas_width,
                                   canvas_height=canvas_height,
                                   margin=margin,
                                   text_obstacles=valid_text_obs)
        if hanan_w:
            return {
                "connector_type": "obstacle_avoidance",
                "waypoints": hanan_w,
                "bends": max(0, len(hanan_w) - 2),
                "obstacles_avoided": direct_colliding,
                "routing_method": "hanan_grid_astar",
                "lane_index": lane_index,
            }

def compute_feedback_loop_route(from_box: List[int], to_box: List[int],
                                obstacles: List[List[int]],
                                clearance: int = 35,
                                prefer_side: Optional[str] = None,
                                canvas_width: Optional[int] = None,
                                canvas_height: Optional[int] = None,
                                margin: int = 10,
                                text_obstacles: Optional[List[List[int]]] = None) -> Optional[Dict[str, Any]]:
    """Compute an outer return loop route for feedback / cycle edges.

    Routes around outer perimeter ports (e.g., bottom-to-bottom or top-to-top),
    avoiding the primary forward axis and intermediate obstacles with 0 overlap.
    """
    fcx = (from_box[0] + from_box[2]) / 2.0
    tcx = (to_box[0] + to_box[2]) / 2.0
    fcy = (from_box[1] + from_box[3]) / 2.0
    tcy = (to_box[1] + to_box[3]) / 2.0

    other_obs = [o for o in obstacles if o != from_box and o != to_box]
    valid_text_obs = []
    for t in (text_obstacles or []):
        if (from_box[0] <= t[0] and t[2] <= from_box[2] and from_box[1] <= t[1] and t[3] <= from_box[3]):
            continue
        if (to_box[0] <= t[0] and t[2] <= to_box[2] and to_box[1] <= t[1] and t[3] <= to_box[3]):
            continue
        valid_text_obs.append(t)

    # 1. Horizontal feedback loop (from_box is downstream to the right of to_box)
    if from_box[0] >= to_box[2] - 10:
        span_boxes = [from_box, to_box] + [
            o for o in other_obs
            if o[0] >= min(to_box[0], from_box[0]) - 10 and o[2] <= max(to_box[2], from_box[2]) + 10
        ]
        max_y2 = max(b[3] for b in span_boxes)
        min_y1 = min(b[1] for b in span_boxes)

        bot_outer_y = int(max_y2 + clearance)
        top_outer_y = int(min_y1 - clearance)

        candidates = []
        # Candidate 1: Bottom loop (exit bottom of from_box, enter bottom of to_box)
        w_bot = [[int(fcx), from_box[3]], [int(fcx), bot_outer_y],
                 [int(tcx), bot_outer_y], [int(tcx), to_box[3]]]
        coll_bot = any(segment_box_collision(w_bot[i], w_bot[i+1], o, margin=2.0)
                       for i in range(len(w_bot)-1) for o in other_obs)
        text_bot = sum(1 for t in valid_text_obs if any(segment_box_collision(w_bot[i], w_bot[i+1], t, margin=1.0) for i in range(len(w_bot)-1)))
        overflow_bot = (canvas_height is not None and bot_outer_y > canvas_height - margin)
        candidates.append((1 if overflow_bot or coll_bot else 0, text_bot,
                           0 if prefer_side in (None, 'bottom') else 1, w_bot, 'bottom'))

        # Candidate 2: Top loop (exit top of from_box, enter top of to_box)
        w_top = [[int(fcx), from_box[1]], [int(fcx), top_outer_y],
                 [int(tcx), top_outer_y], [int(tcx), to_box[1]]]
        coll_top = any(segment_box_collision(w_top[i], w_top[i+1], o, margin=2.0)
                       for i in range(len(w_top)-1) for o in other_obs)
        text_top = sum(1 for t in valid_text_obs if any(segment_box_collision(w_top[i], w_top[i+1], t, margin=1.0) for i in range(len(w_top)-1)))
        overflow_top = (top_outer_y < margin)
        candidates.append((1 if overflow_top or coll_top else 0, text_top,
                           0 if prefer_side == 'top' else 1, w_top, 'top'))

        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        best_w = candidates[0][3]
        return {
            "connector_type": "feedback_loop",
            "waypoints": best_w,
            "bends": 2,
            "loop_side": candidates[0][4],
            "routing_method": "outer_loop",
        }

    # 2. Vertical feedback loop (from_box is downstream below to_box)
    if from_box[1] >= to_box[3] - 10:
        span_boxes = [from_box, to_box] + [
            o for o in other_obs
            if o[1] >= min(to_box[1], from_box[1]) - 10 and o[3] <= max(to_box[3], from_box[3]) + 10
        ]
        max_x2 = max(b[2] for b in span_boxes)
        min_x1 = min(b[0] for b in span_boxes)

        right_outer_x = int(max_x2 + clearance)
        left_outer_x = int(min_x1 - clearance)

        candidates = []
        # Candidate 1: Right loop
        w_right = [[from_box[2], int(fcy)], [right_outer_x, int(fcy)],
                   [right_outer_x, int(tcy)], [to_box[2], int(tcy)]]
        coll_right = any(segment_box_collision(w_right[i], w_right[i+1], o, margin=2.0)
                         for i in range(len(w_right)-1) for o in other_obs)
        text_right = sum(1 for t in valid_text_obs if any(segment_box_collision(w_right[i], w_right[i+1], t, margin=1.0) for i in range(len(w_right)-1)))
        overflow_right = (canvas_width is not None and right_outer_x > canvas_width - margin)
        candidates.append((1 if overflow_right or coll_right else 0, text_right,
                           0 if prefer_side in (None, 'right') else 1, w_right, 'right'))

        # Candidate 2: Left loop
        w_left = [[from_box[0], int(fcy)], [left_outer_x, int(fcy)],
                  [left_outer_x, int(tcy)], [to_box[0], int(tcy)]]
        coll_left = any(segment_box_collision(w_left[i], w_left[i+1], o, margin=2.0)
                        for i in range(len(w_left)-1) for o in other_obs)
        text_left = sum(1 for t in valid_text_obs if any(segment_box_collision(w_left[i], w_left[i+1], t, margin=1.0) for i in range(len(w_left)-1)))
        overflow_left = (left_outer_x < margin)
        candidates.append((1 if overflow_left or coll_left else 0, text_left,
                           0 if prefer_side == 'left' else 1, w_left, 'left'))

        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        best_w = candidates[0][3]
        return {
            "connector_type": "feedback_loop",
            "waypoints": best_w,
            "bends": 2,
            "loop_side": candidates[0][4],
            "routing_method": "outer_loop",
        }

    return None


def detect_elbow_connectors(lines: List[Any], boxes: List[Dict[str, Any]],
                            image_bgr: Optional[np.ndarray] = None,
                            heads: Optional[List[List[int]]] = None,
                            slack: int = 40) -> List[Dict[str, Any]]:
    """Detect multi-directional elbow connectors (L-shaped and Z-shaped) between shape boxes.

    Traces orthogonal line segments meeting at right-angle corner points (拐点):
    - 1-bend L-connectors (Horizontal -> Vertical or Vertical -> Horizontal)
    - 2-bend Z-connectors (H1 -> V -> H2 or V1 -> H -> V2)
    """
    horiz = []
    vert = []
    for l in lines:
        x1, y1, x2, y2 = l
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx >= 30 and dy <= 10:
            horiz.append((min(x1, x2), int((y1 + y2) / 2), max(x1, x2), int((y1 + y2) / 2)))
        elif dy >= 30 and dx <= 10:
            vert.append((int((x1 + x2) / 2), min(y1, y2), int((x1 + x2) / 2), max(y1, y2)))

    elbows = []
    connected_pairs = set()
    head_pts = [((h[0] + h[2]) / 2, (h[1] + h[3]) / 2) for h in (heads or [])]

    for a in boxes:
        for b in boxes:
            if a is b:
                continue

            # 1. L-shaped: H meets V near corner (vx, hy)
            found = False
            for hx1, hy, hx2, _ in horiz:
                for vx, vy1, _, vy2 in vert:
                    if min(abs(hx1 - vx), abs(hx2 - vx)) <= 25 and min(abs(vy1 - hy), abs(vy2 - hy)) <= 25:
                        h_far_x = hx1 if abs(hx2 - vx) < abs(hx1 - vx) else hx2
                        v_far_y = vy1 if abs(vy2 - hy) < abs(vy1 - hy) else vy2
                        p_h = (h_far_x, hy)
                        p_v = (vx, v_far_y)

                        if _dist_to_box(a["bbox"], p_h[0], p_h[1]) <= slack and _dist_to_box(b["bbox"], p_v[0], p_v[1]) <= slack:
                            box_h, box_v = a, b
                        elif _dist_to_box(b["bbox"], p_h[0], p_h[1]) <= slack and _dist_to_box(a["bbox"], p_v[0], p_v[1]) <= slack:
                            box_h, box_v = b, a
                        else:
                            continue

                        head_at_h = any(_dist_to_box(box_h["bbox"], hx, hy_pt) <= 40 and math.hypot(p_h[0] - hx, p_h[1] - hy_pt) <= 50 for hx, hy_pt in head_pts)
                        head_at_v = any(_dist_to_box(box_v["bbox"], hx, hy_pt) <= 40 and math.hypot(p_v[0] - hx, p_v[1] - hy_pt) <= 50 for hx, hy_pt in head_pts)

                        if head_at_h and not head_at_v:
                            from_b, to_b = box_v["bbox"], box_h["bbox"]
                            wpts = [[int(p_v[0]), int(p_v[1])], [int(vx), int(hy)], [int(p_h[0]), int(p_h[1])]]
                            head = True
                        else:
                            from_b, to_b = box_h["bbox"], box_v["bbox"]
                            wpts = [[int(p_h[0]), int(p_h[1])], [int(vx), int(hy)], [int(p_v[0]), int(p_v[1])]]
                            head = head_at_v

                        pair = (from_b[0], from_b[1], to_b[0], to_b[1])
                        rev_pair = (to_b[0], to_b[1], from_b[0], from_b[1])
                        if pair not in connected_pairs and rev_pair not in connected_pairs:
                            elbows.append({
                                "from_box": from_b, "to_box": to_b,
                                "connector_type": "elbow",
                                "waypoints": wpts,
                                "bends": 1,
                                "shaft_confirmed": True,
                                "head_separated": bool(head),
                                "method": "elbow_orthogonal_chain",
                            })
                            connected_pairs.add(pair)
                            found = True
                            break
                if found:
                    break
            if found:
                continue

            # 2. Z-shaped: H1 -> V -> H2
            for h1x1, h1y, h1x2, _ in horiz:
                for vx, vy1, _, vy2 in vert:
                    if min(abs(h1x1 - vx), abs(h1x2 - vx)) <= 25 and min(abs(vy1 - h1y), abs(vy2 - h1y)) <= 25:
                        h1_far_x = h1x1 if abs(h1x2 - vx) < abs(h1x1 - vx) else h1x2
                        v_far_y = vy1 if abs(vy2 - h1y) < abs(vy1 - h1y) else vy2
                        for h2x1, h2y, h2x2, _ in horiz:
                            if h2y == h1y and h2x1 == h1x1:
                                continue
                            if min(abs(h2x1 - vx), abs(h2x2 - vx)) <= 25 and abs(h2y - v_far_y) <= 25:
                                h2_far_x = h2x1 if abs(h2x2 - vx) < abs(h2x1 - vx) else h2x2
                                p1 = (h1_far_x, h1y)
                                p2 = (h2_far_x, h2y)
                                if _dist_to_box(a["bbox"], p1[0], p1[1]) <= slack and _dist_to_box(b["bbox"], p2[0], p2[1]) <= slack:
                                    box1, box2 = a, b
                                elif _dist_to_box(b["bbox"], p1[0], p1[1]) <= slack and _dist_to_box(a["bbox"], p2[0], p2[1]) <= slack:
                                    box1, box2 = b, a
                                else:
                                    continue

                                head_at_p1 = any(_dist_to_box(box1["bbox"], hx, hy_pt) <= 40 for hx, hy_pt in head_pts)
                                head_at_p2 = any(_dist_to_box(box2["bbox"], hx, hy_pt) <= 40 for hx, hy_pt in head_pts)

                                if head_at_p1 and not head_at_p2:
                                    from_b, to_b = box2["bbox"], box1["bbox"]
                                    wpts = [[int(p2[0]), int(p2[1])], [int(vx), int(h2y)], [int(vx), int(h1y)], [int(p1[0]), int(p1[1])]]
                                    head = True
                                else:
                                    from_b, to_b = box1["bbox"], box2["bbox"]
                                    wpts = [[int(p1[0]), int(p1[1])], [int(vx), int(h1y)], [int(vx), int(h2y)], [int(p2[0]), int(p2[1])]]
                                    head = head_at_p2

                                pair = (from_b[0], from_b[1], to_b[0], to_b[1])
                                rev_pair = (to_b[0], to_b[1], from_b[0], from_b[1])
                                if pair not in connected_pairs and rev_pair not in connected_pairs:
                                    elbows.append({
                                        "from_box": from_b, "to_box": to_b,
                                        "connector_type": "elbow",
                                        "waypoints": wpts,
                                        "bends": 2,
                                        "shaft_confirmed": True,
                                        "head_separated": bool(head),
                                        "method": "elbow_orthogonal_chain",
                                    })
                                    connected_pairs.add(pair)
                                    found = True
                                    break
                        if found:
                            break
                if found:
                    break
    return elbows


def detect_arrow_shafts(image_bgr: np.ndarray, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Confirm arrow shafts and multi-directional elbow connectors by Hough lines.

    Returns shaft-confirmed arrows (straight and elbow connectors) with
    head_separated=False unless an arrowhead contour is found near the shaft end.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hl = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                         minLineLength=35, maxLineGap=50)
    lines = []
    if hl is not None:
        for x1, y1, x2, y2 in hl.reshape(-1, 4):
            if math.hypot(x2 - x1, y2 - y1) >= 35:
                lines.append((int(x1), int(y1), int(x2), int(y2)))
    # Triangle/wedge arrowhead detection
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    heads = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 5000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 60 or h > 60 or not (0.4 <= w / max(1, h) <= 2.5):
            continue
        # Exclude glyphs inside shape boxes
        cx, cy = x + w / 2, y + h / 2
        if boxes and any(b["bbox"][0] <= cx <= b["bbox"][2] and b["bbox"][1] <= cy <= b["bbox"][3] for b in boxes):
            continue
        fill_ratio = area / max(1.0, w * h)
        if 0.25 <= fill_ratio <= 0.85:
            heads.append([int(x), int(y), int(x + w), int(y + h)])

    head_pts = [((h[0] + h[2]) / 2, (h[1] + h[3]) / 2) for h in heads]
    arrows = []
    connected_pairs = set()

    for a in boxes:
        for b in boxes:
            if a is b:
                continue
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]

            # 1. Horizontal straight arrow (same row)
            if abs((ay1 + ay2) / 2 - (by1 + by2) / 2) <= 30 and (ax2 < bx1 or bx2 < ax1):
                left_box = a if ax2 < bx1 else b
                right_box = b if ax2 < bx1 else a
                lx2 = left_box["bbox"][2]
                rx1 = right_box["bbox"][0]
                gap = rx1 - lx2
                if gap >= 10:
                    cy = (ay1 + ay2) / 2
                    slack = max(40, int(gap * 0.35))
                    shaft = any(
                        abs(y1 - cy) < 25 and abs(y2 - cy) < 25
                        and min(x1, x2) <= lx2 + slack and max(x1, x2) >= rx1 - slack
                        for x1, y1, x2, y2 in lines
                    )
                    if shaft:
                        head_at_left = any(_dist_to_box(left_box["bbox"], hx, hy) <= 40 for hx, hy in head_pts)
                        head_at_right = any(_dist_to_box(right_box["bbox"], hx, hy) <= 40 for hx, hy in head_pts)
                        if head_at_left and not head_at_right:
                            from_b, to_b = right_box["bbox"], left_box["bbox"]
                            head = True
                        else:
                            from_b, to_b = left_box["bbox"], right_box["bbox"]
                            head = head_at_right
                        pair = (from_b[0], from_b[1], to_b[0], to_b[1])
                        if pair not in connected_pairs:
                            arrows.append({
                                "from_box": from_b, "to_box": to_b,
                                "shaft_confirmed": True, "head_separated": bool(head),
                                "method": "hough_shaft_between_boxes",
                            })
                            connected_pairs.add(pair)
                            continue

            # 2. Vertical straight arrow (same column)
            if abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2) <= 30 and (ay2 < by1 or by2 < ay1):
                top_box = a if ay2 < by1 else b
                bot_box = b if ay2 < by1 else a
                ty2 = top_box["bbox"][3]
                by1_c = bot_box["bbox"][1]
                gap = by1_c - ty2
                if gap >= 10:
                    cx = (ax1 + ax2) / 2
                    slack = max(40, int(gap * 0.35))
                    shaft = any(
                        abs(x1 - cx) < 25 and abs(x2 - cx) < 25
                        and min(y1, y2) <= ty2 + slack and max(y1, y2) >= by1_c - slack
                        for x1, y1, x2, y2 in lines
                    )
                    if shaft:
                        head_at_top = any(_dist_to_box(top_box["bbox"], hx, hy) <= 40 for hx, hy in head_pts)
                        head_at_bot = any(_dist_to_box(bot_box["bbox"], hx, hy) <= 40 for hx, hy in head_pts)
                        if head_at_top and not head_at_bot:
                            from_b, to_b = bot_box["bbox"], top_box["bbox"]
                            head = True
                        else:
                            from_b, to_b = top_box["bbox"], bot_box["bbox"]
                            head = head_at_bot
                        pair = (from_b[0], from_b[1], to_b[0], to_b[1])
                        if pair not in connected_pairs:
                            arrows.append({
                                "from_box": from_b, "to_box": to_b,
                                "shaft_confirmed": True, "head_separated": bool(head),
                                "method": "hough_shaft_between_boxes",
                            })
                            connected_pairs.add(pair)
                            continue

            # 3. Diagonal straight arrow (a above b with gap, non-overlapping)
            if ay2 <= by1 + 10:
                gap = math.hypot(max(0, ax1 - bx2, bx1 - ax2), max(0, ay1 - by2, by1 - ay2))
                if gap >= 30:
                    cax, cay = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                    cbx, cby = (bx1 + bx2) / 2, (by1 + by2) / 2
                    box_ang = math.atan2(cby - cay, cbx - cax)
                    for x1, y1, x2, y2 in lines:
                        d1a = _dist_to_box(a["bbox"], x1, y1)
                        d2b = _dist_to_box(b["bbox"], x2, y2)
                        d2a = _dist_to_box(a["bbox"], x2, y2)
                        d1b = _dist_to_box(b["bbox"], x1, y1)
                        match = False
                        if d1a <= 40 and d2b <= 40:
                            line_ang = math.atan2(y2 - y1, x2 - x1)
                            match = True
                        elif d2a <= 40 and d1b <= 40:
                            line_ang = math.atan2(y1 - y2, x1 - x2)
                            match = True
                        if match:
                            diff = abs(line_ang - box_ang)
                            diff = min(diff, 2 * math.pi - diff)
                            if diff < math.radians(20):
                                head_at_b = any(_dist_to_box(b["bbox"], hx, hy) <= 40 for hx, hy in head_pts)
                                pair = (a["bbox"][0], a["bbox"][1], b["bbox"][0], b["bbox"][1])
                                if pair not in connected_pairs:
                                    arrows.append({
                                        "from_box": a["bbox"], "to_box": b["bbox"],
                                        "shaft_confirmed": True, "head_separated": bool(head_at_b),
                                        "method": "hough_shaft_between_boxes",
                                    })
                                    connected_pairs.add(pair)
                                    break

    # 4. Multi-directional elbow connectors (L-shaped and Z-shaped)
    elbows = detect_elbow_connectors(lines, boxes, image_bgr=image_bgr, heads=heads, slack=40)
    for el in elbows:
        pair = (el["from_box"][0], el["from_box"][1], el["to_box"][0], el["to_box"][1])
        rev_pair = (el["to_box"][0], el["to_box"][1], el["from_box"][0], el["from_box"][1])
        if pair not in connected_pairs and rev_pair not in connected_pairs:
            arrows.append(el)
            connected_pairs.add(pair)
    return arrows


class DetectionContourExtractor:
    """Drop-in shape/arrow extractor for Pipeline step [2].

    process(context) fills context.elements with rectangle elements
    from hole contours and arrow elements for shaft-confirmed
    arrows. It never imports torch or SAM3.
    """

    def __init__(self, config=None, yolo_model_path=None,
                 fastsam_model_path=None):
        self.config = config
        # None = use yolo_detection.DEFAULT_YOLO_MODEL_PATH;
        # "" (empty string) = YOLO disabled, round-2 behaviour.
        self.yolo_model_path = yolo_model_path
        # None = use fastsam_detection.DEFAULT_FASTSAM_MODEL_PATH;
        # "" (empty string) = FastSAM disabled. Missing torch /
        # weights degrades, it never crashes extraction.
        self.fastsam_model_path = fastsam_model_path

    def extract(self, image_path: str, yolo_model_path=None,
                  fastsam_model_path=None) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cannot read image: {image_path}")
        h, w = img.shape[:2]
        boxes = detect_shape_boxes(img)
        arrows = detect_arrow_shafts(img, boxes)
        # Optional YOLO boxes (round-3, v8n default, never yolo11s).
        yolo_path = self.yolo_model_path if yolo_model_path is None else yolo_model_path
        yolo_boxes = []
        yolo_available = False
        yolo_error = None
        if yolo_path:
            try:
                from .yolo_detection import detect_yolo_boxes, iou_xyxy
                raw = detect_yolo_boxes(img, model_path=yolo_path)
                yolo_available = True
                # Drop YOLO boxes duplicating a contour box (IoU > 0.5).
                for d in raw:
                    if any(iou_xyxy(d["bbox"], b["bbox"]) > 0.5 for b in boxes):
                        d = dict(d); d["dropped_as_duplicate"] = True
                        yolo_boxes.append(d)
                    else:
                        yolo_boxes.append(d)
            except Exception as e:  # missing onnxruntime / model file: degrade, do not crash
                yolo_error = f"{type(e).__name__}: {e}"
        kept_yolo = [d for d in yolo_boxes if not d.get("dropped_as_duplicate")]
        # Optional FastSAM masks (round-4 integration, torch lazy).
        fs_path = self.fastsam_model_path if fastsam_model_path is None else fastsam_model_path
        fs_masks, fs_arrows, fs_kept, fs_dropped = [], [], [], []
        fs_available, fs_error = False, None
        if fs_path:
            try:
                from .fastsam_detection import detect_fastsam_masks, merge_fastsam
                raw_fs = detect_fastsam_masks(img, model_path=fs_path)
                merged = merge_fastsam(raw_fs, boxes, yolo_boxes, shafts=arrows)
                fs_masks, fs_arrows = merged["masks"], merged["arrow_instances"]
                fs_kept = merged["shape_candidates_kept"]
                fs_dropped = merged["shape_candidates_dropped"]
                fs_available = True
                # Propagate FastSAM detected heads to arrow shafts
                for a in arrows:
                    for fsa in fs_arrows:
                        if fsa.get("head_separated") and fsa.get("head_bbox"):
                            hbx = fsa["head_bbox"]
                            hcx, hcy = (hbx[0] + hbx[2]) / 2, (hbx[1] + hbx[3]) / 2
                            dx = max(0, a["to_box"][0] - hcx, hcx - a["to_box"][2])
                            dy = max(0, a["to_box"][1] - hcy, hcy - a["to_box"][3])
                            if math.hypot(dx, dy) <= 40:
                                a["head_separated"] = True
                                a["head_bbox"] = fsa["head_bbox"]
                                if fsa.get("orientation_deg") is not None:
                                    a["orientation_deg"] = fsa["orientation_deg"]
                                    a["tip_pt"] = fsa.get("tip_pt")
                                    a["polarity_vec"] = fsa.get("polarity_vec")
            except Exception as e:  # missing torch/weights: degrade, do not crash
                fs_error = f"{type(e).__name__}: {e}"
        return {
            "canvas_width": int(w), "canvas_height": int(h),
            "shapes": boxes, "arrows": arrows,
            "yolo_boxes": yolo_boxes, "yolo_kept": kept_yolo,
            "yolo_available": yolo_available, "yolo_error": yolo_error,
            "yolo_model": yolo_path or None,
            "fastsam_masks": fs_masks,
            "fastsam_arrow_instances": fs_arrows,
            "fastsam_shapes_kept": fs_kept,
            "fastsam_shapes_dropped": fs_dropped,
            "fastsam_available": fs_available,
            "fastsam_error": fs_error,
            "fastsam_model": fs_path or None,
            "ocr_prompt": OCR_PROMPT_NEW, "ocr_params": dict(OCR_PARAMS),
        }

    def process(self, context) -> Any:
        from .data_types import BoundingBox, ElementInfo, ProcessingResult
        result = self.extract(context.image_path)
        elements = []
        for i, b in enumerate(result["shapes"]):
            x1, y1, x2, y2 = b["bbox"]
            elements.append(ElementInfo(
                id=i, element_type="rectangle",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=1.0, polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            ))
        shapes_bboxes = [b["bbox"] for b in result["shapes"]]
        canvas_width = result.get("canvas_width")
        canvas_height = result.get("canvas_height")

        # Collect text obstacles for soft obstacle avoidance
        text_obstacles: List[List[int]] = []
        for elem in getattr(context, "elements", []):
            if getattr(elem, "element_type", "") in {"text", "edge_label"} and getattr(elem, "bbox", None):
                b = elem.bbox
                text_obstacles.append([int(b.x1), int(b.y1), int(b.x2), int(b.y2)])
        if hasattr(context, "intermediate_results") and isinstance(context.intermediate_results, dict):
            for k in ("text_boxes", "ocr_boxes"):
                for b in context.intermediate_results.get(k, []):
                    if isinstance(b, (list, tuple)) and len(b) >= 4:
                        text_obstacles.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            if "text_xml" in context.intermediate_results:
                txml = context.intermediate_results["text_xml"]
                if isinstance(txml, str):
                    for m in re.finditer(r'<mxGeometry\s+[^>]*?x="([\d.]+)"\s+y="([\d.]+)"\s+width="([\d.]+)"\s+height="([\d.]+)"', txml):
                        x, y, w, h = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
                        text_obstacles.append([int(x), int(y), int(x + w), int(y + h)])
        # Check bidirectional / feedback loops
        feedback_arrow_indices = set()
        for idx1, a1 in enumerate(result["arrows"]):
            fb1, tb1 = a1["from_box"], a1["to_box"]
            for idx2, a2 in enumerate(result["arrows"]):
                if idx1 == idx2:
                    continue
                fb2, tb2 = a2["from_box"], a2["to_box"]
                if fb1 == tb2 and tb1 == fb2:
                    fcx1, fcy1 = (fb1[0] + fb1[2]) / 2, (fb1[1] + fb1[3]) / 2
                    tcx1, tcy1 = (tb1[0] + tb1[2]) / 2, (tb1[1] + tb1[3]) / 2
                    if (fcx1 > tcx1 + 20) or (abs(fcx1 - tcx1) <= 20 and fcy1 > tcy1 + 20):
                        feedback_arrow_indices.add(idx1)

        obstacle_lane_counts: Dict[Any, int] = {}
        for j, a in enumerate(result["arrows"]):
            fb = a["from_box"]
            tb = a["to_box"]
            if a.get("connector_type") in {"elbow", "obstacle_avoidance", "feedback_loop"} and a.get("waypoints"):
                wpts = a["waypoints"]
                abox = [min(p[0] for p in wpts), min(p[1] for p in wpts),
                        max(p[0] for p in wpts), max(p[1] for p in wpts)]
                poly = wpts
                arrow_start = tuple(wpts[0])
                arrow_end = tuple(wpts[-1])
                vector_points = wpts[1:-1]
            elif j in feedback_arrow_indices or a.get("is_feedback") or a.get("connector_type") == "feedback_loop":
                fb_route = compute_feedback_loop_route(
                    fb, tb, shapes_bboxes,
                    canvas_width=canvas_width, canvas_height=canvas_height,
                    text_obstacles=text_obstacles
                )
                if fb_route:
                    wpts = fb_route["waypoints"]
                    abox = [min(p[0] for p in wpts), min(p[1] for p in wpts),
                            max(p[0] for p in wpts), max(p[1] for p in wpts)]
                    poly = wpts
                    arrow_start = tuple(wpts[0])
                    arrow_end = tuple(wpts[-1])
                    vector_points = wpts[1:-1]
                    a["connector_type"] = "feedback_loop"
                    a["waypoints"] = wpts
                    a["bends"] = fb_route["bends"]
                    a["loop_side"] = fb_route["loop_side"]
                else:
                    abox = [min(fb[0], tb[0]), min(fb[1], tb[1]), max(fb[2], tb[2]), max(fb[3], tb[3])]
                    arrow_start = ((fb[0] + fb[2]) // 2, (fb[1] + fb[3]) // 2)
                    arrow_end = ((tb[0] + tb[2]) // 2, (tb[1] + tb[3]) // 2)
                    poly = []
                    vector_points = []
            else:
                detour = compute_obstacle_avoidance_route(
                    fb, tb, shapes_bboxes,
                    canvas_width=canvas_width, canvas_height=canvas_height,
                    text_obstacles=text_obstacles
                )
                if detour:
                    obs_key = tuple(sorted((o[0], o[1], o[2], o[3]) for o in detour["obstacles_avoided"]))
                    lane_idx = obstacle_lane_counts.get(obs_key, 0)
                    obstacle_lane_counts[obs_key] = lane_idx + 1
                    if lane_idx > 0:
                        detour = compute_obstacle_avoidance_route(
                            fb, tb, shapes_bboxes, lane_index=lane_idx,
                            canvas_width=canvas_width, canvas_height=canvas_height,
                            text_obstacles=text_obstacles
                        ) or detour
                    wpts = detour["waypoints"]
                    abox = [min(p[0] for p in wpts), min(p[1] for p in wpts),
                            max(p[0] for p in wpts), max(p[1] for p in wpts)]
                    poly = wpts
                    arrow_start = tuple(wpts[0])
                    arrow_end = tuple(wpts[-1])
                    vector_points = wpts[1:-1]
                    a["connector_type"] = "obstacle_avoidance"
                    a["waypoints"] = wpts
                    a["bends"] = detour["bends"]
                    a["obstacles_avoided"] = detour["obstacles_avoided"]
                    a["routing_method"] = detour.get("routing_method", "orthogonal_detour")
                    a["lane_index"] = detour.get("lane_index", lane_idx)
                else:
                    fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                    if abs(fcy - tcy) <= 30 and fb[2] < tb[0]:
                        abox = [fb[2], min(fb[1], tb[1]), tb[0], max(fb[3], tb[3])]
                        arrow_start = (fb[2], int((fb[1] + fb[3]) / 2))
                        arrow_end = (tb[0], int((tb[1] + tb[3]) / 2))
                    elif abs(fcx - tcx) <= 30 and fb[3] < tb[1]:
                        abox = [min(fb[0], tb[0]), fb[3], max(fb[2], tb[2]), tb[1]]
                        arrow_start = (int((fb[0] + fb[2]) / 2), fb[3])
                        arrow_end = (int((tb[0] + tb[2]) / 2), tb[1])
                    else:
                        ex_x = fb[0] if tcx < fcx else fb[2]
                        ex_y = fb[1] if tcy < fcy else fb[3]
                        en_x = tb[2] if tcx < fcx else tb[0]
                        en_y = tb[3] if tcy < fcy else tb[1]
                        abox = [min(ex_x, en_x), min(ex_y, en_y), max(ex_x, en_x), max(ex_y, en_y)]
                        arrow_start = (ex_x, ex_y)
                        arrow_end = (en_x, en_y)
                    poly = []
                    vector_points = []
            source_id = None
            target_id = None
            for s_idx, sb in enumerate(result["shapes"]):
                if sb["bbox"] == a["from_box"]:
                    source_id = str(s_idx)
                if sb["bbox"] == a["to_box"]:
                    target_id = str(s_idx)
            is_rounded = bool(vector_points and len(vector_points) >= 1)
            elements.append(ElementInfo(
                id=len(elements), element_type="arrow",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=0.5 if a.get("head_separated") else 0.3,
                polygon=poly,
                arrow_start=arrow_start,
                arrow_end=arrow_end,
                vector_points=vector_points,
                source_id=source_id,
                target_id=target_id,
                is_rounded=is_rounded,
            ))
        # FastSAM arrow instances: a mask that spans the same gap
        # as a shaft-confirmed arrow CONFIRMS that arrow as an
        # instance; it is NOT a second arrow element. Only an
        # instance with no matching shaft gap becomes a new element.
        # This keeps the synth element count at shapes + arrows
        # instead of double-counting the same 2 arrows.
        fs_matched, fs_new = 0, 0
        for a in result.get("fastsam_arrow_instances", []):
            x1, y1, x2, y2 = a["bbox"]
            cy = (y1 + y2) / 2
            matched = any(
                (a.get("from_box") == sa.get("from_box") and a.get("to_box") == sa.get("to_box")) or
                (x1 <= sa["from_box"][2] + 40
                 and x2 >= sa["to_box"][0] - 40
                 and abs(cy - (sa["from_box"][1] + sa["from_box"][3]) / 2) <= 30)
                for sa in result.get("arrows", []))
            if matched:
                fs_matched += 1
                continue
            fs_new += 1
            elements.append(ElementInfo(
                id=len(elements), element_type="arrow",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=0.5,
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            ))
        result["_fastsam_matched_shaft"] = fs_matched
        result["_fastsam_new"] = fs_new
        for d in result.get("yolo_kept", []):
            x1, y1, x2, y2 = d["bbox"]
            elements.append(ElementInfo(
                id=len(elements), element_type="picture",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=float(d.get("conf", 0.0)),
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            ))
        context.elements = elements
        context.canvas_width = result["canvas_width"]
        context.canvas_height = result["canvas_height"]
        context.intermediate_results["detection_contours"] = result
        return ProcessingResult(
            success=True, elements=elements,
            canvas_width=result["canvas_width"],
            canvas_height=result["canvas_height"],
            metadata={
                "shapes": len(result["shapes"]),
                "arrows_shaft_confirmed": len(result["arrows"]),
                "arrows_head_separated": sum(1 for a in result["arrows"] if a.get("head_separated")),
                "arrows_elbow": sum(1 for a in result["arrows"] if a.get("connector_type") == "elbow"),
                "arrows_obstacle_avoided": sum(1 for a in result["arrows"] if a.get("connector_type") == "obstacle_avoidance"),
                "arrows_feedback_loop": sum(1 for a in result["arrows"] if a.get("connector_type") == "feedback_loop"),
                "yolo_available": result.get("yolo_available", False),
                "yolo_error": result.get("yolo_error"),
                "fastsam_masks": len(result.get("fastsam_masks", [])),
                "fastsam_arrow_instances": len(result.get("fastsam_arrow_instances", [])),
                "fastsam_shapes_kept": len(result.get("fastsam_shapes_kept", [])),
                "fastsam_available": result.get("fastsam_available", False),
                "fastsam_error": result.get("fastsam_error"),
                "fastsam_arrow_instances_matched_shaft": result.get("_fastsam_matched_shaft", 0),
                "fastsam_arrow_instances_new": result.get("_fastsam_new", 0),
                "sam3_used": False,
            },
        )

    def save_visualization(self, context, path: str) -> None:
        img = cv2.imread(context.image_path)
        if img is None:
            return
        for e in context.elements:
            cv2.rectangle(img, (e.bbox.x1, e.bbox.y1), (e.bbox.x2, e.bbox.y2), (0, 0, 255), 2)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, img)

    def save_metadata(self, context, path: str) -> None:
        import json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context.intermediate_results.get("detection_contours", {}),
                      f, ensure_ascii=False, indent=1)
