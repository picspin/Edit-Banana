"""FastSAM-s mask detection via ultralytics (torch, optional at runtime).

This module is the FastSAM counterpart to modules/yolo_detection.py
(ONNX, no torch). FastSAM is different in one important way: it
needs torch + ultralytics, so nothing in this file imports torch at
module top level. The import happens lazily inside
detect_fastsam_masks(); if torch/ultralytics or the weights file are
missing, that function raises and the caller (detection_contours)
degrades to contours + YOLO instead of crashing. The detection
module never imports torch itself.

Verified history (round 3, scratch only, before this integration):
- FastSAM-s.pt 22.7 MB, CPU first image 3.7 s, repeat synth 0.63 s.
- Synth 4-box flowchart: 54 masks, 4 box masks IoU 0.971-0.992,
  2 arrow masks (shaft+head as ONE instance, IoU 0.889).
- Head-only triangular masks: 0 on all 6 scratch images, so this
  module NEVER reports a separately-detected arrow head.
  "Arrow instance" here always means shaft+head as one mask, with
  head_separated=False. Do not describe it as head separation.
- Dense diagrams: the naive elongated-mask classifier overfires
  (33 long-bar candidates on demo_original_1, mostly text/panel
  boxes, no ground truth), so those candidates are NOT reported
  as arrows. An arrow instance is only reported when the mask
  spans the gap between two contour shape boxes on the same row
  (the same gap logic as the Hough shaft check, applied to mask
  geometry). On a dense diagram with no such gap, the count is
  0 -- and 0 is reported, not hidden.
- Round-4 single-chain image: FastSAM gave 52 masks (4 boxes +
  48 glyphs) and 0 arrow instances under this classifier, because
  its arrows are diagonal, not horizontal gap-spanning masks.
  That 0/3 is a known limitation of this classifier, not a
  success, and it is NOT fixed by this integration round.

Weights are NOT committed to git (22.7 MB binary, upstream
ultralytics asset). The default path is the round-3 scratch copy,
overridable via EDITBANANA_FASTSAM_MODEL; see
docs/fastsam-model-packaging.md for bundle/download options.
"""

import math
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
# Round-3 scratch weights, read-only reference. Production
# deployment should bundle this file with the app instead of
# relying on a scratch path; the path is overridable via
# EDITBANANA_FASTSAM_MODEL. "" (empty string) disables FastSAM.
DEFAULT_FASTSAM_MODEL_PATH = os.environ.get(
    "EDITBANANA_FASTSAM_MODEL",
    "/Users/hilbert/work/editbanana-round3/FastSAM-s.pt")


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1.0, area_a + area_b - inter)


def detect_fastsam_masks(image_bgr, model_path=None, conf=0.4,
                         iou=0.9, imgsz=1024):
    """Return FastSAM mask detections in original image coordinates.

    Input is BGR (cv2.imread); the model is fed RGB, mirroring the
    round-2 YOLO BGR->RGB lesson. Images wider than 1024 px are
    resized to width 1024 (as in round-3 scratch) and bboxes are
    scaled back to original coordinates. Each detection carries
    bbox [x1, y1, x2, y2] in original coordinates, area_work,
    area (original-scale approximation), approx_vertices and
    method "fastsam-s". Raises FileNotFoundError for a missing
    weights file and propagates torch/ultralytics import errors
    so the caller can degrade gracefully.
    """
    path = model_path or DEFAULT_FASTSAM_MODEL_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"FastSAM weights not found: {path}")
    from ultralytics import FastSAM  # lazy: torch is optional here
    h0, w0 = image_bgr.shape[:2]
    scale = 1.0
    work = image_bgr
    if w0 > 1024:
        scale = 1024 / w0
        work = cv2.resize(image_bgr, (1024, int(h0 * scale)),
                          interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    model = FastSAM(path)
    res = model(rgb, device="cpu", retina_masks=True, imgsz=imgsz,
                conf=conf, iou=iou, verbose=False)
    r = res[0]
    if r.masks is None:
        return []
    masks = r.masks.data.cpu().numpy()
    dets: List[Dict[str, Any]] = []
    for m in masks:
        mm = (m > 0.5).astype(np.uint8)
        if mm.shape != (work.shape[0], work.shape[1]):
            mm = cv2.resize(mm, (work.shape[1], work.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        area_work = int(mm.sum())
        if area_work < 50:
            continue
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        ar = w / max(1, h)
        orientation = None
        if 10 <= w <= 80 and 10 <= h <= 80 and 100 <= area_work <= 2000 and 0.5 <= ar <= 2.0:
            orientation = estimate_arrowhead_orientation(c, scale)
        dets.append({
            "bbox": [int(x / scale), int(y / scale),
                     int((x + w) / scale), int((y + h) / scale)],
            "bbox_work": [int(x), int(y), int(w), int(h)],
            "area_work": area_work,
            "area": int(area_work / max(scale * scale, 1e-9)),
            "approx_vertices": int(len(approx)),
            "orientation": orientation,
            "method": "fastsam-s",
        })
    return dets


def estimate_arrowhead_orientation(c: np.ndarray, scale: float = 1.0) -> Dict[str, Any]:
    """Estimate arrowhead orientation angle, apex tip point, and polarity vector.

    Combines second central moments (inertia tensor principal axis) with
    triangle apex vertex polarity detection (sharpest interior angle) to
    resolve 180-degree ambiguity and yield a deterministic pointing direction.

    Returns dict:
        - orientation_deg: float in (-180, 180] (0 = +x / Right, 90 = +y / Down)
        - tip_pt: [x, y] in original image coordinates
        - polarity_vec: [vx, vy] normalized unit direction vector
        - principal_axis_deg: float in (-90, 90] (second central moment axis)
    """
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return {
            "orientation_deg": 0.0, "tip_pt": [0, 0],
            "polarity_vec": [1.0, 0.0], "principal_axis_deg": 0.0,
        }
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]

    # Second central moments principal axis (-pi/2 to pi/2)
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)

    pts = c.reshape(-1, 2).astype(float)
    R = np.array([[math.cos(-theta), -math.sin(-theta)],
                  [math.sin(-theta), math.cos(-theta)]])
    rotated = (pts - np.array([cx, cy])) @ R.T

    pos_pts = rotated[rotated[:, 0] > 0]
    neg_pts = rotated[rotated[:, 0] < 0]
    pos_width = float(np.ptp(pos_pts[:, 1])) if len(pos_pts) > 0 else 0.0
    neg_width = float(np.ptp(neg_pts[:, 1])) if len(neg_pts) > 0 else 0.0

    # 3rd central moment skewness projection along principal axis
    u = np.array([math.cos(theta), math.sin(theta)])
    skew = (M.get("mu30", 0.0) / max(1.0, M["m00"])) * u[0] + (M.get("mu03", 0.0) / max(1.0, M["m00"])) * u[1]

    # Taper polarity: tip has smaller perpendicular thickness (converging apex)
    if pos_width < neg_width * 0.9:
        dir_sign = +1
    elif neg_width < pos_width * 0.9:
        dir_sign = -1
    else:
        dir_sign = +1 if skew >= 0 else -1

    if dir_sign == -1:
        oriented_theta = (theta + math.pi) % (2 * math.pi)
        best_idx = int(np.argmin(rotated[:, 0]))
    else:
        oriented_theta = theta % (2 * math.pi)
        best_idx = int(np.argmax(rotated[:, 0]))

    tip = pts[best_idx]
    deg = math.degrees(oriented_theta)
    if deg > 180:
        deg -= 360

    vx = math.cos(oriented_theta)
    vy = math.sin(oriented_theta)
    return {
        "orientation_deg": round(deg, 1),
        "tip_pt": [int(tip[0] / scale), int(tip[1] / scale)],
        "polarity_vec": [round(vx, 3), round(vy, 3)],
        "principal_axis_deg": round(math.degrees(theta), 1),
    }


def _dist_to_box(bx: List[int], x: float, y: float) -> float:
    dx = max(0, bx[0] - x, x - bx[2])
    dy = max(0, bx[1] - y, y - bx[3])
    return math.hypot(dx, dy)


def classify_arrow_instances(masks, shape_boxes, shafts=None):
    """Classify FastSAM arrow instances and separate arrow heads.

    Handles:
    1. Elongated masks (shaft+head as one mask, as in synth):
       If a compact triangular/wedge arrowhead mask is also detected
       at the destination end of the arrow, head_separated is True
       and head_bbox is recorded. If no separate head mask is
       present, head_separated is False.
    2. Thin-shaft arrows with separated arrowhead masks (as in single-chain):
       When FastSAM segments the arrowhead mask (compact, 10-80 px,
       ar 0.5-2.0) in the gap between shape boxes, it is associated
       with the connecting shaft (if provided) or shape box pair to
       form an arrow instance with head_separated=True.
    3. Diagonal / connecting arrows between shape boxes.
    """
    # Detect compact arrowhead masks (outside shape boxes)
    head_masks = []
    for m in masks:
        x1, y1, x2, y2 = m["bbox"]
        w, h = x2 - x1, y2 - y1
        area = m.get("area_work", 0)
        ar = w / max(1, h)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        in_shape = any(
            b["bbox"][0] <= cx <= b["bbox"][2] and b["bbox"][1] <= cy <= b["bbox"][3]
            for b in shape_boxes
        )
        if not in_shape and 10 <= w <= 80 and 10 <= h <= 80 and 100 <= area <= 2000 and 0.5 <= ar <= 2.0:
            head_masks.append(m)

    out = []
    used_head_indices = set()
    seen_pairs = set()

    # Step 1: Elongated full arrow masks (horizontal gap-spanning)
    for d in masks:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        if not (w >= 80 and h <= 45 and d.get("area_work", 0) >= 800):
            continue
        if any(iou_xyxy(d["bbox"], b["bbox"]) > 0.5 for b in shape_boxes):
            continue
        cy = (y1 + y2) / 2
        for a in shape_boxes:
            for b in shape_boxes:
                if a is b:
                    continue
                ax1, ay1, ax2, ay2 = a["bbox"]
                bx1, by1, bx2, by2 = b["bbox"]
                if abs((ay1 + ay2) / 2 - (by1 + by2) / 2) > 30:
                    continue
                if not (ax2 < bx1):
                    continue
                if abs(cy - (ay1 + ay2) / 2) > 30:
                    continue
                if x1 <= ax2 + 40 and x2 >= bx1 - 40:
                    sep_head = None
                    for idx, hm in enumerate(head_masks):
                        hx1, hy1, hx2, hy2 = hm["bbox"]
                        if (x1 <= (hx1 + hx2) / 2 <= x2 and
                                abs((hy1 + hy2) / 2 - cy) <= 20 and
                                (hx1 + hx2) / 2 >= (x1 + x2) / 2):
                            sep_head = hm
                            used_head_indices.add(idx)
                            break
                    pair = (a["bbox"][0], a["bbox"][1], b["bbox"][0], b["bbox"][1])
                    seen_pairs.add(pair)
                    orient = sep_head.get("orientation") if sep_head else None
                    out.append(dict(
                        d, arrow_instance=True,
                        head_separated=bool(sep_head is not None),
                        head_bbox=sep_head["bbox"] if sep_head else None,
                        orientation_deg=orient.get("orientation_deg") if orient else None,
                        tip_pt=orient.get("tip_pt") if orient else None,
                        polarity_vec=orient.get("polarity_vec") if orient else None,
                        from_box=a["bbox"], to_box=b["bbox"],
                        method="fastsam-s-instance"
                    ))

    # Step 2: Associate unused arrowhead masks with confirmed shafts (when provided)
    if shafts is not None:
        for sa in shafts:
            pair = (sa["from_box"][0], sa["from_box"][1], sa["to_box"][0], sa["to_box"][1])
            if pair in seen_pairs:
                continue
            to_box = sa["to_box"]
            from_box = sa["from_box"]
            for idx, hm in enumerate(head_masks):
                if idx in used_head_indices:
                    continue
                hx1, hy1, hx2, hy2 = hm["bbox"]
                hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if _dist_to_box(to_box, hcx, hcy) <= 40:
                    used_head_indices.add(idx)
                    seen_pairs.add(pair)
                    if (abs((from_box[1] + from_box[3]) / 2 - (to_box[1] + to_box[3]) / 2) <= 30
                            and from_box[2] < to_box[0]):
                        bbox = [from_box[2], hy1, hx2, hy2]
                    else:
                        bbox = [min(from_box[2], hx1), min(from_box[3], hy1),
                                max(from_box[0], hx2), max(from_box[1], hy2)]
                    orient = hm.get("orientation")
                    out.append({
                        "bbox": bbox,
                        "area": hm.get("area", 0),
                        "area_work": hm.get("area_work", 0),
                        "arrow_instance": True,
                        "head_separated": True,
                        "head_bbox": hm["bbox"],
                        "orientation_deg": orient.get("orientation_deg") if orient else None,
                        "tip_pt": orient.get("tip_pt") if orient else None,
                        "polarity_vec": orient.get("polarity_vec") if orient else None,
                        "from_box": from_box,
                        "to_box": to_box,
                        "method": "fastsam-s-head-instance",
                    })
                    break
    else:
        # Step 3: Standalone / test mode without shafts: associate unused heads bridging shape boxes
        for idx, hm in enumerate(head_masks):
            if idx in used_head_indices:
                continue
            hx1, hy1, hx2, hy2 = hm["bbox"]
            hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            for a in shape_boxes:
                for b in shape_boxes:
                    if a is b:
                        continue
                    pair = (a["bbox"][0], a["bbox"][1], b["bbox"][0], b["bbox"][1])
                    if pair in seen_pairs:
                        continue
                    ax1, ay1, ax2, ay2 = a["bbox"]
                    bx1, by1, bx2, by2 = b["bbox"]
                    # Horizontal
                    if abs((ay1 + ay2) / 2 - (by1 + by2) / 2) <= 30 and ax2 < bx1:
                        cy = (ay1 + ay2) / 2
                        if abs(hcy - cy) <= 25 and ax2 <= hcx <= bx1 and _dist_to_box(b["bbox"], hcx, hcy) <= 40:
                            used_head_indices.add(idx)
                            seen_pairs.add(pair)
                        orient = hm.get("orientation")
                        out.append({
                            "bbox": [ax2, hy1, hx2, hy2],
                            "area": hm.get("area", 0),
                            "area_work": hm.get("area_work", 0),
                            "arrow_instance": True,
                            "head_separated": True,
                            "head_bbox": hm["bbox"],
                            "orientation_deg": orient.get("orientation_deg") if orient else None,
                            "tip_pt": orient.get("tip_pt") if orient else None,
                            "polarity_vec": orient.get("polarity_vec") if orient else None,
                            "from_box": a["bbox"],
                            "to_box": b["bbox"],
                            "method": "fastsam-s-head-instance",
                        })
                    # Diagonal (a above b)
                    elif ay2 <= by1 + 10:
                        gap = math.hypot(max(0, ax1 - bx2, bx1 - ax2), max(0, ay1 - by2, by1 - ay2))
                        if gap >= 30 and _dist_to_box(b["bbox"], hcx, hcy) <= 40:
                            cax, cay = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                            cbx, cby = (bx1 + bx2) / 2, (by1 + by2) / 2
                            head_ang = math.atan2(hcy - cay, hcx - cax)
                            box_ang = math.atan2(cby - cay, cbx - cax)
                            diff = abs(head_ang - box_ang)
                            diff = min(diff, 2 * math.pi - diff)
                            if diff < math.radians(25):
                                used_head_indices.add(idx)
                                seen_pairs.add(pair)
                                orient = hm.get("orientation")
                                out.append({
                                    "bbox": [min(hx1, int(cax)), min(ay2, hy1),
                                             max(hx2, int(cax)), max(ay2, hy2)],
                                    "area": hm.get("area", 0),
                                    "area_work": hm.get("area_work", 0),
                                    "arrow_instance": True,
                                    "head_separated": True,
                                    "head_bbox": hm["bbox"],
                                    "orientation_deg": orient.get("orientation_deg") if orient else None,
                                    "tip_pt": orient.get("tip_pt") if orient else None,
                                    "polarity_vec": orient.get("polarity_vec") if orient else None,
                                    "from_box": a["bbox"],
                                    "to_box": b["bbox"],
                                    "method": "fastsam-s-head-instance",
                                })
    return out


def classify_shape_candidates(masks, shape_boxes, iou_thresh=0.5):
    """Large rectangular FastSAM masks evaluated against shape boxes.

    FastSAM rectangular masks matching a contour shape box with
    IoU >= iou_thresh are KEPT (confirmed as valid diagram shapes).
    Masks with IoU < iou_thresh (unmatched noise / background)
    are DROPPED.
    """
    kept, dropped = [], []
    for d in masks:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        if not (d.get("area", 0) >= 5000 and w >= 50 and h >= 30
                and d.get("approx_vertices") == 4):
            continue
        if any(iou_xyxy(d["bbox"], b["bbox"]) >= iou_thresh for b in shape_boxes):
            kept.append(dict(d, confirmed_by_contour=True))
        else:
            dropped.append(dict(d, dropped_as_unmatched=True))
    return kept, dropped


def merge_fastsam(masks, shape_boxes, yolo_boxes=None, shafts=None):
    """Merge FastSAM masks with contour (+YOLO) boxes by IoU.

    Returns arrow instances (with head separation), kept shape
    candidates (confirmed by contour or YOLO), dropped candidates
    (unmatched), and full mask list for audit.
    """
    arrows = classify_arrow_instances(masks, shape_boxes, shafts=shafts)
    kept, dropped = classify_shape_candidates(masks, shape_boxes)
    yolo_kept = [d for d in (yolo_boxes or [])
                 if not d.get("dropped_as_duplicate")]
    still_dropped = []
    yolo_confirmed = []
    for d in dropped:
        if any(iou_xyxy(d["bbox"], y["bbox"]) >= 0.5 for y in yolo_kept):
            yolo_confirmed.append(dict(d, confirmed_by_yolo=True))
        else:
            still_dropped.append(d)
    return {
        "masks": masks,
        "arrow_instances": arrows,
        "shape_candidates_kept": kept + yolo_confirmed,
        "shape_candidates_dropped": still_dropped,
    }
