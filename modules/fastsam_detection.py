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

import os
from typing import Any, Dict, List

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
        dets.append({
            "bbox": [int(x / scale), int(y / scale),
                     int((x + w) / scale), int((y + h) / scale)],
            "bbox_work": [int(x), int(y), int(w), int(h)],
            "area_work": area_work,
            "area": int(area_work / max(scale * scale, 1e-9)),
            "approx_vertices": int(len(approx)),
            "method": "fastsam-s",
        })
    return dets


def classify_arrow_instances(masks, shape_boxes):
    """Gap-spanning FastSAM arrow instances only.

    A mask is an arrow instance only if it is elongated
    (width >= 80, height <= 45, area_work >= 800, as in round 3),
    does not duplicate a contour shape box (IoU <= 0.5), and
    spans the gap between two shape boxes on the same row
    (centre-line difference <= 30, mask reaches within 40 px of
    both gap edges). head_separated is always False: FastSAM
    gives shaft+head as one mask, never a separate head mask.
    """
    out = []
    for d in masks:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        if not (w >= 80 and h <= 45 and d.get("area_work", 0) >= 800):
            continue
        if any(iou_xyxy(d["bbox"], b["bbox"]) > 0.5 for b in shape_boxes):
            continue
        cy = (y1 + y2) / 2
        spans = False
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
                    spans = True
        if not spans:
            continue
        out.append(dict(d, arrow_instance=True, head_separated=False,
                        method="fastsam-s-instance"))
    return out


def classify_shape_candidates(masks, shape_boxes):
    """Large rectangular FastSAM masks not duplicating a shape box.

    These are reported for audit (kept) but are NOT promoted to
    pipeline elements by detection_contours.process(): on dense
    diagrams most such masks are panels/text, not flowchart
    boxes, and promoting them would overfire. Duplicates
    (IoU > 0.5 vs a contour shape) are dropped.
    """
    kept, dropped = [], []
    for d in masks:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        if not (d.get("area", 0) >= 5000 and w >= 50 and h >= 30
                and d.get("approx_vertices") == 4):
            continue
        if any(iou_xyxy(d["bbox"], b["bbox"]) > 0.5 for b in shape_boxes):
            dropped.append(dict(d, dropped_as_duplicate=True))
        else:
            kept.append(d)
    return kept, dropped


def merge_fastsam(masks, shape_boxes, yolo_boxes=None):
    """Merge FastSAM masks with contour (+YOLO) boxes by IoU.

    Returns arrow instances, kept shape candidates, dropped
    duplicates (also dropped when duplicating a kept YOLO box),
    and the full mask list for audit. YOLO boxes have no arrow
    class, so they only affect shape dedup, never arrows.
    """
    arrows = classify_arrow_instances(masks, shape_boxes)
    kept, dropped = classify_shape_candidates(masks, shape_boxes)
    yolo_kept = [d for d in (yolo_boxes or [])
                 if not d.get("dropped_as_duplicate")]
    extra_dropped = []
    still_kept = []
    for d in kept:
        if any(iou_xyxy(d["bbox"], y["bbox"]) > 0.5 for y in yolo_kept):
            extra_dropped.append(dict(d, dropped_as_duplicate=True))
        else:
            still_kept.append(d)
    return {
        "masks": masks,
        "arrow_instances": arrows,
        "shape_candidates_kept": still_kept,
        "shape_candidates_dropped": dropped + extra_dropped,
    }
