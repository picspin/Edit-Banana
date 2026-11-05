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

import math
import os
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
    seen_pairs = set()

    for a in boxes:
        for b in boxes:
            if a is b:
                continue
            pair_key = (min(tuple(a["bbox"]), tuple(b["bbox"])), max(tuple(a["bbox"]), tuple(b["bbox"])))
            if pair_key in seen_pairs:
                continue

            # 1. L-shaped: H meets V near corner (vx, hy)
            found = False
            for hx1, hy, hx2, _ in horiz:
                for vx, vy1, _, vy2 in vert:
                    corner_x = min(abs(hx1 - vx), abs(hx2 - vx))
                    corner_y = min(abs(vy1 - hy), abs(vy2 - hy))
                    if corner_x <= 25 and corner_y <= 25:
                        h_far_x = hx1 if abs(hx2 - vx) < abs(hx1 - vx) else hx2
                        v_far_y = vy1 if abs(vy2 - hy) < abs(vy1 - hy) else vy2
                        # Direction: A -> H -> Corner -> V -> B
                        if _dist_to_box(a["bbox"], h_far_x, hy) <= slack and _dist_to_box(b["bbox"], vx, v_far_y) <= slack:
                            head = any(
                                _dist_to_box(b["bbox"], hx + hw / 2, hy + hh / 2) < 40
                                for hx, hy, hw, hh in (heads or [])
                            )
                            elbows.append({
                                "from_box": a["bbox"], "to_box": b["bbox"],
                                "connector_type": "elbow",
                                "waypoints": [[int(h_far_x), int(hy)], [int(vx), int(hy)], [int(vx), int(v_far_y)]],
                                "bends": 1,
                                "shaft_confirmed": True,
                                "head_separated": bool(head),
                                "method": "elbow_orthogonal_chain",
                            })
                            seen_pairs.add(pair_key)
                            found = True
                            break
                        # Direction: A -> V -> Corner -> H -> B
                        elif _dist_to_box(a["bbox"], vx, v_far_y) <= slack and _dist_to_box(b["bbox"], h_far_x, hy) <= slack:
                            head = any(
                                _dist_to_box(b["bbox"], hx + hw / 2, hy + hh / 2) < 40
                                for hx, hy, hw, hh in (heads or [])
                            )
                            elbows.append({
                                "from_box": a["bbox"], "to_box": b["bbox"],
                                "connector_type": "elbow",
                                "waypoints": [[int(vx), int(v_far_y)], [int(vx), int(hy)], [int(h_far_x), int(hy)]],
                                "bends": 1,
                                "shaft_confirmed": True,
                                "head_separated": bool(head),
                                "method": "elbow_orthogonal_chain",
                            })
                            seen_pairs.add(pair_key)
                            found = True
                            break
                if found:
                    break
            if found:
                continue

            # 2. Z-shaped: H1 -> V -> H2
            for h1x1, h1y, h1x2, _ in horiz:
                for vx, vy1, _, vy2 in vert:
                    c1_x = min(abs(h1x1 - vx), abs(h1x2 - vx))
                    c1_y = min(abs(vy1 - h1y), abs(vy2 - h1y))
                    if c1_x <= 25 and c1_y <= 25:
                        h1_far_x = h1x1 if abs(h1x2 - vx) < abs(h1x1 - vx) else h1x2
                        v_far_y = vy1 if abs(vy2 - h1y) < abs(vy1 - h1y) else vy2
                        for h2x1, h2y, h2x2, _ in horiz:
                            if h2y == h1y and h2x1 == h1x1:
                                continue
                            c2_x = min(abs(h2x1 - vx), abs(h2x2 - vx))
                            c2_y = abs(h2y - v_far_y)
                            if c2_x <= 25 and c2_y <= 25:
                                h2_far_x = h2x1 if abs(h2x2 - vx) < abs(h2x1 - vx) else h2x2
                                if (_dist_to_box(a["bbox"], h1_far_x, h1y) <= slack
                                        and _dist_to_box(b["bbox"], h2_far_x, h2y) <= slack):
                                    head = any(
                                        _dist_to_box(b["bbox"], hx + hw / 2, hy + hh / 2) < 40
                                        for hx, hy, hw, hh in (heads or [])
                                    )
                                    elbows.append({
                                        "from_box": a["bbox"], "to_box": b["bbox"],
                                        "connector_type": "elbow",
                                        "waypoints": [[int(h1_far_x), int(h1y)], [int(vx), int(h1y)],
                                                      [int(vx), int(h2y)], [int(h2_far_x), int(h2y)]],
                                        "bends": 2,
                                        "shaft_confirmed": True,
                                        "head_separated": bool(head),
                                        "method": "elbow_orthogonal_chain",
                                    })
                                    seen_pairs.add(pair_key)
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
    hl = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                         minLineLength=60, maxLineGap=50)
    lines = []
    if hl is not None:
        for x1, y1, x2, y2 in hl.reshape(-1, 4):
            if math.hypot(x2 - x1, y2 - y1) >= 60:
                lines.append((int(x1), int(y1), int(x2), int(y2)))
    # Square-like triangle heads (v2 filter from round 2)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    heads = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 80 or area > 5000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) != 3:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 60 or h > 60 or not (0.5 <= w / max(1, h) <= 2.0):
            continue
        heads.append([int(x), int(y), int(w), int(h)])



    arrows = []
    for a in boxes:
        for b in boxes:
            if a is b:
                continue
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            # 1. Horizontal arrow (same row, a left of b, with a gap)
            if abs((ay1 + ay2) / 2 - (by1 + by2) / 2) <= 30 and ax2 < bx1:
                gap = bx1 - ax2
                if gap < 10:
                    continue
                cy = (ay1 + ay2) / 2
                slack = max(40, int(gap * 0.35))
                shaft = any(
                    abs(y1 - cy) < 25 and abs(y2 - cy) < 25
                    and min(x1, x2) <= ax2 + slack and max(x1, x2) >= bx1 - slack
                    for x1, y1, x2, y2 in lines
                )
                if shaft:
                    head = any(
                        abs(hx + hw / 2 - bx1) < 40 and abs(hy + hh / 2 - cy) < 40
                        for hx, hy, hw, hh in heads
                    )
                    arrows.append({
                        "from_box": a["bbox"], "to_box": b["bbox"],
                        "shaft_confirmed": True, "head_separated": bool(head),
                        "method": "hough_shaft_between_boxes",
                    })
                    continue

            # 2. Diagonal / connecting arrow (a above b with gap, non-overlapping)
            if ay2 <= by1 + 10:
                gap = math.hypot(max(0, ax1 - bx2, bx1 - ax2), max(0, ay1 - by2, by1 - ay2))
                if gap < 30:
                    continue
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
                            head = any(
                                _dist_to_box(b["bbox"], hx + hw / 2, hy + hh / 2) < 40
                                for hx, hy, hw, hh in heads
                            )
                            arrows.append({
                                "from_box": a["bbox"], "to_box": b["bbox"],
                                "shaft_confirmed": True, "head_separated": bool(head),
                                "method": "hough_shaft_between_boxes",
                            })
                            break

    # 3. Multi-directional elbow connectors (L-shaped and Z-shaped)
    straight_pairs = {(a["from_box"][0], a["from_box"][1], a["to_box"][0], a["to_box"][1]) for a in arrows}
    elbows = detect_elbow_connectors(lines, boxes, image_bgr=image_bgr, heads=heads, slack=40)
    for el in elbows:
        pair = (el["from_box"][0], el["from_box"][1], el["to_box"][0], el["to_box"][1])
        if pair not in straight_pairs:
            arrows.append(el)
            straight_pairs.add(pair)
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
        for j, a in enumerate(result["arrows"]):
            if a.get("connector_type") == "elbow" and a.get("waypoints"):
                wpts = a["waypoints"]
                abox = [min(p[0] for p in wpts), min(p[1] for p in wpts),
                        max(p[0] for p in wpts), max(p[1] for p in wpts)]
                poly = wpts
            else:
                fb = a["from_box"]
                tb = a["to_box"]
                fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
                tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
                if abs(fcy - tcy) <= 30 and fb[2] < tb[0]:
                    abox = [fb[2], min(fb[1], tb[1]), tb[0], max(fb[3], tb[3])]
                elif abs(fcx - tcx) <= 30 and fb[3] < tb[1]:
                    abox = [min(fb[0], tb[0]), fb[3], max(fb[2], tb[2]), tb[1]]
                else:
                    ex_x = fb[0] if tcx < fcx else fb[2]
                    ex_y = fb[1] if tcy < fcy else fb[3]
                    en_x = tb[2] if tcx < fcx else tb[0]
                    en_y = tb[3] if tcy < fcy else tb[1]
                    abox = [min(ex_x, en_x), min(ex_y, en_y), max(ex_x, en_x), max(ex_y, en_y)]
                poly = []
            x1, y1, x2, y2 = abox
            elements.append(ElementInfo(
                id=len(elements), element_type="arrow",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=0.5 if a.get("head_separated") else 0.3,
                polygon=poly,
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
                "yolo_boxes": len(result.get("yolo_boxes", [])),
                "yolo_kept": len(result.get("yolo_kept", [])),
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
