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
"""

import math
import os
from typing import Any, Dict, List

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


def detect_arrow_shafts(image_bgr: np.ndarray, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Confirm arrow shafts by Hough lines between shape boxes.

    Returns shaft-confirmed arrows with head_separated=False unless
    a square-like triangular head contour is found near the shaft
    end. A shaft-confirmed arrow is NOT a SAM-style instance mask.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hl = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                         minLineLength=60, maxLineGap=10)
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
            # same row, a left of b, with a gap
            if abs((ay1 + ay2) / 2 - (by1 + by2) / 2) > 30:
                continue
            if not (ax2 < bx1):
                continue
            gap = bx1 - ax2
            if gap < 10:
                continue
            cy = (ay1 + ay2) / 2
            # Inner boxes are inset ~4 px, so the shaft line ends
            # ~27 px before the inner box edge; allow 40 px slack
            # for matching only (returned bboxes stay inner).
            shaft = any(
                abs(y1 - cy) < 20 and abs(y2 - cy) < 20
                and min(x1, x2) <= ax2 + 40 and max(x1, x2) >= bx1 - 40
                for x1, y1, x2, y2 in lines
            )
            if not shaft:
                continue
            head = any(
                abs(hx + hw / 2 - bx1) < 40 and abs(hy + hh / 2 - cy) < 40
                for hx, hy, hw, hh in heads
            )
            arrows.append({
                "from_box": a["bbox"], "to_box": b["bbox"],
                "shaft_confirmed": True, "head_separated": bool(head),
                "method": "hough_shaft_between_boxes",
            })
    return arrows


class DetectionContourExtractor:
    """Drop-in shape/arrow extractor for Pipeline step [2].

    process(context) fills context.elements with rectangle elements
    from hole contours and arrow elements for shaft-confirmed
    arrows. It never imports torch or SAM3.
    """

    def __init__(self, config=None):
        self.config = config

    def extract(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cannot read image: {image_path}")
        h, w = img.shape[:2]
        boxes = detect_shape_boxes(img)
        arrows = detect_arrow_shafts(img, boxes)
        return {
            "canvas_width": int(w), "canvas_height": int(h),
            "shapes": boxes, "arrows": arrows,
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
            x1 = a["from_box"][2]; y1 = a["from_box"][1]
            x2 = a["to_box"][0]; y2 = a["to_box"][3]
            elements.append(ElementInfo(
                id=len(elements), element_type="arrow",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                score=0.5 if a["head_separated"] else 0.3,
                polygon=[],
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
                "arrows_head_separated": sum(1 for a in result["arrows"] if a["head_separated"]),
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
