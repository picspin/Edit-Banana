#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides upload and conversion API. Run with: python server_pa.py
Server runs at http://localhost:8000
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import urllib.request
import json
import base64
import time
import cv2
import numpy as np

app = FastAPI(
    title="Edit Banana API",
    description="Image to editable DrawIO (XML) and interactive canvas editing.",
    version="2.0.0",
)

# Mount static directory
static_dir = os.path.join(PROJECT_ROOT, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount output directory for image serving
output_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(output_dir, exist_ok=True)
app.mount("/output", StaticFiles(directory=output_dir), name="output")

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gemini-3.1-flash-image"
    size: str = "1024x1024"


class ParseRequest(BaseModel):
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    prompt_hints: Optional[str] = None


class ParseRegionRequest(BaseModel):
    image_path: Optional[str] = "output/_tmp_parse.png"
    crop_box: List[int] # [x1, y1, x2, y2]


class ExportRequest(BaseModel):
    elements: List[dict]
    canvas_width: int = 1024
    canvas_height: int = 1024


class RenderImageRequest(BaseModel):
    elements: List[dict]
    canvas_width: int = 1024
    canvas_height: int = 1024
    format: str = "png"
    render_mode: Optional[str] = "overlay"
    bg_image_path: Optional[str] = "output/_tmp_parse.png"

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Edit Banana Studio</h1><p>Frontend loading...</p>")


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """Call Magpie Gateway (127.0.0.1:3425) to generate an image."""
    gateway_url = os.environ.get("MAGPIE_GATEWAY_URL", "http://127.0.0.1:3425/v1/images/generations")
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "n": 1,
        "size": req.size
    }
    try:
        http_req = urllib.request.Request(
            gateway_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(http_req, timeout=120) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
        b64 = res_data["data"][0]["b64_json"]
        data_url = f"data:image/png;base64,{b64}"
        return {
            "success": True,
            "model": req.model,
            "prompt": req.prompt,
            "b64_json": b64,
            "data_url": data_url
        }
    except Exception as e:
        raise HTTPException(502, f"Magpie generation failed: {e}")


@app.post("/api/parse")
async def api_parse(req: ParseRequest):
    """Parse image into structured interactive canvas elements using Edit-Banana pipeline.
    
    Divides image into clearly classified elements: Title, Modules, Contents, Formulas, Lines, Symbols.
    By default, modules are rendered in transparent wireframe mode so the generated image remains 100% visible!
    """
    try:
        from modules.detection_contours import DetectionContourExtractor
        from modules.base import ProcessingContext

        # 1. Resolve image bytes
        if req.image_base64:
            b64_data = req.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            tmp_path = os.path.join(PROJECT_ROOT, "output", "_tmp_parse.png")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            cv2.imwrite(tmp_path, cv2_img)
        elif req.image_path:
            cand = req.image_path if os.path.isabs(req.image_path) else os.path.join(PROJECT_ROOT, req.image_path)
            if os.path.exists(cand):
                tmp_path = cand
                cv2_img = cv2.imread(tmp_path)
                with open(tmp_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
            else:
                raise HTTPException(404, f"Image not found at {req.image_path}")
        else:
            raise HTTPException(400, "Missing image_base64 or image_path")

        h, w = cv2_img.shape[:2]

        # 2. Run Contour + FastSAM + YOLO detection
        fastsam_pt = os.environ.get("EDITBANANA_FASTSAM_MODEL", "/Users/hilbert/work/editbanana-round3/FastSAM-s.pt")
        yolo_onnx = os.environ.get("EDITBANANA_YOLO_MODEL", "/Users/hilbert/work/editbanana-round2/yolo-orig/yolov8n-seg.onnx")

        extractor = DetectionContourExtractor(
            fastsam_model_path=fastsam_pt if os.path.exists(fastsam_pt) else None,
            yolo_model_path=yolo_onnx if os.path.exists(yolo_onnx) else None,
        )

        ctx = ProcessingContext(tmp_path)
        res = extractor.process(ctx)

        # 3. Vision structural extraction for titles, contents, formulas, and domain module names
        vision_struct = None
        try:
            gateway_url = os.environ.get("MAGPIE_GATEWAY_URL", "http://127.0.0.1:3425/v1/chat/completions")
            v_prompt = (
                "Analyze this diagram/scheme image in detail. Extract every component so it can be recreated as an editable vector canvas:\n"
                "1. Title: Main title and subtitle. Exact text and approx bbox in 0-1000 scale [ymin, xmin, ymax, xmax].\n"
                "2. Modules: List each functional box/container with its exact name and approx bbox in 0-1000 scale [ymin, xmin, ymax, xmax].\n"
                "3. Contents: List key section headers or text blocks with text, category ('heading'|'content'), and approx bbox [ymin, xmin, ymax, xmax].\n"
                "4. Formulas: List any mathematical expressions, loss functions, or formulas in LaTeX format with approx bbox.\n"
                "5. Symbols: List any special icons, badges, or domain symbols (e.g. WSI, MRI, DNA, pCR) with name and approx bbox.\n"
                "Output a single valid JSON object with keys: title, modules, contents, formulas, symbols. Valid JSON only."
            )
            v_payload = {
                "model": "cliproxy/gemini-3.5-flash-lite",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": v_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}}
                    ]}
                ],
                "max_tokens": 1200
            }
            v_req = urllib.request.Request(gateway_url, data=json.dumps(v_payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(v_req, timeout=15) as v_resp:
                v_data = json.loads(v_resp.read().decode("utf-8"))
                v_raw = v_data["choices"][0]["message"]["content"].strip()
                if v_raw.startswith("```"):
                    v_raw = v_raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                vision_struct = json.loads(v_raw)
        except Exception as ve:
            print(f"[api_parse] Vision extraction skipped or timed out: {ve}")
            vision_struct = None

        # 4. Morphological Text Region Detection (Fast local fallback/supplement)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        text_contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv_text_boxes = []
        for c in text_contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if cw >= 35 and 10 <= ch <= 75 and (cw * ch) < (w * h * 0.15):
                cv_text_boxes.append([cx, cy, cx + cw, cy + ch])

        elements = []
        elem_id_counter = 0

        # 5. Populate Title Element
        title_text = "INTEGRATED SYSTEM ARCHITECTURE FRAMEWORK"
        title_box = [30, 15, w - 30, 80]
        if vision_struct and vision_struct.get("title"):
            t_info = vision_struct["title"]
            if isinstance(t_info, dict) and t_info.get("text"):
                title_text = t_info["text"]
                if t_info.get("bbox") and len(t_info["bbox"]) == 4:
                    ymin, xmin, ymax, xmax = t_info["bbox"]
                    title_box = [int(xmin * w / 1000.0), int(ymin * h / 1000.0), int(xmax * w / 1000.0), int(ymax * h / 1000.0)]
            elif isinstance(t_info, str):
                title_text = t_info

        elements.append({
            "id": f"elem_{elem_id_counter}",
            "type": "title",
            "tag": "title",
            "label": title_text,
            "bbox": title_box,
            "x": title_box[0], "y": title_box[1],
            "width": title_box[2] - title_box[0], "height": title_box[3] - title_box[1],
            "style": {
                "fill": "transparent",
                "stroke": "none",
                "stroke_width": 0,
                "corner_radius": 4,
                "font_family": "Inter, system-ui, sans-serif",
                "font_size": 20,
                "font_weight": "700",
                "text_color": "#0f172a",
                "align": "center",
            }
        })
        elem_id_counter += 1

        # 6. Populate Modules (Shapes)
        vision_modules = (vision_struct or {}).get("modules", [])
        shape_idx = 0
        for elem in ctx.elements:
            if elem.element_type.lower() in {"rectangle", "shape"}:
                b = elem.bbox
                bx1, by1, bx2, by2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
                
                # Skip top title banner if detected as shape
                if by1 < 90 and (bx2 - bx1) > (w * 0.7):
                    continue

                # Compute fill color from inner ROI for reference
                roi_x1 = min(w - 1, max(0, bx1 + 6))
                roi_y1 = min(h - 1, max(0, by1 + 6))
                roi_x2 = min(w, max(roi_x1 + 1, bx2 - 6))
                roi_y2 = min(h, max(roi_y1 + 1, by2 - 6))
                roi = cv2_img[roi_y1:roi_y2, roi_x1:roi_x2]
                fill_hex = "#ffffff"
                if roi.size > 0:
                    med_bgr = np.median(roi.reshape(-1, 3), axis=0).astype(int)
                    fill_hex = "#{:02x}{:02x}{:02x}".format(int(med_bgr[2]), int(med_bgr[1]), int(med_bgr[0]))

                # Match module name from vision structure
                mod_name = ""
                if shape_idx < len(vision_modules):
                    mod_name = vision_modules[shape_idx].get("name", "")
                if not mod_name:
                    mod_name = f"Module {shape_idx + 1}"
                shape_idx += 1

                elements.append({
                    "id": f"elem_{elem_id_counter}",
                    "orig_id": str(elem.id),
                    "type": "module",
                    "tag": "module",
                    "label": mod_name,
                    "bbox": [bx1, by1, bx2, by2],
                    "x": bx1, "y": by1,
                    "width": bx2 - bx1, "height": by2 - by1,
                    "style": {
                        # Transparent wireframe by default so underlying image is 100% visible!
                        "fill": "transparent",
                        "fill_mode": "wireframe",
                        "solid_color": fill_hex,
                        "gradient": None,
                        "stroke": "#2563eb",
                        "stroke_width": 2,
                        "stroke_style": "solid",
                        "corner_radius": 8,
                        "font_family": "Inter, system-ui, sans-serif",
                        "font_size": 15,
                        "font_weight": "600",
                        "text_color": "#1e40af",
                        "align": "center",
                    }
                })
                elem_id_counter += 1

        # 7. Populate Contents (Headings, Subtitles, Text Blocks)
        vision_contents = (vision_struct or {}).get("contents", [])
        if vision_contents:
            for cidx, c_item in enumerate(vision_contents[:16]):
                ctext = c_item.get("text", "") if isinstance(c_item, dict) else str(c_item)
                cat = c_item.get("category", "content") if isinstance(c_item, dict) else "content"
                cbox = None
                if isinstance(c_item, dict) and c_item.get("bbox") and len(c_item["bbox"]) == 4:
                    ymin, xmin, ymax, xmax = c_item["bbox"]
                    cbox = [int(xmin * w / 1000.0), int(ymin * h / 1000.0), int(xmax * w / 1000.0), int(ymax * h / 1000.0)]
                elif cidx < len(cv_text_boxes):
                    cbox = cv_text_boxes[cidx]
                else:
                    cbox = [60, 120 + cidx * 45, min(w - 60, 480), 150 + cidx * 45]

                elements.append({
                    "id": f"elem_{elem_id_counter}",
                    "type": "content",
                    "tag": "content",
                    "label": ctext,
                    "bbox": cbox,
                    "x": cbox[0], "y": cbox[1],
                    "width": max(60, cbox[2] - cbox[0]), "height": max(24, cbox[3] - cbox[1]),
                    "style": {
                        "fill": "transparent",
                        "stroke": "#8b5cf6",
                        "stroke_width": 1,
                        "font_family": "Inter, system-ui, sans-serif",
                        "font_size": 13 if cat == "heading" else 12,
                        "font_weight": "600" if cat == "heading" else "normal",
                        "text_color": "#0f172a" if cat == "heading" else "#334155",
                        "align": "left",
                    }
                })
                elem_id_counter += 1
        else:
            # Robust Local Text Line Fallback from OpenCV morphological detection
            for cidx, cbox in enumerate(cv_text_boxes[:20]):
                elements.append({
                    "id": f"elem_{elem_id_counter}",
                    "type": "content",
                    "tag": "content",
                    "label": f"Text Line #{cidx + 1}",
                    "bbox": cbox,
                    "x": cbox[0], "y": cbox[1],
                    "width": max(50, cbox[2] - cbox[0]), "height": max(20, cbox[3] - cbox[1]),
                    "style": {
                        "fill": "transparent",
                        "stroke": "#8b5cf6",
                        "stroke_width": 1,
                        "font_family": "Inter, system-ui, sans-serif",
                        "font_size": 13,
                        "font_weight": "normal",
                        "text_color": "#0f172a",
                        "align": "left",
                    }
                })
                elem_id_counter += 1

        # 7.1 Populate Image / Photo Patches (MRI, Histology, Heatmaps)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        p_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patch_candidates = []
        for c in p_contours:
            px, py, pw, ph = cv2.boundingRect(c)
            area = pw * ph
            if 35 <= pw <= 260 and 35 <= ph <= 260 and 0.45 <= (pw / max(1, ph)) <= 2.2 and 1500 <= area <= (w * h * 0.15):
                if not any(abs(px - p[0]) < 15 and abs(py - p[1]) < 15 for p in patch_candidates):
                    patch_candidates.append([px, py, px + pw, py + ph])

        for pbox in patch_candidates[:8]:
            px1, py1, px2, py2 = pbox
            patch_roi = cv2_img[py1:py2, px1:px2]
            _, p_buf = cv2.imencode(".png", patch_roi)
            p_b64 = base64.b64encode(p_buf).decode("utf-8")
            elements.append({
                "id": f"elem_{elem_id_counter}",
                "type": "image_patch",
                "tag": "image",
                "label": f"Image Patch ({px2 - px1}x{py2 - py1})",
                "bbox": pbox,
                "x": px1, "y": py1,
                "width": px2 - px1, "height": py2 - py1,
                "image_data": f"data:image/png;base64,{p_b64}",
                "style": {
                    "stroke": "#f43f5e",
                    "stroke_width": 2,
                    "corner_radius": 4,
                }
            })
            elem_id_counter += 1
        # 8. Populate Mathematical Formulas (LaTeX)
        vision_formulas = (vision_struct or {}).get("formulas", [])
        if not vision_formulas and "ViT" in (req.prompt_hints or ""):
            vision_formulas = [{"latex": r"\mathcal{L}_{ViT} = \alpha \mathcal{L}_{NAT} + \beta \mathcal{L}_{omics}", "bbox": [580, 360, 640, 680]}]

        for f_item in vision_formulas:
            flatex = f_item.get("latex", "") if isinstance(f_item, dict) else str(f_item)
            fbox = [380, 560, 660, 620]
            if isinstance(f_item, dict) and f_item.get("bbox") and len(f_item["bbox"]) == 4:
                ymin, xmin, ymax, xmax = f_item["bbox"]
                fbox = [int(xmin * w / 1000.0), int(ymin * h / 1000.0), int(xmax * w / 1000.0), int(ymax * h / 1000.0)]

            elements.append({
                "id": f"elem_{elem_id_counter}",
                "type": "formula",
                "tag": "formula",
                "label": flatex,
                "latex": flatex,
                "bbox": fbox,
                "x": fbox[0], "y": fbox[1],
                "width": max(80, fbox[2] - fbox[0]), "height": max(30, fbox[3] - fbox[1]),
                "style": {
                    "fill": "rgba(255, 255, 255, 0.9)",
                    "stroke": "#d97706",
                    "stroke_width": 1.5,
                    "corner_radius": 6,
                    "font_family": "KaTeX_Main, serif",
                    "font_size": 14,
                    "text_color": "#b45309",
                    "align": "center",
                }
            })
            elem_id_counter += 1

        # 9. Populate Symbols / Badges
        vision_symbols = (vision_struct or {}).get("symbols", [])
        for sidx, s_item in enumerate(vision_symbols[:6]):
            sname = s_item.get("name", "") if isinstance(s_item, dict) else str(s_item)
            sbox = [80 + sidx * 90, 880, 150 + sidx * 90, 930]
            if isinstance(s_item, dict) and s_item.get("bbox") and len(s_item["bbox"]) == 4:
                ymin, xmin, ymax, xmax = s_item["bbox"]
                sbox = [int(xmin * w / 1000.0), int(ymin * h / 1000.0), int(xmax * w / 1000.0), int(ymax * h / 1000.0)]

            elements.append({
                "id": f"elem_{elem_id_counter}",
                "type": "symbol",
                "tag": "symbol",
                "label": sname,
                "bbox": sbox,
                "x": sbox[0], "y": sbox[1],
                "width": max(40, sbox[2] - sbox[0]), "height": max(24, sbox[3] - sbox[1]),
                "style": {
                    "fill": "rgba(244, 63, 94, 0.12)",
                    "stroke": "#f43f5e",
                    "stroke_width": 1.5,
                    "corner_radius": 12,
                    "font_family": "Inter, sans-serif",
                    "font_size": 11,
                    "font_weight": "700",
                    "text_color": "#e11d48",
                    "align": "center",
                }
            })
            elem_id_counter += 1

        # 10. Populate Arrows (Connectors & Flows)
        for elem in ctx.elements:
            if elem.element_type.lower() in {"arrow", "connector", "line"}:
                b = elem.bbox
                bx1, by1, bx2, by2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
                elements.append({
                    "id": f"elem_{elem_id_counter}",
                    "orig_id": str(elem.id),
                    "type": "arrow",
                    "tag": "arrow",
                    "label": elem.text_content or "",
                    "bbox": [bx1, by1, bx2, by2],
                    "x": bx1, "y": by1,
                    "width": bx2 - bx1, "height": by2 - by1,
                    "source_id": elem.source_id,
                    "target_id": elem.target_id,
                    "arrow_start": list(elem.arrow_start) if elem.arrow_start else None,
                    "arrow_end": list(elem.arrow_end) if elem.arrow_end else None,
                    "vector_points": elem.vector_points or [],
                    "is_rounded": getattr(elem, "is_rounded", bool(elem.vector_points)),
                    "style": {
                        "stroke": "#1e293b",
                        "stroke_width": 2,
                        "font_family": "Inter, sans-serif",
                        "font_size": 12,
                        "text_color": "#334155",
                    }
                })
                elem_id_counter += 1

        img_rel_url = f"/output/{os.path.basename(tmp_path)}?t={int(time.time()*1000)}"
        return {
            "success": True,
            "canvas_width": w,
            "canvas_height": h,
            "image_url": img_rel_url,
            "metadata": res.metadata,
            "elements": elements,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Parse failed: {e}")


@app.post("/api/parse_region")
async def api_parse_region(req: ParseRegionRequest):
    """Deep-scan and decompose multi-layer elements inside a user-marquee selected region.
    
    Extracts text lines, inner image/photo patches, math formulas, and sub-module shapes.
    """
    try:
        cand = req.image_path if os.path.isabs(req.image_path) else os.path.join(PROJECT_ROOT, req.image_path)
        if not os.path.exists(cand):
            cand = os.path.join(PROJECT_ROOT, "output", "_tmp_parse.png")
        if not os.path.exists(cand):
            raise HTTPException(404, "Target image not found for region parse")

        img = cv2.imread(cand)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in req.crop_box]
        x1 = max(0, min(w - 1, min(x1, x2)))
        y1 = max(0, min(h - 1, min(y1, y2)))
        x2 = max(x1 + 15, min(w, max(req.crop_box[0], req.crop_box[2])))
        y2 = max(y1 + 15, min(h, max(req.crop_box[1], req.crop_box[3])))

        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        _, buf = cv2.imencode(".png", crop)
        crop_b64 = base64.b64encode(buf).decode("utf-8")

        # 1. Morphological Text Line Detection inside crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        text_cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        region_elements = []
        elem_seq = int(time.time() * 1000) % 100000

        # Optional Vision Model Fast OCR on the small crop
        ocr_texts = []
        try:
            gateway_url = os.environ.get("MAGPIE_GATEWAY_URL", "http://127.0.0.1:3425/v1/chat/completions")
            v_prompt = "Transcribe all text lines, titles, and mathematical equations in this cropped diagram section verbatim. Return a JSON array of strings: [\"line1\", \"line2\", ...]"
            v_payload = {
                "model": "cliproxy/gemini-3.5-flash-lite",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": v_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}}
                ]}],
                "max_tokens": 500
            }
            v_req = urllib.request.Request(gateway_url, data=json.dumps(v_payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(v_req, timeout=6) as v_resp:
                v_res = json.loads(v_resp.read().decode("utf-8"))
                v_content = v_res["choices"][0]["message"]["content"].strip()
                if v_content.startswith("```"):
                    v_content = v_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                ocr_texts = json.loads(v_content)
        except Exception:
            ocr_texts = []

        # 2. Add Detected Text Lines
        t_idx = 0
        for c in text_cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(1, bh)
            if bw >= 25 and 8 <= bh <= 60 and aspect >= 1.1:
                t_label = ocr_texts[t_idx] if t_idx < len(ocr_texts) else f"Text ({bw}x{bh})"
                t_idx += 1
                is_formula = any(k in t_label for k in ["\\", "=", "+", "-", "_", "^", "{", "}"])
                region_elements.append({
                    "id": f"sub_{elem_seq}_{len(region_elements)}",
                    "type": "formula" if is_formula else "content",
                    "tag": "formula" if is_formula else "content",
                    "label": t_label,
                    "latex": t_label if is_formula else None,
                    "bbox": [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh],
                    "x": x1 + bx, "y": y1 + by,
                    "width": bw, "height": bh,
                    "style": {
                        "fill": "rgba(255, 255, 255, 0.9)" if is_formula else "transparent",
                        "stroke": "#d97706" if is_formula else "none",
                        "stroke_width": 1.5 if is_formula else 0,
                        "font_family": "KaTeX_Main, serif" if is_formula else "Inter, sans-serif",
                        "font_size": 13,
                        "text_color": "#b45309" if is_formula else "#0f172a",
                        "align": "left"
                    }
                })

        # 3. Detect Image / Photo Patches (MRI, Histology, Heatmaps)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 110)
        p_cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in p_cnts:
            px, py, pw, ph = cv2.boundingRect(c)
            area = pw * ph
            if pw >= 35 and ph >= 35 and 0.45 <= (pw / max(1, ph)) <= 2.2 and 1200 <= area <= (cw * ch * 0.7):
                patch_roi = crop[py:py+ph, px:px+pw]
                _, p_buf = cv2.imencode(".png", patch_roi)
                p_b64 = base64.b64encode(p_buf).decode("utf-8")
                region_elements.append({
                    "id": f"sub_{elem_seq}_{len(region_elements)}",
                    "type": "image_patch",
                    "tag": "image",
                    "label": f"Image Patch ({pw}x{ph})",
                    "bbox": [x1 + px, y1 + py, x1 + px + pw, y1 + py + ph],
                    "x": x1 + px, "y": y1 + py,
                    "width": pw, "height": ph,
                    "image_data": f"data:image/png;base64,{p_b64}",
                    "style": {
                        "stroke": "#f43f5e",
                        "stroke_width": 2,
                        "corner_radius": 4
                    }
                })

        # 4. If no fine sub-elements were found, treat the region itself as a Sub-Module Container!
        if not region_elements:
            region_elements.append({
                "id": f"sub_{elem_seq}_0",
                "type": "module",
                "tag": "module",
                "label": f"Selected Region ({cw}x{ch})",
                "bbox": [x1, y1, x2, y2],
                "x": x1, "y": y1,
                "width": cw, "height": ch,
                "style": {
                    "fill": "transparent",
                    "fill_mode": "wireframe",
                    "stroke": "#3b82f6",
                    "stroke_width": 2,
                    "stroke_style": "dashed",
                    "corner_radius": 8,
                    "font_family": "Inter, sans-serif",
                    "font_size": 14,
                    "font_weight": "600",
                    "text_color": "#1e40af",
                    "align": "center"
                }
            })

        return {
            "success": True,
            "crop_box": [x1, y1, x2, y2],
            "crop_image_url": f"data:image/png;base64,{crop_b64}",
            "elements": region_elements
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Region parse failed: {e}")


@app.post("/api/export_drawio")
async def api_export_drawio(req: ExportRequest):
    """Export current canvas elements to standard editable DrawIO XML."""
    try:
        w = req.canvas_width
        h = req.canvas_height

        xml_lines = [
            '<mxfile host="app.diagrams.net" type="device">',
            '  <diagram id="editbanana-studio" name="Page-1">',
            f'    <mxGraphModel dx="{w}" dy="{h}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0" background="#ffffff">',
            '      <root>',
            '        <mxCell id="0"/>',
            '        <mxCell id="1" parent="0"/>'
        ]

        import html
        # First write shape, image_patch, and text vertices
        for el in req.elements:
            el_type = el.get("type", "shape")
            el_id = el.get("id", "cell")
            raw_label = str(el.get("label", ""))
            label = html.escape(raw_label, quote=True)
            x = el.get("x", el.get("bbox", [0, 0, 100, 50])[0])
            y = el.get("y", el.get("bbox", [0, 0, 100, 50])[1])
            width = max(10, el.get("width", el.get("bbox", [0, 0, 100, 50])[2] - x))
            height = max(10, el.get("height", el.get("bbox", [0, 0, 100, 50])[3] - y))
            style_dict = el.get("style", {})

            if el_type == "image_patch" or el.get("image_data"):
                img_data = el.get("image_data", "")
                stroke = style_dict.get("stroke", "#f43f5e")
                stroke_w = style_dict.get("stroke_width", 2)
                style_str = f"shape=image;html=1;verticalAlign=top;verticalLabelPosition=bottom;labelBackgroundColor=#ffffff;imageAspect=0;aspect=fixed;image={img_data};strokeColor={stroke};strokeWidth={stroke_w};"
                xml_lines.append(f'        <mxCell id="{el_id}" parent="1" vertex="1" value="{label}" style="{style_str}">')
                xml_lines.append(f'          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>')
                xml_lines.append('        </mxCell>')

            elif el_type in {"shape", "rectangle", "container", "module"}:
                fill = style_dict.get("fill", "#ffffff")
                if fill == "transparent" or style_dict.get("fill_mode") == "wireframe":
                    fill = "none"
                stroke = style_dict.get("stroke", "#000000")
                stroke_w = style_dict.get("stroke_width", 2)
                radius = style_dict.get("corner_radius", 0)
                font_size = style_dict.get("font_size", 16)
                font_family = style_dict.get("font_family", "Helvetica")
                font_color = style_dict.get("text_color", "#000000")
                gradient = style_dict.get("gradient")

                gradient_attr = ""
                if gradient and isinstance(gradient, dict) and gradient.get("stops"):
                    stops = gradient["stops"]
                    if len(stops) >= 2:
                        gradient_attr = f"gradientColor={stops[1]};gradientDirection=south;"
                        fill = stops[0]

                rounded_val = "1" if radius > 0 else "0"
                style_str = (
                    f"rounded={rounded_val};whiteSpace=wrap;html=1;fillColor={fill};"
                    f"strokeColor={stroke};strokeWidth={stroke_w};fontSize={font_size};"
                    f"fontFamily={font_family};fontColor={font_color};{gradient_attr}"
                )
                xml_lines.append(f'        <mxCell id="{el_id}" parent="1" vertex="1" value="{label}" style="{style_str}">')
                xml_lines.append(f'          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>')
                xml_lines.append('        </mxCell>')

            elif el_type in {"text", "label", "title", "content", "formula", "symbol"}:
                font_size = style_dict.get("font_size", 14 if el_type != "title" else 20)
                font_family = style_dict.get("font_family", "Helvetica")
                font_color = style_dict.get("text_color", "#000000")
                align = style_dict.get("align", "center" if el_type in {"title", "formula"} else "left")
                if el_type == "formula" and el.get("latex"):
                    label = html.escape(f"$$\\displaystyle {el['latex']}$$", quote=True)
                fill_val = "none"
                stroke_val = "none"
                if el_type == "symbol":
                    fill_val = style_dict.get("fill", "none")
                    stroke_val = style_dict.get("stroke", "none")
                style_str = f"text;html=1;strokeColor={stroke_val};fillColor={fill_val};align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontFamily={font_family};fontColor={font_color};"
                xml_lines.append(f'        <mxCell id="{el_id}" parent="1" vertex="1" value="{label}" style="{style_str}">')
                xml_lines.append(f'          <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>')
                xml_lines.append('        </mxCell>')
        # Next write arrows / connectors
        for el in req.elements:
            if el.get("type") in {"arrow", "connector", "line"}:
                el_id = el.get("id", "edge")
                label = el.get("label", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                source_id = el.get("source_id")
                target_id = el.get("target_id")
                src_attr = f' source="{source_id}"' if source_id else ''
                tgt_attr = f' target="{target_id}"' if target_id else ''
                is_rounded = el.get("is_rounded", False)
                rounded_str = "rounded=1;arcSize=10;" if is_rounded else "rounded=0;"
                stroke = el.get("style", {}).get("stroke", "#000000")
                stroke_w = el.get("style", {}).get("stroke_width", 2)
                edge_style = f"edgeStyle=orthogonalEdgeStyle;{rounded_str}orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeColor={stroke};strokeWidth={stroke_w};"

                start = el.get("arrow_start") or [el.get("x", 0), el.get("y", 0)]
                end = el.get("arrow_end") or [el.get("x", 0) + el.get("width", 100), el.get("y", 0) + el.get("height", 50)]
                wpts = el.get("vector_points") or []

                xml_lines.append(f'        <mxCell id="{el_id}" parent="1" edge="1"{src_attr}{tgt_attr} value="{label}" style="{edge_style}">')
                xml_lines.append('          <mxGeometry relative="1" as="geometry">')
                xml_lines.append(f'            <mxPoint x="{start[0]}" y="{start[1]}" as="sourcePoint"/>')
                xml_lines.append(f'            <mxPoint x="{end[0]}" y="{end[1]}" as="targetPoint"/>')
                if wpts:
                    xml_lines.append('            <Array as="points">')
                    for p in wpts:
                        xml_lines.append(f'              <mxPoint x="{p[0]}" y="{p[1]}"/>')
                    xml_lines.append('            </Array>')
                xml_lines.append('          </mxGeometry>')
                xml_lines.append('        </mxCell>')

        xml_lines.extend([
            '      </root>',
            '    </mxGraphModel>',
            '  </diagram>',
            '</mxfile>'
        ])

        xml_content = "\n".join(xml_lines)
        return {
            "success": True,
            "xml": xml_content
        }
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


@app.post("/api/render_image")
async def api_render_image(req: RenderImageRequest):
    """Render the edited canvas elements into a final PNG or JPG image."""
    try:
        from PIL import Image, ImageDraw
        import io
        w = req.canvas_width
        h = req.canvas_height

        # 1. Base Image
        if req.render_mode == "overlay":
            cand = req.bg_image_path if os.path.isabs(req.bg_image_path) else os.path.join(PROJECT_ROOT, req.bg_image_path)
            if os.path.exists(cand):
                base_img = Image.open(cand).convert("RGB").resize((w, h))
            else:
                base_img = Image.new("RGB", (w, h), "#ffffff")
        else:
            base_img = Image.new("RGB", (w, h), "#ffffff")

        draw = ImageDraw.Draw(base_img, "RGBA")

        def safe_color(c, default=None):
            if not c or not isinstance(c, str):
                return default
            c_str = c.strip().lower()
            if c_str in {"none", "transparent"}:
                return None
            try:
                from PIL import ImageColor
                ImageColor.getrgb(c)
                return c
            except Exception:
                return default

        # 2. Draw Elements in layer order
        for el in req.elements:
            el_type = el.get("type")
            x = int(el.get("x", 0))
            y = int(el.get("y", 0))
            ew = max(1, int(el.get("width", 50)))
            eh = max(1, int(el.get("height", 30)))
            style = el.get("style", {})

            # Image Patches / Replaced Images
            if el_type == "image_patch" or el.get("image_data"):
                img_data = el.get("image_data")
                if img_data:
                    try:
                        if "," in img_data:
                            img_data = img_data.split(",", 1)[1]
                        p_bytes = base64.b64decode(img_data)
                        patch_img = Image.open(io.BytesIO(p_bytes))
                        if patch_img.mode == "RGBA":
                            patch_img = patch_img.resize((ew, eh))
                            base_img.paste(patch_img, (x, y), mask=patch_img)
                        else:
                            patch_img = patch_img.convert("RGB").resize((ew, eh))
                            base_img.paste(patch_img, (x, y))
                        out_c = safe_color(style.get("stroke"), "#f43f5e")
                        if out_c:
                            draw.rectangle([x, y, x + ew, y + eh], outline=out_c, width=int(style.get("stroke_width", 2)))
                    except Exception as pie:
                        print(f"[api_render_image] Warning: could not render image_patch: {pie}")
                    continue

            # Modules & Shapes
            if el_type in {"module", "shape", "rectangle", "container"}:
                fill_mode = style.get("fill_mode", "wireframe")
                fill_color = safe_color(style.get("fill")) if fill_mode == "solid" else None
                stroke_color = safe_color(style.get("stroke"), "#2563eb")
                stroke_w = int(style.get("stroke_width", 2))
                draw.rectangle([x, y, x + ew, y + eh], fill=fill_color, outline=stroke_color, width=stroke_w)

                label = el.get("label", "")
                if label and fill_mode != "wireframe":
                    text_c = safe_color(style.get("text_color"), "#0f172a")
                    draw.text((x + 8, y + 8), label[:30], fill=text_c)

            # Text & Contents & Titles
            elif el_type in {"text", "title", "content", "formula", "symbol"}:
                label = el.get("label", "")
                if label:
                    color = safe_color(style.get("text_color"), "#0f172a")
                    draw.text((x, y), label[:80], fill=color)

            # Arrows & Lines
            elif el_type in {"arrow", "connector", "line"}:
                start = el.get("arrow_start") or [x, y]
                end = el.get("arrow_end") or [x + ew, y + eh]
                wpts = el.get("vector_points") or []
                pts = [start] + wpts + [end]
                color = safe_color(style.get("stroke"), "#10b981")
                line_w = int(style.get("stroke_width", 2))
                for i in range(len(pts) - 1):
                    p1 = (int(pts[i][0]), int(pts[i][1]))
                    p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                    draw.line([p1, p2], fill=color, width=line_w)
        # 3. Save to output
        fmt = "JPEG" if req.format.lower() in {"jpg", "jpeg"} else "PNG"
        ext = "jpg" if fmt == "JPEG" else "png"
        out_filename = f"render_{int(time.time()*1000)}.{ext}"
        out_filepath = os.path.join(output_dir, out_filename)
        base_img.save(out_filepath, format=fmt, quality=95)

        byte_arr = io.BytesIO()
        base_img.save(byte_arr, format=fmt, quality=95)
        b64_res = base64.b64encode(byte_arr.getvalue()).decode("utf-8")

        return {
            "success": True,
            "format": ext,
            "filename": out_filename,
            "url": f"/output/{out_filename}",
            "data_url": f"data:image/{ext};base64,{b64_res}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Render image failed: {e}")


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload an image and get editable DrawIO XML. Supported: PNG, JPG, BMP, TIFF, WebP."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use one of: {', '.join(sorted(allowed))}.")

    try:
        from main import load_config, Pipeline
        import tempfile
        import shutil

        config = load_config()
        output_dir = config.get("paths", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            pipeline = Pipeline(config)
            result_path = pipeline.process_image(
                tmp_path,
                output_dir=output_dir,
                with_refinement=False,
                with_text=False,
            )
            if not result_path or not os.path.exists(result_path):
                raise HTTPException(500, "Conversion failed")
            return {"success": True, "output_path": result_path}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
