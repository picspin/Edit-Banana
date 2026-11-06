# Detection-contour branch (round 2, local only)

Branch: `editbanana/detection-contours` (from main 35b003e). Local commit only.
No push, no merge, no PR, no GitHub description change.

## What changed
- New `modules/detection_contours.py` (no torch, no SAM import):
  shape boxes from OpenCV RETR_TREE hole contours, arrow shafts
  confirmed only by Hough lines between shape boxes, VLM OCR prompt
  verified in round 2 (see below).
- `main.py` step [2] now defaults to detection contours. SAM3 is
  opt-in only via `EDITBANANA_USE_SAM3=1`; group extraction still
  requires SAM3 and fails loudly otherwise instead of silently
  falling back.
- `modules/__init__.py` exports `DetectionContourExtractor`.

## Verified improvements (round 2, real numbers)
- OCR prompt: old prompt + max_tokens=1500 gave content=None on
  11/20 dense-diagram calls. New prompt (explicit "output ONLY the
  final OCR result, no reasoning") + max_tokens=4000 + temperature=0
  gave content on 10/10 rerun calls: demo_original_1/2/3 +
  demo_frontend + synth, both PaddleOCR-VL-1.5 and MinerU2.5-Pro,
  all finish=stop, all with boxes. MinerU native API was NOT used
  and NOT needed; no new account/key was created or requested.
  Caveat: VLM boxes remain estimates. On synth, PaddleOCR boxes
  are narrow vertical strips and Process/Check positions are
  unreliable; boxes must be matched to OpenCV boxes by centre
  point, not used as measured geometry.
- Shape boxes: hole contours give exactly 4 boxes on synth
  (inner bboxes, about 4 px inset). Top-level contours give 6
  shapes on the same image and are NOT used here.
- YOLO original export: yolov8n-seg.pt (6.7 MB) exported to ONNX
  (13.2 MB) with torch ultralytics in scratch (Python 3.13 venv;
  Python 3.15 venv failed on onnxruntime wheel availability).
  On 10 images v8n gave 9 detections vs yolo11s 18; v8n missed
  all 4 cow photos that yolo11s found, but was cleaner on cat
  (no tv false positive). Arrow detections: 0 for both models
  on all 10 images, as expected (COCO has no arrow class).

## Verified limitations (do NOT overclaim)
- Arrow heads: geometric triangle+line prototype gives synth
  0/2 separated heads (head merges into the row outline) and,
  after deduplication and square-head filtering, only
  demo_original_3 has 2 head-like triangles. Arrows in this
  branch are shaft-confirmed, NOT instance masks, and must not
  be described as SAM-equivalent arrow detection.
- Embedding full vocabulary (local 8082, /v1/embeddings):
  9/14 crops top-1 correct (label 2/4, title 2/3, photo 2/4,
  arrow 2/2, background 1/1). Margins are mostly <0.02;
  title/label/icon/photo are confusable with generic English
  phrases. Specific-text matching (e.g. Start vs Process) was
  4/4 in round 1 and remains the stronger use of embedding;
  generic class typing is not reliable enough for DrawIO typing
  on its own. No intended crops existed for description, formula
  or icon, so those classes have no accuracy number here.
- Generated single-chain image: BLOCKED. Codex image_gen hit
  a usage limit (retry at Oct 31 02:25 as reported by Codex);
  no single-chain image was generated in round 2. The pipeline
  was instead run on the earlier Codex test2 image (parallel
  pairs, not a single chain): OCR 4/4 text with new prompt,
  YOLO 0, triangles 0. Do not describe test2 as single-chain
  or as NanoBanana/gpt-image-2.5 output (backend model name was
  not exposed by the tool).
- Human-eye drag verification in diagrams.net was NOT done;
  DrawIO conclusions remain XML-structure conclusions.
