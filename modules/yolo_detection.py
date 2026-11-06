"""YOLOv8n-seg box detection via ONNX Runtime (no torch at runtime).

Production default model is yolov8n-seg (13.2 MB ONNX, exported in
round 2 from yolov8n-seg.pt 6.7 MB with scratch torch ultralytics).
yolo11s-seg (39 MB) is NOT the default; it is only a comparison
reference in round-2 scratch data. COCO classes have no arrow class,
so YOLO boxes can never be arrow instances -- arrows still come
from detection_contours (shaft-confirmed, head_separated=False on
synth) or, in future, FastSAM masks. This module only adds photo /
object boxes (cat, cow, motorcycle, etc.) that contour detection
misses, and it never presents a YOLO box as an arrow.

Decoding mirrors round-2 yolo_compare.py exactly (letterbox 640,
conf 0.25, NMS IoU 0.45, boxes scaled back to original size), so
numbers are comparable with round-2 (v8n total 9 detections on the
10-image set, vs yolo11s 18).
"""
import os
from typing import Any, Dict, List

import cv2
import numpy as np

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]

# Round-2 exported model, read-only reference. Production deployment
# should bundle this file with the app instead of relying on a
# scratch path; the path is overridable via EDITBANANA_YOLO_MODEL.
DEFAULT_YOLO_MODEL_PATH = os.environ.get(
    "EDITBANANA_YOLO_MODEL",
    "/Users/hilbert/work/editbanana-round2/yolo-orig/yolov8n-seg.onnx")


def _letterbox(img, size=640):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    pad[(size - nh) // 2:(size - nh) // 2 + nh,
        (size - nw) // 2:(size - nw) // 2 + nw] = r
    return pad, s, (size - nw) // 2, (size - nh) // 2


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1.0, area_a + area_b - inter)


def detect_yolo_boxes(image_bgr, model_path=None, conf_thresh=0.25,
                       iou_thresh=0.45):
    """Return YOLO box detections in original image coordinates.

    Input is BGR (cv2.imread). The ONNX model expects RGB, as in
    round-2 yolo_compare.py (which fed RGB); feeding BGR directly
    changes confidences and adds false positives (e.g. laptop on
    cat.jpg), so convert BGR->RGB here. Geometry is unchanged.
    """
    import onnxruntime as ort
    path = model_path or DEFAULT_YOLO_MODEL_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"YOLO ONNX model not found: {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lb, s, px, py = _letterbox(image_rgb)
    inp = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    o0, _o1 = sess.run(None, {"images": inp})
    pred = o0[0].T
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:84]
    conf = cls_scores.max(axis=1)
    cls = cls_scores.argmax(axis=1)
    idx = np.where(conf > conf_thresh)[0]
    dets: List[Dict[str, Any]] = []
    if len(idx):
        b = boxes_xywh[idx]; c = conf[idx]; cl = cls[idx]
        x1 = b[:, 0] - b[:, 2] / 2; y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2; y2 = b[:, 1] + b[:, 3] / 2
        boxes_list = [[float(a), float(bb), float(cc - a), float(d - bb)]
                      for a, bb, cc, d in zip(x1, y1, x2, y2)]
        nms = cv2.dnn.NMSBoxes(boxes_list, c.tolist(), conf_thresh,
                               iou_thresh)
        sel = [int(i) for i in (nms.flatten() if len(nms) else [])]
        for j in sel:
            ox1 = (x1[j] - px) / s; oy1 = (y1[j] - py) / s
            ox2 = (x2[j] - px) / s; oy2 = (y2[j] - py) / s
            dets.append({
                "class": COCO[int(cl[j])],
                "conf": round(float(c[j]), 3),
                "bbox": [int(round(ox1)), int(round(oy1)),
                         int(round(ox2)), int(round(oy2))],
                "method": "yolov8n-seg-onnx",
                "model": os.path.basename(path),
            })
    return dets
