# Edit-Banana round 3 summary (local only, not pushed)
- FastSAM-s: ultralytics FastSAM, FastSAM-s.pt 22.7MB, scratch only.
  Synth: 54 masks, 4 box masks IoU 0.971-0.992, 2 arrow masks (shaft+head) IoU 0.889.
  Head-only triangular masks: 0 on all 6 images. Arrow-instance (shaft+head) synth: 2/2.
  Dense diagrams: no verified arrow instances (no ground truth; elongated-candidate classifier overfires on text/panel boxes, not reported as arrows).
- YOLO v8n: branch editbanana/detection-contours-v8n, modules/yolo_detection.py (ONNX, no torch at runtime; BGR->RGB fix reproduced round-2 numbers).
  Synth: shapes 4, arrows shaft 2, head 0, YOLO 0, elements 6. Cat: shapes 6 + YOLO cat/keyboard = 8 elements.
  Original_2: shapes 21 + arrows 1 + YOLO person 1 = 23 elements. Original_4: shapes 4 + YOLO motorcycle x2 = 6 elements.
- Embedding Chinese vocab: 11/14 vs English 9/14. label 3/4, title 2/3, arrow 2/2, background 1/1, photo 3/4.
  Chinese is better but margins are still tiny; do not use generic vocab alone for DrawIO typing.
- No Codex generation in this round (left to 02:25 scheduled task). No human-eye DrawIO validation in this round.
