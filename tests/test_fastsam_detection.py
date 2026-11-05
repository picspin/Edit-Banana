"""Tests for modules/fastsam_detection.py + extractor integration.

No torch import at test-module top level: FastSAM tests are
skipped when torch/ultralytics or the weights file are absent,
so the contour/YOLO path stays testable without torch. The
merge/dedup tests use synthetic mask dicts and need no model.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.fastsam_detection import (
    classify_arrow_instances,
    classify_shape_candidates,
    detect_fastsam_masks,
    iou_xyxy,
    merge_fastsam,
)
from modules.detection_contours import DetectionContourExtractor

SYNTH = "/Users/hilbert/work/editbanana-real-test/images/synth_flow.png"
SINGLE = "/Users/hilbert/work/editbanana-round2/generated/single-chain.png"
WEIGHTS = os.environ.get(
    "EDITBANANA_FASTSAM_MODEL",
    "/Users/hilbert/work/editbanana-round3/FastSAM-s.pt")
# Inner contour shape boxes from round-2 detection (synth, 4 boxes).
SYNTH_BOXES = [
    {"bbox": [114, 114, 427, 237]},
    {"bbox": [590, 114, 910, 237]},
    {"bbox": [114, 594, 430, 720]},
    {"bbox": [590, 594, 910, 720]},
]


def fastsam_available():
    if not os.path.exists(WEIGHTS):
        return False
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


class TestMergeLogicNoModel(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(
            iou_xyxy([0, 0, 100, 100], [0, 0, 100, 100]), 1.0)
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_gap_arrow_kept_and_head_separated_when_head_present(self):
        masks = [
            {"bbox": [432, 167, 590, 185], "area_work": 1414,
             "area": 1414, "approx_vertices": 4, "method": "fastsam-s"},
            {"bbox": [569, 167, 590, 186], "area_work": 266,
             "area": 266, "approx_vertices": 6, "method": "fastsam-s"},
        ]
        out = classify_arrow_instances(masks, SYNTH_BOXES)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["arrow_instance"])
        self.assertTrue(out[0]["head_separated"])
        self.assertEqual(out[0]["head_bbox"], [569, 167, 590, 186])

    def test_gap_arrow_without_head_mask_has_head_separated_false(self):
        masks = [{"bbox": [432, 167, 590, 185], "area_work": 1414,
                  "area": 1414, "approx_vertices": 4,
                  "method": "fastsam-s"}]
        out = classify_arrow_instances(masks, SYNTH_BOXES)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["arrow_instance"])
        self.assertFalse(out[0]["head_separated"])

    def test_dense_bar_without_gap_not_an_arrow(self):
        # Long bar on a diagram with no shape-box gap: not reported.
        masks = [{"bbox": [100, 100, 300, 140], "area_work": 5000,
                  "area": 5000, "approx_vertices": 4,
                  "method": "fastsam-s"}]
        out = classify_arrow_instances(masks, SYNTH_BOXES)
        self.assertEqual(len(out), 0)

    def test_shape_candidate_kept_when_matches_shape(self):
        masks = [{"bbox": [114, 114, 427, 237], "area_work": 100000,
                  "area": 100000, "approx_vertices": 4,
                  "method": "fastsam-s"}]
        kept, dropped = classify_shape_candidates(masks, SYNTH_BOXES)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 0)

    def test_unmatched_shape_candidate_dropped(self):
        masks = [{"bbox": [1, 500, 100, 600], "area_work": 10000,
                  "area": 10000, "approx_vertices": 4,
                  "method": "fastsam-s"}]
        kept, dropped = classify_shape_candidates(masks, SYNTH_BOXES)
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(dropped), 1)
    def test_merge_returns_audit_fields(self):
        m = merge_fastsam([], SYNTH_BOXES)
        self.assertEqual(m["arrow_instances"], [])
        self.assertEqual(m["shape_candidates_kept"], [])


class TestExtractorDegradeNoModel(unittest.TestCase):
    def test_missing_model_raises_in_detect(self):
        import cv2
        img = cv2.imread(SYNTH)
        self.assertIsNotNone(img)
        with self.assertRaises(FileNotFoundError):
            detect_fastsam_masks(img, model_path="/nonexistent/x.pt")

    def test_disabled_fastsam_still_returns_shapes(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        r = ex.extract(SYNTH)
        self.assertEqual(len(r["shapes"]), 4)
        self.assertFalse(r["fastsam_available"])
        self.assertEqual(r["fastsam_arrow_instances"], [])

    def test_missing_model_degrades_not_crash(self):
        ex = DetectionContourExtractor(
            fastsam_model_path="/nonexistent/x.pt")
        r = ex.extract(SYNTH)
        self.assertEqual(len(r["shapes"]), 4)
        self.assertFalse(r["fastsam_available"])
        self.assertIn("FileNotFoundError", r["fastsam_error"])


@unittest.skipUnless(fastsam_available(),
                     "FastSAM torch/weights unavailable")
class TestFastsamLive(unittest.TestCase):
    def test_synth_boxes_iou_and_two_arrow_instances(self):
        import cv2
        img = cv2.imread(SYNTH)
        masks = detect_fastsam_masks(img, model_path=WEIGHTS)
        self.assertGreaterEqual(len(masks), 4)
        for b in SYNTH_BOXES:
            best = max(iou_xyxy(m["bbox"], b["bbox"]) for m in masks)
            self.assertGreaterEqual(
                best, 0.90,
                f"synth box {b['bbox']} best FastSAM IoU {best:.3f}")
        arrows = classify_arrow_instances(masks, SYNTH_BOXES)
        self.assertEqual(len(arrows), 2)
        got = sorted(tuple(a["bbox"]) for a in arrows)
        for got_box, want in zip(
                got, [(432, 167, 590, 185), (432, 647, 590, 665)]):
            for g, w in zip(got_box, want):
                self.assertLessEqual(abs(g - w), 10)
        self.assertTrue(all(a["head_separated"] for a in arrows))

    def test_single_chain_arrow_instances_detected(self):
        import cv2
        img = cv2.imread(SINGLE)
        masks = detect_fastsam_masks(img, model_path=WEIGHTS)
        from modules.detection_contours import detect_shape_boxes, detect_arrow_shafts
        shapes = detect_shape_boxes(img)
        shafts = detect_arrow_shafts(img, shapes)
        self.assertEqual(len(shafts), 3)
        arrows = classify_arrow_instances(masks, shapes, shafts=shafts)
        self.assertEqual(len(arrows), 3)
        self.assertTrue(all(a["head_separated"] for a in arrows))

class TestFastsamProcessDedup(unittest.TestCase):
    @unittest.skipUnless(fastsam_available(),
                         "FastSAM torch/weights unavailable")
    def test_synth_process_does_not_double_count_arrows(self):
        from modules.detection_contours import DetectionContourExtractor
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 0
                self.canvas_height = 0
                self.intermediate_results = {}
        ctx = Ctx()
        ex = DetectionContourExtractor(
            yolo_model_path="",
            fastsam_model_path=WEIGHTS)
        res = ex.process(ctx)
        # 4 shapes + 2 shaft arrows, FastSAM instances confirm
        # the same 2 arrows, they are NOT extra elements.
        self.assertEqual(len(ctx.elements), 6)
        self.assertEqual(
            res.metadata["fastsam_arrow_instances_matched_shaft"], 2)
        self.assertEqual(res.metadata["fastsam_arrow_instances_new"], 0)
        self.assertEqual(res.metadata["arrows_head_separated"], 2)

    @unittest.skipUnless(fastsam_available(),
                         "FastSAM torch/weights unavailable")
    def test_single_chain_process_all_arrows_detected(self):
        from modules.detection_contours import DetectionContourExtractor
        class Ctx:
            def __init__(self):
                self.image_path = SINGLE
                self.elements = []
                self.canvas_width = 0
                self.canvas_height = 0
                self.intermediate_results = {}
        ctx = Ctx()
        ex = DetectionContourExtractor(
            yolo_model_path="",
            fastsam_model_path=WEIGHTS)
        res = ex.process(ctx)
        # 4 shapes + 3 shaft arrows (all confirmed by FastSAM)
        self.assertEqual(len(ctx.elements), 7)
        self.assertEqual(
            res.metadata["fastsam_arrow_instances_matched_shaft"], 3)
        self.assertEqual(res.metadata["fastsam_arrow_instances_new"], 0)
        self.assertEqual(res.metadata["arrows_head_separated"], 3)

if __name__ == "__main__":
    unittest.main()
