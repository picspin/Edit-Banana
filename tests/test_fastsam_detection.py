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
    estimate_arrowhead_orientation,
    iou_xyxy,
    merge_fastsam,
)
from modules.detection_contours import (
    DetectionContourExtractor,
    bind_edge_label_to_arrow,
    compute_feedback_loop_route,
    compute_obstacle_avoidance_route,
    detect_arrow_shafts,
    detect_elbow_connectors,
    detect_shape_boxes,
    hanan_grid_route,
    segment_box_collision,
)
from modules.xml_merger import XMLMerger, XMLFragment
from modules.data_types import LayerLevel, BoundingBox, ElementInfo

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
        # Direction certainty: synth arrows point horizontally to the right
        self.assertTrue(all(a.get("orientation_deg") is not None for a in arrows))
        self.assertTrue(all(abs(a["orientation_deg"]) < 15.0 for a in arrows))
        self.assertTrue(all(a["polarity_vec"][0] > 0.9 for a in arrows))
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
        # All 3 arrows have estimated orientation and apex tip
        self.assertTrue(all(a.get("orientation_deg") is not None for a in arrows))
        self.assertTrue(all(a.get("tip_pt") is not None for a in arrows))
        # Horizontal arrows (idx 0 and 2) point right (+x)
        self.assertGreater(arrows[0]["polarity_vec"][0], 0.9)
        self.assertGreater(arrows[2]["polarity_vec"][0], 0.9)
        # Diagonal arrow (idx 1) points down-left (negative x, positive y)
        self.assertLess(arrows[1]["polarity_vec"][0], -0.5)
        self.assertGreater(arrows[1]["polarity_vec"][1], 0.3)
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


class TestArrowheadOrientation(unittest.TestCase):
    def test_cardinal_directions(self):
        import numpy as np
        # Right (+x, 0 deg)
        c_right = np.array([[10, 10], [10, 30], [35, 20]], dtype=np.int32)
        r_right = estimate_arrowhead_orientation(c_right)
        self.assertAlmostEqual(r_right["orientation_deg"], 0.0, places=0)
        self.assertEqual(r_right["tip_pt"], [35, 20])
        self.assertGreater(r_right["polarity_vec"][0], 0.9)

        # Down (+y, 90 deg)
        c_down = np.array([[10, 10], [30, 10], [20, 35]], dtype=np.int32)
        r_down = estimate_arrowhead_orientation(c_down)
        self.assertAlmostEqual(r_down["orientation_deg"], 90.0, places=0)
        self.assertEqual(r_down["tip_pt"], [20, 35])
        self.assertGreater(r_down["polarity_vec"][1], 0.9)

        # Left (-x, 180 deg)
        c_left = np.array([[35, 10], [35, 30], [10, 20]], dtype=np.int32)
        r_left = estimate_arrowhead_orientation(c_left)
        self.assertAlmostEqual(abs(r_left["orientation_deg"]), 180.0, places=0)
        self.assertEqual(r_left["tip_pt"], [10, 20])
        self.assertLess(r_left["polarity_vec"][0], -0.9)

        # Up (-y, -90 deg)
        c_up = np.array([[10, 35], [30, 35], [20, 10]], dtype=np.int32)
        r_up = estimate_arrowhead_orientation(c_up)
        self.assertAlmostEqual(r_up["orientation_deg"], -90.0, places=0)
        self.assertEqual(r_up["tip_pt"], [20, 10])
        self.assertLess(r_up["polarity_vec"][1], -0.9)


class TestElbowConnectors(unittest.TestCase):
    def test_l_shaped_elbow_connector(self):
        lines = [(150, 100, 350, 100), (350, 100, 350, 300)]
        boxes = [{"bbox": [50, 50, 150, 150]}, {"bbox": [300, 300, 400, 400]}]
        elbows = detect_elbow_connectors(lines, boxes, slack=40)
        self.assertEqual(len(elbows), 1)
        el = elbows[0]
        self.assertEqual(el["connector_type"], "elbow")
        self.assertEqual(el["bends"], 1)
        self.assertEqual(len(el["waypoints"]), 3)
        self.assertEqual(el["from_box"], [50, 50, 150, 150])
        self.assertEqual(el["to_box"], [300, 300, 400, 400])

    def test_z_shaped_elbow_connector(self):
        lines = [(150, 100, 250, 100), (250, 100, 250, 300), (250, 300, 350, 300)]
        boxes = [{"bbox": [50, 50, 150, 150]}, {"bbox": [350, 250, 450, 350]}]
        elbows = detect_elbow_connectors(lines, boxes, slack=40)
        self.assertEqual(len(elbows), 1)
        el = elbows[0]
        self.assertEqual(el["connector_type"], "elbow")
        self.assertEqual(el["bends"], 2)
        self.assertEqual(len(el["waypoints"]), 4)
        self.assertEqual(el["from_box"], [50, 50, 150, 150])
        self.assertEqual(el["to_box"], [350, 250, 450, 350])

class TestPortAndEdgeLabel(unittest.TestCase):
    def test_drawio_port_source_target_binding(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 0
                self.canvas_height = 0
                self.intermediate_results = {}
        ctx = Ctx()
        ex.process(ctx)
        arrows = [e for e in ctx.elements if e.element_type == "arrow"]
        self.assertGreaterEqual(len(arrows), 2)
        for a in arrows:
            self.assertIsNotNone(a.source_id)
            self.assertIsNotNone(a.target_id)
            self.assertNotEqual(a.source_id, a.target_id)

    def test_bind_edge_label_to_arrow_straight_and_elbow(self):
        # Straight arrow: (100, 200) -> (500, 200), text at (300, 180)
        res_straight = bind_edge_label_to_arrow([280, 170, 320, 190], (100, 200), (500, 200))
        self.assertIsNotNone(res_straight)
        self.assertEqual(res_straight, (0, -20))

        # Elbow arrow: (100, 100) -> (300, 100) -> (300, 300), text at (315, 200)
        res_elbow = bind_edge_label_to_arrow([305, 190, 325, 210], (100, 100), (300, 300),
                                            vector_points=[[300, 100]])
        self.assertIsNotNone(res_elbow)
        self.assertEqual(res_elbow, (15, 100))

        # Far away text: not bound
        res_far = bind_edge_label_to_arrow([50, 50, 80, 70], (100, 200), (500, 200))
        self.assertIsNone(res_far)

    def test_xml_merger_bind_text_to_edges(self):
        merger = XMLMerger()
        edge_xml = '<mxCell id="4" parent="1" edge="1"><mxGeometry relative="1" as="geometry"><mxPoint x="100" y="200" as="sourcePoint"/><mxPoint x="500" y="200" as="targetPoint"/></mxGeometry></mxCell>'
        text_xml = '<mxCell id="text_1" value="Yes" parent="1" vertex="1"><mxGeometry x="280" y="180" width="40" height="20" as="geometry"/></mxCell>'
        frags = [
            XMLFragment(element_id="4", xml_content=edge_xml, layer_level=LayerLevel.ARROW.value, bbox=BoundingBox(100, 180, 500, 220), element_type="arrow"),
            XMLFragment(element_id="text_1", xml_content=text_xml, layer_level=LayerLevel.TEXT.value, bbox=BoundingBox(280, 180, 320, 200), element_type="text"),
        ]
        updated = merger.bind_text_to_edges(frags, max_dist=60.0)
        text_frag = [f for f in updated if f.element_id == "text_1"][0]
        self.assertEqual(text_frag.element_type, "edge_label")
        self.assertIn('parent="4"', text_frag.xml_content)
        self.assertIn('connectable="0"', text_frag.xml_content)
        self.assertIn('as="offset"', text_frag.xml_content)

    def test_xml_merger_remapping_source_target_parent(self):
        import xml.etree.ElementTree as ET
        merger = XMLMerger()
        shape0_xml = '<mxCell id="0" parent="1" vertex="1" style="rounded=0;"><mxGeometry x="50" y="50" width="100" height="80" as="geometry"/></mxCell>'
        shape1_xml = '<mxCell id="1" parent="1" vertex="1" style="rounded=0;"><mxGeometry x="300" y="50" width="100" height="80" as="geometry"/></mxCell>'
        edge_xml = '<mxCell id="4" parent="1" edge="1" source="0" target="1" style="edgeStyle=orthogonalEdgeStyle;"><mxGeometry relative="1" as="geometry"><mxPoint x="150" y="90" as="sourcePoint"/><mxPoint x="300" y="90" as="targetPoint"/></mxGeometry></mxCell>'
        label_xml = '<mxCell id="text_1" value="Next" parent="4" vertex="1" connectable="0"><mxGeometry relative="1" as="geometry"><mxPoint x="0" y="-15" as="offset"/></mxGeometry></mxCell>'

        frags = [
            XMLFragment(element_id="0", xml_content=shape0_xml, layer_level=LayerLevel.BASIC_SHAPE.value, bbox=BoundingBox(50, 50, 150, 130), element_type="rectangle"),
            XMLFragment(element_id="1", xml_content=shape1_xml, layer_level=LayerLevel.BASIC_SHAPE.value, bbox=BoundingBox(300, 50, 400, 130), element_type="rectangle"),
            XMLFragment(element_id="4", xml_content=edge_xml, layer_level=LayerLevel.ARROW.value, bbox=BoundingBox(150, 80, 300, 100), element_type="arrow"),
            XMLFragment(element_id="text_1", xml_content=label_xml, layer_level=LayerLevel.TEXT.value, bbox=BoundingBox(200, 70, 240, 90), element_type="edge_label"),
        ]

        sorted_f = merger._sort_fragments(frags)
        root = merger._build_xml_structure(500, 500, sorted_f)

        edge_cells = [c for c in root.iter("mxCell") if c.get("edge") == "1"]
        self.assertEqual(len(edge_cells), 1)
        edge = edge_cells[0]
        self.assertNotIn(edge.get("source"), ["0", "1", None])
        self.assertNotIn(edge.get("target"), ["0", "1", None])
        all_ids = {c.get("id") for c in root.iter("mxCell")}
        self.assertIn(edge.get("source"), all_ids)
        self.assertIn(edge.get("target"), all_ids)

        label_cells = [c for c in root.iter("mxCell") if c.get("connectable") == "0"]
        self.assertEqual(len(label_cells), 1)
        self.assertEqual(label_cells[0].get("parent"), edge.get("id"))
class TestW2DetectionGaps(unittest.TestCase):
    def test_w2_cases_no_reversed_and_z_detected(self):
        cases_dir = "/Users/hilbert/work/editbanana-fastsam-integration/cases"
        if not os.path.exists(os.path.join(cases_dir, "manifest.json")):
            self.skipTest("w2 cases not present")
        import cv2
        import json
        # 1. arrow_left: must be from right box to left box
        img_left = cv2.imread(os.path.join(cases_dir, "arrow_left.png"))
        boxes_left = detect_shape_boxes(img_left)
        arrows_left = detect_arrow_shafts(img_left, boxes_left)
        self.assertEqual(len(arrows_left), 1)
        self.assertLess(arrows_left[0]["to_box"][0], 200)
        self.assertGreater(arrows_left[0]["from_box"][0], 500)

        # 2. elbow_up: must be from bottom-left box to top-right box
        img_up = cv2.imread(os.path.join(cases_dir, "elbow_up.png"))
        boxes_up = detect_shape_boxes(img_up)
        arrows_up = detect_arrow_shafts(img_up, boxes_up)
        self.assertEqual(len(arrows_up), 1)
        self.assertEqual(arrows_up[0]["connector_type"], "elbow")
        self.assertLess(arrows_up[0]["to_box"][1], 200)
        self.assertGreater(arrows_up[0]["from_box"][1], 500)

        # 3. elbow_Z_right: must be detected with bends = 2
        img_z = cv2.imread(os.path.join(cases_dir, "elbow_Z_right.png"))
        boxes_z = detect_shape_boxes(img_z)
        arrows_z = detect_arrow_shafts(img_z, boxes_z)
        self.assertEqual(len(arrows_z), 1)
        self.assertEqual(arrows_z[0]["connector_type"], "elbow")
        self.assertEqual(arrows_z[0]["bends"], 2)


class TestObstacleAvoidance(unittest.TestCase):
    def test_segment_box_collision(self):
        box_o = [250, 150, 350, 250]
        self.assertTrue(segment_box_collision((100, 200), (500, 200), box_o))
        self.assertFalse(segment_box_collision((100, 100), (500, 100), box_o))
        self.assertFalse(segment_box_collision((100, 300), (500, 300), box_o))

    def test_compute_obstacle_avoidance_route_horizontal(self):
        box_a = [50, 200, 150, 300]
        box_b = [450, 200, 550, 300]
        box_o = [250, 180, 350, 320]
        route = compute_obstacle_avoidance_route(box_a, box_b, [box_o], clearance=25)
        self.assertIsNotNone(route)
        self.assertEqual(route["connector_type"], "obstacle_avoidance")
        self.assertEqual(route["bends"], 4)
        wpts = route["waypoints"]
        self.assertEqual(len(wpts), 6)
        for i in range(len(wpts) - 1):
            self.assertFalse(segment_box_collision(wpts[i], wpts[i + 1], box_o, margin=0))

    def test_compute_obstacle_avoidance_route_vertical(self):
        box_a = [200, 50, 300, 150]
        box_b = [200, 450, 300, 550]
        box_o = [180, 250, 320, 350]
        route = compute_obstacle_avoidance_route(box_a, box_b, [box_o], clearance=25)
        self.assertIsNotNone(route)
        self.assertEqual(route["connector_type"], "obstacle_avoidance")
        self.assertEqual(route["bends"], 4)
        wpts = route["waypoints"]
        for i in range(len(wpts) - 1):
            self.assertFalse(segment_box_collision(wpts[i], wpts[i + 1], box_o, margin=0))

    def test_process_applies_obstacle_avoidance_to_drawio_xml(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        box_a = [50, 200, 150, 300]
        box_o = [250, 180, 350, 320]
        box_b = [450, 200, 550, 300]
        shapes = [{"bbox": box_a}, {"bbox": box_o}, {"bbox": box_b}]
        arrows = [{
            "from_box": box_a, "to_box": box_b,
            "shaft_confirmed": True, "head_separated": False,
            "method": "hough_shaft_between_boxes"
        }]

        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 600
                self.canvas_height = 500
                self.intermediate_results = {}
        ctx = Ctx()
        original_extract = ex.extract
        ex.extract = lambda path, **kwargs: {
            "canvas_width": 600, "canvas_height": 500,
            "shapes": shapes, "arrows": arrows,
            "yolo_boxes": [], "yolo_kept": [],
            "yolo_available": False, "yolo_error": None, "yolo_model": None,
            "fastsam_masks": [], "fastsam_arrow_instances": [],
            "fastsam_shapes_kept": [], "fastsam_shapes_dropped": [],
            "fastsam_available": False, "fastsam_error": None, "fastsam_model": None,
        }
        try:
            res = ex.process(ctx)
            self.assertEqual(res.metadata["arrows_obstacle_avoided"], 1)
            arrow_elem = [e for e in ctx.elements if e.element_type == "arrow"][0]
            self.assertIsNotNone(arrow_elem.vector_points)
            self.assertGreaterEqual(len(arrow_elem.vector_points), 3)
            self.assertEqual(arrow_elem.source_id, "0")
            self.assertEqual(arrow_elem.target_id, "2")
        finally:
            ex.extract = original_extract

    def test_channel_separation_lane_offsets(self):
        box_a = [50, 200, 150, 300]
        box_b = [450, 200, 550, 300]
        box_o = [250, 180, 350, 320]
        r0 = compute_obstacle_avoidance_route(box_a, box_b, [box_o], clearance=25, lane_index=0)
        r1 = compute_obstacle_avoidance_route(box_a, box_b, [box_o], clearance=25, lane_index=1, lane_spacing=15)
        self.assertIsNotNone(r0)
        self.assertIsNotNone(r1)
        self.assertEqual(r0["lane_index"], 0)
        self.assertEqual(r1["lane_index"], 1)
        # r1's detour track is offset by 15 px further away from obstacle than r0
        self.assertEqual(abs(r0["waypoints"][2][1] - r1["waypoints"][2][1]), 15)
        for r in [r0, r1]:
            for i in range(len(r["waypoints"]) - 1):
                self.assertFalse(segment_box_collision(r["waypoints"][i], r["waypoints"][i+1], box_o, margin=0))

    def test_detour_selects_unblocked_candidate_side(self):
        box_a = [50, 200, 150, 300]
        box_b = [450, 200, 550, 300]
        box_o1 = [250, 180, 350, 320]
        box_o2 = [250, 80, 350, 175]  # blocks top candidate detour
        route = compute_obstacle_avoidance_route(box_a, box_b, [box_o1, box_o2], clearance=25)
        self.assertIsNotNone(route)
        self.assertEqual(route["routing_method"], "orthogonal_detour")
        # Must have chosen bottom detour
        self.assertGreaterEqual(route["waypoints"][2][1], 345)
        for i in range(len(route["waypoints"]) - 1):
            self.assertFalse(segment_box_collision(route["waypoints"][i], route["waypoints"][i+1], box_o1, margin=0))
            self.assertFalse(segment_box_collision(route["waypoints"][i], route["waypoints"][i+1], box_o2, margin=0))

    def test_hanan_grid_astar_maze_fallback(self):
        box_a = [50, 200, 150, 300]
        box_b = [450, 200, 550, 300]
        box_o1 = [250, 180, 350, 320]
        box_o2 = [250, 80, 350, 175]   # blocks top
        box_o3 = [250, 325, 350, 420]  # blocks bottom
        route = compute_obstacle_avoidance_route(box_a, box_b, [box_o1, box_o2, box_o3], clearance=25)
        self.assertIsNotNone(route)
        self.assertEqual(route["routing_method"], "hanan_grid_astar")
        for i in range(len(route["waypoints"]) - 1):
            p1 = route["waypoints"][i]
            p2 = route["waypoints"][i+1]
            for obs in [box_o1, box_o2, box_o3]:
                self.assertFalse(segment_box_collision(p1, p2, obs, margin=0))

    def test_diagonal_obstacle_avoidance_hanan_grid(self):
        start_pt = (150, 100)
        end_pt = (450, 500)
        box_o = [200, 200, 350, 350]
        # Direct diagonal segment collides
        self.assertTrue(segment_box_collision(start_pt, end_pt, box_o))
        route = hanan_grid_route(start_pt, end_pt, [box_o], clearance=25)
        self.assertIsNotNone(route)
        self.assertEqual(route[0], [150, 100])
        self.assertEqual(route[-1], [450, 500])
        for i in range(len(route) - 1):
            self.assertFalse(segment_box_collision(route[i], route[i+1], box_o, margin=0))

    def test_process_multiple_arrows_channel_separation(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        box_a = [50, 200, 150, 300]
        box_o = [250, 180, 350, 320]
        box_b = [450, 200, 550, 300]
        shapes = [{"bbox": box_a}, {"bbox": box_o}, {"bbox": box_b}]
        # Two arrows avoiding the same obstacle
        arrows = [
            {"from_box": box_a, "to_box": box_b, "shaft_confirmed": True, "head_separated": False, "method": "test"},
            {"from_box": box_a, "to_box": box_b, "shaft_confirmed": True, "head_separated": False, "method": "test"},
        ]
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 600
                self.canvas_height = 500
                self.intermediate_results = {}
        ctx = Ctx()
        original_extract = ex.extract
        ex.extract = lambda path, **kwargs: {
            "canvas_width": 600, "canvas_height": 500,
            "shapes": shapes, "arrows": arrows,
            "yolo_boxes": [], "yolo_kept": [],
            "yolo_available": False, "yolo_error": None, "yolo_model": None,
            "fastsam_masks": [], "fastsam_arrow_instances": [],
            "fastsam_shapes_kept": [], "fastsam_shapes_dropped": [],
            "fastsam_available": False, "fastsam_error": None, "fastsam_model": None,
        }
        try:
            res = ex.process(ctx)
            self.assertEqual(res.metadata["arrows_obstacle_avoided"], 2)
            arrow_elems = [e for e in ctx.elements if e.element_type == "arrow"]
            self.assertEqual(len(arrow_elems), 2)
            w0 = arrow_elems[0].vector_points
            w1 = arrow_elems[1].vector_points
            # Channel separation ensures waypoints are distinct and separated by 15 px
            self.assertNotEqual(w0, w1)
            self.assertEqual(abs(w0[1][1] - w1[1][1]), 15)
        finally:
            ex.extract = original_extract

    def test_canvas_bounds_top_overflow_selects_bottom(self):
        # Box O near top edge y=10, canvas_height=500, margin=10
        # Top detour (10 - 25 = -15 < margin) overflows, so bottom detour (150 + 25 = 175) is selected
        box_a = [50, 60, 150, 140]
        box_b = [450, 60, 550, 140]
        box_o = [250, 10, 350, 150]
        route = compute_obstacle_avoidance_route(
            box_a, box_b, [box_o], clearance=25,
            canvas_width=600, canvas_height=500, margin=10
        )
        self.assertIsNotNone(route)
        wpts = route["waypoints"]
        # All waypoints must be strictly within canvas bounds [10, 490]
        for p in wpts:
            self.assertGreaterEqual(p[1], 10)
            self.assertLessEqual(p[1], 490)
        # Detour y must be bottom detour
        self.assertEqual(wpts[2][1], 175)

    def test_canvas_bounds_bottom_overflow_selects_top(self):
        # Box O near bottom edge y=480, canvas_height=500, margin=10
        # Bottom detour (480 + 25 = 505 > 490) overflows, so top detour is selected
        box_a = [50, 360, 150, 440]
        box_b = [450, 360, 550, 440]
        box_o = [250, 350, 350, 480]
        route = compute_obstacle_avoidance_route(
            box_a, box_b, [box_o], clearance=25,
            canvas_width=600, canvas_height=500, margin=10
        )
        self.assertIsNotNone(route)
        wpts = route["waypoints"]
        for p in wpts:
            self.assertGreaterEqual(p[1], 10)
            self.assertLessEqual(p[1], 490)
        # Detour y must be top detour (350 - 25 = 325)
        self.assertEqual(wpts[2][1], 325)

    def test_text_obstacle_avoidance_switches_detour_side(self):
        box_a = [50, 200, 150, 300]
        box_b = [450, 200, 550, 300]
        box_o = [250, 180, 350, 320]
        # Text label sitting right on the top detour path y=155
        box_text = [220, 130, 380, 170]
        route = compute_obstacle_avoidance_route(
            box_a, box_b, [box_o], clearance=25,
            text_obstacles=[box_text]
        )
        self.assertIsNotNone(route)
        wpts = route["waypoints"]
        # Should choose bottom detour y=345 to avoid piercing text box
        self.assertEqual(wpts[2][1], 345)
        for i in range(len(wpts) - 1):
            self.assertFalse(segment_box_collision(wpts[i], wpts[i+1], box_text, margin=1.0))

    def test_hanan_grid_astar_soft_text_avoidance(self):
        box_o = [250, 180, 350, 320]
        # Text label blocking top path
        text_top = [220, 130, 380, 170]
        wpts = hanan_grid_route(
            (150, 250), (450, 250), [box_o], clearance=25,
            text_obstacles=[text_top]
        )
        self.assertIsNotNone(wpts)
        # Hanan A* selects bottom path y=345
        self.assertEqual(wpts[1][1], 345)
        for i in range(len(wpts) - 1):
            self.assertFalse(segment_box_collision(wpts[i], wpts[i+1], text_top, margin=1.0))
            self.assertFalse(segment_box_collision(wpts[i], wpts[i+1], box_o, margin=0))

    def test_process_with_text_xml_extracts_and_avoids_labels(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        box_a = [50, 200, 150, 300]
        box_o = [250, 180, 350, 320]
        box_b = [450, 200, 550, 300]
        shapes = [{"bbox": box_a}, {"bbox": box_o}, {"bbox": box_b}]
        arrows = [{
            "from_box": box_a, "to_box": box_b,
            "shaft_confirmed": True, "head_separated": False, "method": "test"
        }]
        text_xml = '''
        <mxGraphModel>
          <root>
            <mxCell id="10" value="Top Status Label" vertex="1">
              <mxGeometry x="220" y="130" width="160" height="40" as="geometry"/>
            </mxCell>
          </root>
        </mxGraphModel>
        '''
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 600
                self.canvas_height = 500
                self.intermediate_results = {"text_xml": text_xml}
        ctx = Ctx()
        original_extract = ex.extract
        ex.extract = lambda path, **kwargs: {
            "canvas_width": 600, "canvas_height": 500,
            "shapes": shapes, "arrows": arrows,
            "yolo_boxes": [], "yolo_kept": [],
            "yolo_available": False, "yolo_error": None, "yolo_model": None,
            "fastsam_masks": [], "fastsam_arrow_instances": [],
            "fastsam_shapes_kept": [], "fastsam_shapes_dropped": [],
            "fastsam_available": False, "fastsam_error": None, "fastsam_model": None,
        }
        try:
            res = ex.process(ctx)
            self.assertEqual(res.metadata["arrows_obstacle_avoided"], 1)
            arrow_elem = [e for e in ctx.elements if e.element_type == "arrow"][0]
            # Must route around bottom to avoid the text label at y in [130, 170]
            self.assertGreaterEqual(arrow_elem.vector_points[1][1], 345)
        finally:
            ex.extract = original_extract

    def test_compute_feedback_loop_route_horizontal(self):
        box_a = [50, 100, 150, 200]
        box_b = [450, 100, 550, 200]
        route = compute_feedback_loop_route(box_b, box_a, [box_a, box_b], clearance=35)
        self.assertIsNotNone(route)
        self.assertEqual(route["connector_type"], "feedback_loop")
        self.assertEqual(route["bends"], 2)
        self.assertEqual(route["loop_side"], "bottom")
        self.assertEqual(route["waypoints"], [[500, 200], [500, 235], [100, 235], [100, 200]])
        # Waypoints clear both box interiors
        for i in range(len(route["waypoints"]) - 1):
            p1, p2 = route["waypoints"][i], route["waypoints"][i+1]
            self.assertFalse(segment_box_collision(p1, p2, box_a, margin=0))
            self.assertFalse(segment_box_collision(p1, p2, box_b, margin=0))

    def test_compute_feedback_loop_route_vertical(self):
        box_a = [100, 50, 200, 150]
        box_b = [100, 450, 200, 550]
        route = compute_feedback_loop_route(box_b, box_a, [box_a, box_b], clearance=35)
        self.assertIsNotNone(route)
        self.assertEqual(route["connector_type"], "feedback_loop")
        self.assertEqual(route["bends"], 2)
        self.assertEqual(route["loop_side"], "right")
        self.assertEqual(route["waypoints"], [[200, 500], [235, 500], [235, 100], [200, 100]])

    def test_process_detects_bidirectional_feedback_loop(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        box_a = [50, 100, 150, 200]
        box_b = [450, 100, 550, 200]
        shapes = [{"bbox": box_a}, {"bbox": box_b}]
        # Bidirectional arrows: A->B and B->A
        arrows = [
            {"from_box": box_a, "to_box": box_b, "shaft_confirmed": True, "head_separated": False, "method": "test"},
            {"from_box": box_b, "to_box": box_a, "shaft_confirmed": True, "head_separated": False, "method": "test"},
        ]
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 600
                self.canvas_height = 500
                self.intermediate_results = {}
        ctx = Ctx()
        original_extract = ex.extract
        ex.extract = lambda path, **kwargs: {
            "canvas_width": 600, "canvas_height": 500,
            "shapes": shapes, "arrows": arrows,
            "yolo_boxes": [], "yolo_kept": [],
            "yolo_available": False, "yolo_error": None, "yolo_model": None,
            "fastsam_masks": [], "fastsam_arrow_instances": [],
            "fastsam_shapes_kept": [], "fastsam_shapes_dropped": [],
            "fastsam_available": False, "fastsam_error": None, "fastsam_model": None,
        }
        try:
            res = ex.process(ctx)
            self.assertEqual(res.metadata["arrows_feedback_loop"], 1)
            arrow_elems = [e for e in ctx.elements if e.element_type == "arrow"]
            self.assertEqual(len(arrow_elems), 2)
            # Forward arrow
            self.assertEqual(arrow_elems[0].source_id, "0")
            self.assertEqual(arrow_elems[0].target_id, "1")
            # Feedback arrow has waypoints routing underneath
            self.assertEqual(arrow_elems[1].source_id, "1")
            self.assertEqual(arrow_elems[1].target_id, "0")
            self.assertIsNotNone(arrow_elems[1].vector_points)
            self.assertTrue(arrow_elems[1].is_rounded)
            self.assertEqual(arrow_elems[1].vector_points, [[500, 235], [100, 235]])
        finally:
            ex.extract = original_extract

    def test_drawio_xml_rounded_corners_for_multibend_connectors(self):
        import types
        if "yaml" not in sys.modules:
            sys.modules["yaml"] = types.ModuleType("yaml")
        from main import Pipeline
        pipeline = Pipeline.__new__(Pipeline)
        class Ctx:
            def __init__(self):
                self.elements = [
                    # Multi-bend arrow with waypoints
                    ElementInfo(
                        id=0, element_type="arrow",
                        bbox=BoundingBox(50, 100, 550, 235),
                        arrow_start=(500, 200), arrow_end=(100, 200),
                        vector_points=[[500, 235], [100, 235]],
                        source_id="1", target_id="0", is_rounded=True,
                    ),
                    # Straight arrow
                    ElementInfo(
                        id=1, element_type="arrow",
                        bbox=BoundingBox(150, 150, 450, 150),
                        arrow_start=(150, 150), arrow_end=(450, 150),
                        vector_points=[],
                        source_id="0", target_id="1", is_rounded=False,
                    ),
                ]
        ctx = Ctx()
        pipeline._generate_xml_fragments(ctx)
        # Multi-bend connector has rounded=1;arcSize=10;
        self.assertIn("rounded=1;arcSize=10;", ctx.elements[0].xml_fragment)
        # Straight connector has rounded=0;
        self.assertIn("rounded=0;", ctx.elements[1].xml_fragment)


class TestHananCanvasClampRegression(unittest.TestCase):
    """Regressions for the canvas-clamp / A*-endpoint fixes (Major 1-3).

    These pin the three reproduced defects in hanan_grid_route: a clamped
    terminal could lose a route (M1), mix raw and snapped coordinates into
    a non-orthogonal segment (M2), or append an end point that pierced an
    obstacle (M3). Each test asserts the invariant, not just one output.
    """

    @staticmethod
    def _orthogonal(waypoints):
        return all(p[0] == q[0] or p[1] == q[1]
                   for p, q in zip(waypoints, waypoints[1:]))

    @staticmethod
    def _collisions(waypoints, obstacles, margin=0.0):
        return [(p, q, o) for p, q in zip(waypoints, waypoints[1:])
                for o in obstacles
                if segment_box_collision(p, q, o, margin=margin)]

    def test_m1_clamped_terminal_does_not_lose_route(self):
        # Terminal y=4 would clamp to margin y=10, inside obstacle y[7,64].
        obstacles = [[0, 7, 126, 64]]
        route = hanan_grid_route((83, 4), (525, 127), obstacles,
                                 canvas_width=600, canvas_height=500, margin=10)
        self.assertIsNotNone(
            route, "canvas clamp must not drop a route that exists on-canvas")
        self.assertEqual(route[0], [83, 4])
        self.assertEqual(route[-1], [525, 127])
        self.assertEqual(self._collisions(route, obstacles), [])

    def test_m2_no_diagonal_segment_from_clamped_terminal(self):
        # Terminal x=2 is inside the margin band; the grid snaps it inward.
        obstacles = [[300, 300, 500, 500]]
        route = hanan_grid_route((2, 200), (700, 700), obstacles,
                                 canvas_width=1024, canvas_height=1024, margin=10)
        self.assertIsNotNone(route)
        self.assertTrue(self._orthogonal(route),
                        f"route must stay orthogonal: {route}")
        self.assertEqual(self._collisions(route, obstacles), [])

    def test_m3_appended_end_does_not_pierce_obstacle(self):
        # Without the guard the final (250,490)->(250,700) hop pierces the box.
        obstacles = [[150, 600, 350, 700]]
        route = hanan_grid_route((250, 100), (250, 700), obstacles,
                                 canvas_width=600, canvas_height=500, margin=10)
        self.assertIsNotNone(route)
        self.assertEqual(self._collisions(route, obstacles), [])

    def test_clamp_never_returns_worse_than_no_canvas(self):
        # A route that exists without canvas bounds and lies fully on-canvas
        # must also be found with bounds: clamping may not create a false None.
        cases = [
            (((83, 4), (525, 127), [[0, 7, 126, 64]]), 600, 500, 10),
            (((127, 436), (39, 122),
              [[6, 199, 157, 362], [104, 562, 206, 607]]), 210, 607, 15),
        ]
        for (start, end, obstacles), w, h, margin in cases:
            plain = hanan_grid_route(start, end, obstacles)
            bounded = hanan_grid_route(start, end, obstacles,
                                       canvas_width=w, canvas_height=h,
                                       margin=margin)
            if plain is not None and all(0 <= p[0] <= w and 0 <= p[1] <= h
                                         for p in plain):
                self.assertIsNotNone(
                    bounded, f"on-canvas route lost with canvas: {plain}")

    def test_public_api_route_is_orthogonal_and_clear(self):
        # Same defect through the public API: a border-flush port makes the
        # Hanan fallback return a diagonal lead-in segment in the connector
        # polyline. (Reproduces via compute_obstacle_avoidance_route.)
        from_box = [242, 0, 328, 44]
        to_box = [222, 0, 277, 41]
        obstacles = [[110, 28, 268, 74], [13, 30, 87, 179], [240, 133, 302, 186]]
        route = compute_obstacle_avoidance_route(
            from_box, to_box, obstacles,
            canvas_width=302, canvas_height=704, margin=10)
        self.assertIsNotNone(route)
        wpts = route["waypoints"]
        self.assertTrue(self._orthogonal(wpts),
                        f"waypoints must be orthogonal: {wpts}")
        others = [o for o in obstacles if o != from_box and o != to_box]
        self.assertEqual(self._collisions(wpts, others), [])


class TestM4AndMinorsRegression(unittest.TestCase):
    """Regressions for M4 text performance and m1-m4, opt findings."""

    def test_m4_dense_text_obstacles_performance(self):
        import random, time
        random.seed(42)
        text_obs = [[random.randint(50, 500), random.randint(50, 500),
                     random.randint(50, 500) + 40, random.randint(50, 500) + 20]
                    for _ in range(100)]
        obs = [[250, 180, 350, 320]]
        t0 = time.perf_counter()
        route = hanan_grid_route((150, 250), (450, 250), obs, clearance=25, text_obstacles=text_obs)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 0.25, f"Dense text routing took {dt:.3f}s; expected < 0.25s")

    def test_m1_negative_coordinates_text_xml(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 600
                self.canvas_height = 500
                self.intermediate_results = {
                    "text_xml": '<mxCell id="1"><mxGeometry x="-15.5" y="-10.5" width="80" height="30"/></mxCell>'
                }
        ctx = Ctx()
        original_extract = ex.extract
        ex.extract = lambda path, **kwargs: {
            "canvas_width": 600, "canvas_height": 500,
            "shapes": [{"bbox": [50, 100, 150, 200]}, {"bbox": [450, 100, 550, 200]}],
            "arrows": [],
            "yolo_boxes": [], "yolo_kept": [], "yolo_available": False, "yolo_error": None, "yolo_model": None,
            "fastsam_masks": [], "fastsam_arrow_instances": [],
            "fastsam_shapes_kept": [], "fastsam_shapes_dropped": [],
            "fastsam_available": False, "fastsam_error": None, "fastsam_model": None,
        }
        try:
            res = ex.process(ctx)
            self.assertTrue(res.success)
        finally:
            ex.extract = original_extract

    def test_m4_distinct_lanes_at_canvas_margin(self):
        box_a = [50, 10, 150, 50]
        box_b = [450, 10, 550, 50]
        box_o = [250, 5, 350, 55]
        r0 = compute_obstacle_avoidance_route(box_a, box_b, [box_o], clearance=25, lane_index=0,
                                             canvas_width=600, canvas_height=100, margin=10)
        r1 = compute_obstacle_avoidance_route(box_a, box_b, [box_o], clearance=25, lane_index=1,
                                             canvas_width=600, canvas_height=100, margin=10)
        self.assertIsNotNone(r0)
        self.assertIsNotNone(r1)
        self.assertNotEqual(r0["waypoints"][2][1], r1["waypoints"][2][1])

    def test_opt_arrow_bbox_matches_abox(self):
        ex = DetectionContourExtractor(fastsam_model_path="")
        box_a = [50, 100, 150, 200]
        box_b = [450, 100, 550, 200]
        class Ctx:
            def __init__(self):
                self.image_path = SYNTH
                self.elements = []
                self.canvas_width = 600
                self.canvas_height = 500
                self.intermediate_results = {}
        ctx = Ctx()
        original_extract = ex.extract
        ex.extract = lambda path, **kwargs: {
            "canvas_width": 600, "canvas_height": 500,
            "shapes": [{"bbox": box_a}, {"bbox": box_b}],
            "arrows": [{
                "from_box": box_a, "to_box": box_b,
                "shaft_confirmed": True, "head_separated": False, "method": "test"
            }],
            "yolo_boxes": [], "yolo_kept": [], "yolo_available": False, "yolo_error": None, "yolo_model": None,
            "fastsam_masks": [], "fastsam_arrow_instances": [],
            "fastsam_shapes_kept": [], "fastsam_shapes_dropped": [],
            "fastsam_available": False, "fastsam_error": None, "fastsam_model": None,
        }
        try:
            res = ex.process(ctx)
            arrow_elem = [e for e in ctx.elements if e.element_type == "arrow"][0]
            self.assertNotEqual(arrow_elem.bbox.to_list(), box_b)
            self.assertEqual(arrow_elem.bbox.x1, 150)
            self.assertEqual(arrow_elem.bbox.x2, 450)
        finally:
            ex.extract = original_extract


if __name__ == "__main__":
    unittest.main()
