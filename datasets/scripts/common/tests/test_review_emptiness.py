"""Synthetic checks for the unlabelled-target evidence module.

Every image is drawn in memory on a flat background; no file is read.

Two scenes are used:

* a small 220×140 scene with three identical squares, which pins the round-one
  behaviour (boxes, NMS, coverage, template matching, empty-template rule);
* a 640×360 standee scene with frontal boards, distant small boards and one
  back-side board, which pins the target-spec semantics: only standees at or
  above the minimum target scale and facing the camera can be missing labels,
  while small and back-side boards are recorded separately and do not stop the
  frame from being clean background.

The standee scene pastes exact template copies, so a real candidate scores ~1.0
while a cross-scale partial match of the plain grey back board scores ~0.64.
Those scenarios therefore pass ``suspect_min_score=0.85`` to count exact
matches only; the module default stays 0.62.
"""

import unittest

import cv2
import numpy as np

from datasets.scripts.common.review import emptiness
from datasets.scripts.common.review.schema import REASON_CODES, Box, EmptinessReport

BACKGROUND = 60
FOREGROUND = (235, 235, 235)
TARGET_SIDE = 16
TEMPLATE_PAD = 4

#: Three targets of equal size; only the first is ever annotated in the tests.
TARGETS = (
    (20, 25, 36, 41),
    (100, 30, 116, 46),
    (160, 85, 176, 101),
)
FRAME_SHAPE = (140, 220)

#: Target spec of the person task: a standee below 28 % of the frame height is
#: not a target, a back side is a hard negative.
STANDEE_SPEC = {
    "min_height_fraction": 0.28,
    "back_side": "hard_negative",
    "below_min_scale": "not_a_target",
}
#: Exact template copies score ~1.0 in the synthetic scene; 0.85 keeps the
#: low-contrast cross-scale partial matches out of the counts.
EXACT_MATCH_PARAMS = {"suspect_min_score": 0.85}

STANDEE_FRAME_SHAPE = (360, 640)
FRONTAL_PLACEMENTS = ((30, 200), (200, 200), (520, 190))
SMALL_PLACEMENTS = ((120, 320), (400, 310))
BACK_PLACEMENT = (300, 200)
MIXED_PLACEMENT = (300, 200)


def synthetic_frame() -> np.ndarray:
    image = np.full((FRAME_SHAPE[0], FRAME_SHAPE[1], 3), BACKGROUND, np.uint8)
    for x1, y1, x2, y2 in TARGETS:
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), FOREGROUND, -1)
    return image


def synthetic_template() -> np.ndarray:
    side = TARGET_SIDE + 2 * TEMPLATE_PAD
    template = np.full((side, side, 3), BACKGROUND, np.uint8)
    cv2.rectangle(
        template,
        (TEMPLATE_PAD, TEMPLATE_PAD),
        (TEMPLATE_PAD + TARGET_SIDE - 1, TEMPLATE_PAD + TARGET_SIDE - 1),
        FOREGROUND,
        -1,
    )
    return template


def padded_box(box, margin: int = 8) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = float(width * height)
    union = float((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 0.0 else 0.0


def frontal_patch(height: int = 117, width: int = 60) -> np.ndarray:
    """Printed standee: saturated board, face band with eyes, striped body."""

    patch = np.full((height, width, 3), BACKGROUND, np.uint8)
    patch[2:height - 2, 2:width - 2] = (40, 90, 210)
    band = int(height * 0.4)
    patch[6:band, 10:width - 10] = (150, 190, 235)
    patch[14:18, 18:22] = (30, 30, 30)
    patch[14:18, 38:42] = (30, 30, 30)
    patch[26:29, 24:36] = (60, 60, 120)
    for y in range(band + 6, height - 8, 8):
        patch[y:y + 3, 6:width - 6] = (80, 130, 60)
    return patch


def back_patch(height: int = 117, width: int = 60) -> np.ndarray:
    """White back: grey board with a thin printed outline, no colour."""

    patch = np.full((height, width, 3), BACKGROUND, np.uint8)
    cv2.rectangle(patch, (2, 2), (width - 3, height - 3), (170, 170, 170), 1)
    cv2.rectangle(patch, (4, 4), (width - 5, height - 5), (203, 203, 203), -1)
    return patch


def small_patch(height: int = 36, width: int = 18) -> np.ndarray:
    """Distant standee: dark board with a light rim, 0.10 of the frame height."""

    patch = np.full((height, width, 3), BACKGROUND, np.uint8)
    patch[1:height - 1, 1:width - 1] = (75, 75, 85)
    patch[0, :] = (130, 130, 130)
    patch[-1, :] = (130, 130, 130)
    patch[:, 0] = (130, 130, 130)
    patch[:, -1] = (130, 130, 130)
    return patch


def mixed_patch(height: int = 117, width: int = 60, seed: int = 7) -> np.ndarray:
    """Half flat, half noise: a contaminated patch, not a usable standee box."""

    patch = np.full((height, width, 3), 203, np.uint8)
    rng = np.random.default_rng(seed)
    patch[:, width // 2:] = rng.integers(0, 256, size=(height, width - width // 2, 3), dtype=np.uint8)
    return patch


FRONTAL_PATCH = frontal_patch()
BACK_PATCH = back_patch()
SMALL_PATCH = small_patch()
MIXED_PATCH = mixed_patch()


def standee_frame(frontal=(), small=(), back=False, mixed=False) -> np.ndarray:
    frame = np.full((STANDEE_FRAME_SHAPE[0], STANDEE_FRAME_SHAPE[1], 3), BACKGROUND, np.uint8)
    for x, y in frontal:
        frame[y:y + FRONTAL_PATCH.shape[0], x:x + FRONTAL_PATCH.shape[1]] = FRONTAL_PATCH
    for x, y in small:
        frame[y:y + SMALL_PATCH.shape[0], x:x + SMALL_PATCH.shape[1]] = SMALL_PATCH
    if back:
        x, y = BACK_PLACEMENT
        frame[y:y + BACK_PATCH.shape[0], x:x + BACK_PATCH.shape[1]] = BACK_PATCH
    if mixed:
        x, y = MIXED_PLACEMENT
        frame[y:y + MIXED_PATCH.shape[0], x:x + MIXED_PATCH.shape[1]] = MIXED_PATCH
    return frame


def back_box() -> tuple[int, int, int, int]:
    x, y = BACK_PLACEMENT
    return (x, y, x + BACK_PATCH.shape[1], y + BACK_PATCH.shape[0])


class EmptinessScanTest(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()
        self.template = synthetic_template()

    def test_two_unlabelled_targets_are_reported(self):
        report = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[self.template],
        )
        self.assertIsInstance(report, EmptinessReport)
        self.assertFalse(report.clean)
        self.assertEqual(report.reason_codes, ["unlabelled_target_suspected"])
        self.assertGreaterEqual(len(report.suspects), 2)
        for target in TARGETS[1:]:
            with self.subTest(target=target):
                self.assertTrue(
                    any(iou(suspect["xyxy"], target) > 0.3 for suspect in report.suspects),
                    "no suspect overlaps the unlabelled target %r: %r"
                    % (target, report.suspects),
                )
        # The annotated target is explained by its box and must not be reported.
        self.assertFalse(any(iou(suspect["xyxy"], TARGETS[0]) > 0.3 for suspect in report.suspects))
        self.assertGreaterEqual(report.covered_area_fraction, 0.0)
        self.assertLess(report.covered_area_fraction, 1.0)

    def test_suspect_entries_expose_the_documented_shape(self):
        report = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[self.template],
        )
        height, width = self.frame.shape[:2]
        for suspect in report.suspects:
            with self.subTest(suspect=suspect):
                self.assertEqual(sorted(suspect), ["method", "score", "xyxy"])
                self.assertEqual(suspect["method"], "template")
                self.assertIsInstance(suspect["score"], float)
                self.assertGreater(suspect["score"], 0.62)
                x1, y1, x2, y2 = suspect["xyxy"]
                self.assertTrue(all(isinstance(value, int) for value in suspect["xyxy"]))
                self.assertTrue(0 <= x1 < x2 <= width)
                self.assertTrue(0 <= y1 < y2 <= height)
        scores = [suspect["score"] for suspect in report.suspects]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fully_annotated_frame_is_clean(self):
        boxes = [Box("target", padded_box(target)) for target in TARGETS]
        report = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=boxes,
            templates_bgr=[self.template],
        )
        self.assertTrue(report.clean)
        self.assertEqual(report.suspects, [])
        self.assertEqual(report.reason_codes, ["no_unlabelled_target"])
        self.assertGreater(report.covered_area_fraction, 0.05)

    def test_boundary_matches_are_dropped_when_the_annotation_explains_them(self):
        # Tight boxes still mask a template-sized neighbourhood of every target,
        # so only shifted matches whose top-left corner is unexplained survive
        # the search mask; their centre is inside the annotation.
        boxes = [Box("target", target) for target in TARGETS]
        without_rule = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=boxes,
            templates_bgr=[self.template],
            params={"drop_explained_candidates": False},
        )
        self.assertFalse(without_rule.clean, "the shifted matches near the boxes must be visible")
        with_rule = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=boxes,
            templates_bgr=[self.template],
        )
        self.assertTrue(with_rule.clean)

    def test_empty_templates_never_report_clean(self):
        for boxes in ([], [Box("target", TARGETS[0])]):
            with self.subTest(boxes=len(boxes)):
                report = emptiness.scan_unlabelled_targets(
                    self.frame,
                    boxes=boxes,
                    templates_bgr=[],
                )
                self.assertFalse(report.clean)
                self.assertEqual(report.suspects, [])
                self.assertEqual(report.reason_codes, ["emptiness_unproven"])

    def test_empty_templates_keep_the_measured_coverage(self):
        boxes = [Box("target", padded_box(TARGETS[0]))]
        report = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=boxes,
            templates_bgr=[],
        )
        expected = emptiness.covered_area_fraction(emptiness.mask_boxes(FRAME_SHAPE, [padded_box(TARGETS[0])]))
        self.assertAlmostEqual(report.covered_area_fraction, expected)

    def test_gray_input_matches_bgr_input(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        from_bgr = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[self.template],
        )
        from_gray = emptiness.scan_unlabelled_targets(
            gray,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[gray_template],
        )
        self.assertEqual(
            [suspect["xyxy"] for suspect in from_bgr.suspects],
            [suspect["xyxy"] for suspect in from_gray.suspects],
        )

    def test_frame_without_targets_is_clean(self):
        empty_frame = np.full((FRAME_SHAPE[0], FRAME_SHAPE[1], 3), BACKGROUND, np.uint8)
        report = emptiness.scan_unlabelled_targets(
            empty_frame,
            boxes=[],
            templates_bgr=[self.template],
        )
        self.assertTrue(report.clean)
        self.assertEqual(report.suspects, [])
        self.assertEqual(report.reason_codes, ["no_unlabelled_target"])
        self.assertEqual(report.covered_area_fraction, 0.0)

    def test_suspect_floor_is_consumed(self):
        # A floor above every possible score disables the suspect check; this is
        # the documented limitation of a score-only guard.
        report = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[self.template],
            params={"suspect_min_score": 1.5},
        )
        self.assertTrue(report.clean)

    def test_mask_margin_parameter_is_consumed(self):
        narrow = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[self.template],
            params={"mask_margin_px": 0},
        )
        wide = emptiness.scan_unlabelled_targets(
            self.frame,
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[self.template],
            params={"mask_margin_px": 24},
        )
        self.assertLess(narrow.covered_area_fraction, wide.covered_area_fraction)

    def test_reason_codes_stay_inside_the_shared_vocabulary(self):
        cases = [
            emptiness.scan_unlabelled_targets(
                self.frame, boxes=[Box("target", TARGETS[0])], templates_bgr=[self.template]
            ),
            emptiness.scan_unlabelled_targets(
                self.frame,
                boxes=[Box("target", padded_box(target)) for target in TARGETS],
                templates_bgr=[self.template],
            ),
            emptiness.scan_unlabelled_targets(self.frame, boxes=[], templates_bgr=[]),
        ]
        for report in cases:
            for code in report.reason_codes:
                self.assertIn(code, REASON_CODES)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            emptiness.scan_unlabelled_targets(
                self.frame, boxes=[], templates_bgr=[self.template], params={"mask_margin_px": -1}
            )
        with self.assertRaises(ValueError):
            emptiness.scan_unlabelled_targets(
                self.frame, boxes=[], templates_bgr=[self.template], params={"nms_iou": 1.5}
            )
        with self.assertRaises(ValueError):
            emptiness.scan_unlabelled_targets(
                self.frame, boxes=[], templates_bgr=[self.template], params={"scales": ()}
            )


class TargetSpecScanTest(unittest.TestCase):
    """Small and back-side standees must not block a frame as background."""

    def test_frontal_suspects_are_separated_from_small_and_back_standees(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(frontal=FRONTAL_PLACEMENTS, small=SMALL_PLACEMENTS, back=True),
            boxes=[Box("person", (30, 200, 90, 317))],
            templates_bgr=[FRONTAL_PATCH, SMALL_PATCH, BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec=STANDEE_SPEC,
        )
        self.assertFalse(report.clean)
        self.assertEqual(len(report.suspects), 2)
        self.assertEqual(len(report.below_scale), 2)
        self.assertEqual(len(report.hard_negatives), 1)
        self.assertEqual(
            report.reason_codes,
            ["below_target_scale", "hard_negative_present", "unlabelled_target_suspected"],
        )
        # The two unlabelled frontal boards are the suspects.
        for placement in FRONTAL_PLACEMENTS[1:]:
            x, y = placement
            expected = (x, y, x + FRONTAL_PATCH.shape[1], y + FRONTAL_PATCH.shape[0])
            with self.subTest(placement=placement):
                self.assertTrue(any(iou(suspect["xyxy"], expected) > 0.5 for suspect in report.suspects))
        # Small boards stay below scale and keep their measured height fraction.
        for entry in report.below_scale:
            with self.subTest(entry=entry):
                self.assertEqual(sorted(entry), ["height_fraction", "score", "xyxy"])
                self.assertLess(entry["height_fraction"], STANDEE_SPEC["min_height_fraction"])
                self.assertAlmostEqual(entry["height_fraction"], 0.10, places=2)
                self.assertFalse(any(iou(entry["xyxy"], suspect["xyxy"]) > 0.3 for suspect in report.suspects))
        # The grey back board is a hard negative, never a suspect.
        self.assertEqual(report.hard_negatives[0]["side"], "back")
        self.assertEqual(sorted(report.hard_negatives[0]), ["score", "side", "xyxy"])
        self.assertGreater(iou(report.hard_negatives[0]["xyxy"], back_box()), 0.8)
        self.assertFalse(any(iou(entry["xyxy"], back_box()) > 0.3 for entry in report.suspects))

    def test_frame_with_only_small_and_back_standees_is_clean(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(small=SMALL_PLACEMENTS, back=True),
            boxes=[],
            templates_bgr=[SMALL_PATCH, BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec=STANDEE_SPEC,
        )
        self.assertTrue(report.clean)
        self.assertEqual(report.suspects, [])
        self.assertEqual(len(report.below_scale), 2)
        self.assertEqual(len(report.hard_negatives), 1)
        self.assertEqual(
            report.reason_codes,
            ["below_target_scale", "hard_negative_present", "no_unlabelled_target"],
        )
        for entry in report.below_scale:
            self.assertLess(iou(entry["xyxy"], back_box()), 0.3)
        self.assertAlmostEqual(report.hard_negatives[0]["score"], 1.0, places=4)

    def test_unknown_side_stays_a_suspect(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(mixed=True),
            boxes=[],
            templates_bgr=[MIXED_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec=STANDEE_SPEC,
        )
        self.assertFalse(report.clean)
        self.assertEqual(len(report.suspects), 1)
        self.assertEqual(report.hard_negatives, [])
        self.assertEqual(report.reason_codes, ["unlabelled_target_suspected"])
        self.assertGreater(iou(report.suspects[0]["xyxy"], (300, 200, 360, 317)), 0.8)

    def test_below_scale_policy_suspect_keeps_small_candidates(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(small=SMALL_PLACEMENTS),
            boxes=[],
            templates_bgr=[SMALL_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec={"min_height_fraction": 0.28, "below_min_scale": "suspect"},
        )
        self.assertFalse(report.clean)
        self.assertEqual(len(report.suspects), 2)
        self.assertEqual(report.below_scale, [])
        self.assertEqual(report.reason_codes, ["unlabelled_target_suspected"])

    def test_below_scale_hit_records_back_side_proposal_without_clearing_a_partial_match(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(back=True),
            boxes=[],
            templates_bgr=[BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec={**STANDEE_SPEC, "min_height_fraction": 0.4, "below_min_scale": "suspect"},
        )
        self.assertFalse(report.clean)
        self.assertEqual(report.hard_negatives, [])
        self.assertEqual(len(report.suspects), 1)
        self.assertEqual(report.suspects[0]["side_observation"]["side"], "back")

    def test_back_side_policy_suspect_keeps_the_candidate(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(back=True),
            boxes=[],
            templates_bgr=[BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec={"min_height_fraction": 0.28},
        )
        self.assertFalse(report.clean)
        self.assertEqual(len(report.suspects), 1)
        self.assertEqual(report.hard_negatives, [])
        self.assertEqual(report.reason_codes, ["unlabelled_target_suspected"])

    def test_default_spec_keeps_the_round_one_verdict(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(frontal=FRONTAL_PLACEMENTS, small=SMALL_PLACEMENTS, back=True),
            boxes=[Box("person", (30, 200, 90, 317))],
            templates_bgr=[FRONTAL_PATCH, SMALL_PATCH, BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
        )
        self.assertFalse(report.clean)
        self.assertEqual(len(report.suspects), 5)
        self.assertEqual(report.below_scale, [])
        self.assertEqual(report.hard_negatives, [])
        self.assertEqual(report.target_spec, emptiness.DEFAULT_SPEC)

    def test_target_spec_is_echoed_for_audit(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(small=SMALL_PLACEMENTS, back=True),
            boxes=[],
            templates_bgr=[SMALL_PATCH, BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec=STANDEE_SPEC,
        )
        self.assertEqual(report.target_spec, STANDEE_SPEC)
        partial = emptiness.scan_unlabelled_targets(
            standee_frame(back=True),
            boxes=[],
            templates_bgr=[BACK_PATCH],
            params=EXACT_MATCH_PARAMS,
            spec={"min_height_fraction": 0.3},
        )
        self.assertEqual(partial.target_spec["min_height_fraction"], 0.3)
        self.assertEqual(partial.target_spec["back_side"], emptiness.DEFAULT_SPEC["back_side"])
        self.assertEqual(partial.target_spec["below_min_scale"], emptiness.DEFAULT_SPEC["below_min_scale"])

    def test_target_spec_is_echoed_for_unproven_frames(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(),
            boxes=[],
            templates_bgr=[],
            spec=STANDEE_SPEC,
        )
        self.assertFalse(report.clean)
        self.assertEqual(report.reason_codes, ["emptiness_unproven"])
        self.assertEqual(report.below_scale, [])
        self.assertEqual(report.hard_negatives, [])
        self.assertEqual(report.target_spec, STANDEE_SPEC)

    def test_gray_frame_keeps_the_back_board_as_a_suspect(self):
        # Without colour the side cannot be proven, so the safe outcome is a
        # suspect instead of a hard negative and a clean frame.
        gray_frame = cv2.cvtColor(standee_frame(back=True), cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(BACK_PATCH, cv2.COLOR_BGR2GRAY)
        report = emptiness.scan_unlabelled_targets(
            gray_frame,
            boxes=[],
            templates_bgr=[gray_template],
            params=EXACT_MATCH_PARAMS,
            spec=STANDEE_SPEC,
        )
        self.assertFalse(report.clean)
        self.assertEqual(len(report.suspects), 1)
        self.assertEqual(report.hard_negatives, [])
        self.assertEqual(report.reason_codes, ["unlabelled_target_suspected"])

    def test_invalid_spec_is_rejected(self):
        frame = standee_frame()
        for spec in (
            {"min_height_fraction": 1.5},
            {"min_height_fraction": -0.1},
            {"back_side": "maybe"},
            {"below_min_scale": "ignore"},
        ):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    emptiness.scan_unlabelled_targets(
                        frame, boxes=[], templates_bgr=[FRONTAL_PATCH], spec=spec
                    )

    def test_new_reason_codes_are_in_the_shared_vocabulary(self):
        for code in (
            emptiness.REASON_BELOW_SCALE,
            emptiness.REASON_HARD_NEGATIVE,
            emptiness.REASON_CLEAN,
            emptiness.REASON_SUSPECT,
            emptiness.REASON_UNPROVEN,
        ):
            self.assertIn(code, REASON_CODES)


class CliChannelSpecTest(unittest.TestCase):
    """The CLI forwards only ``params``, so spec keys may travel inside it."""

    def test_spec_inside_params_activates_the_policy(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(small=SMALL_PLACEMENTS, back=True),
            boxes=[],
            templates_bgr=[SMALL_PATCH, BACK_PATCH],
            params=dict(EXACT_MATCH_PARAMS, **STANDEE_SPEC),
        )
        self.assertTrue(report.clean)
        self.assertEqual(len(report.below_scale), 2)
        self.assertEqual(len(report.hard_negatives), 1)
        self.assertEqual(report.target_spec, STANDEE_SPEC)

    def test_explicit_spec_wins_over_params(self):
        report = emptiness.scan_unlabelled_targets(
            standee_frame(small=SMALL_PLACEMENTS),
            boxes=[],
            templates_bgr=[SMALL_PATCH],
            params={"suspect_min_score": 0.85, "min_height_fraction": 0.28},
            spec={"min_height_fraction": 0.0},
        )
        self.assertEqual(report.target_spec["min_height_fraction"], 0.0)
        self.assertEqual(report.below_scale, [])
        self.assertFalse(report.clean)

    def test_params_without_spec_keys_keep_the_default_spec(self):
        report = emptiness.scan_unlabelled_targets(
            synthetic_frame(),
            boxes=[Box("target", TARGETS[0])],
            templates_bgr=[synthetic_template()],
            params={"mask_margin_px": 8},
        )
        self.assertEqual(report.target_spec, emptiness.DEFAULT_SPEC)


class StandeeSideTest(unittest.TestCase):
    def test_saturated_textured_board_is_frontal(self):
        result = emptiness.classify_standee_side(FRONTAL_PATCH)
        self.assertEqual(result["side"], "frontal")
        self.assertGreater(result["saturation_mean"], 0.3)
        self.assertGreater(result["edge_density"], 0.03)
        self.assertEqual(result["reason"], "saturated_print")

    def test_uniform_grey_board_is_back(self):
        result = emptiness.classify_standee_side(BACK_PATCH)
        self.assertEqual(result["side"], "back")
        self.assertAlmostEqual(result["saturation_mean"], 0.0, places=3)
        self.assertEqual(result["reason"], "uniform_low_saturation")

    def test_half_noise_patch_is_unknown(self):
        result = emptiness.classify_standee_side(MIXED_PATCH)
        self.assertEqual(result["side"], "unknown")
        self.assertEqual(result["reason"], "mixed_texture")

    def test_uniform_grey_without_outline_is_back(self):
        flat = np.full((60, 30, 3), 200, np.uint8)
        result = emptiness.classify_standee_side(flat)
        self.assertEqual(result["side"], "back")
        self.assertEqual(result["edge_density"], 0.0)

    def test_tiny_patch_is_unknown(self):
        result = emptiness.classify_standee_side(np.full((4, 4, 3), 200, np.uint8))
        self.assertEqual(result["side"], "unknown")
        self.assertEqual(result["reason"], "patch_too_small")
        self.assertEqual(result["saturation_mean"], 0.0)

    def test_gray_patch_is_never_declared_back(self):
        # A 2-D patch has no saturation, so the decisive colour evidence is
        # missing: the classifier must stay unknown instead of demoting the
        # standee to a hard negative.
        gray = cv2.cvtColor(FRONTAL_PATCH, cv2.COLOR_BGR2GRAY)
        result = emptiness.classify_standee_side(gray)
        self.assertEqual(result["side"], "unknown")
        self.assertEqual(result["reason"], "colour_unavailable")
        self.assertGreater(result["edge_density"], 0.0)

    def test_thresholds_come_from_params(self):
        raised = emptiness.classify_standee_side(
            FRONTAL_PATCH, params={"side_saturation_frontal_min": 0.99}
        )
        self.assertEqual(raised["side"], "unknown")
        strict_back = emptiness.classify_standee_side(
            BACK_PATCH, params={"side_edge_back_max": 0.01}
        )
        self.assertNotEqual(strict_back["side"], "back")

    def test_empty_patch_is_rejected(self):
        with self.assertRaises(ValueError):
            emptiness.classify_standee_side(np.zeros((0, 0, 3), np.uint8))

    def test_result_exposes_the_documented_keys(self):
        result = emptiness.classify_standee_side(FRONTAL_PATCH)
        self.assertEqual(
            sorted(result),
            ["edge_density", "face_texture", "reason", "saturation_mean", "side"],
        )
        for key in ("saturation_mean", "edge_density", "face_texture"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)


class PlanAdapterTest(unittest.TestCase):
    """The review CLI measures a frame through ``check_emptiness(image, plan)``."""

    def test_plan_without_templates_stays_unproven(self):
        report = emptiness.check_emptiness(synthetic_frame(), {"boxes": [list(TARGETS[0])]})
        self.assertFalse(report.clean)
        self.assertEqual(report.suspects, [])
        self.assertEqual(report.reason_codes, ["emptiness_unproven"])

    def test_missing_plan_is_unproven_not_clean(self):
        report = emptiness.check_emptiness(synthetic_frame())
        self.assertFalse(report.clean)
        self.assertEqual(report.reason_codes, ["emptiness_unproven"])

    def test_plan_with_boxes_and_templates_reports_suspects(self):
        plan = {
            "boxes": [{"class_name": "target", "xyxy": list(TARGETS[0])}],
            "templates_bgr": [synthetic_template()],
        }
        report = emptiness.check_emptiness(synthetic_frame(), plan)
        self.assertFalse(report.clean)
        self.assertEqual(report.reason_codes, ["unlabelled_target_suspected"])
        self.assertGreaterEqual(len(report.suspects), 2)

    def test_plan_forwards_emptiness_thresholds(self):
        plan = {
            "boxes": [list(TARGETS[0])],
            "templates_bgr": [synthetic_template()],
            "emptiness": {"suspect_min_score": 1.5},
        }
        self.assertTrue(emptiness.check_emptiness(synthetic_frame(), plan).clean)

    def test_plan_forwards_the_target_spec(self):
        plan = {
            "templates_bgr": [SMALL_PATCH, BACK_PATCH],
            "emptiness": EXACT_MATCH_PARAMS,
            "spec": STANDEE_SPEC,
        }
        report = emptiness.check_emptiness(standee_frame(small=SMALL_PLACEMENTS, back=True), plan)
        self.assertTrue(report.clean)
        self.assertEqual(report.target_spec, STANDEE_SPEC)
        self.assertEqual(len(report.below_scale), 2)
        self.assertEqual(len(report.hard_negatives), 1)

    def test_plan_accepts_target_spec_key_as_well(self):
        plan = {
            "templates_bgr": [BACK_PATCH],
            "emptiness": EXACT_MATCH_PARAMS,
            "target_spec": {"min_height_fraction": 0.28, "back_side": "hard_negative"},
        }
        report = emptiness.check_emptiness(standee_frame(back=True), plan)
        self.assertTrue(report.clean)
        self.assertEqual(report.target_spec["back_side"], "hard_negative")

    def test_plan_template_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            emptiness.check_emptiness(synthetic_frame(), {"templates_bgr": ["template.png"]})


class NmsTest(unittest.TestCase):
    def test_overlapping_boxes_keep_only_the_highest_score(self):
        boxes = [[10, 10, 60, 60], [14, 14, 64, 64]]
        scores = [0.9, 0.7]
        self.assertEqual(emptiness.nms(boxes, scores), [0])
        self.assertEqual(emptiness.nms(boxes, [0.5, 0.8]), [1])
        self.assertEqual(emptiness.nms(boxes[::-1], scores[::-1]), [1])

    def test_disjoint_boxes_are_all_kept(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        self.assertEqual(sorted(emptiness.nms(boxes, [0.9, 0.8])), [0, 1])

    def test_threshold_decides_suppression(self):
        boxes = [[0, 0, 100, 100], [0, 0, 100, 40]]  # IoU == 0.4
        self.assertEqual(emptiness.nms(boxes, [0.9, 0.8], 0.3), [0])
        self.assertEqual(sorted(emptiness.nms(boxes, [0.9, 0.8], 0.5)), [0, 1])

    def test_degenerate_inputs(self):
        self.assertEqual(emptiness.nms([], []), [])
        self.assertEqual(emptiness.nms([[5, 5, 5, 10]], [0.9]), [0])
        with self.assertRaises(ValueError):
            emptiness.nms([[0, 0, 1, 1]], [])
        with self.assertRaises(ValueError):
            emptiness.nms([[0, 0, 1, 1]], [float("nan")])


class MaskBoxesTest(unittest.TestCase):
    def test_coverage_grows_with_margin(self):
        shape = (64, 80)
        box = [(16, 16, 32, 32)]
        tight = emptiness.covered_area_fraction(emptiness.mask_boxes(shape, box, 0))
        padded = emptiness.covered_area_fraction(emptiness.mask_boxes(shape, box, 4))
        self.assertAlmostEqual(tight, 256.0 / (64 * 80))
        self.assertAlmostEqual(padded, 576.0 / (64 * 80))
        self.assertLess(tight, padded)

    def test_mask_is_uint8_with_binary_values(self):
        mask = emptiness.mask_boxes((32, 32), [(4, 4, 12, 12)], 2)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (32, 32))
        self.assertEqual(sorted(np.unique(mask).tolist()), [0, 255])
        self.assertTrue(np.all(mask[2:14, 2:14] == 255))

    def test_boxes_are_clipped_to_the_frame(self):
        mask = emptiness.mask_boxes((32, 32), [(0, 0, 8, 8)], 8)
        self.assertAlmostEqual(emptiness.covered_area_fraction(mask), 256.0 / (32 * 32))
        self.assertTrue(np.all(mask[:16, :16] == 255))

    def test_zero_area_box_and_empty_mask(self):
        self.assertEqual(emptiness.covered_area_fraction(emptiness.mask_boxes((16, 16), [(4, 4, 4, 4)], 0)), 0.0)
        self.assertEqual(emptiness.covered_area_fraction(np.zeros((8, 8), np.uint8)), 0.0)
        self.assertEqual(emptiness.covered_area_fraction(np.full((8, 8), 7, np.uint8)), 1.0)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            emptiness.mask_boxes((16, 16), [(0, 0, 4, 4)], -1)
        with self.assertRaises(ValueError):
            emptiness.mask_boxes((16,), [(0, 0, 4, 4)])
        with self.assertRaises(ValueError):
            emptiness.mask_boxes((16, 16), [(0, 0, 4)])


class MultiScaleMatchTest(unittest.TestCase):
    def setUp(self):
        self.gray = cv2.cvtColor(synthetic_frame(), cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(synthetic_template(), cv2.COLOR_BGR2GRAY)

    def test_returns_parallel_sorted_lists_with_usable_boxes(self):
        boxes, scores = emptiness.multi_scale_match(self.gray, self.template)
        self.assertTrue(boxes)
        self.assertEqual(len(boxes), len(scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        height, width = self.gray.shape[:2]
        for box, score in zip(boxes, scores):
            with self.subTest(box=box):
                self.assertGreater(score, 0.5)
                self.assertLessEqual(score, 1.0)
                x1, y1, x2, y2 = box
                self.assertTrue(0 <= x1 < x2 <= width)
                self.assertTrue(0 <= y1 < y2 <= height)
                self.assertGreaterEqual(x2 - x1, emptiness.MIN_BOX_PX)
                self.assertGreaterEqual(y2 - y1, emptiness.MIN_BOX_PX)

    def test_exact_alignment_is_found_at_scale_one(self):
        boxes, scores = emptiness.multi_scale_match(self.gray, self.template, scales=(1.0,))
        expected = [16, 21, 40, 45]
        self.assertIn(expected, boxes)
        self.assertAlmostEqual(max(scores), 1.0, places=5)

    def test_search_mask_removes_the_masked_candidate(self):
        allowed = np.full(self.gray.shape, 255, np.uint8)
        for x1, y1, _, _ in TARGETS:  # the exact-alignment peaks of all three targets
            allowed[y1 - TEMPLATE_PAD, x1 - TEMPLATE_PAD] = 0
        boxes, scores = emptiness.multi_scale_match(
            self.gray, self.template, scales=(1.0,), search_mask=allowed
        )
        self.assertNotIn([16, 21, 40, 45], boxes)
        self.assertLess(max(scores), 1.0)
        self.assertTrue(all(score > 0.5 for score in scores))

    def test_scale_changes_the_box_size(self):
        boxes, _ = emptiness.multi_scale_match(self.gray, self.template, scales=(1.3,))
        self.assertTrue(boxes)
        for box in boxes:
            self.assertEqual(box[2] - box[0], 31)
            self.assertEqual(box[3] - box[1], 31)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            emptiness.multi_scale_match(self.gray, self.template, scales=(0.0,))
        with self.assertRaises(ValueError):
            emptiness.multi_scale_match(self.gray, self.template, scales=(-1.0,))
        with self.assertRaises(ValueError):
            emptiness.multi_scale_match(self.gray, self.template, search_mask=np.ones((3, 3), np.uint8))
        with self.assertRaises(ValueError):
            emptiness.multi_scale_match(np.zeros((0, 4), np.uint8), self.template)


class VarianceMapTest(unittest.TestCase):
    def test_flat_region_has_no_local_deviation(self):
        flat = np.full((40, 40), 100, np.uint8)
        variance = emptiness.variance_map(flat, 8)
        self.assertEqual(variance.shape, (40, 40))
        self.assertEqual(variance.dtype, np.float32)
        self.assertLess(float(variance.max()), 1e-3)

    def test_edge_produces_high_local_deviation(self):
        frame = np.full((40, 40), 100, np.uint8)
        frame[18:22, 18:22] = 200
        variance = emptiness.variance_map(frame, 8)
        self.assertGreater(float(variance.max()), 10.0)
        self.assertAlmostEqual(float(variance[0, 0]), 0.0, places=5)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError):
            emptiness.variance_map(np.zeros((10, 10), np.uint8), 1)


if __name__ == "__main__":
    unittest.main()
