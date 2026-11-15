"""Deterministic checks for review ground-plane geometry.

Every scene is drawn in memory from a pinhole ground-plane model: a camera one unit
above the floor, focal length 300 px, horizon row 150, yawed 12 degrees so that both
ground directions have finite vanishing points. The truth used by the assertions
comes from that model, not from the estimator.
"""

import math
import unittest

import cv2
import numpy as np

from datasets.scripts.common.review import geometry
from datasets.scripts.common.review.schema import GroundObservation, GroundPlaneObservation


WIDTH = 640
HEIGHT = 360
HORIZON = 150
FOCAL = 300.0
YAW = math.radians(12.0)
# Ground directions in the image: forward (longitudinal) and across (transverse).
VP_FORWARD = (WIDTH / 2.0 + FOCAL * math.tan(YAW), HORIZON)
VP_LATERAL = (WIDTH / 2.0 - FOCAL / math.tan(YAW), HORIZON)
WALL_BASE = 170


def ground_scene(horizon: float = HORIZON, focal: float = FOCAL,
                 noise: float = 3.0, yaw_degrees: float = 12.0,
                 wall_base: float = None) -> np.ndarray:
    """Perspective floor with two converging line families and lane markings.

    With ``wall_base`` the upper part becomes a vertical face and a bright junction
    line is drawn at that row, which is the ground/vertical-surface boundary the floor
    top edge must follow.
    """

    yaw = math.radians(yaw_degrees)
    forward = (WIDTH / 2.0 + focal * math.tan(yaw), horizon)
    lateral = (WIDTH / 2.0 - focal / math.tan(yaw), horizon)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    rows = np.arange(HEIGHT)[:, None]
    ground = np.clip(70 + (rows - horizon) * 0.18, 0, 255).astype(np.uint8)
    sky = np.clip(180 - (horizon - rows) * 0.1, 0, 255).astype(np.uint8)
    shade = np.where(rows < horizon, sky, ground)
    if wall_base is not None:
        face = np.clip(130 - (rows - horizon) * 0.06, 0, 255).astype(np.uint8)
        shade = np.where(rows < wall_base, face, ground)
    image[:] = np.repeat(shade, WIDTH, axis=1)[:, :, None].repeat(3, axis=2)
    if wall_base is not None:
        cv2.line(image, (0, wall_base), (WIDTH - 1, wall_base), (252, 252, 252), 4, cv2.LINE_AA)
        cv2.line(image, (0, wall_base + 6), (WIDTH - 1, wall_base + 6), (25, 25, 25), 4, cv2.LINE_AA)
    for start in range(-160, WIDTH + 160, 46):
        cv2.line(image, (start, HEIGHT - 1),
                 (int(round(forward[0])), int(round(horizon))), (238, 238, 238), 3, cv2.LINE_AA)
    for depth in (2.0, 2.6, 3.4, 4.4, 5.8, 7.6, 10.0, 13.0, 17.0, 22.0):
        row = horizon + focal / depth
        if row >= HEIGHT - 1:
            continue
        slope = (row - horizon) / (0.0 - lateral[0])
        cv2.line(image, (0, int(round(row))),
                 (WIDTH - 1, int(round(row + slope * (WIDTH - 1)))), (150, 150, 150), 2, cv2.LINE_AA)
    rng = np.random.default_rng(11)
    return np.clip(image.astype(np.float64) + rng.normal(0.0, noise, image.shape), 0, 255).astype(np.uint8)


def noise_scene(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def angle_from_centre(point, truth, centre=(WIDTH / 2.0, HEIGHT / 2.0)) -> float:
    left = np.array(point, dtype=float) - np.array(centre, dtype=float)
    right = np.array(truth, dtype=float) - np.array(centre, dtype=float)
    cosine = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def inside(polygon, point) -> float:
    contour = np.array(polygon, np.int32).reshape(-1, 1, 2)
    return float(cv2.pointPolygonTest(contour, (int(point[0]), int(point[1])), False))


def quad_checks(patch) -> dict:
    """Shape facts about a patch quad, computed independently of the estimator."""

    points = [(float(x), float(y)) for x, y in patch]
    crosses = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    tiny = 1e-9
    area = abs(sum(points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
                   for i in range(4))) / 2.0

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - first[1])
                - (first[1] - origin[1]) * (second[0] - first[0]))

    def proper(first, second, third, fourth):
        left = cross(first, second, third)
        right = cross(first, second, fourth)
        up = cross(third, fourth, first)
        down = cross(third, fourth, second)
        return left * right < 0 and up * down < 0

    return {
        "convex": all(value > tiny for value in crosses) or all(value < -tiny for value in crosses),
        "collinear": any(abs(value) <= tiny for value in crosses),
        "self_intersecting": (proper(points[0], points[1], points[2], points[3])
                              or proper(points[1], points[2], points[3], points[0])),
        "clockwise": area > 0.0,
        "area": area,
        "top_width": math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]),
        "bottom_width": math.hypot(points[2][0] - points[3][0], points[2][1] - points[3][1]),
        "left_height": math.hypot(points[3][0] - points[0][0], points[3][1] - points[0][1]),
        "right_height": math.hypot(points[1][0] - points[2][0], points[1][1] - points[2][1]),
    }


class VanishingPointTest(unittest.TestCase):
    def test_two_ground_directions_give_both_vanishing_points(self):
        plane = geometry.estimate_ground_plane(ground_scene())
        self.assertTrue(plane.ok, plane.reason_codes)
        self.assertEqual(len(plane.vanishing_points), 2)
        truth = [VP_FORWARD, VP_LATERAL]
        for estimate in plane.vanishing_points:
            with self.subTest(vp=estimate):
                self.assertLessEqual(min(angle_from_centre(estimate, point) for point in truth), 5.0)
        for point in truth:
            with self.subTest(truth=point):
                self.assertLessEqual(min(angle_from_centre(estimate, point)
                                         for estimate in plane.vanishing_points), 5.0)
        self.assertIsNotNone(plane.horizon_y)
        self.assertLessEqual(abs(float(plane.horizon_y) - HORIZON), 6.0)
        self.assertGreaterEqual(plane.residual, 0.0)
        self.assertLessEqual(plane.residual, 1.0)

    def test_horizon_row_holds_for_other_camera_rows(self):
        for horizon in (110, 150, 200):
            with self.subTest(horizon=horizon):
                plane = geometry.estimate_ground_plane(ground_scene(horizon=horizon))
                self.assertTrue(plane.ok, plane.reason_codes)
                self.assertLessEqual(abs(float(plane.horizon_y) - horizon), 6.0)

    def test_homography_is_invertible_and_maps_the_ground(self):
        plane = geometry.estimate_ground_plane(ground_scene())
        matrix = np.array(plane.homography, dtype=np.float64)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertGreater(abs(float(np.linalg.det(matrix))), 1e-9)
        inverse = np.linalg.inv(matrix)
        point = np.array([[[120.0, 300.0]]], np.float64)
        back = cv2.perspectiveTransform(cv2.perspectiveTransform(point, matrix), inverse)
        self.assertAlmostEqual(float(back[0, 0, 0]), 120.0, places=6)
        self.assertAlmostEqual(float(back[0, 0, 1]), 300.0, places=6)


class FloorAndPatchTest(unittest.TestCase):
    def test_floor_region_spans_the_frame_below_the_horizon(self):
        plane = geometry.estimate_ground_plane(ground_scene())
        self.assertEqual(len(plane.floor_region), 4)
        top_left, top_right, bottom_right, bottom_left = plane.floor_region
        self.assertEqual(top_left[0], 0)
        self.assertEqual(top_right[0], WIDTH - 1)
        self.assertEqual(bottom_left[1], HEIGHT - 1)
        self.assertEqual(bottom_right[1], HEIGHT - 1)
        self.assertGreater(top_left[1], HORIZON)
        self.assertGreater(top_right[1], HORIZON)
        self.assertLess(max(top_left[1], top_right[1]), HEIGHT - 1)

    def test_floor_top_follows_the_wall_junction(self):
        plane = geometry.estimate_ground_plane(ground_scene(wall_base=WALL_BASE))
        self.assertTrue(plane.ok, plane.reason_codes)
        top_left, top_right = plane.floor_region[:2]
        for corner in (top_left, top_right):
            with self.subTest(corner=corner):
                self.assertLessEqual(abs(corner[1] - WALL_BASE), 8)

    def test_placement_patch_is_a_clockwise_convex_rectangle(self):
        plane = geometry.estimate_ground_plane(ground_scene())
        patch = plane.placement_patch
        self.assertEqual(len(patch), 4, plane.reason_codes)
        facts = quad_checks(patch)
        self.assertTrue(facts["convex"], facts)
        self.assertFalse(facts["collinear"], facts)
        self.assertFalse(facts["self_intersecting"], facts)
        self.assertTrue(facts["clockwise"], facts)
        self.assertGreaterEqual(facts["area"], max(1200.0, 0.0025 * HEIGHT * WIDTH))
        for first, second in (("top_width", "bottom_width"), ("left_height", "right_height")):
            ratio = facts[first] / facts[second]
            self.assertGreaterEqual(ratio, 0.25, facts)
            self.assertLessEqual(ratio, 4.0, facts)
        for corner in patch:
            with self.subTest(corner=corner):
                self.assertGreaterEqual(inside(plane.floor_region, corner), 0)
                self.assertGreaterEqual(corner[1], float(plane.horizon_y))
                self.assertTrue(0 <= corner[0] <= WIDTH - 1 and 0 <= corner[1] <= HEIGHT - 1)

    def test_placement_patch_stays_below_the_horizon_for_the_declared_target(self):
        plane = geometry.estimate_ground_plane(ground_scene())
        top_left, top_right, _, _ = plane.placement_patch
        minimum = HORIZON + geometry.DEFAULT_PARAMS["target_height_fraction"] * HEIGHT
        self.assertGreaterEqual(min(top_left[1], top_right[1]), minimum - 2.0)

    def test_larger_target_never_grows_the_patch(self):
        scene = ground_scene()
        small = geometry.estimate_ground_plane(scene, params={"target_height_fraction": 0.20})
        default = geometry.estimate_ground_plane(scene)
        large = geometry.estimate_ground_plane(scene, params={"target_height_fraction": 0.45})

        def area(plane):
            return geometry.polygon_area_ratio(plane.placement_patch, (HEIGHT, WIDTH)) if plane.placement_patch else 0.0

        self.assertGreaterEqual(area(small), area(default))
        self.assertGreaterEqual(area(default), area(large))

    def test_patch_disappears_when_the_target_cannot_fit(self):
        # A target wider than the frame can never be placed, whatever the floor span.
        plane = geometry.estimate_ground_plane(ground_scene(), params={"target_aspect": 40.0})
        self.assertTrue(plane.ok)
        self.assertEqual(plane.placement_patch, [])
        self.assertIn("geometry_rejected", plane.reason_codes)

    def test_crossed_quad_is_rejected_by_the_acceptance_rule(self):
        plane = geometry.estimate_ground_plane(ground_scene())
        bow_tie = [[100, 300], [400, 340], [110, 340], [390, 300]]
        problems = geometry._patch_problems(bow_tie, plane.floor_region,
                                            np.array([0.0, 1.0, -HORIZON]), (HEIGHT, WIDTH))
        self.assertTrue(problems, "a crossed quad must not be accepted")


class LevelHorizonTest(unittest.TestCase):
    def test_tilted_vanishing_point_pair_is_replaced_by_a_level_row(self):
        # Two ground directions whose vanishing points sit on different rows put the
        # line through them at ~8 degrees; the camera is mounted level, so the reported
        # horizon must still be level.
        scene = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        scene[:HORIZON] = (178, 178, 178)
        scene[HORIZON:] = (72, 72, 72)
        forward = (int(round(VP_FORWARD[0])), HORIZON)
        transverse = (-300, 250)
        for start in range(-160, WIDTH + 160, 46):
            cv2.line(scene, (start, HEIGHT - 1), forward, (238, 238, 238), 3, cv2.LINE_AA)
        for anchor in range(-800, 1200, 90):
            cv2.line(scene, (anchor, HEIGHT - 1), transverse, (150, 150, 150), 2, cv2.LINE_AA)
        rng = np.random.default_rng(11)
        scene = np.clip(scene.astype(np.float64) + rng.normal(0.0, 3.0, scene.shape), 0, 255).astype(np.uint8)
        plane = geometry.estimate_ground_plane(scene)
        self.assertTrue(plane.ok, plane.reason_codes)
        self.assertIn("horizon_forced_horizontal", plane.reason_codes)
        top_left, top_right = plane.floor_region[:2]
        slope = math.degrees(math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
        self.assertLessEqual(abs(slope), geometry.DEFAULT_PARAMS["horizon_max_slope_deg"])
        self.assertLessEqual(abs(top_left[1] - top_right[1]), 0.02 * HEIGHT)

    def test_painted_wall_foot_line_sets_the_floor_top(self):
        plane = geometry.estimate_ground_plane(ground_scene(wall_base=WALL_BASE))
        self.assertTrue(plane.ok, plane.reason_codes)
        top_left, top_right = plane.floor_region[:2]
        for corner in (top_left, top_right):
            with self.subTest(corner=corner):
                self.assertLessEqual(abs(corner[1] - WALL_BASE), 8)

    def test_frame_without_bright_line_or_transverse_family_never_gets_a_sloped_horizon(self):
        # Only longitudinal lines: no transverse family, so no horizon row can be
        # measured. The estimate must be rejected rather than guessed.
        scene = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        scene[:] = (60, 60, 60)
        scene[:HORIZON] = (170, 170, 170)
        for start in range(0, WIDTH, 40):
            cv2.line(scene, (start, HEIGHT - 1), (WIDTH // 2, HORIZON), (235, 235, 235), 2, cv2.LINE_AA)
        plane = geometry.estimate_ground_plane(scene)
        if plane.ok:
            top_left, top_right = plane.floor_region[:2]
            slope = math.degrees(math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0]))
            self.assertLessEqual(abs(slope), geometry.DEFAULT_PARAMS["horizon_max_slope_deg"])
            self.assertLessEqual(abs(top_left[1] - top_right[1]), 0.02 * HEIGHT)
        else:
            self.assertIn("geometry_rejected", plane.reason_codes)

    def test_slope_limit_is_a_parameter(self):
        scene = ground_scene()
        tight = geometry.estimate_ground_plane(scene, params={"horizon_max_slope_deg": 0.01})
        self.assertTrue(tight.ok, tight.reason_codes)
        top_left, top_right = tight.floor_region[:2]
        self.assertLessEqual(abs(top_left[1] - top_right[1]), 0.02 * HEIGHT)
        with self.assertRaises(ValueError):
            geometry.estimate_ground_plane(scene, params={"horizon_max_slope_deg": 0.0})


class DegenerateInputTest(unittest.TestCase):
    def test_noise_has_no_ground_plane(self):
        plane = geometry.estimate_ground_plane(noise_scene())
        self.assertFalse(plane.ok)
        self.assertEqual(plane.reason_codes, ["geometry_rejected"])
        self.assertEqual(plane.floor_region, [])
        self.assertEqual(plane.placement_patch, [])
        self.assertEqual(plane.vanishing_points, [])
        self.assertIsNone(plane.homography)
        self.assertIsNone(plane.horizon_y)

    def test_diagonal_band_gives_a_legal_rectangle_or_nothing(self):
        # A steep yaw turns the usable feet into a diagonal band in ground coordinates,
        # which is where a per-axis extreme outline used to cross itself.
        for yaw in (20.0, 30.0, 40.0):
            with self.subTest(yaw=yaw):
                plane = geometry.estimate_ground_plane(ground_scene(yaw_degrees=yaw))
                patch = plane.placement_patch
                if not patch:
                    self.assertIn("geometry_rejected", plane.reason_codes)
                    continue
                facts = quad_checks(patch)
                self.assertFalse(facts["self_intersecting"], facts)
                self.assertTrue(facts["convex"], facts)
                self.assertTrue(facts["clockwise"], facts)
                self.assertGreaterEqual(facts["area"], max(1200.0, 0.0025 * HEIGHT * WIDTH))

    def test_flat_and_tiny_images_are_rejected(self):
        for image in (np.zeros((HEIGHT, WIDTH, 3), np.uint8),
                      np.full((HEIGHT, WIDTH, 3), 128, np.uint8),
                      np.zeros((16, 16, 3), np.uint8)):
            with self.subTest(shape=image.shape):
                plane = geometry.estimate_ground_plane(image)
                self.assertFalse(plane.ok)
                self.assertIn("geometry_rejected", plane.reason_codes)

    def test_unknown_and_invalid_parameters_raise(self):
        scene = ground_scene()
        with self.assertRaises(ValueError):
            geometry.estimate_ground_plane(scene, params={"bottom_fraction": 0.3})
        with self.assertRaises(ValueError):
            geometry.estimate_ground_plane(scene, params={"target_aspect": 0.0})
        with self.assertRaises(ValueError):
            geometry.estimate_ground_plane(scene, params={"ransac_iters": -1})

    def test_same_parameters_reproduce_the_same_plane(self):
        scene = ground_scene()
        first = geometry.estimate_ground_plane(scene)
        second = geometry.estimate_ground_plane(scene)
        self.assertEqual(first.floor_region, second.floor_region)
        self.assertEqual(first.placement_patch, second.placement_patch)
        self.assertEqual(first.vanishing_points, second.vanishing_points)
        another = geometry.estimate_ground_plane(scene, params={"seed": 7, "ransac_iters": 200})
        self.assertEqual(another.ok, first.ok)


class CompatibilityTest(unittest.TestCase):
    def test_region_record_maps_the_plane_record(self):
        scene = ground_scene()
        plane = geometry.estimate_ground_plane(scene)
        region = geometry.estimate_ground_region(scene)
        self.assertIsInstance(region, GroundObservation)
        self.assertIsInstance(plane, GroundPlaneObservation)
        self.assertEqual(region.ok, plane.ok)
        self.assertEqual(region.polygon, plane.floor_region)
        self.assertEqual(region.horizon_y, plane.horizon_y)
        self.assertEqual(region.residual, plane.residual)
        self.assertEqual(region.reason_codes, plane.reason_codes)
        bottom_middle = geometry.foot_from_polygon(plane.placement_patch, 0.5)
        self.assertEqual(region.foot_xy, bottom_middle)

    def test_region_record_rejects_like_the_plane_record(self):
        region = geometry.estimate_ground_region(noise_scene())
        self.assertFalse(region.ok)
        self.assertEqual(region.polygon, [])
        self.assertIsNone(region.foot_xy)
        self.assertEqual(region.reason_codes, ["geometry_rejected"])


class HelperTest(unittest.TestCase):
    def test_foot_sits_on_the_bottom_edge(self):
        polygon = [[100, 200], [300, 200], [380, 340], [20, 340]]
        self.assertEqual(geometry.foot_from_polygon(polygon), [200, 340])
        self.assertEqual(geometry.foot_from_polygon(polygon, 0.0), [20, 340])
        self.assertEqual(geometry.foot_from_polygon(polygon, 1.0), [380, 340])
        self.assertIsNone(geometry.foot_from_polygon([]))
        self.assertIsNone(geometry.foot_from_polygon([[0, 0], [1, 1]]))

    def test_area_ratio_uses_the_shoelace_formula(self):
        frame = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]
        self.assertAlmostEqual(geometry.polygon_area_ratio(frame, (HEIGHT, WIDTH)), 1.0, places=6)
        self.assertEqual(geometry.polygon_area_ratio([], (HEIGHT, WIDTH)), 0.0)
        self.assertEqual(geometry.polygon_area_ratio([[0, 0], [10, 0]], (HEIGHT, WIDTH)), 0.0)
        with self.assertRaises(ValueError):
            geometry.polygon_area_ratio(frame, (0, WIDTH))


if __name__ == "__main__":
    unittest.main()
