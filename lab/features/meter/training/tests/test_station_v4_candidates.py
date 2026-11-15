"""Check 640x480 station projections and candidate-only outputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import uuid

from PIL import Image


ROOT = Path(__file__).resolve().parents[5]
BLENDER = os.environ.get('BLENDER_EXECUTABLE') or shutil.which('blender')
SCENE = ROOT/'datasets/assets/blender/meter/blender_9011_station_v4_tilt65_640_preview/meter_9011_power_meter_station.blend'
REFERENCE = SCENE.parent/'prototype_manifest.json'
SCRIPT = ROOT/'datasets/scripts/meter/render_meter_candidates.py'
ENDPOINTS = 'lab/features/meter/training/config/generated_face_endpoints_v1.json'
OUTER_SCENE = ROOT/'datasets/assets/blender/meter/blender_9011_station_v4_outer_tick_preview_20261112/meter_9011_power_meter_station.blend'
OUTER_ENDPOINTS = 'lab/features/meter/training/config/generated_face_endpoints_v2.json'
KEYPOINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
             'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')


@unittest.skipUnless(BLENDER and SCENE.is_file(), 'Blender and local station prototype are required')
class StationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.output = 'runs/meter/test_station_preflight_' + uuid.uuid4().hex

    def run_preflight(self, *options):
        command = [str(BLENDER), '--background', str(SCENE), '--python-exit-code', '1',
                   '--python', str(SCRIPT), '--', '--station-v4', '--preflight', '--count', '2',
                   '--device', 'CPU', '--samples', '1', '--output', self.output,
                   '--endpoints', ENDPOINTS, *options]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=40,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def test_station_transform_matches_saved_prototype_projection(self):
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith('STATION_V4_PREFLIGHT_OK '))
        actual = json.loads(line.split(' ', 1)[1])
        expected = json.loads(REFERENCE.read_text(encoding='utf-8'))['previews'][0]
        self.assertEqual(actual['resolution'], [640, 480])
        self.assertAlmostEqual(actual['ground_angle_deg'], 65.0, places=3)
        for index, value in enumerate(actual['bbox_xyxy']):
            self.assertAlmostEqual(value * (640 if index % 2 == 0 else 480), expected['dial_rim_xyxy_pixels'][index], places=3)
        for name in KEYPOINTS:
            for actual_value, expected_value in zip(actual['keypoints'][name][:2], expected['dial_points_xy_normalized'][name]):
                self.assertAlmostEqual(actual_value, expected_value, places=5)
        for point, reference in zip(actual['display_quad'], expected['screen_roi_xy_normalized']):
            for actual_value, expected_value in zip(point[:2], reference):
                self.assertAlmostEqual(actual_value, expected_value, places=5)
        self.assertFalse((ROOT/self.output).exists())

    def test_station_rejects_resolution_and_camera_overrides_outside_contract(self):
        for options in (('--resolution', '1280'), ('--station-camera-z', '3')):
            with self.subTest(options=options):
                result = self.run_preflight(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((ROOT/self.output).exists())

    def test_fitted_near_view_is_native_resolution_and_larger(self):
        result = self.run_preflight('--station-camera-z', '9.65')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith('STATION_V4_PREFLIGHT_OK '))
        value = json.loads(line.split(' ', 1)[1])
        width = (value['bbox_xyxy'][2] - value['bbox_xyxy'][0]) * 640
        self.assertEqual(value['resolution'], [640, 480])
        self.assertGreater(width, 120)
        self.assertLess(width, 145)

    def test_room_context_is_bounded_and_keeps_landmarks_in_frame(self):
        bounded = self.run_preflight('--station-room-v4', '--station-camera-z', '9.65')
        self.assertEqual(bounded.returncode, 0, bounded.stdout + bounded.stderr)
        line = next(line for line in bounded.stdout.splitlines() if line.startswith('STATION_V4_PREFLIGHT_OK '))
        value = json.loads(line.split(' ', 1)[1])
        self.assertEqual(set(value['keypoints']), set(KEYPOINTS))
        self.assertTrue(all(0 <= coordinate <= 1 for p in value['keypoints'].values() for coordinate in p[:2]))
        rejected = self.run_preflight('--station-room-v4', '--count', '3')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((ROOT/self.output).exists())


@unittest.skipUnless(BLENDER and OUTER_SCENE.is_file(), 'Blender and the local outer-tick station prototype are required')
class StationOuterTickTests(unittest.TestCase):
    def setUp(self):
        self.output = 'runs/meter/test_station_outer_preflight_' + uuid.uuid4().hex

    def run_preflight(self, endpoint_path, station_path=None):
        station = str(OUTER_SCENE) if station_path is None else station_path
        command = [str(BLENDER), '--background', str(OUTER_SCENE), '--python-exit-code', '1',
                   '--python', str(SCRIPT), '--', '--station-v4', '--preflight', '--count', '1',
                   '--device', 'CPU', '--samples', '1', '--output', self.output,
                   '--station-source', station, '--endpoints', endpoint_path]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=40,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def test_matching_calibration_matches_saved_prototype(self):
        result = self.run_preflight(OUTER_ENDPOINTS)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith('STATION_V4_PREFLIGHT_OK '))
        actual = json.loads(line.split(' ', 1)[1])
        expected = json.loads((OUTER_SCENE.parent/'prototype_manifest.json').read_text(encoding='utf-8'))['previews'][0]
        self.assertEqual(actual['resolution'], [640, 480])
        self.assertAlmostEqual(actual['ground_angle_deg'], 65.0, places=3)
        for name in KEYPOINTS:
            for value, reference in zip(actual['keypoints'][name][:2], expected['dial_points_xy_normalized'][name]):
                self.assertAlmostEqual(value, reference, places=5)
        self.assertFalse((ROOT/self.output).exists())

    def test_old_calibration_and_unrelated_station_are_rejected(self):
        old = self.run_preflight(ENDPOINTS)
        self.assertNotEqual(old.returncode, 0)
        self.assertIn('station landmark marker parent or coordinates differ from calibration', old.stdout + old.stderr)
        wrong_station = self.run_preflight(OUTER_ENDPOINTS, str(ROOT/'README.md'))
        self.assertNotEqual(wrong_station.returncode, 0)
        self.assertIn('station-v4 source must be an existing blend under meter assets or annotations',
                      wrong_station.stdout + wrong_station.stderr)
        self.assertFalse((ROOT/self.output).exists())


@unittest.skipUnless(os.environ.get('AEC_METER_V4_OUTPUT'), 'A locally rendered candidate directory is required')
class StationRenderedPairTests(unittest.TestCase):
    def test_pair_has_native_images_and_unknown_visibility(self):
        path = (ROOT / os.environ['AEC_METER_V4_OUTPUT']).resolve()
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'complete_candidates')
        self.assertEqual(manifest['completed_count'], 2)
        self.assertEqual(manifest['resolution'], [640, 480])
        self.assertEqual(manifest['device'], 'CPU')
        records = []
        for item in manifest['samples']:
            with Image.open(path/item['image']) as image:
                self.assertEqual(image.size, (640, 480))
                image.verify()
            record = json.loads((path/item['annotation']).read_text(encoding='utf-8'))
            self.assertEqual(record['annotation_status'], 'candidate')
            self.assertEqual(record['split'], 'unassigned')
            self.assertEqual(record['keypoint_visibility'], {name: 'unknown' for name in KEYPOINTS})
            self.assertEqual(record['display_roi_visibility'], 'unknown')
            self.assertEqual(len(record['display_roi_quad_xy_depth_normalized_candidate']), 4)
            self.assertEqual(record['image_processing']['resize'], 'none; native 640x480 render')
            records.append(record)
        self.assertLess(records[0]['camera_offset_x'], records[1]['camera_offset_x'])
        self.assertGreater(records[0]['keypoints_xy_depth_normalized_candidate']['temperature_pivot'][0],
                           records[1]['keypoints_xy_depth_normalized_candidate']['temperature_pivot'][0])
        self.assertFalse(list(path.rglob('*.txt')), 'Candidate output must not contain YOLO labels')


@unittest.skipUnless(os.environ.get('AEC_METER_V4_OUTER_OUTPUT'), 'A locally rendered outer-tick pair is required')
class StationOuterTickRenderedTests(unittest.TestCase):
    def test_reviewed_pair_is_regression_and_keeps_visibility_unknown(self):
        path = (ROOT / os.environ['AEC_METER_V4_OUTER_OUTPUT']).resolve()
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'complete_candidates')
        self.assertEqual(manifest['completed_count'], 2)
        self.assertEqual(manifest['split'], 'regression')
        self.assertEqual(manifest['endpoint_config'], OUTER_ENDPOINTS)
        self.assertEqual(manifest['resolution'], [640, 480])
        for item in manifest['samples']:
            with Image.open(path/item['image']) as image:
                self.assertEqual(image.size, (640, 480))
                image.verify()
            with Image.open(path/item['overlay']) as overlay:
                self.assertEqual(overlay.size, (640, 480))
                overlay.verify()
            sample = json.loads((path/item['annotation']).read_text(encoding='utf-8'))
            self.assertEqual(sample['split'], 'regression')
            self.assertEqual(sample['keypoint_visibility'], {name: 'unknown' for name in KEYPOINTS})
            self.assertEqual(sample['display_roi_visibility'], 'unknown')
        self.assertFalse(list(path.rglob('*.txt')), 'Regression candidates must not contain YOLO labels')


if __name__ == '__main__':
    unittest.main()
