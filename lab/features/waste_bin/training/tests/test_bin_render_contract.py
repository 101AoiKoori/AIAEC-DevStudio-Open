import importlib.util
import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/waste_bin/training/scripts/bin_render_contract.py'
spec = importlib.util.spec_from_file_location('bin_render_contract', SCRIPT)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class MouthContrastTest(unittest.TestCase):
    def test_dark_opening_passes_bright_or_missing_opening_fails(self):
        body = np.ones((10, 10), dtype=bool)
        mouth = np.zeros((10, 10), dtype=bool)
        mouth[:5] = True
        rgb = np.full((10, 10, 4), .8)
        self.assertFalse(contract.mouth_pixel_contrast(rgb, mouth, body)['passed'])
        rgb[:5, :, :3] = .1
        self.assertTrue(contract.mouth_pixel_contrast(rgb, mouth, body)['passed'])
        self.assertFalse(contract.mouth_pixel_contrast(rgb, np.zeros_like(mouth), body)['passed'])


def mock_materials():
    result = {}
    for name in contract.BIN_NAMES:
        base = SimpleNamespace(default_value=[.006, .035, .48, 1.], is_linked=False)
        rough = SimpleNamespace(default_value=.18, is_linked=False)
        node = SimpleNamespace(inputs={'Base Color': base, 'Roughness': rough})
        result[name + ' adjustable body color'] = SimpleNamespace(
            diffuse_color=(.012, .07, .8, 1), node_tree=SimpleNamespace(nodes={'Principled BSDF': node}))
    return result


class BinRenderContractTest(unittest.TestCase):
    def test_standalone_plan_rejects_unimplemented_and_duplicate_states(self):
        state = {'id': 'a', 'conditions': [], 'planned_count': 1,
                 'sources': ['bin.blend'], 'background': {'scene': 'bin.blend', 'floor_gray': .1, 'backdrop_gray': .3},
                 'target': {'bins': {'blue': 'open'}, 'open_pose': 'hanging_back'},
                 'labels': {'mode': 'candidate', 'open': 0, 'closed': 1}, 'augmentation': {'mode': 'none'},
                 'parameters': {'camera_location': [0, -6, 3], 'camera_aim': [0, 0, 1],
                                'yaw_degrees': 0, 'lens_mm': 35, 'key_energy': 800, 'fill_energy': 400,
                                'color_factor': 1, 'roughness_delta': 0}}
        other = copy.deepcopy(state)
        other['id'] = 'b'
        other['target']['bins']['blue'] = 'closed'
        plan = {'model_key': 'waste_bin_state', 'budget': 2,
                'constraints': {'resolution': [640, 480], 'label_mode': 'candidate',
                                'min_visibility': .7, 'min_visible_pixels': 24},
                'strata': [state, other]}
        self.assertEqual(len(contract.standalone_planned_jobs(plan, 'bin.blend', 640, 480, .7, 24)), 2)
        other['target']['bins']['blue'] = 'open'
        other['parameters']['camera_location'][0] = 0.0
        with self.assertRaises(ValueError):
            contract.standalone_planned_jobs(plan, 'bin.blend', 640, 480, .7, 24)
        other['target']['bins']['blue'] = 'closed'
        state['planned_count'] = 2
        with self.assertRaises(ValueError):
            contract.standalone_planned_jobs(plan, 'bin.blend', 640, 480, .7, 24)
        state['planned_count'] = 1
        state['target']['bins'] = {}
        other['target']['bins'] = {}
        other['target']['open_pose'] = 'upright'
        other['parameters']['color_factor'] = .9
        with self.assertRaises(ValueError):
            contract.standalone_planned_jobs(plan, 'bin.blend', 640, 480, .7, 24)
        state['augmentation'] = {'mode': 'blur'}
        with self.assertRaises(ValueError):
            contract.standalone_planned_jobs(plan, 'bin.blend', 640, 480, .7, 24)

    def test_arena_plan_drives_quotas_and_rejects_unsupported_states(self):
        state = {'id': 'open', 'conditions': [], 'planned_count': 1,
                 'background': {'scene': 'arena.blend'}, 'sources': ['arena.blend', 'bin.blend'],
                 'target': {'bins': {'blue': 'open', 'red': 'closed'}, 'open_pose': 'upright'},
                 'labels': {'mode': 'candidate', 'open': 0, 'closed': 1},
                 'augmentation': {'exposure_ev': -.5, 'saturation': .8},
                 'parameters': {'bin_location': [0, 1, 0], 'camera_location': [0, .64, .18],
                                'camera_aim': [0, 1, .075]}}
        second = copy.deepcopy(state)
        second['id'] = 'closed'
        second['target']['bins']['blue'] = 'closed'
        plan = {'model_key': 'waste_bin_state', 'budget': 2,
                'constraints': {'resolution': [640, 480], 'label_mode': 'candidate'}, 'strata': [state, second]}
        self.assertEqual([x['id'] for x in contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')],
                         ['open', 'closed'])
        second['target']['bins']['blue'] = 'open'
        second['parameters']['bin_location'][0] = 0.0
        with self.assertRaises(ValueError):
            contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')
        second['target']['bins']['blue'] = 'closed'
        state['planned_count'] = 2
        with self.assertRaises(ValueError):
            contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')
        state['planned_count'] = 1
        state['target']['bins'] = {}
        second['target']['bins'] = {}
        second['parameters']['bin_location'][0] = 1
        with self.assertRaises(ValueError):
            contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')
        state['target']['bins'] = {'blue': 'open', 'red': 'closed'}
        second['target']['bins'] = {'blue': 'closed', 'red': 'closed'}
        state['augmentation']['blur'] = 1
        with self.assertRaises(ValueError):
            contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')
        del state['augmentation']['blur']
        state['target']['open_pose'] = 'hanging_back'
        with self.assertRaises(ValueError):
            contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')
        state['target']['open_pose'] = 'upright'
        plan['budget'] = 3
        with self.assertRaises(ValueError):
            contract.arena_planned_jobs(plan, 'arena.blend', 'bin.blend')

    def test_pbr_baseline_uses_shader_and_does_not_drift(self):
        materials = mock_materials()
        baseline = contract.capture_material_baseline(materials)
        materials['blue adjustable body color'].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value[0] = .9
        self.assertEqual(baseline['blue']['base_color'], (.006, .035, .48))
        self.assertEqual(contract.vary_material(baseline['blue'], 1.0, 0.0), ((.006, .035, .48), .18))
        self.assertNotEqual(baseline['blue']['base_color'][0], materials['blue adjustable body color'].diffuse_color[0])
        materials['red adjustable body color'].node_tree.nodes['Principled BSDF'].inputs['Roughness'].is_linked = True
        with self.assertRaises(ValueError):
            contract.capture_material_baseline(materials)

    def test_masks_produce_bounding_box_and_reject_clipping_or_occlusion(self):
        rgba = np.zeros((10, 20, 4), dtype=float)
        rgba[2:8, 3:12] = [1, 0, 0, 1]
        visible = contract.decode_mask(rgba, 'blue')
        solo = visible.copy()
        result = contract.measure_visibility(visible, solo, True, min_pixels=1)
        self.assertTrue(result['label_eligible'])
        self.assertEqual(result['candidate_xyxy_normalized'], [3/20, 2/10, 12/20, 8/10])
        clipped = contract.measure_visibility(visible, solo, False, min_pixels=1)
        self.assertFalse(clipped['label_eligible'])
        visible[:, 5:] = False
        occluded = contract.measure_visibility(visible, solo, True, min_pixels=1)
        self.assertIn('occluded_below_threshold', occluded['rejection_reasons'])
        self.assertTrue(math.isfinite(occluded['visible_fraction']))

    def test_projection_requires_all_points_in_front_and_inside_margin(self):
        self.assertTrue(contract.projected_points_fit([(.1, .1, 1), (.9, .9, 2)], .05))
        self.assertFalse(contract.projected_points_fit([(.01, .5, 1)], .05))
        self.assertFalse(contract.projected_points_fit([(.5, .5, 0)], .05))
        self.assertFalse(contract.projected_points_fit([(.5, float('nan'), 1)], .05))
        self.assertFalse(contract.projected_points_fit([], .05))

    def test_whole_bin_and_state_evidence_are_both_required(self):
        mask = np.ones((8, 8), dtype=bool)
        target = contract.measure_visibility(mask, mask, True, min_pixels=1)
        hidden = contract.measure_visibility(np.zeros_like(mask), mask, True, min_pixels=1)
        open_item = contract.attach_state_evidence(target, hidden, 'open')
        self.assertFalse(open_item['label_eligible'])
        self.assertEqual(open_item['state_evidence']['kind'], 'mouth_proxy')
        self.assertIn('state_evidence:occluded_below_threshold', open_item['rejection_reasons'])
        closed_item = contract.attach_state_evidence(target, target, 'closed')
        self.assertTrue(closed_item['label_eligible'])
        self.assertEqual(closed_item['state_evidence']['kind'], 'lid')
        self.assertEqual(contract.OPEN_POSES['hanging_back'], -270)
        self.assertEqual(contract.OPEN_POSES['upright'], -90)

    def test_hinge_shift_clears_wall_and_rejects_unexpected_geometry(self):
        self.assertAlmostEqual(contract.required_hinge_shift(.37, .31), .075)
        self.assertEqual(contract.required_hinge_shift(.37, .40), 0)
        with self.assertRaises(ValueError):
            contract.required_hinge_shift(.37, 0)
        with self.assertRaises(ValueError):
            contract.required_hinge_shift(float('nan'), .31)

    def test_color_codes_and_balanced_pairs(self):
        image = np.zeros((1, 4, 4))
        for idx, name in enumerate(contract.BIN_NAMES):
            image[0, idx] = [*contract.MASK_COLORS[name], 1]
        for idx, name in enumerate(contract.BIN_NAMES):
            self.assertEqual(np.flatnonzero(contract.decode_mask(image, name)).tolist(), [idx])
        for count in (2, 4, 6, 8, 10, 12, 14, 16):
            combos = contract.combinations(count)
            for bit in range(4):
                self.assertEqual(sum((c >> bit) & 1 for c in combos), count // 2)


if __name__ == '__main__':
    unittest.main()
