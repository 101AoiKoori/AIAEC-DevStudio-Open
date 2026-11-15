"""Read-only current-source plans and in-memory geometry contracts."""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.common.config import load_task_config
from datasets.scripts.sign import compose_sign_v4_measured as sign
from datasets.scripts.waste_bin_state.bin_render_contract import standalone_planned_jobs
from datasets.scripts.waste_bin_state import plan_current_sources as bin_planner
from datasets.scripts.waste_sticker import generate_sticker_real_photo_v4 as surface
from datasets.scripts.waste_sticker.package_real_stickers import prepare_inputs
from datasets.scripts.waste_sticker.source_inputs import real_source_catalog, surface_background_catalog

ROOT = Path(__file__).resolve().parents[3]


def config_plan(task, name='production_plan'):
    _, sources = load_task_config(task)
    return sources, json.loads((ROOT/sources[name]).read_text(encoding='utf8'))


class CurrentSourcePlans(unittest.TestCase):
    def test_sign_400_includes_ten_backgrounds_and_designed_pole_crops(self):
        sources, plan = config_plan('sign')
        self.assertEqual(plan_production(plan, root=ROOT)['budget'], 400)
        self.assertEqual(len({s['background']['image'] for s in plan['strata']}), 10)
        self.assertEqual(sum(s['parameters'].get('retain_lower_half', False) for s in plan['strata']), 32)
        masks = json.loads((sign.base.ANNOTATED/'manifest.json').read_text(encoding='utf8'))
        self.assertEqual(len(sign._prepare(plan, {}, plan['background_inputs'], masks, ROOT)), 400)
        self.assertNotIn('pole_only', sources['current_selection']['excluded_branches'])

    def test_pole_crop_bbox_uses_remaining_alpha_without_writing(self):
        sprite = np.full((100, 20, 4), 255, np.uint8)
        placement = {'height': 100, 'left': 100, 'top': 100,
            'ground_polygon': [[90,190],[130,190],[130,210],[90,210]],
            'occlusion_polygons': [], 'visible_kind': 'identifiable_pole',
            'visibility_evidence': 'designed crop, not a photographed condition', 'retain_lower_half': True}
        _, box, actual = sign.compose(np.zeros((480,640,3), np.uint8), sprite, 'placed',
                                     'normal', random.Random(0), None, placement)
        self.assertEqual(box, [100,150,120,200])
        self.assertEqual(actual['visible_alpha_pixels'], 1000)

    def test_direct_selection_retains_candidate_and_excludes_42(self):
        sources, plan = config_plan('waste_sticker', 'direct_real_plan')
        catalog = real_source_catalog(sources)
        self.assertEqual(Counter(row['selected_for_current_batch'] for row in catalog), {False:42, True:1})
        self.assertEqual(len(sources['current_selection']['excluded_direct_images']), 42)
        selected = next(row for row in catalog if row['selected_for_current_batch'])
        self.assertEqual(selected['frame_number'], 2016)
        self.assertEqual(selected['annotation']['annotation_level'], 'candidate')
        self.assertEqual(len(prepare_inputs(plan, catalog)), 1)

    def test_surface_plan_balances_all_artworks_on_two_real_backgrounds(self):
        sources, plan = config_plan('waste_sticker')
        self.assertEqual(plan_production(plan, root=ROOT)['budget'], 192)
        geometry = surface.load_geometry(ROOT/sources['real_photo_geometry_manifest'])
        names = [item['name'] for item in yaml.safe_load((ROOT/sources['artwork_toolkit']/'configs/classes.yaml').read_text(encoding='utf8'))['items']]
        surface.validate_surface_plan(plan, names, geometry, sources['real_photo_geometry_manifest'])
        roles = surface_background_catalog(sources)
        self.assertEqual(len(roles), 2)
        self.assertEqual(len({row['source_video'] for row in roles.values()}), 2)
        self.assertEqual({state['parameters']['photo'] for state in plan['strata']}, set(roles))
        counts = Counter(placement['class_id'] for state in plan['strata'] for placement in state['parameters']['placements'])
        self.assertEqual(counts, dict.fromkeys(range(24), 8))

    def test_blender_plan_is_valid_without_importing_blender(self):
        sources, plan = config_plan('waste_sticker', 'blender_plan')
        tree = ast.parse((ROOT/'datasets/scripts/waste_sticker/render_sticker_arena_v4.py').read_text(encoding='utf8'))
        nodes = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name == 'validate') or
                 (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('ASSET','COLORS','CONSTRAINTS') for t in node.targets))]
        scope = {'math': math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<renderer-contract>', 'exec'), scope)
        scope['validate'](plan, sources['arena_scene'])
        self.assertEqual(plan_production(plan, root=ROOT)['budget'], 26)
        self.assertEqual(plan['target_split'], 'test')
        schedule = json.loads((ROOT/sources['animated_source_plan']).read_text(encoding='utf8'))
        frames = {state['parameters']['frame'] for state in plan['strata']}
        classes = {item['class_id'] for row in schedule['samples'] if row['frame'] in frames for item in row['bins']}
        self.assertEqual(classes, set(range(24)))

    def test_bin_renderer_plan_and_missing_real_mask_dependency(self):
        sources, plan = config_plan('waste_bin_state', 'render_plan')
        planned = plan_production(plan, root=ROOT)
        jobs = standalone_planned_jobs(planned, sources['standalone_scene'], 640,480,.7,24)
        self.assertEqual(len(jobs), 19)
        self.assertEqual(len(sources['current_selection']['background_ids']), 10)
        self.assertEqual(len(sources['current_selection']['excluded_background_ids']), 4)
        with patch.object(Path, 'is_file', return_value=False):
            with self.assertRaisesRegex(ValueError, 'actual render manifest is missing'):
                bin_planner.build()


if __name__ == '__main__':
    unittest.main()
