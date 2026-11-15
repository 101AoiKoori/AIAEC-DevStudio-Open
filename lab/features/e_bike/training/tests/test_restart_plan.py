"""Validate selected bike plans using existing labels and alpha masks only."""
import json
from pathlib import Path
import unittest

import numpy as np

from datasets.scripts.common.bayesian import plan_generation
from datasets.scripts.common.perspective import planar_warp
from datasets.scripts.e_bike import compose_v4_grounded_train as entry
from datasets.scripts.e_bike import prepare_v4_low_angle_real as direct
from lab.scripts.data.v4_source_gate import build_queue


class BikeRestartPlanTest(unittest.TestCase):
    def test_actual_selected_inputs_and_mask_contact(self):
        root=entry.ROOT
        backgrounds=json.loads(entry.BACKGROUND.read_text(encoding='utf8'))['backgrounds']
        _,vehicles=entry.bike.sources(selection=entry.bike.SELECTION,include_people=False)
        queue=build_queue(root)
        plan=json.loads((root/'datasets/scripts/e_bike/config/production_plan.json').read_text(encoding='utf8'))
        entry.validate_ground_plan(plan,backgrounds,vehicles,queue)
        allocated=plan_generation(plan,root)
        self.assertEqual(sum(s['planned_count'] for s in allocated['strata']),plan['budget'])
        self.assertGreater(len({s['background']['image'] for s in plan['strata']}),1)
        for state in plan['strata']:
            bg=next(b for b in backgrounds if b['id']==state['background']['id'])
            for target,spec in zip(state['target']['vehicles'],state['parameters']['actors']):
                _,sprite=next(v for v in vehicles[target['class']] if v[0]['path']==target['source'])
                sprite=planar_warp(sprite,spec['yaw'],spec['pitch'],spec['roll'])
                sprite=entry.bike.scene.assets.resized(sprite,**({'height':spec['size_px']} if target['class']=='upright' else {'width':spec['size_px']}))
                actor={'sprite':sprite,'foot_x':spec['foot_x'],'foot_y':spec['foot_y']}
                sh,sw=sprite.shape[:2];left,top=round(spec['foot_x']-sw/2),round(spec['foot_y']-sh)
                self.assertTrue(0<=left<left+sw<=640 and 0<=top<top+sh<=480)
                self.assertTrue(entry.grounded(actor,entry.ground_zone(bg)))
                self.assertGreaterEqual(np.count_nonzero(sprite[:,:,3]>127),120)
        plan=json.loads((root/'datasets/scripts/e_bike/config/direct_real_plan.json').read_text(encoding='utf8'))
        rows=json.loads((direct.REVIEW/'review_manifest.json').read_text(encoding='utf8'))['rows']
        direct.validate_declaration(plan)
        for state in plan['strata']:
            direct.check_state(state,rows,queue)
        allocated=plan_generation(plan,root)
        self.assertEqual(sum(s['planned_count'] for s in allocated['strata']),plan['budget'])


if __name__ == '__main__':
    unittest.main()
