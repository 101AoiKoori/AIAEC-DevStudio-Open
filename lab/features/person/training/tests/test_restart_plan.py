"""Validate the selected restart inputs and alpha geometry without compositing."""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from datasets.scripts.common.bayesian import plan_generation
from datasets.scripts.person.compose_planned_ground_person import ROOT, prepare_batch
from datasets.scripts.person.person_scene_plan import _foreground
from datasets.scripts.common.composite import visible_masks


class PersonRestartPlanTest(unittest.TestCase):
    def test_selected_sources_labels_allocation_and_occlusion_geometry(self):
        plan = json.loads((ROOT/'datasets/scripts/person/config/production_plan.json').read_text(encoding='utf8'))
        prepared = prepare_batch(plan)
        allocated = plan_generation(plan, ROOT)
        self.assertEqual(sum(s['planned_count'] for s in allocated['strata']), plan['budget'])
        self.assertLess(plan['budget'], len(prepared))
        self.assertGreater(len({s['background']['image'] for s in plan['strata']}), 1)
        for item in prepared.values():
            scene = item['scene']
            masks, identity_masks, ids = [], [], []
            width,height = item['state']['parameters']['resolution']
            protected = [o['xyxy'] for o in item['existing']] + item['protected']
            for actor in scene['actors']:
                sprite, identity = _foreground(item['foregrounds'][actor['id']],
                                               item['identity_masks'].get(actor['id']), actor)
                sh,sw = sprite.shape[:2]
                x,y = actor['foot_xy'];left,top = round(x-sw/2),round(y-sh)
                self.assertTrue(0 <= left < left+sw <= width and 0 <= top < top+sh <= height)
                ground=np.asarray(scene['ground_polygon'],np.float32)
                for contact_x in (x-actor['contact_width']/2,x+actor['contact_width']/2):
                    self.assertGreaterEqual(cv2.pointPolygonTest(ground,(contact_x,float(y)),False),0)
                mask=np.zeros((height,width),bool);mask[top:top+sh,left:left+sw]=sprite[:,:,3]>127
                for x1,y1,x2,y2 in protected:
                    self.assertFalse(mask[y1:y2,x1:x2].any())
                marker=np.zeros_like(mask)
                if identity is not None:
                    marker[top:top+sh,left:left+sw]=identity & (sprite[:,:,3]>127)
                ids.append(actor['id']);masks.append(mask);identity_masks.append(marker)
            shown=visible_masks(masks)
            for back,front in scene['required_overlaps']:
                self.assertTrue((masks[ids.index(back)] & masks[ids.index(front)]).any())
            for actor,marker,visible in zip(scene['actors'],identity_masks,shown):
                if actor['kind']=='person':
                    self.assertGreater(marker.sum(),0)
                    self.assertGreaterEqual((marker&visible).sum()/marker.sum(),actor['min_identity_visible'])


if __name__ == '__main__':
    unittest.main()
