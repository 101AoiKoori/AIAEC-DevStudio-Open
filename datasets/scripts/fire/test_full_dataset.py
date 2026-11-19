"""Contracts for parent-isolated production and photo-camera compositing."""
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from datasets.scripts.common.config import load_task_config
from datasets.scripts.fire.build_dataset import camera_uses, render_plan, production_plan
from datasets.scripts.fire.photo_geometry import blend_render, fit_photo_camera, choose_photo_placement


class FireFullProduction(unittest.TestCase):
    def test_full_overview_stays_within_jpeg_dimensions_and_covers_branches(self):
        from datasets.scripts.fire.compose_real_facades import overview_indices
        branches=['direct_real','real_background_composite','blender_render','blender_real_background_composite']
        rows=[{'branch':branches[i%4],'frame_id':str(i%50),'building_counts':{'A':{'visible_count':i%3}},'camera':{}}
              for i in range(3000)]
        selected=overview_indices(rows)
        self.assertLessEqual(len(selected),240)
        self.assertEqual({rows[i]['branch'] for i in selected},set(branches))
        self.assertLessEqual(((len(selected)+3)//4)*320,65500)

    def test_photo_floor_placement_keeps_complete_body_and_footprint(self):
        import random
        source=load_task_config('fire')[1]
        reviews=json.loads(Path(source['hybrid_geometry_review']).read_text(encoding='utf8'))['backgrounds']
        for review in reviews:
            for depth in (.126,.204):
                for seed in range(12):
                    placement=choose_photo_placement(review,[[-.101,.101],[-depth/2,depth/2],[0,.4004]],random.Random(seed))
                    x1,y1,x2,y2=placement['body_bbox_photo_px']
                    self.assertTrue(0<x1<x2<449 and 0<y1<y2<600)
                    polygon=np.float32(review['ground_polygon_px'])
                    for point in placement['footprint_photo_px']:
                        self.assertGreaterEqual(cv2.pointPolygonTest(polygon,tuple(point),True),0)

    def test_actual_parent_allocation_and_all_full_quotas(self):
        source=load_task_config('fire')[1]
        allocation=json.loads(Path(source['full_dataset_plan']).read_text(encoding='utf8'))
        parents={}
        for split in ('train','val','test'):
            uses=camera_uses(source,allocation,split,True)
            renderer=render_plan(source,allocation,True) if split=='train' else None
            plan=production_plan(source,uses,renderer,split)
            self.assertEqual(plan['budget'],allocation['totals'][split])
            for use in uses.values():
                row=use['row'];parents.setdefault(row['parent_group'],set()).add(row['split'])
                self.assertEqual(row['split'],split)
        self.assertTrue(all(len(s)==1 for s in parents.values()))
        self.assertEqual(len(parents),36)

    def test_hybrid_requires_both_current_train_parents(self):
        source=load_task_config('fire')[1]
        allocation=json.loads(Path(source['full_dataset_plan']).read_text(encoding='utf8'))
        plan=render_plan(source,allocation,True)
        hybrids=[s for s in plan['strata'] if s['background'].get('photo_geometry_id')]
        self.assertEqual(sum(s['min_count'] for s in hybrids),240)
        assigned={r['parent_group']:r['split'] for r in allocation['frames']}
        for state in hybrids:
            self.assertEqual(assigned[state['background']['parent_group']],'train')
        self.assertEqual(allocation['blender_split'],'train')

    def test_known_camera_recovery_and_coordinate_convention(self):
        world=np.float64([[-.1,-.063,.4],[.1,-.063,.4],[.1,.063,.4],[-.1,-.063,0],[.1,-.063,0],[.1,.063,0]])
        rotation=cv2.Rodrigues(np.array([2.8,.2,.1]))[0]
        eye=np.array([.2,-1.,.25]);translation=-rotation@eye
        # Construct a camera looking at the building rather than relying on a
        # particular Rodrigues orientation near the optical-axis singularity.
        forward=np.array([0,0,.2])-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right);rotation=np.array([right,down,forward]);translation=-rotation@eye
        k=np.array([[600.,0,224.5],[0,600.,300.],[0,0,1]])
        pixels=cv2.projectPoints(world,cv2.Rodrigues(rotation)[0],translation,k,None)[0][:,0]
        result=fit_photo_camera(world,pixels,[449,600])
        self.assertLess(result['reprojection_rms_px'],1e-6)
        np.testing.assert_allclose(result['camera_position_m'],eye,atol=1e-6)

    def test_alpha_blend_keeps_photo_and_uses_linear_light(self):
        bg=np.full((2,2,3),128,np.uint8)
        foreground=np.zeros((2,2,4),np.uint8)
        np.testing.assert_array_equal(blend_render(bg,foreground),bg)
        foreground[:,:,:3]=255;foreground[:,:,3]=255
        np.testing.assert_array_equal(blend_render(bg,foreground),np.full_like(bg,255))
        foreground[:,:,3]=128;bg[:]=0
        self.assertTrue(np.all((blend_render(bg,foreground)>=187)&(blend_render(bg,foreground)<=188)))


if __name__=='__main__':
    unittest.main()
