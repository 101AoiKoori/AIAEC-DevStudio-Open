"""Check the requested cross coverage on the actual current sample plan."""
import itertools
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]


class MaterialSampleCoverage(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT/'datasets/scripts/meter/config/sample_plan.json').read_text())
        self.rows = [s['parameters'] for s in self.plan['strata'] if s['parameters']['branch']=='blender_render']

    def test_every_camera_has_paired_materials_and_independent_reading_grid(self):
        self.assertEqual(len({p['view'] for p in self.rows}),27)
        levels=(0,.2,.4,.6,.8,1)
        for material in ('white','colored'):
            actual={(p['temperature_fraction'],p['humidity_fraction']) for p in self.rows if p['face_material']==material}
            self.assertEqual(actual,set(itertools.product(levels,repeat=2)))
        for row in self.rows:
            other={**row,'face_material':'colored' if row['face_material']=='white' else 'white'}
            self.assertIn(other,self.rows)

    def test_current_preview_is_one_image_without_html(self):
        folder=ROOT/'datasets/samples/meter'
        if not (folder/'manifest.json').exists():self.skipTest('No local sample')
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest.get('generation_plan',{}).get('render_seed')!=20261118:
            self.skipTest('Previous recipe is still published')
        self.assertTrue((folder/'contact_sheet.jpg').is_file())
        self.assertFalse(list(folder.glob('*.html')))

    def test_every_glare_has_same_camera_reading_and_material_control(self):
        glare=[p for p in self.rows if p['glare']!='none']
        self.assertEqual(len(glare),18)
        self.assertEqual({p['glare'] for p in glare},{'temperature_tip','humidity_tip','temperature_max'})
        for row in glare:self.assertIn({**row,'glare':'none'},self.rows)
        self.assertEqual(self.plan['budget'],433)
        self.assertEqual(sum(s['parameters']['branch']=='direct_task_review' for s in self.plan['strata']),1)
        self.assertEqual(sum(s['parameters']['branch']=='direct_task_pose' for s in self.plan['strata']),54)
        self.assertEqual(self.plan['target_split'],'unassigned')

    def test_published_boxes_points_and_display_roi_use_final_image_coordinates(self):
        folder=ROOT/'datasets/samples/meter'
        if not (folder/'manifest.json').exists():self.skipTest('No local published sample')
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest.get('status')!='review_only_paired_material_camera_reading_glare':
            self.skipTest('Current sample uses another recipe')
        rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        for row in rows:
            annotation=json.loads((folder/row['annotation']).read_text())
            if row['parameters']['branch']=='direct_task_review':
                self.assertIsNone(row['pose_label'])
                self.assertFalse(row['training_eligible'])
                self.assertTrue(row['image'].startswith('review_images/'))
                self.assertEqual(len(row['observed_keypoints']),8)
                continue
            label=list(map(float,(folder/row['pose_label']).read_text().split()))
            cx,cy,w,h=label[1:5]
            iw,ih=row['image_size'];box=[(cx-w/2)*iw,(cy-h/2)*ih,(cx+w/2)*iw,(cy+h/2)*ih]
            for actual,expected in zip(annotation['bbox_xyxy_pixels'],box):
                self.assertAlmostEqual(actual,expected,delta=1e-4)
            for index,key in enumerate(annotation['keypoint_order']):
                for axis in range(2):
                    self.assertAlmostEqual(annotation['keypoints_xy_depth_normalized_candidate'][key][axis],label[5+3*index+axis],delta=1e-8)
            if row['transform']:
                t=row['transform'];dx,dy=t['offset_xy'];scale=t['scale']
                for original,current in zip(row['foreground_geometry']['display_roi_quad'],annotation['display_roi_quad']):
                    self.assertAlmostEqual(current[0],original[0]*scale+dx/640)
                    self.assertAlmostEqual(current[1],original[1]*scale+dy/480)


if __name__=='__main__':unittest.main()
