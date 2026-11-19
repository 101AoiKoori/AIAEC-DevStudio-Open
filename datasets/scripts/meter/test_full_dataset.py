"""Meaningful full-batch geometry, split and image degradation contracts."""
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image,ImageDraw

from datasets.scripts.meter.full_variations import random_placement,degrade
from datasets.scripts.meter.compose_meter_real_background_v4 import place_foreground
from datasets.scripts.meter.build_full_dataset import valid_cache
from datasets.scripts.meter.full_plan import digest
from datasets.scripts.common.frame_environment import image_digest

ROOT=Path(__file__).resolve().parents[3]


class FullMeter(unittest.TestCase):
    def test_resume_rejects_changed_input_and_changed_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as name:
            folder=Path(name);source=folder/'input.json';output=folder/'image.png'
            source.write_text('{"point":1}')
            Image.new('RGB',(8,8),'white').save(output)
            expected={'id':'one'}
            record={'plan_sha256':digest(expected),'token':'fixture',
                    'inputs':{source.relative_to(ROOT).as_posix():image_digest(source)},
                    'files':{'image.png':image_digest(output)}}
            (folder/'completed.json').write_text(json.dumps(record))
            with patch('datasets.scripts.meter.build_full_dataset.receipt_completed',return_value=True):
                self.assertTrue(valid_cache(folder,'fixture',expected))
                source.write_text('{"point":2}')
                self.assertFalse(valid_cache(folder,'fixture',expected))
                source.write_text('{"point":1}')
                Image.new('RGB',(8,8),'black').save(output)
                self.assertFalse(valid_cache(folder,'fixture',expected))

    def test_completed_composition_chunk_has_exact_export_inputs(self):
        root=ROOT/'datasets/annotated/meter/full_production/train/composites'
        completed=sorted(root.glob('chunk_*/completed.json'))
        if not completed:self.skipTest('No local composition cache')
        folder=completed[0].parent
        for line in (folder/'samples.jsonl').read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            image=ROOT/row['composition_input'];base=image.parent.parent
            self.assertEqual(base,folder)
            self.assertTrue((base/row['pose_label']).is_file())
            self.assertEqual(json.loads((base/row['annotation']).read_text(encoding='utf8')),row)

    def test_layout_changes_box_and_all_eight_points_with_same_affine(self):
        fg=Image.new('RGBA',(640,480));ImageDraw.Draw(fg).rectangle((260,80,380,420),fill=(90,80,60,255))
        bg={'image_size':[640,480],'geometry':{'regions':[{'kind':'full_frame_empty_scene'}]}}
        values=[0,.5,.3,.15,.15]+[v for _ in range(8) for v in (.5,.3,2)]
        centers=[]
        for x in (.05,.5,.95):
            proposed,_=random_placement(bg,fg,{'x_fraction':x,'y_fraction':.5,'scale_factor':.8})
            result,label,transform=place_foreground(fg,values,proposed)
            self.assertTrue(result.getchannel('A').getbbox())
            expected_x=.5*transform['scale']+transform['offset_xy'][0]/640
            expected_y=.3*transform['scale']+transform['offset_xy'][1]/480
            centers.append(label[1])
            for i in range(8):
                self.assertAlmostEqual(label[5+3*i],expected_x)
                self.assertAlmostEqual(label[6+3*i],expected_y)
            self.assertAlmostEqual(label[3],values[3]*transform['scale'])
        self.assertGreater(max(centers)-min(centers),.4)

    def test_degradation_is_repeatable_and_keeps_pixel_coordinates(self):
        source=np.zeros((48,64,3),dtype='uint8');source[12:36,20:44]=[190,140,90]
        controls={'kind':'blur_jpeg','blur_radius':.5,'jpeg_quality':72,'noise_sigma':2.,'seed':16}
        a=np.asarray(degrade(Image.fromarray(source),controls));b=np.asarray(degrade(Image.fromarray(source),controls))
        self.assertTrue(np.array_equal(a,b));self.assertEqual(a.shape,source.shape)
        self.assertFalse(np.array_equal(a,source))
        self.assertLess(abs(np.unravel_index(a[:,:,0].argmax(),a[:,:,0].shape)[0]-24),14)

    def test_exact_quotas_and_split_specific_images_readings(self):
        pairs=[];backgrounds=[]
        for split,total,native,negative in [('train',6480,45,648),('val',1080,7,108),('test',1080,2,108)]:
            path=ROOT/f'datasets/scripts/meter/config/full_{split}_plan.json'
            if not path.is_file():self.skipTest('Compile full plans first')
            plan=json.loads(path.read_text());states=plan['strata']
            self.assertEqual(plan['budget'],total);self.assertEqual(len(states),total)
            self.assertEqual(sum(s['parameters']['branch']=='direct_task_pose' for s in states),native)
            self.assertEqual(sum(s['parameters']['branch']=='empty_negative' for s in states),negative)
            render=json.loads((ROOT/plan['render_plan']).read_text())['strata']
            self.assertEqual(len({s['parameters']['view'] for s in render}),27)
            self.assertGreater(sum(s['parameters']['reading_sampling']=='continuous' for s in render)/len(render),.6)
            pairs.append({(s['parameters']['temperature_fraction'],s['parameters']['humidity_fraction']) for s in render})
            inventory=json.loads((ROOT/plan['background_inventory']).read_text())['backgrounds']
            backgrounds.append({r['image'] for r in inventory})
        for i in range(3):
            for j in range(i):
                self.assertFalse(pairs[i]&pairs[j]);self.assertFalse(backgrounds[i]&backgrounds[j])

    def test_pilot_has_real_changes_and_round_tripped_eight_point_labels(self):
        folder=ROOT/'datasets/annotated/meter/full_production/production_inputs/pilot'
        if not (folder/'manifest.json').is_file():self.skipTest('Render pilot first')
        rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        centers=[];sizes=[]
        for row in rows:
            label=list(map(float,(folder/row['pose_label']).read_text().split()))
            self.assertEqual(len(label),29)
            self.assertTrue(np.allclose(label,row['label_values'],atol=1e-8,rtol=0))
            for i,key in enumerate(row['keypoint_order']):
                self.assertTrue(np.allclose(label[5+3*i:7+3*i],row['keypoints_xy_depth_normalized_candidate'][key][:2],atol=1e-8,rtol=0))
            centers.append(label[1]);sizes.append(label[3])
        self.assertGreater(max(centers)-min(centers),.3)
        self.assertGreater(max(sizes)/min(sizes),1.7)


if __name__=='__main__':unittest.main()
