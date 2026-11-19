"""Check pool eligibility, paired photometry and real alpha/pose invariants."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from datasets.scripts.meter.backgrounds import select_backgrounds, photometric_schedule, adjust
from lab.scripts.contracts.state_coverage import _meter_axes

ROOT = Path(__file__).resolve().parents[3]


class MeterBackgrounds(unittest.TestCase):
    def test_selection_excludes_wrong_split_and_incomplete_or_positive_labels(self):
        base={'split':'train','status':'eligible','source_kind':'blender_render','scene':'room','route':'meter',
              'image':'a.png','render_frame':0,'record_id':'a','content_sha256':'hash',
              'label':{'presence':'absent','completeness':'complete'}}
        rows=[]
        for i in range(24):
            rows.append({**copy.deepcopy(base),'record_id':str(i),'render_frame':i,'image':f'{i}.png'})
        for updates in ({'split':'val'},{'status':'unknown'},{'label':{'presence':'present','completeness':'complete'}},
                        {'label':{'presence':'absent','completeness':'partial'}}):
            rows.append({**base,**updates})
        with patch('datasets.scripts.meter.backgrounds.image_digest',return_value='hash'):
            selected=select_backgrounds({'split':'train','frames_per_scene_route':4},rows=rows)
        self.assertEqual({r['render_frame'] for r in selected},{0,8,15,23})
        self.assertEqual(len(selected),4)

    def test_schedule_crosses_three_controls_and_is_repeatable(self):
        recipe=json.loads((ROOT/'datasets/scripts/meter/config/sample_recipe.json').read_text())
        schedule=photometric_schedule(recipe,54)
        self.assertEqual(schedule,photometric_schedule(recipe,54))
        self.assertEqual(len({tuple(sorted(r.items())) for r in schedule[:27]}),27)
        self.assertEqual(schedule[:27].count(schedule[0]),1)

    def test_rgb_changes_keep_exact_alpha_and_shape(self):
        pixels=np.zeros((14,19,4),dtype=np.uint8)
        pixels[:,:,:3]=[190,80,30];pixels[2:12,4:15,3]=255
        source=Image.fromarray(pixels)
        result=adjust(source,saturation=.15,brightness=.78)
        self.assertTrue(np.array_equal(np.asarray(result)[:,:,3],pixels[:,:,3]))
        rgb=np.asarray(result)[:,:,:3]
        self.assertLess(rgb.max(axis=2).mean()-rgb.min(axis=2).mean(),160)
        self.assertEqual(source.getchannel('A').getbbox(),result.getchannel('A').getbbox())

    def test_axis_consumer_reports_declared_photometry(self):
        p={'saturation':'very_low','exposure':'dark','light_relation':'foreground_brighter'}
        self.assertEqual(_meter_axes({'parameters':p}),{k:[v] for k,v in p.items()})

    def test_published_material_pairs_have_identical_background_tone_and_pose(self):
        folder=ROOT/'datasets/samples/meter'
        if not (folder/'manifest.json').is_file():self.skipTest('No local sample')
        records=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        pairs={}
        composites=[r for r in records if r.get('photometry')]
        if not composites:self.skipTest('Expanded sample not published yet')
        for row in composites:
            p={k:v for k,v in row['parameters'].items() if k not in ('face_material','foreground_id')}
            key=json.dumps(p,sort_keys=True)
            pairs.setdefault(key,[]).append(row)
        self.assertTrue(all(len(pair)==2 for pair in pairs.values()))
        for a,b in pairs.values():
            self.assertEqual(a['label_values'][:7],b['label_values'][:7])
            self.assertEqual(a['transform'],b['transform'])
            self.assertEqual(a['label_values'][5:],b['label_values'][5:])
        used={r['background']['record_id'] for r in composites}
        self.assertEqual(len(used),99)
        self.assertEqual({r['background']['scene'] for r in composites if r['background']['source_kind']=='blender_render'},
                         {'Arena','Arena Detailed','Arena Indoor Soft Light'})


if __name__=='__main__':unittest.main()
