"""Verify split budgets, structural controls and counterfactual mask semantics."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets.scripts.building_name_paper import full_plan as full,prepare_full_dataset as planner
from datasets.scripts.building_name_paper import build_full_dataset as builder
from lab.scripts.contracts.state_coverage import _building_name_paper_targets


class FullDatasetTests(unittest.TestCase):
    def test_derived_image_identity_is_distinct_from_its_background(self):
        first={'branch':'blender_photo_composite','image':'val/images/a.png','source_photo':'raw/shared.png'}
        second={**first,'image':'val/images/b.png'}
        a=builder.source_identity(first);b=builder.source_identity(second)
        self.assertNotEqual(a['workspace_image'],b['workspace_image'])
        self.assertEqual(a['background_image'],b['background_image'])
        direct=builder.source_identity({'branch':'direct_real','image':'val/images/real.jpg','source_image':'raw/real.jpg'})
        self.assertEqual(direct['workspace_image'],'raw/real.jpg')

    def test_compiler_covers_all_26_building_paper_states_in_each_split(self):
        recipe=full.config();plates=list(full.plates());templates=full.templates()['strata']
        expected={(a,''.join(c)) for a in recipe['building_combinations']
                  for n in range(len(a)+1) for c in itertools.combinations(a,n)}
        with tempfile.TemporaryDirectory(dir=full.WORK) as directory:
            with patch.object(planner,'destination',side_effect=lambda key:Path(directory)/(key+'.json')):
                for split,total in [('train',3000),('val',300),('test',300)]:
                    planner.compile_split(split,plates,templates,recipe)
                    plan=json.loads((Path(directory)/('full_'+split+'_plan.json')).read_text(encoding='utf8'))
                    self.assertEqual(len(plan['strata']),total)
                    states={(s['target']['buildings'],s['target']['papers']) for s in plan['strata']
                            if isinstance(s['target'].get('papers'),str)}
                    self.assertEqual(states,expected)

    def test_photo_background_parents_are_disjoint_by_split(self):
        parents={}
        for plate in full.plates():
            prior=parents.setdefault(plate['parent_group'],plate['split'])
            self.assertEqual(prior,plate['split'])
        self.assertEqual(len(list(full.plates())),13)
        self.assertEqual(len(parents),4)

    def test_zero_paper_is_explicit_complete_negative_in_coverage(self):
        stats=_building_name_paper_targets({'target':{'buildings':'ABC','papers':''}})
        self.assertEqual(stats['presence'],['absent'])
        self.assertEqual(stats['instance_count'],[0])
        partial=_building_name_paper_targets({'target':{'buildings':'ABC','papers':'AC'}})
        self.assertEqual(partial['instance_count'],[2])

    def test_sensor_transform_keeps_masks_paired_and_empty_state_empty(self):
        recipe=full.config()['augmentation'];image=np.full((120,160,3),140,np.uint8)
        mask=np.zeros((120,160),bool);mask[20:30,50:100]=True
        transformed,masks,_=builder.augment(image,{'A':mask},43,recipe)
        self.assertEqual(transformed.shape,image.shape)
        self.assertGreaterEqual(int(masks['A'].sum()),12)
        _,empty,_=builder.augment(image,{},44,recipe)
        self.assertEqual(empty,{})

    def test_no_extra_direct_records_does_not_select_other_splits(self):
        from datasets.scripts.building_name_paper.build_current_sample import direct_records
        rows,_,_=direct_records({'direct_split':'train','additional_direct_records':[]})
        self.assertTrue(rows)
        self.assertEqual({row[2]['source_split'] for row in rows},{'train'})
