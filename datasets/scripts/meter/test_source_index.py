"""Verify that source-index semantics agree with actual meter generation branches."""
import unittest
from datasets.scripts.meter.source_index import source_record
from lab.training_common.engine.data import reviewed_strata


class MeterSourceIndex(unittest.TestCase):
    def row(self,branch,instances):
        return {'id':'one','image':'images/one.png','pose_label':'labels/one.txt','annotation':'annotations/one.json',
                'parameters':{'branch':branch,'degradation':{'kind':'jpeg'}},'instance_count':instances,
                'sources':['scene.blend','background.png'],'background':{'source_kind':'blender_render',
                'parent_group':'datasets/assets/blender/aiaec_dataset.blend','image':'background.png'},
                'foreground_id':'fg','foreground_annotation':'evidence.json'}

    def test_positive_and_negative_get_complete_but_distinct_supervision(self):
        for n in (0,1):
            row=self.row('blender_background_composite' if n else 'empty_negative',n)
            record=source_record(row,'train',{},{});strata=reviewed_strata(record,'meter',flat=True)
            self.assertEqual(strata['supervision'],'complete')
            self.assertEqual(strata['label_state'],'complete_positive' if n else 'complete_negative')
            self.assertEqual(strata['image_domain'],'other_synthetic')
            self.assertEqual(record['label'],'datasets/exports/meter/train/labels/one.txt')

    def test_real_background_composite_is_not_native_camera_data(self):
        row=self.row('real_background_composite',1);row['background']['source_kind']='video_frame'
        record=source_record(row,'train',{},{});strata=reviewed_strata(record,'meter',flat=True)
        self.assertEqual(strata['image_domain'],'real_scene_composite')


if __name__=='__main__':unittest.main()
