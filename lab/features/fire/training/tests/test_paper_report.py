"""Exercise paper localization, count errors, unknown views and metadata binding."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'lab/training_common'))
spec=importlib.util.spec_from_file_location('paper_report',Path(__file__).resolve().parents[1]/'scripts/paper_report.py')
paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)


class PaperReportTest(unittest.TestCase):
    def fixture(self):
        truth=[{'class_id':0,'xyxy':[10,10,30,14]},{'class_id':0,'xyxy':[50,20,90,40]}]
        record={'image':'val/a.png','source_id':'unit-parent',
            'paper':{'schema_version':1,'image_size':[100,80],
                'instances':[{'xyxy':t['xyxy'],'building_id':b} for t,b in zip(truth,'AC')],
                'scene':{'branch':'blender_photo_composite','buildings':'ABC','papers':'AC',
                    'yaw_degrees':-50,'camera_height_m':.14,'view_basis':'render_control_with_estimated_camera'}}}
        row={'source_image':'val/a.png','shape_hw':[80,100],'ground_truth':truth,
             'predictions':[truth[0],{'class_id':0,'xyxy':[10,50,20,60]}]}
        return row,record

    def test_equal_counts_do_not_hide_missed_paper_and_extra_detection(self):
        row,record=self.fixture();report=paper.build_paper_report([row],{'val/a.png':record},100,.5)
        total=report['images'];self.assertEqual(total['count_correct_rate'],1)
        self.assertEqual(total['exact_image_rate'],0);self.assertEqual(total['fp'],1);self.assertEqual(total['fn'],1)
        self.assertEqual(report['image_groups']['paper_state']['partial']['images'],1)
        self.assertEqual(report['instance_groups']['input_short_side_px']['0_8']['matched'],1)
        self.assertEqual(report['instance_groups']['input_short_side_px']['16_32']['missed'],1)
        self.assertEqual(report['instance_groups']['building_id']['C']['missed'],1)

    def test_empty_images_and_unknown_real_view_keep_their_semantics(self):
        row,record=self.fixture();row['ground_truth']=[];record['paper']['instances']=[]
        record['paper']['scene']={'branch':'direct_real','buildings':None,'papers':None,'view_basis':'unknown'}
        report=paper.build_paper_report([row],{'val/a.png':record},100,.5)
        self.assertEqual(report['images']['negative_false_positive_rate'],1)
        self.assertEqual(report['image_groups']['yaw_control']['unknown']['images'],1)
        self.assertIsNone(report['images']['recall'])

    def test_counterfactual_building_without_any_paper_is_explicit(self):
        row,record=self.fixture();row['ground_truth']=[];row['predictions']=[]
        record['paper']['instances']=[];record['paper']['scene']['papers']=''
        report=paper.build_paper_report([row],{'val/a.png':record},100,.5)
        self.assertEqual(report['image_groups']['paper_state']['none']['exact_image_rate'],1)

    def test_stale_box_order_wrong_size_and_contradictory_controls_are_rejected(self):
        row,record=self.fixture()
        for mutate in (lambda r:r['paper']['instances'].reverse(),
                       lambda r:r['paper'].update(image_size=[101,80]),
                       lambda r:r['paper']['scene'].update(papers='AB')):
            bad=deepcopy(record);mutate(bad)
            with self.assertRaises(ValueError):paper.build_paper_report([row],{'val/a.png':bad},100,.5)

    def test_duplicate_alias_and_invalid_predictions_are_rejected(self):
        row,record=self.fixture();other={**row,'source_image':'alias.png'}
        with self.assertRaises(ValueError):
            paper.build_paper_report([row,other],{'val/a.png':record,'alias.png':record},100,.5)
        row['predictions'][0]={'class_id':1,'xyxy':[1,1,2,2]}
        with self.assertRaises(ValueError):paper.build_paper_report([row],{'val/a.png':record},100,.5)

    def test_border_rounding_is_allowed_but_real_out_of_bounds_is_rejected(self):
        row,record=self.fixture()
        row['ground_truth'][0]={'class_id':0,'xyxy':[-0.00005,10,30,14]}
        record['paper']['instances'][0]['xyxy']=[0,10,30,14]
        paper.build_paper_report([row],{'val/a.png':record},100,.5)
        row['ground_truth'][0]['xyxy'][0]=-0.01
        record['paper']['instances'][0]['xyxy'][0]=-0.01
        with self.assertRaisesRegex(ValueError,'Invalid evaluated paper box'):
            paper.build_paper_report([row],{'val/a.png':record},100,.5)


if __name__=='__main__':unittest.main()
