import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'lab/training_common'))
sys.path.insert(0,str(ROOT/'lab/features/sign/training/scripts'))
from sign_report import build_sign_report,load_metadata,write_sign_report


class SignReportTest(unittest.TestCase):
    def setUp(self):
        a={'class_id':0,'xyxy':[0,0,20,40]};b={'class_id':0,'xyxy':[50,10,90,80]}
        self.rows=[{'source_image':'positive.png','shape_hw':[100,100],
                    'ground_truth':[a,b],'predictions':[a,a]},
                   {'source_image':'negative.png','shape_hw':[100,100],'ground_truth':[],'predictions':[a]}]
        self.metadata={'positive.png':[{'source_group':'p','sign':{'schema_version':1,'image_size':[100,100],
            'instances':[{'xyxy':a['xyxy'],'view':'oblique','visibility':'partial','layout':'partial','origin':'composite'},
                         {'xyxy':b['xyxy'],'view':'unknown','visibility':'unknown','layout':'unknown','origin':'retained_background'}],
            'scene':{'quota_group':'partial','negative_context':'none'}}}],
            'negative.png':[{'source_group':'p','sign':{'schema_version':1,'image_size':[100,100],'instances':[],
                'scene':{'quota_group':'ordinary_pole','negative_context':'ordinary_signal_supports'}}}]}

    def test_misses_duplicate_predictions_and_negative_false_positives_have_separate_denominators(self):
        report=build_sign_report(self.rows,self.metadata,.5)
        self.assertEqual(report['instances'],{'instances':2,'matched':1,'missed':1,'recall':.5})
        self.assertEqual(report['images']['fp'],2)
        self.assertEqual(report['images']['negative_false_positive_rate'],1)
        self.assertEqual(report['instance_groups']['view']['oblique']['recall'],1)
        self.assertEqual(report['instance_groups']['view']['unknown']['missed'],1)
        self.assertIsNone(report['instance_groups']['view']['front']['recall'])
        self.assertEqual(report['instance_groups']['short_side_px']['under32']['instances'],1)
        self.assertEqual(report['image_groups']['quota_group']['ordinary_pole']['negative_images'],1)

    def test_inherited_box_order_and_missing_metadata_are_rejected(self):
        bad=copy.deepcopy(self.metadata)
        bad['positive.png'][0]['sign']['instances'].reverse()
        with self.assertRaisesRegex(ValueError,'box order'):build_sign_report(self.rows,bad,.5)
        with self.assertRaisesRegex(ValueError,'ambiguous'):build_sign_report(self.rows,{},.5)
        with self.assertRaisesRegex(ValueError,'Duplicate'):build_sign_report(self.rows+[self.rows[0]],self.metadata,.5)

    def test_relocated_package_aliases_and_report_cli_inputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);manifest=root/'dataset/workspace_sources.json';manifest.parent.mkdir()
            records=[]
            for image,items in self.metadata.items():
                records.append({**items[0],'split':'val','image':'dataset/images/val/'+image,'workspace_image':image})
            manifest.write_text(json.dumps({'model_key':'sign','records':records}),encoding='utf8')
            evaluation=root/'runs/sign/check/evaluation/val';(evaluation/'predictions').mkdir(parents=True)
            (evaluation/'summary.json').write_text(json.dumps({'model_key':'sign','split':'val','diagnostics':{'match_iou':.5}}),encoding='utf8')
            (evaluation/'predictions/predictions.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows),encoding='utf8')
            self.assertIn('dataset/images/val/positive.png',load_metadata(manifest,root,'val'))
            report=json.loads(write_sign_report(evaluation,manifest,root=root).read_text(encoding='utf8'))
            self.assertEqual(report['images']['images'],2)
            self.assertEqual(report['split'],'val')


if __name__=='__main__':unittest.main()
