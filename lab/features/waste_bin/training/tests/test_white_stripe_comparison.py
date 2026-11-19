import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('white_stripe_compare',Path(__file__).resolve().parents[1]/'scripts/compare_white_stripe_finetune.py')
compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)


class GroupComparisonTests(unittest.TestCase):
    def test_background_denominator_and_mismatched_sample_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
            folder=Path(temporary)
            summary={'split':'val','evaluation_config':{'diagnostic_conf':.25},'metrics':{},
                     'diagnostics':dict(true_positives=1,false_positives=2,false_negatives=0,
                                        negative_images=1,negative_images_with_false_positives=1,nms_iou=.7)}
            (folder/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
            row={'source_image':'fixture/background.png','ground_truth':'0',
                 'true_positives':'0','false_positives':'2','false_negatives':'0'}
            with (folder/'per_image.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            records=[{'image':row['source_image'],'group':'stripe_negative','split':'val'}]
            report=compare.group_report(folder,records)
            self.assertEqual(report['evaluation_config']['nms_iou'],.7)
            result=report['groups']['stripe_negative']
            self.assertEqual(result['false_boxes_per_100_images'],200)
            self.assertEqual(result['negative_false_positive_rate'],1)
            self.assertIsNone(result['recall'])
            with self.assertRaisesRegex(ValueError,'images differ'):
                compare.group_report(folder,[{**records[0],'image':'fixture/another.png'}])


if __name__=='__main__':unittest.main()
