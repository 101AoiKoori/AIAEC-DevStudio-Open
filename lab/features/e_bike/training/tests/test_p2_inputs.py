import unittest
from unittest.mock import patch
from lab.features.e_bike.training.scripts.p2_detector_experiment import select_fit


def real(name,number):
    return {'source_record':{'source_image':name},'count':number}


def synthetic(state,background,number):
    return {'source_record':{'state_id':state,'background':{'id':background}},'count':number}


class P2InputTest(unittest.TestCase):
    def test_fit_selection_keeps_mirrors_together_and_covers_backgrounds(self):
        records=[real('a',7),real('a',7),real('b',4),
                 synthetic('low_000001','A',8),synthetic('low_000002','A',8),
                 synthetic('low_000003','A',8),synthetic('low_000005','B',2)]
        with patch('lab.features.e_bike.training.scripts.p2_detector_experiment.truths',lambda row:[None]*row['count']):
            selected=select_fit(records,2,2)
        self.assertEqual(len(selected),4)
        self.assertEqual([r['source_record']['source_image'] for r in selected[:2]],['a','b'])
        self.assertEqual({r['source_record']['background']['id'] for r in selected[2:]},{'A','B'})

    def test_fit_selection_reports_actual_missing_families(self):
        with patch('lab.features.e_bike.training.scripts.p2_detector_experiment.truths',lambda row:[None]*row['count']):
            with self.assertRaisesRegex(ValueError,'Insufficient distinct'):
                select_fit([real('a',1),real('a',1)],2,0)


if __name__=='__main__':unittest.main()
