import unittest
from lab.features.e_bike.training.scripts.quality_rank_experiment import association,geometry,rank_pairs


class QualityRankTest(unittest.TestCase):
    def test_merged_candidate_association_stays_unknown(self):
        self.assertIsNone(association([.571,.530]))
        self.assertEqual(association([.75,.2]),0)
        self.assertIsNone(association([.1,.2]))

    def test_geometry_has_no_absolute_center_shortcut(self):
        first=geometry([10,20,30,60],.7,(100,100,3))
        second=geometry([40,30,60,70],.7,(100,100,3))
        self.assertEqual(first,second)
        self.assertEqual(geometry([-10,0,10,20],.7,(100,100,3))[-1],.5)

    def test_rank_pairs_require_same_image_instance_and_large_quality_gap(self):
        records=[{'source_image':'a','associated_gt':0,'quality_target':.9},
                 {'source_image':'a','associated_gt':0,'quality_target':.5},
                 {'source_image':'a','associated_gt':1,'quality_target':.2},
                 {'source_image':'b','associated_gt':0,'quality_target':.3},
                 {'source_image':'a','associated_gt':None,'quality_target':.1}]
        pairs=rank_pairs(records,list(range(5)),{'rank_pair_min_iou_gap':.15,'max_pairs_per_image':12})
        self.assertEqual(pairs,[(0,1)])


if __name__=='__main__':unittest.main()
