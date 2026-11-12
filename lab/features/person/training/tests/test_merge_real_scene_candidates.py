import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from merge_real_scene_candidates import approved_team, select_singles


class MergeCandidatesTest(unittest.TestCase):
    def test_selection_preserves_strata_and_is_repeatable(self):
        rows=[]
        for i in range(875):
            rows.append({'image':f'{i:04}.png',
                         'lighting':{'stratum':[None,None,None,i%2], 'level':('dim','normal','bright')[i%3]},
                         'background':{'kind':('ground','track','sign')[(i//3)%3]}})
        selected=select_singles(rows,846,7901)
        self.assertEqual(846,len(selected))
        self.assertEqual([r['image'] for r in selected],[r['image'] for r in select_singles(rows,846,7901)])
        key=lambda r:(r['lighting']['stratum'][3],r['background']['kind'],r['lighting']['level'])
        original=Counter(map(key,rows)); chosen=Counter(map(key,selected))
        self.assertEqual(set(original),set(chosen))
        for k in original:
            self.assertLessEqual(abs(chosen[k]-original[k]*846/875),1)

    def test_missing_reviewed_items_are_rejected(self):
        with self.assertRaises(ValueError): approved_team([])

    def test_impossible_quota_is_rejected(self):
        with self.assertRaises(ValueError): select_singles([],1,7901)


if __name__=='__main__': unittest.main()
