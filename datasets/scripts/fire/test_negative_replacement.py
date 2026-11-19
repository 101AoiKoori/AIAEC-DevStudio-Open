"""Selection/plan invariants for destructive generated-pair substitution."""
import unittest
from datasets.scripts.fire.replace_empty_negatives import select_retired,current_plan


def row(name,branch,count,group='source'):
    return {'image':name,'branch':branch,'instances':[{}]*count,'building_counts':{'A':{}},
            'frame_id':group,'production_state_id':group}


class NegativeReplacement(unittest.TestCase):
    def test_selection_preserves_all_positives_and_direct_real(self):
        rows=[row('positive','real_background_composite',1),row('real','direct_real',0),
              row('a','real_background_composite',0,'first'),row('b','real_background_composite',0,'second')]
        cfg={'retire_eval_branch':'real_background_composite','replace_images':{'val':2}}
        self.assertEqual({r['image'] for r in select_retired(rows,'val',cfg)},{'a','b'})

    def test_shortage_does_not_fall_back_to_positive_or_real_rows(self):
        rows=[row('positive','real_background_composite',1),row('real','direct_real',0)]
        with self.assertRaises(ValueError):
            select_retired(rows,'val',{'retire_eval_branch':'real_background_composite','replace_images':{'val':1}})

    def test_plan_removes_retired_state_without_changing_retained_budget(self):
        original={'budget':3,'strata':[{'id':'keep','min_count':2,'max_count':2},{'id':'remove','min_count':1,'max_count':1}]}
        retained=[row('x','blender_render',1,'keep'),row('y','blender_render',1,'keep')]
        addition={'production_state_id':'new','branch':'blender_render','context':{'distractor':'red_signal_lamp'},
                  'source_group':'mother','scene_snapshot':'snapshot.blend','sample_seed':1,'source_blend_sha256':'hash'}
        updated=current_plan(original,retained,[addition])
        self.assertEqual(updated['budget'],3)
        self.assertEqual([s['id'] for s in updated['strata']],['keep','new'])
        self.assertEqual(original['strata'][1]['id'],'remove')


if __name__=='__main__':unittest.main()
