"""No-render contracts for the task-owned Blender branch."""
import copy
import unittest
from datasets.scripts.building_name_paper import render_blender_names as entry
from datasets.scripts.common.bayesian import plan_production

class NameRendererContracts(unittest.TestCase):
    def test_three_branches_allocate_independent_task_states(self):
        request=entry.build_plan({'single':3,'multiple':2,'negative':1},seed=7)
        planned=plan_production(request)
        self.assertEqual(planned['model_key'],'building_name_paper')
        self.assertEqual(planned['planned_total'],6)
        counts=sorted(len(s['target']['texts']) for s in planned['strata'])
        self.assertEqual(counts,[0,1,1,1,2,2])
        self.assertTrue(all(s['planned_count']==1 for s in planned['strata']))
        self.assertEqual(request,entry.build_plan({'single':3,'multiple':2,'negative':1},seed=7))

    def test_source_and_label_contract_reject_cross_task_fields(self):
        p=entry.build_plan({'single':1,'multiple':0,'negative':0})
        p['model_key']='fire_sticker'
        with self.assertRaises(ValueError): entry.validate_plan(p)
        p['model_key']='building_name_paper'
        p['strata'][0]['target']={'fire_counts':{'A':1}}
        with self.assertRaises(ValueError): entry.validate_plan(p)

    def test_boundaries_and_unknown_buildings_are_rejected(self):
        p=entry.build_plan({'single':1,'multiple':0,'negative':0})
        bad=copy.deepcopy(p);bad['strata'][0]['parameters']['yaw_degrees']=90
        with self.assertRaises(ValueError): entry.validate_plan(bad)
        bad=copy.deepcopy(p);bad['strata'][0]['background']['buildings']=['untracked']
        with self.assertRaises(ValueError): entry.validate_plan(bad)
        bad=copy.deepcopy(p);bad['strata'][0]['sources']=['invented']
        with self.assertRaises(ValueError): entry.validate_plan(bad)

    def test_planner_uses_name_renderer_directly(self):
        from lab.features.building_name_paper.training.scripts.plan_rooftop_batch import blender_plan
        p=blender_plan({'combinations':[{'id':'B_SINGLE_NAME','quota':1},{'id':'B_MULTI_NAME','quota':1},{'id':'B_NO_PAPER','quota':1}]})
        self.assertEqual(p['model_key'],entry.TASK)
        self.assertEqual(p['budget'],3)

    def test_external_plan_can_allocate_elastic_quotas(self):
        request = entry.build_plan({'single': 1, 'multiple': 1, 'negative': 0})
        request['budget'] = 8
        for state in request['strata']:
            state.update(min_count=0, max_count=8)
        entry.validate_plan(request)
        first = plan_production(request)
        request['strata'][0]['failure_cost'] = 20
        second = plan_production(request)
        self.assertEqual(second['planned_total'], 8)
        self.assertNotEqual([s['planned_count'] for s in first['strata']],
                            [s['planned_count'] for s in second['strata']])

if __name__=='__main__': unittest.main()
