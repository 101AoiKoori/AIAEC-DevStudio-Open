import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import state_coverage

ROOT = Path(__file__).resolve().parents[4]


class MeterMaterialAxesTest(unittest.TestCase):
    def test_explicit_camera_material_and_glare_controls(self):
        row={'parameters':{'face_material':'colored','distance_m':1.35,
             'yaw_deg':0,'elevation_deg':8,'glare':'humidity_tip',
             'branch':'blender_render'}}
        self.assertEqual(state_coverage._meter_axes(row),{
            'face_material':['colored'],'distance':['1.35'],'view':['oblique'],
            'glare':['present'],'background':['open_neutral_preview']})

    def test_legacy_missing_controls_remain_unknown(self):
        self.assertEqual(state_coverage._meter_axes({'parameters':{}}),{})


class TaskDesignViewsTest(unittest.TestCase):
    def test_finetune_budget_is_registered_as_a_separate_use(self):
        target=state_coverage.load_targets(ROOT/'config/task_targets.yaml')['tasks']['waste_sticker']
        source = yaml.safe_load((ROOT / 'datasets/scripts/waste_sticker/config/source.yaml').read_text(encoding='utf-8'))
        ref = (ROOT / source['finetune_plan']).relative_to(ROOT / 'datasets/scripts/waste_sticker').as_posix()
        self.assertIn(ref,target['plans'])
        self.assertEqual(target['plan_usage'][ref],'fine_tune')
        axes=state_coverage.load_axes(ROOT/'config/combination_axes.yaml')['tasks']['waste_sticker']['axes']
        report=state_coverage._summarize_plan(ROOT,'waste_sticker',ref,target,axes)
        self.assertEqual(report['usage'],'fine_tune')
        current_plan = json.loads((ROOT / source['finetune_plan']).read_text(encoding='utf-8'))
        self.assertEqual(report['planned_total'], current_plan['budget'])
        self.assertIsNone(report['error'])
        legacy = (ROOT / source['legacy_finetune_plan']).relative_to(ROOT / 'datasets/scripts/waste_sticker').as_posix()
        self.assertNotIn(legacy, target['plans'])

    def test_staged_view_must_match_source(self):
        targets = state_coverage.load_targets(ROOT / 'config/task_targets.yaml')
        axes = state_coverage.load_axes(ROOT / 'config/combination_axes.yaml')
        class Result:
            returncode = 0
            stdout = b''
        def git_show(args, **kwargs):
            result = Result()
            path = ROOT / args[-1][1:]
            result.stdout = path.read_bytes()
            return result
        with patch.object(state_coverage.subprocess, 'run', side_effect=git_show):
            state_coverage.check_staged_task_designs(ROOT, targets, axes)
            targets['tasks']['person']['annotation'] = '暂存内容尚未包含的目标修改'
            with self.assertRaisesRegex(ValueError, '暂存任务说明尚未同步'):
                state_coverage.check_staged_task_designs(ROOT, targets, axes)

    def test_eight_views_stay_in_sync_and_scope_changes_to_one_task(self):
        targets = state_coverage.load_targets(ROOT / 'config/task_targets.yaml')
        axes = state_coverage.load_axes(ROOT / 'config/combination_axes.yaml')
        self.assertEqual(state_coverage.update_task_designs(ROOT, targets, axes, write=False), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in state_coverage.TASK_ORDER:
                original = ROOT / 'datasets/scripts' / task / 'README.md'
                destination = root / 'datasets/scripts' / task / 'README.md'
                destination.parent.mkdir(parents=True)
                destination.write_text(original.read_text(encoding='utf-8'), encoding='utf-8')
            targets['tasks']['person']['annotation'] = '测试用的人物标注变更'
            changed = state_coverage.update_task_designs(root, targets, axes, write=False)
            self.assertEqual([item.parent.name for item in changed], ['person'])
            before = changed[0].read_text(encoding='utf-8')
            state_coverage.update_task_designs(root, targets, axes, write=True)
            after = changed[0].read_text(encoding='utf-8')
            self.assertIn('测试用的人物标注变更', after)
            self.assertEqual(before.split(state_coverage.TASK_START)[0], after.split(state_coverage.TASK_START)[0])
            self.assertEqual(before.split(state_coverage.TASK_END)[1], after.split(state_coverage.TASK_END)[1])
            self.assertEqual(state_coverage.update_task_designs(root, targets, axes, write=False), [])


def plan(task, name='production_plan.json'):
    path = ROOT / 'datasets/scripts' / task / 'config' / name
    return json.loads(path.read_text(encoding='utf-8'))


class ExtractTest(unittest.TestCase):
    def test_sign_background_with_retained_labels_is_positive_not_empty(self):
        row={'target':{'state':'negative','mask':'none'},
             'labels':{'class_id':0,'kind':'retained_background_boxes'}}
        self.assertEqual(state_coverage._sign_targets(row),{'classes':['sign_fixture'],'presence':['placed']})
        self.assertIsNone(state_coverage._sign_axes(row)['view'])

    def test_person_target_and_axis_are_separate(self):
        stratum = {'target': {'actors': [
            {'id': 'front', 'kind': 'person', 'view': 'front', 'category': 'community'},
        ]}}
        self.assertEqual(state_coverage._person_targets(stratum)['classes'], ['community'])
        self.assertEqual(state_coverage._person_targets(stratum)['count'], ['single'])
        axes = state_coverage._person_axes(stratum)
        self.assertEqual(axes['vehicle_occlusion'], ['none'])
        self.assertEqual(axes['view'], ['front'])

    def test_e_bike_pose_is_target_not_axis(self):
        stratum = next(item for item in plan('e_bike')['strata']
                       if (item.get('target') or {}).get('vehicles'))
        targets = state_coverage._e_bike_targets(stratum)
        self.assertTrue(targets['classes'])
        self.assertTrue(set(targets['classes']) <= {'upright', 'fallen'})
        self.assertNotIn('e_bike', state_coverage.AXIS_EXTRACTORS)
        direct = plan('e_bike', 'direct_real_plan.json')['strata'][0]
        self.assertEqual(state_coverage._e_bike_targets(direct)['count'], ['multi'])

    def test_partial_bike_review_keeps_confirmed_pose_but_not_total_count(self):
        target={'classes':['upright']*3,'complete_camera_label':False}
        self.assertEqual(state_coverage._e_bike_targets({'target':target}),
                         {'classes':['upright'],'count':None})

    def test_fire_count_and_facade(self):
        stratum = plan('fire')['strata'][0]
        self.assertEqual(state_coverage._fire_targets(stratum),
                         {'classes': [], 'count': ['zero']})
        self.assertIn('facade', state_coverage._fire_axes(stratum))

    def test_fire_building_counts_and_legacy_crop_keep_their_meaning(self):
        multi = {'target': {'fire_counts': {'A': 0, 'B': 2}}}
        self.assertEqual(state_coverage._fire_targets(multi),
                         {'classes': ['fire_sticker'], 'count': ['multi']})
        self.assertEqual(state_coverage._fire_axes(multi)['facade'], ['A', 'B'])
        direct = {'target': {'building': 'C', 'count': 0}}
        self.assertEqual(state_coverage._fire_axes(direct),
                         {'facade': ['C'], 'position': ['not_applicable']})
        legacy = {'target': {'kind': 'fire_sticker', 'count': 1},
                  'parameters': {'source_id': 'facade_A_04'}}
        self.assertEqual(state_coverage._fire_targets(legacy)['count'], ['single'])
        self.assertEqual(state_coverage._fire_axes(legacy)['facade'], ['facade_A_04'])
        self.assertEqual(state_coverage._fire_targets({'target': {'building': 'A'}}), {})
        self.assertEqual(state_coverage._fire_targets({'target': {'fire_counts': {'A': -1}}}), {})

    def test_fire_count_does_not_require_building_identity(self):
        for target in ({'building': None, 'count': 0}, {'count': 0}):
            with self.subTest(target=target):
                row = {'target': target, 'conditions': ['direct_real']}
                self.assertEqual(state_coverage._fire_targets(row),
                                 {'classes': [], 'count': ['zero']})
                self.assertEqual(state_coverage._fire_axes(row)['position'], ['not_applicable'])
                self.assertNotIn('facade', state_coverage._fire_axes(row))
        for count in (None, True, -1, '0'):
            self.assertEqual(state_coverage._fire_targets({'target': {'count': count}}), {})
        self.assertEqual(state_coverage._fire_targets({'target': {'count': 2}})['count'], ['multi'])

    def test_building_name_paper_target_and_building_axis(self):
        strata = plan('building_name_paper')['strata']
        source_positive = next(item for item in strata if
                               item['id'] == 'direct_frame_005101')
        replaced = next(item for item in strata if item['id'] == 'replace_0')
        negative = next(item for item in strata if item['id'] == 'direct_frame_004921')
        self.assertEqual(state_coverage._building_name_paper_targets(source_positive),
                         {'classes': ['paper'], 'presence': ['present'], 'instance_count': [1],
                          'text_count': [0], 'ocr': None})
        self.assertEqual(state_coverage._building_name_paper_targets(replaced)['instance_count'], [1])
        self.assertEqual(state_coverage._building_name_paper_targets(negative)['presence'], ['absent'])
        blender = plan('building_name_paper', 'blender_plan.json')['strata'][0]
        self.assertIn('building_count', state_coverage._building_name_paper_axes(blender))

    def test_paper_count_is_independent_of_text(self):
        stratum = {'target': {'texts': {'A': '同名', 'B': '同名'}}}
        targets = state_coverage._building_name_paper_targets(stratum)
        self.assertEqual(targets['instance_count'], [2])
        self.assertEqual(targets['text_count'], [2])

    def test_person_back_only_is_empty_and_vehicle_is_not_mutual_occlusion(self):
        stratum = {'target': {'actors': [{'id': 'back', 'kind': 'person', 'view': 'back'}]}}
        self.assertEqual(state_coverage._person_targets(stratum), {'classes': [], 'count': ['empty']})
        stratum = {
            'target': {'actors': [
                {'id': 'front', 'kind': 'person', 'view': 'front', 'category': 'community'},
                {'id': 'vehicle', 'kind': 'vehicle', 'category': 'upright'},
            ]},
            'parameters': {'required_overlaps': [['front', 'vehicle']]},
        }
        axes = state_coverage._person_axes(stratum)
        self.assertEqual(axes['mutual_occlusion'], ['not_applicable'])
        self.assertEqual(axes['vehicle_occlusion'], ['upright'])

    def test_sign_truncation_is_axis(self):
        stratum = next(item for item in plan('sign')['strata']
                       if (item.get('parameters') or {}).get('visible_kind') == 'identifiable_pole')
        self.assertEqual(state_coverage._sign_targets(stratum)['classes'], ['sign_fixture'])
        self.assertEqual(state_coverage._sign_axes(stratum)['truncation'], ['pole_only'])

    def test_sign_layouts_and_actual_geometry_preserve_unknowns(self):
        for layout, expected in [('small', 'none'), ('medium', 'none'), ('occluded', 'none'), ('partial', 'partial'),
                                 ('full', 'none'), ('far_full', 'none'), ('near_full', 'none'),
                                 ('top_clipped', 'partial'), ('pole_only', 'pole_only')]:
            with self.subTest(layout=layout):
                parameters = {'layout': layout, 'visible_kind':
                              'identifiable_pole' if layout == 'pole_only' else 'fixture'}
                planned = {'target': {'state': 'placed'}, 'conditions': ['front'],
                           'parameters': parameters}
                actual = {'actual': {'state': 'placed', 'view': 'front', 'layout': layout,
                                     'geometry': parameters}}
                self.assertEqual(state_coverage._sign_axes(planned)['truncation'], [expected])
                self.assertEqual(state_coverage._sign_axes(actual)['truncation'], [expected])
        legacy = {'target': {'state': 'placed'}, 'parameters': {'retain_lower_half': True}}
        self.assertEqual(state_coverage._sign_axes(legacy)['truncation'], ['pole_only'])
        self.assertIsNone(state_coverage._sign_axes({'target': {'state': 'placed'}})['truncation'])
        measured = {'image_size': [640, 480], 'actual': {
            'state': 'placed', 'view': 'front', 'layout': 'full', 'geometry': {
                'left': 10, 'top': -5, 'sprite_width_px': 30, 'sprite_height_px': 100}}}
        self.assertEqual(state_coverage._sign_axes(measured)['truncation'], ['partial'])
        measured['actual']['geometry']['top'] = 10
        self.assertEqual(state_coverage._sign_axes(measured)['truncation'], ['none'])

    def test_waste_bin_state_is_target_and_color_is_axis(self):
        stratum = next(item for item in plan('waste_bin_state', 'render_plan.json')['strata']
                       if (item.get('target') or {}).get('bins'))
        targets = state_coverage._waste_bin_state_targets(stratum)
        self.assertTrue(set(targets['classes']) <= {'open', 'closed'})
        axes = state_coverage._waste_bin_state_axes(stratum)
        self.assertTrue(set(axes['color']) <= {'blue', 'red', 'green', 'gray'})

    def test_waste_sticker_class_is_target_and_carrier_is_axis(self):
        stratum = plan('waste_sticker')['strata'][0]
        targets = state_coverage._waste_sticker_targets(stratum)
        self.assertTrue(targets['classes'])
        self.assertEqual(targets['presence'], ['present'])
        self.assertEqual(state_coverage._waste_sticker_targets({'target': {'placements': []}})['presence'],
                         ['absent'])
        self.assertEqual(state_coverage._waste_sticker_axes(stratum), {'carrier': ['bin_face']})

    def test_meter_needle_is_target_and_tilt_does_not_imply_distance(self):
        stratum = plan('meter', 'render_plan.json')['strata'][0]
        targets = state_coverage._meter_targets(stratum)
        self.assertEqual(targets['presence'], ['single'])
        self.assertTrue(targets['needle_position'])
        self.assertEqual(stratum['parameters']['ground_angle_deg'], 65)
        self.assertNotIn('distance', state_coverage._meter_axes(stratum))
        self.assertNotIn('view', state_coverage._meter_axes(stratum))

    def test_batch_request_is_separate_from_executable_budget(self):
        declaration = {'requested_budget': 40, 'executable_budget': 36, 'remaining_shortfall': 4}
        expected = {'requested_budget': 40, 'remaining_shortfall': 4}
        for field in ('allocation_evidence', 'quota_gap'):
            with self.subTest(field=field):
                self.assertEqual(state_coverage._batch_request({'budget': 36, field: declaration}), expected)
                with self.assertRaisesRegex(ValueError, '本批缺额不一致'):
                    state_coverage._batch_request({'budget': 39, field: declaration})
        self.assertEqual(state_coverage._batch_request({'budget': 36}), {})


class ActualBatchTest(unittest.TestCase):
    def test_declared_current_manifest_uses_exports_instead_of_old_sample(self):
        import shutil
        base=self.write_batch('sign',[{'image':'one.png','label':'one.txt','split':'train'}],
                              ['0 0.5 0.5 0.2 0.2\n'])
        export=self.root/'datasets/exports/sign'
        shutil.copytree(base,export)
        (base/'one.txt').write_text('',encoding='utf8')
        (self.root/'datasets/scripts/sign/config/dataset.yaml').write_text(
            'sample_output: datasets/samples/sign\nactual_manifest: datasets/exports/sign/manifest.json\n',encoding='utf8')
        result=state_coverage._summarize_actual_batch(self.root,'sign')
        self.assertIsNone(result['error'])
        self.assertEqual(result['path'],'datasets/exports/sign/manifest.json')
        self.assertEqual((result['images'],result['instances']),(1,1))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_batch(self, task, records, labels):
        from PIL import Image

        base = self.root / 'datasets/samples' / task
        base.mkdir(parents=True)
        config = self.root / 'datasets/scripts' / task / 'config/dataset.yaml'
        config.parent.mkdir(parents=True)
        config.write_text('sample_output: datasets/samples/{}\n'.format(task), encoding='utf-8')
        targets = self.root / 'config/task_targets.yaml'
        targets.parent.mkdir(parents=True)
        targets.write_text(yaml.safe_dump({'version': 1, 'tasks': {
            task: {'plans': ['config/production_plan.json'], 'classes': ['target']}}}), encoding='utf-8')
        for record, content in zip(records, labels):
            Image.new('RGB', (64, 48)).save(base / record['image'])
            if content is not None:
                (base / record['label']).write_text(content, encoding='utf-8')
        (base / 'manifest.json').write_text(json.dumps({'records': records}), encoding='utf-8')
        return base

    def test_fire_actual_uses_paired_labels_instead_of_requested_counts(self):
        records = [
            {'image': 'one.png', 'label': 'one.txt', 'requested_fire_counts': {'A': 99},
             'building_counts': {'A': {'visible_count': 1, 'whole_building_count': None}}},
            {'image': 'zero.png', 'label': 'zero.txt', 'requested_fire_counts': {'A': 3},
             'building_counts': {'A': {'visible_count': 0}, 'B': {'visible_count': 0}}},
        ]
        self.write_batch('fire', records, ['0 0.5 0.5 0.1 0.1\n', ''])
        result = state_coverage._summarize_actual_batch(self.root, 'fire')
        self.assertIsNone(result['error'])
        self.assertEqual((result['images'], result['instances']), (2, 1))
        self.assertEqual(result['counts_by_instances'], {'0': 1, '1': 1})
        self.assertEqual(result['building_counts']['A'], {'0': 1, '1': 1})
        self.assertEqual(result['axes']['facade'], {'A': 2, 'B': 1})
        self.assertEqual(result['unknown']['position'], 1)

    def test_actual_class_ids_follow_the_configured_class_order(self):
        records = [{'image': 'one.png', 'label': 'one.txt'}]
        self.write_batch('fire', records, ['1 0.5 0.5 0.1 0.1\n'])
        target = {'classes': ['first', 'second']}
        result = state_coverage._summarize_actual_batch(self.root, 'fire', target)
        self.assertIsNone(result['error'])
        self.assertEqual(result['class_instances'], {'second': 1})

    def test_sign_actual_keeps_missing_pairs_out_of_negative_counts(self):
        records = [
            {'image': 'partial.png', 'label': 'partial.txt', 'actual': {
                'state': 'placed', 'view': 'front', 'layout': 'top_clipped',
                'geometry': {'visible_kind': 'fixture'}}},
            {'image': 'empty.png', 'label': 'empty.txt', 'actual': {'state': 'negative'}},
            {'image': 'missing.png', 'label': 'missing.txt', 'actual': {'state': 'negative'}},
            {'image': 'retained.png', 'label': 'retained.txt', 'split': 'val',
             'actual': {'state': 'background_only', 'view': 'none'}},
        ]
        self.write_batch('sign', records, ['0 0.5 0.5 0.1 0.1\n', '', None, '0 0.5 0.5 0.1 0.1\n'])
        result = state_coverage._summarize_actual_batch(self.root, 'sign')
        self.assertIsNone(result['error'])
        self.assertEqual((result['recorded_images'], result['images'], result['instances']), (4, 3, 2))
        self.assertEqual(result['counts_by_instances'], {'0': 1, '1': 2})
        self.assertEqual(result['view_truncation'], {'front/partial': 1})
        self.assertEqual(result['not_applicable'], {'view': 1, 'truncation': 1})
        self.assertEqual(result['unknown'], {'view': 1, 'truncation': 1})
        self.assertEqual(result['splits']['val'], {'images': 1, 'instances': 1, 'negative_images': 0})
        self.assertEqual(len(result['missing_files']), 1)

    def test_unreadable_manifest_reports_unknown_instead_of_zero(self):
        base = self.write_batch('fire', [], [])
        (base / 'manifest.json').write_text('{', encoding='utf-8')
        result = state_coverage._summarize_actual_batch(self.root, 'fire')
        self.assertTrue(result['error'])
        self.assertIsNone(result['images'])
        self.assertIsNone(result['instances'])


class ReshootFineTunePlanSummaryTest(unittest.TestCase):
    def summarize_fixture(self, with_annotations=True, *, grouped=False,
                          recipe='all_color_combinations_all_binary_states_equal_repetitions', pair_images=192):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            root = Path(temporary)
            relative = 'config/reshoot_finetune_plan.json'
            declaration = {'purpose': 'four_color_real_appearance_finetune',
                           'annotation_manifest': 'annotations/reshoot.json',
                           'train': {'total': 544, 'source_photos': 32, 'real_cutout_composites': 384,
                                     'composites_per_color_state': 48, 'replay_images': 128}}
            if grouped:
                declaration['train'] = {
                    'total': 36 + 256 + pair_images + 128 + 192,
                    'source_photos': 36, 'real_cutout_composites': 256 + pair_images + 128,
                    'replay_images': 192, 'replay_empty_images': 64,
                    'composition_recipe': recipe,
                    'composition_groups': [{'count': 1, 'images': 256, 'heights_px': [90, 110, 150, 200]},
                                           {'count': 2, 'images': pair_images, 'heights_px': [90, 110, 140, 170]},
                                           {'count': 4, 'images': 128, 'heights_px': [90, 110, 125, 140]}]}
            declaration_path = root / 'datasets/scripts/waste_bin_state' / relative
            declaration_path.parent.mkdir(parents=True)
            declaration_path.write_text(json.dumps(declaration), encoding='utf-8')
            if with_annotations:
                annotations = root / declaration['annotation_manifest']
                annotations.parent.mkdir(parents=True)
                rows = [{'color': color, 'state': state, 'bins': [{}]}
                        for color in ('gray', 'red', 'blue', 'green')
                        for state in ('open', 'closed') for _ in range(4)]
                if grouped:
                    rows += [{'color': 'gray', 'state': 'closed', 'bins': [{}]} for _ in range(4)]
                annotations.write_text(json.dumps({'records': rows}), encoding='utf-8')
            target = {'classes': ['open', 'closed'], 'counts': ['empty', 'single', 'multi'],
                      'plan_usage': {relative: 'fine_tune'}}
            return state_coverage._summarize_plan(root, 'waste_bin_state', relative, target,
                                                   {'color': {}, 'view': {}, 'neighbor_occlusion': {}})

    def test_current_finetune_budget_preserves_replay_unknowns(self):
        result = self.summarize_fixture()
        self.assertEqual((result['planned_total'], result['budget'], result['split'], result['usage']),
                         (544, 544, 'train', 'fine_tune'))
        self.assertEqual(result['targets']['classes'], {'open': 208, 'closed': 208})
        self.assertEqual(result['target_unknown']['classes'], 128)
        self.assertEqual(result['coverage']['count'], {'single': 416})
        self.assertEqual(result['coverage_unknown']['count'], 128)
        self.assertEqual(result['axes']['color'], {'blue': 104, 'gray': 104, 'green': 104, 'red': 104})
        self.assertEqual(result['unknown'], {'color': 128, 'view': 544, 'neighbor_occlusion': 544})
        self.assertIsNone(result['capacity'])
        self.assertIsNone(result['error'])
        self.assertEqual(result['generation_methods'],
                         {'direct_real': 32, 'real_cutout_composite': 384, 'replay': 128})

    def test_missing_photo_annotations_do_not_invent_photo_classes_or_colors(self):
        result = self.summarize_fixture(with_annotations=False)
        self.assertEqual(result['targets']['classes'], {'open': 192, 'closed': 192})
        self.assertEqual(result['target_unknown']['classes'], 160)
        self.assertEqual(result['axes']['color'], dict.fromkeys(('blue', 'gray', 'green', 'red'), 96))
        self.assertEqual(result['unknown']['color'], 160)

    def test_multi_bin_finetune_keeps_image_presence_separate_from_instance_quotas(self):
        result = self.summarize_fixture(grouped=True)
        self.assertEqual((result['planned_total'], result['budget'], result['usage']), (804, 804, 'fine_tune'))
        # The 192 two-bin frames contain each state in 144 frames; the 128
        # four-bin frames contain each state in 120 frames. Mixed frames count
        # once per state present, while every box contributes an instance.
        self.assertEqual(result['targets']['classes'], {'open': 408, 'closed': 412})
        self.assertEqual(result['target_unknown']['classes'], 128)
        self.assertEqual(result['coverage']['count'], {'single': 292, 'multi': 320, 'empty': 64})
        self.assertEqual(result['coverage_unknown']['count'], 128)
        self.assertEqual(result['axes']['color'], {'blue': 296, 'gray': 300, 'green': 296, 'red': 296})
        self.assertEqual(result['unknown'], {'color': 128, 'view': 804, 'neighbor_occlusion': 804})
        self.assertEqual(result['not_applicable']['color'], 64)
        instances = result['instance_quotas']
        self.assertEqual(instances['classes'], {'open': 592, 'closed': 596})
        self.assertEqual(instances['known_instances'], 1188)
        self.assertEqual(instances['unresolved_images'], 128)
        self.assertEqual(instances['color_state'],
                         {f'{color}:{state}': 152 if (color, state) == ('gray', 'closed') else 148
                          for color in ('blue', 'gray', 'green', 'red') for state in ('open', 'closed')})
        self.assertEqual(result['generation_methods'],
                         {'direct_real': 36, 'real_cutout_composite': 576, 'replay': 192})

    def test_multi_bin_finetune_missing_photos_still_does_not_invent_annotations(self):
        result = self.summarize_fixture(with_annotations=False, grouped=True)
        self.assertEqual(result['targets']['classes'], {'open': 392, 'closed': 392})
        self.assertEqual(result['target_unknown']['classes'], 164)
        self.assertEqual(result['axes']['color'], dict.fromkeys(('blue', 'gray', 'green', 'red'), 288))
        self.assertEqual(result['unknown']['color'], 164)
        self.assertEqual(result['instance_quotas']['classes'], {'open': 576, 'closed': 576})
        self.assertEqual(result['instance_quotas']['known_instances'], 1152)
        self.assertEqual(result['instance_quotas']['unresolved_images'], 164)

    def test_multi_bin_finetune_requires_declared_and_divisible_state_recipe(self):
        with self.assertRaisesRegex(ValueError, '多桶状态组合配方'):
            self.summarize_fixture(grouped=True, recipe='unknown')
        with self.assertRaisesRegex(ValueError, '不能均匀覆盖'):
            self.summarize_fixture(grouped=True, pair_images=191)


class SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = state_coverage.summarize(ROOT)
        cls.body = state_coverage.render_markdown(cls.summary)

    def test_all_tasks_present(self):
        self.assertEqual(set(self.summary['tasks']), set(state_coverage.TASK_ORDER))

    def test_person_targets_exclude_vehicle_occlusion(self):
        person = self.summary['tasks']['person']
        self.assertIn('community', person['targets']['classes'])
        self.assertIn('noncommunity', person['targets']['classes'])
        self.assertIn('vehicle_occlusion', person['axes'])
        declaration = plan('person')
        capacity = sum(item['max_count'] for item in declaration['strata'])
        direct = len(set(declaration.get('direct_capture_ids', [])))
        self.assertEqual(person['train_capacity'], capacity)
        self.assertEqual(person['train_planned_direct'], direct)
        self.assertEqual(person['capacity_gap'], max(0, person['target_images'] - capacity - direct))

    def test_ebike_pose_is_target(self):
        bike = self.summary['tasks']['e_bike']
        self.assertIn('fallen', bike['targets']['classes'])
        self.assertIn('upright', bike['targets']['classes'])
        self.assertNotIn('pose', bike['axes'])

    def test_sticker_classes_are_target(self):
        sticker = self.summary['tasks']['waste_sticker']
        self.assertEqual(set(sticker['targets']['classes']), {str(i) for i in range(24)})
        self.assertNotIn('class', sticker['axes'])
        self.assertNotIn('bin_color', sticker['axes'])
        self.assertNotIn('bin_state', sticker['axes'])

    def test_bin_state_is_target_color_is_axis(self):
        bins = self.summary['tasks']['waste_bin_state']
        self.assertIn('open', bins['targets']['classes'])
        self.assertIn('closed', bins['targets']['classes'])
        self.assertIn('color', bins['axes'])
        self.assertNotIn('open', bins['axes']['color'])

    def test_sign_view_and_truncation_are_axes(self):
        sign = self.summary['tasks']['sign']
        self.assertEqual(set(sign['targets']['classes']), {'sign_fixture'})
        self.assertIn('front', sign['axes']['view'])
        self.assertIn('pole_only', sign['axes']['truncation'])

    def test_gap_reports_capacity_shortfall(self):
        sign = self.summary['tasks']['sign']
        self.assertEqual(sign['target_images'], 2500)
        capacity = sum(row['max_count'] for entry in sign['plans'] if entry['split'] == 'train'
                       for row in plan('sign', Path(entry['path']).name)['strata'])
        self.assertEqual(sign['train_capacity'], capacity)
        self.assertEqual(sign['capacity_gap'], 2500 - capacity)

    def test_source_render_stays_outside_train_budget(self):
        for task, usage, budget in (('meter', 'source_render', 6480),
                                    ('waste_bin_state', 'source_render', 0)):
            with self.subTest(task=task):
                item = self.summary['tasks'][task]
                self.assertIn(usage, {plan['usage'] for plan in item['plans']})
                self.assertEqual(item['train_planned'], budget)
                self.assertEqual(item['budget_gap'], item['target_images'] - budget)

    def test_current_plan_registration_uses_task_sources_and_excludes_legacy_samples(self):
        sources = yaml.safe_load((ROOT / 'datasets/scripts/waste_sticker/config/source.yaml').read_text(encoding='utf-8'))
        targets = state_coverage.load_targets()['tasks']['waste_sticker']
        base = ROOT / 'datasets/scripts/waste_sticker'
        expected = [(ROOT / sources[key]).relative_to(base).as_posix()
                    for key in ('production_plan', 'full_val_plan', 'full_test_plan')]
        fine_tune = (ROOT / sources['finetune_plan']).relative_to(base).as_posix()
        self.assertEqual(targets['plans'], expected + [fine_tune])
        rows = self.summary['tasks']['waste_sticker']['plans']
        self.assertEqual([row['path'] for row in rows if row['usage'] == 'dataset'], expected)
        self.assertEqual([row['path'] for row in rows if row['usage'] == 'fine_tune'], [fine_tune])
        self.assertEqual(sources['sample_manifest'], 'datasets/samples/waste_sticker/manifest.json')
        for key in ('legacy_direct_real_plan', 'legacy_blender_plan'):
            self.assertNotIn((ROOT / sources[key]).relative_to(base).as_posix(), expected)
        sources = yaml.safe_load((ROOT / 'datasets/scripts/e_bike/config/source.yaml').read_text(encoding='utf-8'))
        self.assertEqual(self.summary['tasks']['e_bike']['plans'][0]['path'],
                         (ROOT / sources['sample_plan']).relative_to(ROOT / 'datasets/scripts/e_bike').as_posix())
        self.assertEqual(self.summary['tasks']['e_bike']['train_planned'], 0)
        self.assertEqual(self.summary['tasks']['e_bike']['plans'][0]['usage'], 'sample')

    def test_sign_global_inventory_follows_current_split_plans(self):
        sources = yaml.safe_load((ROOT / 'datasets/scripts/sign/config/source.yaml').read_text(encoding='utf-8'))
        expected = [(ROOT / path).relative_to(ROOT / 'datasets/scripts/sign').as_posix()
                  for path in sources['full_dataset']['plans'].values()]
        self.assertEqual([entry['path'] for entry in self.summary['tasks']['sign']['plans']], expected)
        self.assertEqual({entry['split'] for entry in self.summary['tasks']['sign']['plans']}, set(sources['full_dataset']['plans']))

    def test_plan_inventory_is_not_maintained_in_axes(self):
        axes = state_coverage.load_axes()
        self.assertTrue(all('plans' not in task for task in axes['tasks'].values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'axes.yaml'
            # A stale legacy list cannot override or veto the target inventory.
            axes['tasks']['sign']['plans'] = ['config/production_plan.json']
            path.write_text(yaml.safe_dump(axes, allow_unicode=True), encoding='utf-8')
            summary = state_coverage.summarize(axes_path=path)
            self.assertEqual(summary['tasks']['sign']['plans'], self.summary['tasks']['sign']['plans'])

    def test_source_inventory_uses_split_capture_counts_for_each_role(self):
        negative = {'task': 'person', 'role': 'empty_negative', 'frames': 2, 'captures': 1, 'parents': 1}
        background = {'task': 'person', 'role': 'composite_background', 'frames': 1, 'captures': 1, 'parents': 1}
        result = state_coverage.render_source_status({
            'registered_frames': 2,
            'registered_paths': 2,
            'eligible_stock': [negative, background],
            'eligible_by_split': [
                {**negative, 'split': 'unassigned'},
                {**background, 'split': 'train'},
            ],
        })
        self.assertIn('| person | empty_negative | 2 / 1 / 1 | 2 / 1 / 1 | 0 / 0 / 0 |', result)
        self.assertIn('| person | composite_background | 1 / 1 / 1 | 0 / 0 / 0 | 1 / 1 / 1 |', result)

    def test_scene_counts_and_missing_axis_evidence_are_reported_separately(self):
        paper = self.summary['tasks']['building_name_paper']
        self.assertNotIn('instance_count', paper['targets'])
        counts = paper['plans'][0]['coverage']['instance_count']
        self.assertEqual(sum(counts.values()), paper['plans'][0]['planned_total'])
        self.assertGreater(counts.get('0', 0), 0)
        self.assertGreater(counts.get('3', 0), 0)
        self.assertNotIn('glare', paper['axes_spec'])
        bins = self.summary['tasks']['waste_bin_state']
        fine_tune_empty = sum(json.loads((ROOT / 'datasets/scripts/waste_bin_state' / p['path']).read_text(
            encoding='utf-8')).get('train', {}).get('replay_empty_images', 0)
            for p in bins['plans'] if p['usage'] == 'fine_tune')
        self.assertEqual(bins['not_applicable']['color'], 1 + fine_tune_empty)
        self.assertNotIn('not_applicable', bins['axes']['color'])
        meter = self.summary['tasks']['meter']
        sample = next(p for p in meter['plans'] if p['usage']=='sample')
        self.assertEqual(sample['unknown']['distance'], 55)
        self.assertEqual(sample['axes']['face_material'], {'white': 189, 'colored': 189})
        for axis in ('saturation','exposure','light_relation'):
            self.assertEqual(sum(sample['axes'][axis].values()),252)
            self.assertEqual(sample['unknown'][axis],181)
        self.assertEqual(meter['train_planned'],6480)
        self.assertEqual(meter['budget_gap'],0)

    def test_glare_is_a_design_axis_only_for_meter(self):
        for task, item in self.summary['tasks'].items():
            with self.subTest(task=task):
                axes = item['axes_spec']
                if task == 'meter':
                    self.assertIn('glare', axes)
                else:
                    self.assertNotIn('glare', axes)
                    self.assertTrue(all(not isinstance(values, list) or 'glare' not in values
                                        for values in axes.values()))

    def test_markdown_separates_targets_and_axes(self):
        self.assertIn('**训练目标覆盖', self.body)
        self.assertIn('**组合轴覆盖', self.body)
        target_part, axes_part = self.body.split('**组合轴覆盖', 1)
        self.assertIn('community', target_part)
        self.assertNotIn('community', axes_part)

    def test_target_and_axis_fields_do_not_overlap(self):
        targets = state_coverage.load_targets()
        axes = state_coverage.load_axes()
        target_fields = {'classes', 'counts', 'presence', 'ocr', 'keypoints',
                         'needle_axes', 'negatives', 'notes', 'plans'}
        for task in state_coverage.TASK_ORDER:
            self.assertFalse(
                target_fields & set(axes['tasks'][task]['axes']),
                '{} 的目标字段与组合轴重名'.format(task))
        self.assertIn('community', targets['tasks']['person']['classes'])
        self.assertNotIn('pose', axes['tasks']['e_bike']['axes'])
        self.assertNotIn('class', axes['tasks']['waste_sticker']['axes'])
        self.assertNotIn('state', axes['tasks']['waste_bin_state']['axes'])

    def test_generated_block_is_current(self):
        text = (ROOT / 'docs/DATASET_DESIGN.md').read_text(encoding='utf-8')
        start = text.index(state_coverage.START) + len(state_coverage.START)
        end = text.index(state_coverage.END)
        self.assertEqual(text[start:end].strip('\n'), self.body)

    def test_historical_observations_join_by_stable_id_and_missing_is_unknown(self):
        axes = state_coverage.load_axes()
        first, second = axes['tasks']['person']['crosses']
        original = {cross['id']: cross['evidence'] for cross in self.summary['tasks']['person']['crosses']}
        axes['tasks']['person']['crosses'] = [second, first, {
            'id': 'new_cross', 'purpose': 'new', 'pairs': 'new', 'check': 'new',
        }]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'axes.yaml'
            path.write_text(yaml.safe_dump(axes, allow_unicode=True), encoding='utf-8')
            changed = state_coverage.summarize(ROOT, axes_path=path)
            result = {cross['id']: cross['evidence'] for cross in changed['tasks']['person']['crosses']}
            self.assertEqual(result[first['id']], original[first['id']])
            self.assertEqual(result[second['id']], original[second['id']])
            self.assertEqual(result['new_cross'], '未知')
            missing = state_coverage.summarize(ROOT, observations_path=Path(temp) / 'missing.yaml')
            self.assertEqual(missing['tasks']['person']['crosses'][0]['evidence'], '未知')
            self.assertEqual(missing['observation_provenance']['status'], 'unknown')
        self.assertEqual(self.summary['observation_provenance']['status'], 'historical_snapshot')
        self.assertIsNone(self.summary['observation_provenance']['observed_at'])
        self.assertIn('不代表实时产出', self.body)

    def test_cross_structure_links_task_axes_and_historical_observations(self):
        axes = state_coverage.load_axes()
        for task, spec in axes['tasks'].items():
            self.assertTrue(spec['structure']['objects'])
            dimensions = spec['structure']['dimensions']
            self.assertTrue(dimensions)
            for cross in spec['crosses']:
                reported = next(row for row in self.summary['tasks'][task]['crosses']
                                if row['id'] == cross['id'])
                if 'structured' not in cross:
                    # Descriptive crosses remain valid; do not invent executable relations.
                    self.assertNotIn('structured', reported)
                    self.assertEqual(reported['check'], cross['check'])
                    continue
                structured = cross['structured']
                self.assertEqual(structured['completeness'], 'partial')
                self.assertTrue(structured['contrasts'])
                self.assertTrue(any(item.get('rule') for item in structured['constraints']))
                link = structured['evidence_links'][0]
                self.assertEqual((link['task'], link['cross_id']), (task, cross['id']))
                for relation in structured['contrasts']:
                    self.assertTrue(set(relation['hold'] + relation['vary']) <= dimensions.keys())
                reported = next(row for row in self.summary['tasks'][task]['crosses']
                                if row['id'] == cross['id'])
                self.assertEqual(reported['structured'], structured)
        self.assertIn('尚未自动接入计划生成', self.body)

    def test_invalid_structured_dimension_reference_is_reported(self):
        axes = state_coverage.load_axes()
        axes['tasks']['person']['crosses'][0]['structured']['contrasts'][0]['vary'] = ['missing.axis']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'axes.yaml'
            path.write_text(yaml.safe_dump(axes, allow_unicode=True), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '对照维度无效'):
                state_coverage.load_axes(path)

    def test_incomplete_relations_do_not_misstate_original_design(self):
        axes = state_coverage.load_axes()
        self.assertFalse(axes['structured_semantics']['executable'])
        self.assertEqual(axes['structured_semantics']['multi_dimension_variation'], 'unresolved')
        self.assertFalse(self.summary['structured_semantics']['executable'])
        def cross(task, identifier):
            return next(row for row in axes['tasks'][task]['crosses'] if row['id'] == identifier)['structured']
        for task, identifier in (('sign', 'fixture_background'), ('meter', 'background_glare')):
            item = cross(task, identifier)
            self.assertEqual(item['contrasts'][0]['vary'], ['target.presence'])
            self.assertTrue(item['unresolved'])
        person = cross('person', 'front_back')
        self.assertIn('同框', person['unresolved'][0])
        self.assertIn('薄边', person['unresolved'][0])
        vehicle = cross('e_bike', 'pose_context')
        self.assertIn({'kind': 'applicability', 'dimension': 'target.class',
                       'when': 'target.count=empty', 'result': 'not_applicable'}, vehicle['constraints'])
        meter = cross('meter', 'reading_view_light')
        self.assertIn('scene.incidence_angle', meter['contrasts'][0]['vary'])
        self.assertNotIn('target.view', meter['contrasts'][0]['vary'])
        self.assertEqual(axes['tasks']['meter']['structure']['dimensions']['scene.incidence_angle']['value_source'],
                         'unknown')
        for spec in axes['tasks'].values():
            for row in spec['crosses']:
                for relation in row.get('structured', {}).get('contrasts', []):
                    self.assertEqual(relation['variation'], 'unresolved' if len(relation['vary']) > 1 else 'single')


if __name__ == '__main__':
    unittest.main()
