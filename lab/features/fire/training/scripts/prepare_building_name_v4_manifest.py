"""Reconcile the existing reviewed name-paper selection with its V4 source manifest."""
import copy
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/data')]
from lab.training_common.production_guard import generation_guard
import v4_source_gate as gate

TASK = 'building_name_paper'
INTEGRATED = ROOT / 'datasets/exports/fire/v4_integrated_candidates_20261112/building_name_paper'
REVIEWED = ROOT / 'datasets/exports/fire/building_name_paper_v4_reviewed_20261113'
ARENA = ROOT / 'datasets/exports/fire/building_name_paper_v4_arena_text_20261113'
SOURCE = INTEGRATED / 'source_manifest.json'
DISTRIBUTION = INTEGRATED / 'distribution_manifest.candidate.json'
SELECTION = REVIEWED / 'workspace_sources.json'
ARENA_MANIFEST = ARENA / 'manifest.json'
VARIANTS = ROOT / 'datasets/exports/fire/building_name_paper_v4_arena_variants_20261113'
CONFIG = ROOT / 'lab/features/fire/training/config/building_name_paper_v4.yaml'
ARENA_SCENE = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
RENDER_BATCHES = [ROOT / f'datasets/exports/fire/fire_sticker_v4_arena_batch_{key}_20261113' for key in 'abc']


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def render_views():
    result = {}
    for batch in RENDER_BATCHES:
        for line in (batch / 'manifest.jsonl').read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            if ARENA_SCENE not in row['source_assets']:
                raise ValueError('render does not cite the arena source')
            placement = row['placement']
            name = f"vehicle_d{placement['distance_m']:.2f}_l{placement['lateral_m']:+.2f}"
            result[relative(batch / row['image'])] = {'kind': 'blender', 'source_group': ARENA_SCENE,
                                                       'view_group': name}
    return result


def prepare(source, selection, arena, views):
    records = [r for r in selection['records'] if r.get('source_group') != 'fitted_arena_name_variants']
    originals = {r['id']: r for r in source['samples']}
    by_image = {r.get('image'): r for r in source['samples']}
    arena_rows = {r['source_image']: r for r in arena['records']}
    kept = [copy.deepcopy(r) for r in source['samples']
            if r.get('split') not in ('train', 'val', 'test') and not r['id'].startswith('arena_parent_')]
    pairs = [p for p in source.get('near_duplicate_pairs', [])
             if not any(x.startswith('arena_parent_') for x in p)]
    grouped = {}
    for item in records:
        if item.get('source_group') != 'fitted_arena_shared_building_v13':
            original = originals.get(item['id']) or by_image.get(item.get('workspace_image'))
            if original is None or original.get('split') != item['split']:
                raise ValueError('missing matching prior reviewed source: ' + item['id'])
            row = copy.deepcopy(original)
            row['image'] = item['image']
            row['tasks'][TASK]['label'] = item['label']
            video = row.get('parent_video')
            processing = row.get('image_processing', {})
            if not video or not (ROOT / video).is_file() or processing.get('background_origin') != 'direct_camera':
                raise ValueError('real background provenance missing: ' + item['id'])
            row['background'] = {'kind': 'real', 'source_group': video}
            kept.append(row)
            continue
        prior = arena_rows.get(item.get('parent_image'))
        if not prior or prior['source_instance_pass'] != item['parent_instance_pass']:
            raise ValueError('arena parent or mask mismatch: ' + item['id'])
        if prior['ocr_status'] != 'not_verified' or item['review_state'] != 'complete_positive':
            raise ValueError('unexpected arena review state: ' + item['id'])
        pid = 'arena_parent_' + item['id']
        kept.append({'id': pid, 'image': item['parent_image'], 'split': 'unassigned',
                     'parent_ids': [], 'source_group': item['source_group'], 'source_kind': 'derived_image',
                     'generation_method': 'synthetic_render', 'selected_tasks': [], 'tasks': {}})
        tone = prior['tone']
        conditions = ['low_light', 'low_saturation'] if tone['exposure_ev'] < 0 and tone['saturation'] < 1 else []
        evidence = relative(ARENA_MANIFEST)
        kept.append({'id': item['id'], 'image': item['image'], 'split': item['split'],
                     'parent_ids': [pid], 'source_group': item['source_group'], 'source_kind': 'derived_image',
                     'generation_method': 'augmentation', 'source_assets': item['source_assets'],
                     'physical_assets': None, 'mother_scene': item['source_assets'][0],
                     'image_size': [640, 480], 'image_processing': {'resize': 'none', 'compression': 'PNG lossless',
                         'blur': 'none', 'origin_review': 'confirmed', 'parent_origin': 'synthetic'},
                     'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                     'review_evidence': evidence, 'selected_tasks': [TASK], 'tasks': {TASK: {
                         'state': 'complete_positive', 'label': item['label'], 'review_evidence': evidence,
                         'reviewer': item['review_basis'], 'human_accepted': False,
                         'strata_review': 'confirmed', 'strata': {'difficulty': 'fitted_arena_paper',
                             'conditions': conditions, 'target_states': ['complete_positive']}}},
                     'source_instance_pass': item['parent_instance_pass'], 'generated_text': prior['name_requested'],
                     'ocr_status': 'not_verified', 'tone': tone, 'background': views[item['parent_image']]})
        group = item['parent_pair']
        if group in grouped:
            pairs.append([grouped[group], pid])
        else:
            grouped[group] = pid
    ids = {r['id'] for r in kept}
    source['samples'] = kept
    source['near_duplicate_pairs'] = [p for p in pairs if all(x in ids for x in p)]
    source['models'][TASK]['data_yaml'] = relative(REVIEWED / 'data.yaml')
    source['status'] = 'candidate_not_frozen'
    source['scope'] = ('Existing representative and automated review; 210 fitted arena images reuse '
                       'rendered parents. The configured test is synthetic and shares scene assets. '
                       'This dataset labels paper detection; OCR readability has not been evaluated.')
    return source


def add_variants(source, selection, variants, refs, views, existing=None):
    existing = existing or {}
    selection['records'] = [r for r in selection['records'] if r.get('source_group') != 'fitted_arena_name_variants']
    parents = {r['parent_image']: r for r in selection['records']
               if r.get('source_group') == 'fitted_arena_shared_building_v13'}
    for row in variants['records']:
        parent = parents.get(row['source_image'])
        if not parent or parent['parent_instance_pass'] != row['source_instance_pass']:
            raise ValueError('variant parent not selected: ' + row['id'])
        conditions = (['low_light', 'low_saturation'] if row['tone']['exposure_ev'] < 0
                      and row['tone']['saturation'] < 1 else [])
        evidence = relative(VARIANTS / 'manifest.json')
        task = {'state': 'complete_positive', 'label': row['label'], 'review_evidence': evidence,
                'reviewer': 'instance-mask geometry and representative visual inspection', 'human_accepted': False,
                'strata_review': 'confirmed', 'strata': {'difficulty': 'fitted_arena_paper',
                    'conditions': conditions, 'target_states': ['complete_positive']}}
        item = {'id': 'arena_variant_' + row['id'], 'image': row['image'], 'label': row['label'], 'split': 'train',
                'parent_ids': ['arena_parent_' + parent['id']], 'source_group': 'fitted_arena_name_variants',
                'source_kind': 'derived_image', 'generation_method': 'augmentation', 'physical_assets': None,
                'source_assets': row['source_assets'], 'mother_scene': row['source_assets'][0],
                'image_size': [640, 480], 'image_processing': {'resize': 'none', 'compression': 'PNG lossless',
                    'blur': 'none', 'origin_review': 'confirmed', 'parent_origin': 'synthetic'},
                'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed', 'review_evidence': evidence,
                'selected_tasks': [TASK], 'tasks': {TASK: task}, 'source_instance_pass': row['source_instance_pass'],
                'generated_text': row['name_requested'], 'ocr_status': 'not_verified', 'tone': row['tone'],
                'background': views[row['source_image']],
                'indexed_source_references': existing.get('arena_variant_' + row['id'],
                                                          refs.get(row['source_image'], []))}
        source['samples'].append(item)
        selection['records'].append(copy.deepcopy(item))
    source['scope'] = ('Existing 210 fitted-arena images and 420 projected-text variants share 210 parent renders. '
                       'Synthetic test and scene assets are recorded; OCR readability has not been evaluated.')
    return source, selection


def main():
    source, selection, arena, variants = load(SOURCE), load(SELECTION), load(ARENA_MANIFEST), load(VARIANTS / 'manifest.json')
    views = render_views()
    existing = {row['id']: row['indexed_source_references'] for row in source['samples']
                if row['id'].startswith('arena_variant_') and 'indexed_source_references' in row}
    candidate = prepare(source, selection, arena, views)
    inputs = {ROOT / r['image'] for r in selection['records']}
    inputs.update(ROOT / r['parent_image'] for r in selection['records'] if r.get('parent_image'))
    inputs.update(ROOT / r['parent_instance_pass'] for r in selection['records'] if r.get('parent_instance_pass'))
    labels = {ROOT / r['label'] for r in selection['records']}
    inputs.update(ROOT / r['image'] for r in variants['records'])
    labels.update(ROOT / r['label'] for r in variants['records'])
    procedural = [SOURCE, DISTRIBUTION, SELECTION, CONFIG, ARENA_MANIFEST, VARIANTS / 'manifest.json', REVIEWED / 'data.yaml', *labels]
    procedural.extend(batch / 'manifest.jsonl' for batch in RENDER_BATCHES)
    queue_media, queue_procedural, queue_code = gate.queue_inputs(ROOT)
    inputs.update(queue_media)
    procedural.extend(queue_procedural)
    outputs = [SOURCE, DISTRIBUTION, SELECTION, CONFIG]
    with generation_guard('fire.prepare_building_name_v4_manifest', sorted(inputs), outputs, {'variants': len(variants['records'])},
                          code_paths=[Path(__file__), *queue_code], procedural_inputs=procedural,
                          inplace_paths=outputs, root=ROOT) as run:
        if run.reused:
            print('Existing name-paper manifest reused')
            return
        candidate, selection = add_variants(candidate, selection, variants, run.source_refs, views, existing)
        known = gate.build_queue(ROOT)
        audit = gate.audit(ROOT, candidate, known)
        if audit['status'] != 'passed' or audit['assigned'] != len(selection['records']):
            raise ValueError('source audit failed: ' + json.dumps(audit['errors'][:30], ensure_ascii=False))
        distribution = load(DISTRIBUTION)
        distribution['config'] = 'lab/features/fire/training/config/building_name_paper_v4.yaml'
        distribution['additional_records'] = [relative(ARENA_MANIFEST), relative(VARIANTS / 'manifest.json')]
        distribution['scope'] = ('1244 train, 50 validation, 50 synthetic test candidates; 630 of 1344 images use '
                                 'the fitted Blender arena as background. Scene and parent reuse is recorded. Training and testing not run.')
        config = yaml.safe_load(CONFIG.read_text(encoding='utf8'))
        config['data_overrides'] = {'train': [{'data': relative(REVIEWED / 'data.yaml'), 'split': 'train'},
            {'images': relative(VARIANTS / 'images/unassigned'), 'labels': relative(VARIANTS / 'labels/unassigned'),
             'source_group': 'fitted_arena_name_variants'}]}
        for path, record in ((SOURCE, candidate), (DISTRIBUTION, distribution), (SELECTION, selection)):
            content = json.dumps(record, ensure_ascii=False, indent=2) + '\n'
            if path.read_text(encoding='utf8') != content:
                path.write_text(content, encoding='utf8')
        config_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        if CONFIG.read_text(encoding='utf8') != config_text:
            CONFIG.write_text(config_text, encoding='utf8')
        print(json.dumps({'assigned': audit['assigned'], 'counts': audit['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
