"""Adapt already reviewed fire candidates to the existing V4 distribution manifest.

The original source manifest is extended in place, not replaced by a parallel
lineage ledger.  The unchanged main V4 config is not modified.  Execution is
admitted by the shared database guard and fails before writes on gate errors.
"""
from __future__ import annotations

import copy
import filecmp
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/data'), str(ROOT / 'lab/training_common')]
from lab.training_common.production_guard import generation_guard
import v4_source_gate as gate

DIRECTORY = ROOT / 'datasets/exports/fire/v4_integrated_candidates_20261112/fire_sticker'
ORIGINAL = DIRECTORY / 'source_manifest.json'
DISTRIBUTION = DIRECTORY / 'distribution_manifest.candidate.json'
RECORDS = ROOT / 'datasets/exports/fire/fire_sticker_v4_candidate_entry_20261113/workspace_sources.json'
CONFIG = ROOT / 'lab/features/fire/training/config/fire_sticker_v4.yaml'
DELIVERY = ROOT / 'lab/features/fire/training/config/fire_sticker_v4_delivery_20261113.yaml'
ARENA_EVIDENCE = ROOT / 'datasets/annotated/fire/fire_sticker_v4_arena_handoff_20261113/instance_gate_20261113.json'
DISTRACTOR_EVIDENCE = ROOT / 'datasets/annotated/fire/fire_sticker_v4_arena_handoff_20261113/distractor_v4_gate_20261113.json'
DISTRACTOR_BATCHES = (
    'datasets/exports/fire/fire_sticker_v4_distractor_multiparent_20261113/manifest.json',
    'datasets/exports/fire/fire_sticker_v4_hardneg_coexist_20261113/manifest.json',
)
TRAIN_DIRS = (
    'datasets/exports/fire/v4_integrated_candidates_20261112/fire_sticker/images/train',
    'datasets/exports/fire/fire_sticker_v4_arena_v4_train_20261113/images/train',
    'datasets/exports/fire/fire_sticker_v4_distractors_v4_train_20261113/images/train',
)


def _json(path):
    return json.loads(path.read_text(encoding='utf8'))


def _relative(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError('missing or external input: ' + str(path))
    return resolved.relative_to(ROOT).as_posix()


def _selected(records):
    arena = [r for r in records if r.get('source_id') == 'arena_master_user_20261113_shared_scene']
    distractors = [r for r in records if r.get('source_manifest') in DISTRACTOR_BATCHES]
    if len(arena) != 210 or len(distractors) != 216:
        raise ValueError(f'expected 210 arena and 216 distractor records, got {len(arena)} and {len(distractors)}')
    return arena, distractors


def _inputs():
    records = _json(RECORDS)['records']
    original = _json(ORIGINAL)
    arena, distractors = _selected(records)
    batches = {name: {row['image']: row for row in _json(ROOT / name)['records']}
               for name in DISTRACTOR_BATCHES}
    proofs = (_json(ARENA_EVIDENCE), _json(DISTRACTOR_EVIDENCE))
    if any(proof.get('status') != 'passed' for proof in proofs):
        raise ValueError('candidate instance or source proof is not passed')
    arena_proof = {(item['batch'], item['record']['image']) for item in proofs[0]['selections']}
    distractor_proof = {item['record']['image'] for item in proofs[1]['selections']}
    media = set()
    procedural = {ORIGINAL, DISTRIBUTION, RECORDS, CONFIG, ARENA_EVIDENCE, DISTRACTOR_EVIDENCE}
    procedural.update(ROOT / name for name in DISTRACTOR_BATCHES)
    for row in original['samples']:
        if row.get('image'):
            media.add(ROOT / row['image'])
        if row.get('parent_video'):
            media.add(ROOT / row['parent_video'])
        for task in row.get('tasks', {}).values():
            if task.get('label'):
                procedural.add(ROOT / task['label'])
            if task.get('review_evidence'):
                procedural.add(ROOT / task['review_evidence'])
        if row.get('review_evidence'):
            procedural.add(ROOT / row['review_evidence'])
    for row in arena + distractors:
        media.add(ROOT / row['image'])
        procedural.add(ROOT / row['label'])
        if row in arena:
            batch = Path(row['source_batch']).name
            match = re.fullmatch(r'fire_sticker_v4_arena_batch_([abc])_20261113', batch)
            if not match or (match.group(1), row['source_image']) not in arena_proof:
                raise ValueError('arena sample lacks its recorded instance evidence: ' + row['image'])
            origin = (Path(row['source_batch']) / row['source_image']).as_posix()
        else:
            origin = row['source_image']
            if origin not in distractor_proof or origin not in batches[row['source_manifest']]:
                raise ValueError('distractor sample lacks its recorded evidence: ' + row['image'])
            item = batches[row['source_manifest']][origin]
            for field in ('background_parent', 'source_foreground', 'source_instance_pass'):
                if row.get(field):
                    media.add(ROOT / row[field])
            if item['background'].get('source_video'):
                media.add(ROOT / item['background']['source_video'])
        media.add(ROOT / origin)
        for field in ('source_review_evidence', 'review_evidence', 'source_scene', 'source_buildings'):
            if row.get(field):
                procedural.add(ROOT / row[field])
        if not filecmp.cmp(ROOT / origin, ROOT / row['image'], shallow=False):
            raise ValueError('selected sample differs from recorded original: ' + row['image'])
    queue_media, queue_procedural, queue_code = gate.queue_inputs(ROOT)
    media.update(queue_media)
    procedural.update(queue_procedural)
    procedural.add(ROOT / (gate.SITE_SEQUENCE + '/ffmpeg.log'))
    return original, arena, distractors, batches, sorted(media), sorted(procedural), queue_code


def _candidate(original, arena, distractors, batches, times, source_refs):
    data = copy.deepcopy(original)
    existing = {row['image']: row['id'] for row in data['samples']}
    all_ids = {row['id'] for row in data['samples']}
    parents = {}

    def parent(path):
        if path in existing:
            return existing[path]
        if path not in parents:
            _relative(ROOT / path)
            identifier = 'fire_delivery_parent_' + str(len(parents))
            if identifier in all_ids:
                raise ValueError('parent ID already exists: ' + identifier)
            all_ids.add(identifier)
            node = {'id': identifier, 'image': path, 'split': 'unassigned', 'parent_ids': [],
                    'selected_tasks': [], 'tasks': {}}
            if path.startswith(gate.SITE_SEQUENCE + '/'):
                match = re.fullmatch(r'frame_(\d+)\.png', path[len(gate.SITE_SEQUENCE) + 1:])
                if not match:
                    raise ValueError('invalid site sequence parent: ' + path)
                node.update(source_kind='video_frame', generation_method='provided_video_frame',
                            parent_video=gate.SITE_VIDEO, source_time_seconds=times[int(match.group(1))])
            data['samples'].append(node)
            parents[path] = identifier
        return parents[path]

    for index, row in enumerate(arena + distractors):
        is_arena = index < len(arena)
        origin = ((Path(row['source_batch']) / row['source_image']).as_posix()
                  if is_arena else row['source_image'])
        paths = [origin]
        background = {}
        if not is_arena:
            background = batches[row['source_manifest']][origin]['background']
            paths.append(row['background_parent'])
            if row.get('source_foreground'):
                paths.append(row['source_foreground'])
        identifier = 'fire_delivery_selected_' + str(index)
        if identifier in all_ids or row['image'] in existing:
            raise ValueError('sample already exists: ' + row['image'])
        all_ids.add(identifier)
        assets = ([row['source_scene'], row['source_buildings']] if is_arena else
                  [row[field] for field in ('background_parent', 'source_foreground', 'source_instance_pass')
                   if row.get(field)])
        if row.get('label_status') != 'reviewed' or row.get('review_state') not in ('complete_positive', 'complete_negative'):
            raise ValueError('unconfirmed task label: ' + row['image'])
        sample = {
            'id': identifier, 'image': row['image'], 'split': 'train', 'source_group': row['source_id'],
            'source_kind': row['source_kind'], 'generation_method': row['generation_method'],
            'parent_ids': list(dict.fromkeys(parent(path) for path in paths)),
            'physical_assets': ['site_and_building_props_shared'], 'source_assets': assets,
            'mother_scene': row.get('source_scene'), 'image_size': [640, 480],
            'image_processing': {'resize': ('copy 640x480 rendered output' if is_arena else
                                             'source recorded crop, resize and affine transform'),
                                 'compression': 'PNG lossless', 'blur': 'not recorded'},
            # These provisional values are written only after the machine
            # source gate checks the entire proposed graph successfully.
            'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
            'review_evidence': row['source_review_evidence'], 'selected_tasks': ['fire_sticker'],
            'tasks': {'fire_sticker': {'state': row['review_state'], 'label': row['label'],
                      'review_evidence': row['review_evidence'], 'reviewer': 'AI candidate and source gate',
                      'human_accepted': False, 'strata_review': 'pending'}},
            'source_review_scope': 'Machine source gate on recorded parents, known exposure and exact files; '
                                   'not human acceptance, exhaustive visual duplicate review or independent capture.',
            'indexed_source_references': source_refs[row['image']],
        }
        if background.get('source_video'):
            sample['parent_video'] = background['source_video']
            sample['source_time_seconds'] = background.get('source_time_seconds')
        if row.get('development_contact'):
            sample['development_contact'] = row['development_contact']
        data['samples'].append(sample)
    return data


def _delivery_config(config):
    config = copy.deepcopy(config)
    sources = config.get('data_overrides', {}).get('train', [])
    selected = [row for row in sources if row.get('images') in TRAIN_DIRS]
    if len(selected) != 3 or {row['images'] for row in selected} != set(TRAIN_DIRS):
        raise ValueError('main V4 config does not expose the three intended train sources')
    config['data_overrides']['train'] = selected
    config.setdefault('source_notes', []).append(
        'Delivery subset: original 584 train plus 210 fitted-arena and 216 distractor samples; '
        'remaining source candidates stay in the main V4 config. Same-site and asset reuse recorded.')
    return config


def main():
    original, arena, distractors, batches, media, procedural, queue_code = _inputs()
    config = _delivery_config(yaml.safe_load(CONFIG.read_text(encoding='utf8')))
    descriptor = _json(DISTRIBUTION)
    if DELIVERY.exists() and yaml.safe_load(DELIVERY.read_text(encoding='utf8')) != config:
        raise ValueError('existing delivery configuration does not match the selected inputs')
    with generation_guard('fire.prepare_v4_delivery_manifest_20261113', media,
                          [ORIGINAL, DISTRIBUTION, DELIVERY], {'arena': 210, 'distractor': 216},
                          code_paths=[Path(__file__), *queue_code], procedural_inputs=procedural,
                          inplace_paths=[ORIGINAL, DISTRIBUTION], adopt_outputs=[DELIVERY], root=ROOT) as run:
        if run.reused:
            print('Reused existing V4 delivery manifest and configuration')
            return
        if DELIVERY.exists() and yaml.safe_load(DELIVERY.read_text(encoding='utf8')) != config:
            raise ValueError('delivery configuration changed before production')
        data = _candidate(original, arena, distractors, batches, gate.site_sequence_times(ROOT), run.source_refs)
        result = gate.audit(ROOT, data, gate.build_queue(ROOT))
        if result.get('status') != 'passed' or result.get('assigned') != 1110:
            raise ValueError('source gate rejected delivery subset: ' + json.dumps(result.get('errors', [])[:30], ensure_ascii=False))
        # Records the actual, scoped machine check; it is not a new audit ledger.
        data['delivery_machine_source_check'] = {'status': result['status'], 'assigned': result['assigned'],
            'scope': 'Known exposure, recorded parent links, file identity and split checks; not human or independence certification.'}
        descriptor['config'] = DELIVERY.relative_to(ROOT).as_posix()
        descriptor['scope'] = ('Candidate subset 1010/50/50. Same site, props and model sources can recur; '
                               'test is a synthetic pressure set. No training or evaluation has run.')
        # Write only after every input, label and source-gate validation succeeded.
        try:
            with DELIVERY.open('x', encoding='utf8') as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        except FileExistsError:
            if yaml.safe_load(DELIVERY.read_text(encoding='utf8')) != config:
                raise ValueError('delivery configuration changed during production')
        if _json(ORIGINAL) != data:
            ORIGINAL.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        if _json(DISTRIBUTION) != descriptor:
            DISTRIBUTION.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'assigned': result['assigned'], 'train': 1010, 'val': 50, 'test': 50}, ensure_ascii=False))


if __name__ == '__main__':
    main()
