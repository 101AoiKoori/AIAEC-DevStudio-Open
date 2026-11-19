"""Produce fire candidates from explicitly reviewed facade footprints and a quota plan."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/.")

import argparse
import json
import os
import random
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
import cv2
import numpy as np

ROOT = Path(__file__).absolute().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/scripts/media'))
import media_index_sqlite as media_index
from lab.training_common.production_guard import generation_guard, plan_production
from generate_fire_dataset import _generate, read_image

ALLOWED_ROOTS = ('datasets/raw', 'Archive/Assets', 'datasets/assets/blender')
CLASSIFICATION = 'datasets/annotated/raw_source_classification_manifest_20261114.jsonl'
COPY_MANIFEST = 'datasets/source_copy_manifest.jsonl'
ARTWORK = 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'


def checked_path(value, *, source=False, must_exist=True):
    """Reject noncanonical paths and every existing symlink/reparse component."""
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value or
            PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() or
            any(p in ('', '.', '..') for p in value.split('/')) or PurePosixPath(value).as_posix() != value):
        raise ValueError('path must be canonical workspace-relative POSIX syntax')
    path = ROOT / value
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if must_exist:
                raise ValueError('path does not exist: ' + value)
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('symlink or reparse point is not supported: ' + str(part))
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(ROOT):
        raise ValueError('path escapes workspace')
    if source and not any(resolved.is_relative_to(ROOT / root) for root in ALLOWED_ROOTS):
        raise ValueError('source is outside allowed roots')
    if must_exist and not resolved.is_file():
        raise ValueError('required source or evidence is not a file: ' + value)
    return resolved


def evidence_row(ref, expected_path):
    if (not isinstance(ref, dict) or ref.get('path') != expected_path or
            type(ref.get('record_index')) is not int or ref['record_index'] < 0):
        raise ValueError('unsupported evidence reference')
    file = checked_path(expected_path)
    with file.open(encoding='utf8') as lines:
        for index, line in enumerate(lines):
            if index == ref['record_index']:
                return json.loads(line)
    raise ValueError('evidence record index missing')


def same_file_bytes(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            chunk = a.read(1024 * 1024)
            if chunk != b.read(1024 * 1024):
                return False
            if not chunk:
                return True


def validate_sources(data):
    """Support only recorded raw working copies and the user-provided artwork."""
    if (not isinstance(data, dict) or data.get('schema_version') != 1 or
            data.get('task') != 'fire_sticker' or data.get('purpose') != 'facade_candidate_generation' or
            data.get('allowed_roots') != list(ALLOWED_ROOTS) or not isinstance(data.get('sources'), list)):
        raise ValueError('invalid fire source manifest')
    items = {}
    media, evidence = set(), set()
    for item in data['sources']:
        if not isinstance(item, dict) or item.get('path') in items:
            raise ValueError('duplicate or invalid source')
        path = item.get('path')
        actual = checked_path(path, source=True)
        if item.get('exists_at_recording') is not True:
            raise ValueError('source presence was not recorded')
        classification = evidence_row(item.get('classification_evidence'), CLASSIFICATION)
        if classification.get('path') != path or classification.get('source_kind') != item.get('source_kind'):
            raise ValueError('source classification does not match evidence')
        evidence.add(ROOT / CLASSIFICATION)
        media.add(actual)
        if item['source_kind'] == 'archive_working_copy':
            if (item.get('role') not in ('background', 'reference') or item.get('material_type') != 'camera_photo_working_copy' or
                    item.get('allowed_root') != 'datasets/raw' or not actual.is_relative_to(ROOT / 'datasets/raw')):
                raise ValueError('unsupported photo source role')
            origin = item.get('origin', {})
            original = origin.get('path')
            ancestor = checked_path(original, source=True)
            copy = evidence_row(origin.get('evidence'), COPY_MANIFEST)
            if (origin.get('relation') != 'file_copy' or origin.get('allowed_root') != 'Archive/Assets' or
                    origin.get('exists_at_recording') is not True or not ancestor.is_relative_to(ROOT / 'Archive/Assets') or
                    origin.get('evidence') not in [{'path': r.get('path'), 'record_index': r.get('record_index')}
                                                   for r in classification.get('source_records', [])] or
                    classification.get('original_path') != original or
                    copy.get('path') != path or copy.get('source_image') != original or
                    copy.get('generation_method') != 'file_copy' or not same_file_bytes(actual, ancestor)):
                raise ValueError('copy lineage evidence or current bytes do not match')
            media.add(ancestor)
            evidence.add(ROOT / COPY_MANIFEST)
        elif item['source_kind'] == 'user_document_embedded_image':
            if (path != ARTWORK or item.get('allowed_root') != 'datasets/raw' or item.get('role') != 'artwork' or
                    item.get('material_type') != 'user_digital_artwork' or item.get('origin', {}).get('path') is not None):
                raise ValueError('unsupported artwork origin')
        else:
            raise ValueError('unsupported derived or source type')
        items[path] = item
    if ARTWORK not in items or not any(x.get('role') == 'background' for x in items.values()):
        raise ValueError('artwork and at least one background are required')
    return items, media, evidence


def ready_backgrounds(items):
    """Only admit full-frame negative truth and source-specific placement evidence."""
    ready, evidence = [], set()
    for item in items.values():
        if item.get('role') != 'background':
            continue
        gen = item.get('generation')
        if not isinstance(gen, dict) or gen.get('whole_image_fire_state') != 'complete_negative':
            continue
        ref = gen.get('admission_evidence')
        if (not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or
                not ref['path'].startswith('datasets/annotated/fire/') or not ref['path'].endswith('.jsonl')):
            raise ValueError('complete-negative background requires fire admission evidence')
        record = evidence_row(ref, ref['path'])
        review = record.get('task_review', {}).get('fire_sticker', {})
        if (record.get('path') != item['path'] or record.get('review_status') != 'confirmed' or
                review.get('state') != 'complete_negative' or review.get('exhaustive') is not True or
                review.get('scope') != 'entire_image' or record.get('traffic_signal') != 'not_visible' or
                record.get('placement_quads') != gen.get('quads')):
            raise ValueError('background truth or placement is not supported by admission evidence')
        entry = {**gen, 'image': item['path'], 'review_state': 'verified_empty',
                 'traffic_signal': 'not_observed', 'source_group': item['source_group'],
                 'material_type': item['material_type'], 'view': item['attributes']['view']}
        ready.append(entry)
        evidence.add(checked_path(ref['path']))
    return ready, evidence


def source_state_plan(count, seed, entries):
    """Allocate over feasible photo/count combinations and retain their actual parameters."""
    if count < 1:
        raise ValueError('budget must be positive')
    base = declaration(max(count, 4), seed, [e['image'] for e in entries])
    risk = {n: state for n, state in enumerate(base['strata'])}
    states = []
    for index, entry in enumerate(entries):
        supported = entry.get('supported_counts')
        if (not isinstance(entry.get('quads'), list) or not entry['quads'] or not isinstance(supported, list) or not supported or len(set(supported)) != len(supported) or
                any(type(n) is not int or n not in (0, 1, 2, 3) or n > len(entry['quads']) for n in supported)):
            raise ValueError('unsupported source/count combination')
        for n in supported:
            state = json.loads(json.dumps(risk[n]))
            state.update(id=f'photo_{index:03d}_fire_count_{n}', min_count=0, max_count=count)
            state['conditions'] = [f'count:{n}', f'source:{entry["image"]}']
            state['background'] = {'domain': 'real_facade_photo', 'source_image': entry['image'],
                                   'review_state': entry['review_state'], 'view': entry['view'],
                                   'material_type': entry['material_type']}
            state['target'] = {'kind': 'fire_sticker', 'count': n}
            state['parameters'] = {'base_seed': seed, 'source_image': entry['image'],
                                   'quads': entry['quads'], 'crop_xyxy': entry.get('crop_xyxy')}
            states.append(state)
    if not states:
        raise ValueError('no feasible source and target states')
    base.update(budget=count, strata=states)
    return base


def generate_for_state(state, by_path, sample_seed, item_file, icon, sample_dir, paper):
    """Send the allocated source, target count and footprints to the existing generator."""
    params, target = state['parameters'], state['target']
    chosen = by_path.get(params['source_image'])
    if (chosen is None or target.get('kind') != 'fire_sticker' or
            params['quads'] != chosen['quads'] or params['crop_xyxy'] != chosen.get('crop_xyxy') or
            target['count'] not in chosen['supported_counts']):
        raise ValueError('allocated state differs from source conditions')
    review = chosen['review_state']
    if review not in ('verified_empty', 'ai_reviewed_empty_candidate'):
        raise ValueError('unsupported background review state')
    item_file.write_text(json.dumps([{'image': chosen['image'], 'split': 'train',
        'review_state': review, 'verified_empty': review == 'verified_empty',
        'sticker_count': target['count'], 'quad': q} for q in chosen['quads']],
        ensure_ascii=False), encoding='utf8')
    _generate(item_file, icon, sample_dir, 1, sample_seed, paper_override=paper)
    return chosen


def declaration(count, seed, source_ids):
    states = []
    for n, prior, exposure, cost in [(0, (1, 5), 1.0, 1.0), (1, (2, 4), 1.4, 1.0),
                                      (2, (3, 4), 1.0, 1.3), (3, (4, 4), .8, 1.5)]:
        states.append({'id': f'fire_count_{n}', 'failure_event': 'coverage_risk_assumption',
            'min_count': 1, 'max_count': count - 3, 'prior': {'alpha': prior[0], 'beta': prior[1]},
            'exposure_weight': exposure, 'failure_cost': cost, 'conditions': [f'count:{n}'],
            'background': {'domain': 'real_facade_photo', 'sampling': 'seeded_uniform', 'source_ids': source_ids},
            'target': {'kind': 'fire_sticker', 'count': n},
            'labels': {'class_id': 0, 'format': 'yolo', 'status': 'generated_geometry_candidate'},
            'augmentation': {'method': 'perspective_paper_projection', 'scale': [0.78, 0.88], 'offset': [0, 0.12],
                             'paper': 'single_artwork_fitted_to_observed_paper_occupancy'},
            'sources': ['real_facade_photo', 'fire_sticker_artwork'],
            'parameters': {'base_seed': seed, 'source_sampling': 'one_per_source_then_uniform_with_replacement'}})
    return {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
            'purpose': 'prior-only facade fire sticker composition', 'budget': count,
            'uncertainty_weight': .5, 'independence_basis': 'photo selection units; physical session identity unknown',
            'constraints': {'traffic_signal': 'excluded_from_building_scenes',
                            'unlabelled_existing_fire': 'excluded', 'output': 'candidate'}, 'strata': states}


def select_sources(reviewed, seed):
    """Intersect indexed facade observations with explicitly inspected footprints."""
    manifest = checked_path('datasets/annotated/fire/background_source_manifest.jsonl')
    checked_path('datasets/annotated/media_index/query_cache.sqlite3')
    observations = {}
    for index, line in enumerate(manifest.read_text(encoding='utf8').splitlines()):
        row = json.loads(line)
        path = row['path']
        if (path.startswith(('datasets/raw/fire/', 'Archive/Assets/working_sources/legacy/fire/',
                             'Archive/Assets/captures/fire/')) and not row['observed_tasks']
                and '正面' in row['view']):
            observations[path] = index
    db = media_index.connect(readonly=True)
    try:
        db.execute('BEGIN')
        meta, _, full = media_index.validated(db, False)
        if not full:
            raise ValueError('media catalog is partial')
        pool = []
        for path in sorted(observations):
            row = db.execute('SELECT m.item_json,p.descriptor FROM media m JOIN partitions p '
                             'ON p.name=m.partition_name WHERE m.path_key=?',
                             (media_index.path_key(path),)).fetchall()
            if checked_path(path, source=True).is_file() and any(json.loads(item)['media_type'] == 'image' and
                    json.loads(part)['source_present'] and not media_index.core.lifecycle_reason(json.loads(part)['name'])
                    for item, part in row):
                pool.append(path)
    finally:
        db.close()
    by_path = {row['image']: row for row in reviewed}
    eligible = sorted(set(pool) & set(by_path))
    selected = random.Random(seed).sample(eligible, min(6, len(eligible)))
    if not selected:
        raise ValueError('no indexed and inspected source photos')
    return [by_path[path] for path in selected], {'snapshot_utc': meta['snapshot_utc'],
        'query': 'existing indexed raw/fire image; manifest observed_tasks empty and view includes 正面',
        'source_manifest': manifest.relative_to(ROOT).as_posix(), 'seed': seed,
        'sampling_unit': 'photo path', 'selection': 'uniform without replacement from reviewed intersection',
        'pool': [{'path': path, 'record_index': observations[path]} for path in pool],
        'excluded_from_reviewed_intersection': [path for path in pool if path not in by_path],
        'selected': selected, 'capture_session_independence': 'unknown'}


def frame_and_labels(image_path, label_path, crop):
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('cannot read generated image')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and (x2-x1)*3 == (y2-y1)*4):
        raise ValueError('crop must be an in-bounds 4:3 rectangle')
    rows = []
    for line in label_path.read_text(encoding='utf8').splitlines():
        cls, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        left, right = (cx-bw/2)*w, (cx+bw/2)*w
        top, bottom = (cy-bh/2)*h, (cy+bh/2)*h
        if left < x1 or top < y1 or right > x2 or bottom > y2:
            raise ValueError('crop would truncate a labelled target')
        rows.append(f'{cls} {(left+right-2*x1)/(2*(x2-x1)):.6f} '
                    f'{(top+bottom-2*y1)/(2*(y2-y1)):.6f} '
                    f'{(right-left)/(x2-x1):.6f} {(bottom-top)/(y2-y1):.6f}')
    resized = cv2.resize(image[y1:y2, x1:x2], (640, 480), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.png', resized)
    if not ok:
        raise ValueError('cannot encode framed image')
    encoded.tofile(str(image_path))
    label_path.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf8')
    return {'input_size': [w, h], 'crop_xyxy': crop, 'output_size': [640, 480],
            'interpretation': 'photo crop; not measured vehicle camera field of view'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--seed', type=int, default=20261114)
    a = p.parse_args()
    source = checked_path(a.sources)
    output = checked_path(a.output, must_exist=False)
    allowed = (ROOT / 'runs/fire_sticker').resolve()
    if not source.is_file() or not source.is_relative_to(ROOT) or not output.is_relative_to(allowed) or output.exists():
        raise ValueError('source must exist and candidate output must be new under runs/fire_sticker')
    source_data = json.loads(source.read_text(encoding='utf8'))
    items, media_inputs, evidence_inputs = validate_sources(source_data)
    reviewed, review_evidence = ready_backgrounds(items)
    if not reviewed:
        raise ValueError('no background has complete full-image truth and confirmed placement evidence')
    entries, selection = select_sources(reviewed, a.seed)
    if not entries or a.count < 1:
        raise ValueError('need eligible sources and a positive budget')
    for e in entries:
        if (e.get('review_state') != 'verified_empty' or e.get('traffic_signal') != 'not_observed'
                or not e.get('quads')):
            raise ValueError('every photo needs complete negative truth and supported placement')
    paths = [e['image'] for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError('source paths must be unique')
    icon = checked_path(ARTWORK, source=True)
    fitted_artwork = cv2.resize(read_image(icon), (114, 129), interpolation=cv2.INTER_AREA)
    paper = np.full((140, 100, 3), 255, dtype=np.uint8)
    paper[5:134] = fitted_artwork[:, 7:107]
    spec = source_state_plan(a.count, a.seed, entries)
    planned = plan_production(spec, ROOT)
    source_manifest = checked_path(selection['source_manifest'])
    inputs = sorted({source, source_manifest, *media_inputs, *evidence_inputs, *review_evidence})
    with generation_guard('fire.integrated:' + output.relative_to(ROOT).as_posix(), inputs, [output],
                          {'seed': a.seed, 'budget': a.count}, code_paths=[Path(__file__),
                          ROOT / 'lab/features/fire/training/scripts/generate_fire_dataset.py',
                          ROOT / 'lab/training_common/bayesian_allocation.py'],
                          procedural_inputs=sorted({source, source_manifest, *evidence_inputs, *review_evidence}),
                          production_plan=spec, sample_output_paths=[output / 'images'], root=ROOT) as run:
        if run.reused:
            return
        output.mkdir(parents=True)
        (output / 'plan.json').write_text(json.dumps(planned, ensure_ascii=False, indent=2), encoding='utf8')
        (output / 'source_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf8')
        (output / 'images/train').mkdir(parents=True)
        (output / 'labels/train').mkdir(parents=True)
        (output / 'classes.txt').write_text('fire_sticker\n', encoding='utf8')
        schedule = [s for s in run.plan['strata'] for _ in range(s['planned_count'])]
        random.Random(a.seed).shuffle(schedule)
        by_path = {entry['image']: entry for entry in entries}
        actual_sources = set()
        with (output / 'manifest.jsonl').open('w', encoding='utf8') as log:
            for i, state in enumerate(schedule):
                sample_seed = a.seed + i + 1
                n = state['target']['count']
                sample_dir = output / 'samples' / f'{i:06d}'
                item_file = output / f'source_{i:06d}.json'
                chosen = generate_for_state(state, by_path, sample_seed, item_file, icon, sample_dir, paper)
                actual_sources.add(chosen['image'])
                produced = json.loads((sample_dir / 'manifest.json').read_text(encoding='utf8'))[0]
                image = output / 'images/train' / f'fire_{i:06d}.png'
                label = output / 'labels/train' / f'fire_{i:06d}.txt'
                shutil.move(str(sample_dir / 'images/train/fire_000001.png'), str(image))
                shutil.move(str(sample_dir / 'labels/train/fire_000001.txt'), str(label))
                framing = frame_and_labels(image, label, chosen['crop_xyxy']) if chosen.get('crop_xyxy') else None
                produced['image'] = image.relative_to(ROOT).as_posix()
                produced['label'] = label.relative_to(ROOT).as_posix()
                (sample_dir / 'manifest.json').write_text(json.dumps([produced], ensure_ascii=False, indent=2), encoding='utf8')
                actual = {'sample_seed': sample_seed, 'parent': chosen['image'], 'parent_group': chosen['source_group'],
                          'generation_method': 'real_facade_plus_digital_artwork_projection', 'target_count': n,
                          'paper_artwork': {'icon_resize': [114, 129], 'icon_crop_x': [7, 107],
                                            'paper_y': [5, 134], 'paper_size': [100, 140],
                                            'basis': 'one observed printed reference; approximate, not calibrated'},
                          'projected': produced['projected'], 'background_state': chosen['review_state'],
                          'traffic_signal': chosen['traffic_signal'], 'framing': framing,
                          'placement_quads': chosen['quads'], 'uncertainty': chosen.get('uncertainty', [])}
                run.record_sample(state['id'], output=image, sources=state['sources'],
                                  parameters=state['parameters'], actual=actual)
                log.write(json.dumps({'image': image.relative_to(ROOT).as_posix(), 'label': label.relative_to(ROOT).as_posix(), 'state_id': state['id'],
                                      **actual}, ensure_ascii=False) + '\n')
                log.flush()
        (output / 'source_manifest.json').write_text(json.dumps({
            'schema_version': 1, 'task': 'fire_sticker', 'purpose': 'production_source_record',
            'candidate_manifest': a.sources, 'source_selection': selection,
            'actual_drawn_backgrounds': sorted(actual_sources), 'sample_records': 'manifest.jsonl',
            'sampling_unit': 'source_and_target_state', 'seed': a.seed,
            'validated_and_guarded_inputs': [items[path] for path in sorted(items)],
            'selected_candidate_backgrounds': paths,
            'observation': 'all allocated media inputs and their recorded parents were checked by production_guard'},
            ensure_ascii=False, indent=2), encoding='utf8')
        run.set_result({'count': a.count, 'source_count': len(entries), 'seed': a.seed})
    print('PRODUCED', a.count, output, flush=True)


if __name__ == '__main__':
    main()
