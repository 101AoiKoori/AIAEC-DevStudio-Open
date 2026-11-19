"""Collect existing real-image name-paper candidates without changing their status."""


import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.paths import sample_dir

MANIFESTS = tuple(ROOT / value for value in load_task_config('building_name_paper')[1]['direct_manifests'])

STATES = {'complete_positive', 'complete_negative',
          'candidate_complete_positive', 'candidate_complete_negative'}


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def collect():
    selected, seen = [], set()
    for manifest in MANIFESTS:
        data = json.loads(manifest.read_text(encoding='utf8'))
        records = data.get('images', data.get('samples', []))
        for index, row in enumerate(records):
            task = row.get('tasks', {}).get('building_name_paper', {})
            if task.get('state') not in STATES or not task.get('label'):
                continue
            image = (ROOT / row['image']).resolve()
            label = (ROOT / task['label']).resolve()
            if not image.is_relative_to(ROOT) or not label.is_relative_to(ROOT):
                raise ValueError('source outside workspace')
            if not image.is_file() or not label.is_file():
                raise ValueError('candidate image or label is missing')
            if image in seen:
                raise ValueError('source image occurs in more than one selected record')
            seen.add(image)
            selected.append((manifest, index, row, task, image, label))
    if not selected:
        raise ValueError('no existing real candidates')
    return selected


def collect_selection(selection_path):
    """Select explicit, non-reserved source records without upgrading labels."""
    selection_path = Path(selection_path).resolve()
    if not selection_path.is_relative_to(ROOT) or not selection_path.is_file():
        raise ValueError('selection must be an existing workspace file')
    request = json.loads(selection_path.read_text(encoding='utf8'))
    return collect_selection_request(request)


def bound_direct_review(relation, *, record_id=None):
    """Read the stored full-frame detection decision for this direct source."""
    role = 'direct_target' if relation['source_state'] == 'complete_positive' else 'empty_negative'
    if relation['role'] != role:
        raise ValueError('direct paper role differs from its task label polarity')
    database = ROOT / load_task_config('building_name_paper')[1]['databases']['frame_environment']
    rows = query_frames(task='building_name_paper', role=role, split=relation['split'],
                        record_id=record_id, database=database)
    matches = [row for row in rows if relation['image'] in row['paths']]
    if len(matches) != 1:
        raise ValueError('direct paper source lacks one eligible frame decision: ' + relation['image'])
    row = matches[0]
    label = row['label']
    if (not row['current'] or row['status'] != 'eligible' or row['scope']['kind'] != 'frame' or
            label['completeness'] != 'complete' or
            label['presence'] != ('present' if role == 'direct_target' else 'absent') or
            label.get('label') != relation['source_label'] or
            row['parent_video'] != relation['parent_source'] or
            image_digest(relation['image']) != row['content_sha256']):
        raise ValueError('direct paper source differs from stored task labels or identity')
    return {'role': role, 'frame_id': row['frame_id'], 'record_id': row['record_id'],
            'evidence_ids': [item['evidence_id'] for item in row['evidence']]}


def collect_selection_request(request):
    """Validate an in-memory selection using the same code as the packager."""
    from datasets.scripts.building_name_paper.plan_rooftop_batch import _reserved, _source_record

    if (request.get('model_key') != 'building_name_paper' or request.get('target_split') != 'train' or
            request.get('reservation_plan') != 'datasets/scripts/building_name_paper/config/combination_plan.json'):
        raise ValueError('unsupported direct candidate selection')
    master = ROOT / request['reservation_plan']
    reserved = _reserved(json.loads(master.read_text(encoding='utf8'))['v4_task_combination_plan'])
    rows = request.get('records')
    if not isinstance(rows, list) or not rows:
        raise ValueError('selection must contain candidate records')
    selected, seen = [], set()
    for relation in rows:
        if (relation.get('task') != 'building_name_paper' or relation.get('split') != 'train' or
                relation.get('role') not in ('direct_target', 'empty_negative')):
            raise ValueError('task, split and role must match on each relation')
        if relation.get('source_state') not in ('complete_positive', 'complete_negative'):
            raise ValueError('direct source requires complete detection labels')
        saved_review = relation.get('source_review')
        if (not isinstance(saved_review, dict) or saved_review != bound_direct_review(
                relation, record_id=saved_review.get('record_id'))):
            raise ValueError('direct source review revision differs from the saved selection')
        _, row = _source_record(relation, reserved)
        task = row.get('tasks', {}).get('building_name_paper', {})
        if (task.get('state') not in ('complete_positive', 'complete_negative') or
                task.get('state') != relation.get('source_state') or
                task.get('label') != relation.get('source_label')):
            raise ValueError('direct source label or state differs from its record')
        image = (ROOT / relation['image']).resolve()
        label = (ROOT / task['label']).resolve()
        manifest = (ROOT / relation['evidence']['manifest']).resolve()
        if not label.is_relative_to(ROOT) or not label.is_file() or image in seen:
            raise ValueError('missing or repeated selected image/label')
        lines = [line.split() for line in label.read_text(encoding='utf8').splitlines() if line.strip()]
        if bool(lines) != (task['state'] == 'complete_positive'):
            raise ValueError('existing candidate label does not match its state')
        for line in lines:
            if len(line) != 5 or line[0] != '0':
                raise ValueError('invalid paper candidate label')
            x, y, w, h = [float(value) for value in line[1:]]
            if (not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0 or
                    x-w/2 < 0 or y-h/2 < 0 or x+w/2 > 1 or y+h/2 > 1):
                raise ValueError('paper candidate label is outside image')
        selected.append((manifest, relation['evidence']['record_index'], row, task, image, label))
        seen.add(image)
    return selected


def package(output, selection_path=None):
    output = Path(output).resolve()
    if output != sample_dir('building_name_paper'):
        raise ValueError('direct candidates use the current building-name sample directory')
    selection_path = Path(selection_path or source_path('building_name_paper', 'direct_selection')).resolve()
    selected = collect_selection(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    database = ROOT / load_task_config('building_name_paper')[1]['databases']['frame_environment']
    reviews = {}
    working_images = {}
    for relation in selection['records']:
        review = relation['source_review']
        records = query_frames(task='building_name_paper', role=review['role'], split='train',
                               record_id=review['record_id'], database=database)
        if len(records) != 1:
            raise ValueError('saved direct paper decision cannot be read')
        working = (ROOT / records[0]['working_path']).resolve()
        if not working.is_relative_to(ROOT / 'datasets/raw/building_name_paper/targets') or \
                image_digest(records[0]['working_path']) != records[0]['content_sha256']:
            raise ValueError('direct paper raw copy differs from stored source')
        reviews[relation['image']] = review
        working_images[relation['image']] = working
    images = [item[4] for item in selected] + list(working_images.values())
    metadata = [selection_path, ROOT / 'datasets/scripts/building_name_paper/config/combination_plan.json',
                *(item[0] for item in selected), *(item[5] for item in selected)]
    images += [(ROOT / relation['parent_source']).resolve() for relation in selection['records']]
    code = [Path(__file__), Path(__file__).with_name('plan_rooftop_batch.py'),
            Path(__file__).with_name('compose_rooftop_papers.py'),
            ROOT / 'datasets/scripts/common/frame_environment.py',
            ROOT / 'datasets/scripts/common/paper.py', ROOT / 'datasets/scripts/common/perspective.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'datasets/scripts/common/names.py']
    metadata = list(dict.fromkeys(metadata))
    images = list(dict.fromkeys(images))
    metadata += [ROOT / 'datasets/scripts/building_name_paper/config' / name for name in ('dataset.yaml', 'source.yaml')]
    with generation_guard('building_name.package_real_candidates:' + relative(output),
                          images + metadata, [output], {'selection': relative(selection_path)},
                          code_paths=code, procedural_inputs=metadata, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        if output.exists() and any(output.iterdir()):
            raise ValueError('current sample directory is not empty')
        (output / 'images').mkdir(parents=True, exist_ok=True)
        (output / 'labels').mkdir()
        records = []
        for number, (manifest, index, row, task, image, label) in enumerate(selected):
            stem = f'real_{number:05d}'
            target_image = output / 'images' / (stem + image.suffix.lower())
            target_label = output / 'labels' / (stem + '.txt')
            working_image = working_images[relative(image)]
            shutil.copy2(working_image, target_image)
            shutil.copy2(label, target_label)
            records.append({
                'image': relative(target_image), 'label': relative(target_label),
                'source_image': relative(image), 'source_label': relative(label),
                'working_image': relative(working_image), 'source_review': reviews[relative(image)],
                'source_manifest': relative(manifest), 'record_index': index,
                'source_record': row, 'source_task': task,
                'source_refs': run.source_refs.get(relative(image), []),
                'usage': 'sample_from_eligible_direct_detection_source',
                'generation_method': row.get('generation_method', 'source_record_unspecified'),
                'ocr_status': 'not_verified',
            })
        result = {'status': 'direct_detection_sample',
                  'scope': 'Stored detection labels and history are preserved; original OCR remains unknown.',
                  'selection_plan': relative(selection_path),
                  'count': len(records), 'records': records}
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--selection', help='workspace JSON with explicit per-image candidate relations')
    args = parser.parse_args()
    result = package(ROOT / args.output, ROOT / args.selection if args.selection else None)
    print(json.dumps({'status': result['status'], 'count': result['count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
