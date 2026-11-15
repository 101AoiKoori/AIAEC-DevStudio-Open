"""Collect existing real bin-state image/label candidates without altering them."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SITE_REVIEW = ROOT / 'datasets/annotated/waste_bin/v4_review_20261112/review_manifest.json'
ASSET_REVIEW = ROOT / 'datasets/annotated/waste_bin/v4_real_asset_review_20261112/review_manifest.json'
OLD_REVIEW = ROOT / 'datasets/annotated/waste_bin_state/real_review_20261113/manifest.json'
OUTPUT_ROOT = ROOT / 'runs/waste_bin_state'


def workspace_file(value, relative_to=None):
    if relative_to is None:
        relative_to = ROOT
    path = (relative_to / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('missing or non-workspace input: ' + str(value))
    return path


def collect():
    """Preserve source status and parent links; avoid known duplicate parent frames."""
    rows = []
    data = json.loads(SITE_REVIEW.read_text(encoding='utf8'))
    for item in data['images']:
        task = item['tasks']['waste_bin_state']
        if task.get('state') not in ('candidate_complete_positive', 'candidate_complete_negative') or not task.get('label'):
            continue
        rows.append({'id': 'site_' + item['id'], 'image': workspace_file(item['image']),
                     'label': workspace_file(task['label']), 'source_manifest': SITE_REVIEW,
                     'source_record_id': item['id'], 'source_group': item.get('source_group'),
                     'parent_video': item.get('parent_video'), 'parent_image': item['image'],
                     'generation_method': item.get('generation_method'), 'source_time_seconds': item.get('source_time_seconds'),
                     'prior_role': item.get('split'), 'source_label_state': task['state'],
                     'human_accepted': task.get('human_accepted'), 'expected_boxes': len(task.get('boxes', []))})
    data = json.loads(ASSET_REVIEW.read_text(encoding='utf8'))
    for item in data['samples']:
        task = item['tasks']['waste_bin_state']
        if task.get('state') != 'candidate_complete_positive' or not task.get('label'):
            continue
        rows.append({'id': 'asset_' + item['id'], 'image': workspace_file(item['image'], ASSET_REVIEW.parent),
                     'label': workspace_file(task['label'], ASSET_REVIEW.parent), 'source_manifest': ASSET_REVIEW,
                     'source_record_id': item['id'], 'source_group': item.get('source_group'),
                     'parent_video': item.get('source_video'), 'parent_image': item.get('parent_image'),
                     'generation_method': item.get('generation_method'), 'source_frame': item.get('source_frame'),
                     'prior_role': item.get('split'), 'source_label_state': task['state'],
                     'human_accepted': task.get('human_accepted'), 'expected_boxes': len(task.get('candidate_boxes', []))})
    data = json.loads(OLD_REVIEW.read_text(encoding='utf8'))
    item = next(row for row in data['records'] if row['image'].endswith('/0010.png'))
    rows.append({'id': 'old_0010', 'image': workspace_file(item['image']), 'label': workspace_file(item['label']),
                 'source_manifest': OLD_REVIEW, 'source_record_id': '0010',
                 'source_group': item.get('source_video'), 'parent_video': item.get('source_video'),
                 'parent_image': item.get('parent_image'), 'source_frame': item.get('source_frame'),
                 'source_time_seconds': item.get('source_time_seconds_approx'),
                 'generation_method': 'video_frame_area_downscale', 'prior_role': item.get('role'),
                 'source_label_state': data['status'], 'human_accepted': False,
                 'expected_boxes': len(item['objects'])})
    if len(rows) != 17 or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('source records changed; inspect selection before copying')
    for row in rows:
        lines = row['label'].read_text(encoding='utf8').splitlines()
        if len(lines) != row['expected_boxes'] or any(len(line.split()) != 5 or line.split()[0] not in ('0', '1') for line in lines):
            raise ValueError('source label format or count differs from its manifest: ' + row['id'])
    return rows


def collect_planned(plan_path):
    """Resolve explicitly selected image-use relations without changing labels."""
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    if declaration.get('model_key') != 'waste_bin_state' or declaration.get('target_split') != 'train':
        raise ValueError('planned intake requires waste_bin_state/train')
    relation_path = workspace_file(declaration.get('relation_source', ''))
    source = json.loads(relation_path.read_text(encoding='utf8'))
    relations = source['handoff']['image_task_relations']
    requests = declaration.get('records')
    if not isinstance(requests, list) or not requests:
        raise ValueError('planned intake needs at least one image relation')
    rows, seen = [], set()
    for request in requests:
        index, use_index = request.get('relation_index'), request.get('use_index')
        if type(index) is not int or not 0 <= index < len(relations):
            raise ValueError('invalid image relation index')
        relation = relations[index]
        uses = relation.get('task_use_status', [])
        if type(use_index) is not int or not 0 <= use_index < len(uses):
            raise ValueError('invalid task use index')
        use = uses[use_index]
        if (use.get('task'), use.get('split'), use.get('role'), use.get('status')) not in (
                ('waste_bin_state', 'train', 'direct_sample', 'candidate'),
                ('waste_bin_state', 'train', 'direct_sample', 'ready')):
            raise ValueError('selected relation is not a train direct candidate')
        if use.get('label_status') not in ('candidate_complete_positive', 'candidate_complete_negative'):
            raise ValueError('selected source has incomplete state labels')
        if request.get('image') != relation.get('image') or request.get('label') != use.get('label'):
            raise ValueError('planned image or label differs from selected relation')
        image, label = workspace_file(relation['image']), workspace_file(use['label'])
        if image in seen:
            raise ValueError('duplicate direct image in plan')
        seen.add(image)
        original = relation['original_record']
        manifest = workspace_file(original['manifest'])
        pointer = original.get('json_pointer', '')
        parts = pointer.strip('/').split('/')
        if len(parts) != 2 or parts[0] != 'images' or not parts[1].isdigit():
            raise ValueError('unsupported original review record pointer')
        review = json.loads(manifest.read_text(encoding='utf8'))['images'][int(parts[1])]
        task = review['tasks']['waste_bin_state']
        if (review['image'] != relation['image'] or review.get('split') == 'regression' or
                task.get('label') != use.get('label') or task.get('state') != use.get('label_status') or
                task.get('boxes', []) != use.get('candidate_boxes', [])):
            raise ValueError('source relation differs from original review or is regression')
        lines = label.read_text(encoding='utf8').splitlines()
        expected = len(use.get('candidate_boxes', []))
        if len(lines) != expected or any(len(line.split()) != 5 or line.split()[0] not in ('0', '1') for line in lines):
            raise ValueError('source label format or count differs from relation')
        if use['label_status'] == 'candidate_complete_negative' and expected:
            raise ValueError('negative relation must not contain candidate boxes')
        rows.append({'id': request['id'], 'image': image, 'label': label, 'source_manifest': manifest,
                     'source_record_id': original.get('json_pointer'), 'source_group': relation.get('source_group'),
                     'parent_video': relation.get('parent_video'), 'parent_image': relation['image'],
                     'source_time_seconds': relation.get('source_time_seconds'),
                     'prior_role': use.get('split_basis'), 'source_label_state': use['label_status'],
                     'human_accepted': use.get('human_accepted'), 'expected_boxes': expected,
                     'requested_split': 'train', 'relation_source': str(relation_path.relative_to(ROOT)),
                     'relation_index': index, 'use_index': use_index, 'combination_id': request['combination_id']})
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('planned intake IDs must be unique')
    return rows, relation_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan', help='optional explicit candidate source plan')
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(OUTPUT_ROOT.resolve()) or out == OUTPUT_ROOT.resolve():
        parser.error('candidate output must be a new directory under runs/waste_bin_state')
    if args.plan:
        plan_path = workspace_file(args.plan)
        rows, relation_path = collect_planned(plan_path)
    else:
        plan_path, relation_path = None, None
        rows = collect()
    media = [row['image'] for row in rows]
    procedural = list({row['label'] for row in rows} | {row['source_manifest'] for row in rows})
    if plan_path:
        procedural.extend((plan_path, relation_path))
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin_state.real_candidate_intake:' + out.relative_to(ROOT).as_posix(),
                          media, [out], {'selection': args.plan or 'site_10_asset_6_old_0010', 'split': 'unassigned'},
                          code_paths=[Path(__file__).resolve()], procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        out.mkdir(parents=True, exist_ok=False)
        (out / 'images').mkdir()
        (out / 'labels').mkdir()
        result = []
        for row in rows:
            image_out = out / 'images' / (row['id'] + row['image'].suffix.lower())
            label_out = out / 'labels' / (row['id'] + '.txt')
            shutil.copyfile(row['image'], image_out)
            shutil.copyfile(row['label'], label_out)
            record = {key: value for key, value in row.items() if key not in ('image', 'label', 'source_manifest')}
            record.update({'image': image_out.relative_to(ROOT).as_posix(), 'label': label_out.relative_to(ROOT).as_posix(),
                           'source_image': row['image'].relative_to(ROOT).as_posix(),
                           'source_label': row['label'].relative_to(ROOT).as_posix(),
                           'source_manifest': row['source_manifest'].relative_to(ROOT).as_posix(),
                           'split': 'unassigned'})
            result.append(record)
        report = {'status': 'unreviewed_candidates', 'model_key': 'waste_bin_state',
                  'scope': 'Existing camera images/video frames and original candidate labels; no human acceptance inferred.',
                  'allowed_use': 'candidate source inventory; not an approved training export',
                  'image_count': len(result), 'candidate_instances': sum(row['expected_boxes'] for row in result),
                  'records': result}
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'status': report['status'], 'images': report['image_count'],
                          'candidate_instances': report['candidate_instances'], 'output': out.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
