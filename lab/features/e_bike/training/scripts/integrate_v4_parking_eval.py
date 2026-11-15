"""Register checked shared-source parking composites as V4 val/test data."""
import argparse
import filecmp
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from engine.config import load_config  # noqa: E402
from engine.data import inspect_dataset  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402

OWNER = ROOT / 'datasets/exports/e_bike'
SOURCE = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/staged_source_gate.json'
DISTRIBUTION = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/distribution_manifest.json'
CONFIGS = [ROOT / 'lab/features/e_bike/training/config/e_bike_v4_reviewed_858.yaml',
           ROOT / 'lab/features/e_bike/training/config/e_bike_v4.yaml']
BASE_DATA = 'datasets/exports/e_bike/v4_reviewed_858_20261113/data.yaml'
FOREGROUND_REVIEW = ROOT / 'datasets/exports/e_bike/v4_production_candidate_20261113/positive_parent_review_20261113.json'


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    args = parser.parse_args()
    batch = (ROOT / args.batch).resolve()
    if not batch.is_relative_to(OWNER.resolve()) or batch == OWNER.resolve():
        parser.error('batch must be an e-bike export subdirectory')
    manifest_path = batch / 'manifest.json'
    data = json.loads(manifest_path.read_text(encoding='utf8'))
    original_path = ROOT / data['source_manifest']
    qa_path = ROOT / data['source_qa']
    original = json.loads(original_path.read_text(encoding='utf8'))
    qa = json.loads(qa_path.read_text(encoding='utf8'))
    rows = data['records']
    if (Counter(row['split'] for row in rows) != {'val': 24, 'test': 48} or
            qa.get('error_count') != 0 or qa.get('images') != len(original['records']) or
            qa.get('status') != 'structural_passed_visual_sample_pending'):
        raise ValueError('Expected 24/48 selected samples and complete source structural QA')
    excluded = set(json.loads(FOREGROUND_REVIEW.read_text(encoding='utf8'))['excluded_foreground_stems'])
    if any(Path(obj['foreground_source']['path']).stem in excluded for row in rows for obj in row['objects']
           if obj.get('foreground_source')):
        raise ValueError('Selected sample uses an excluded foreground')
    if any(row['source_manifest'] != data['source_manifest'] for row in rows):
        raise ValueError('Selected sample does not refer to the recorded source manifest')
    config = load_config(ROOT, rel(CONFIGS[0]))
    _, active, _ = inspect_dataset(ROOT, config)
    active_records = active['train'] + active['val']
    media = [ROOT / item['image'] for item in active_records]
    labels = [ROOT / item['label'] for item in active_records]
    media.append(ROOT / qa['visual_evidence'])
    reviews = set()
    for row in rows:
        media.extend(ROOT / row[key] for key in ('image', 'source_image', 'background_image',
                                                  'background_parent', 'background_source_video'))
        labels.extend(ROOT / row[key] for key in ('label', 'source_label'))
        reviews.add(ROOT / row['original_review'])
        for obj in row['objects']:
            if obj.get('visible_mask'):
                media.append(ROOT / obj['visible_mask'])
            fg = obj.get('foreground_source')
            if fg:
                media.extend(ROOT / fg[key] for key in ('path', 'source_frame', 'source_video'))
        media.extend(ROOT / actor['source'] for actor in row.get('other_actors', []))
    procedural = [SOURCE, DISTRIBUTION, *CONFIGS, manifest_path, original_path, qa_path, FOREGROUND_REVIEW,
                  ROOT / config['data'], *reviews, *labels]
    output = [SOURCE, DISTRIBUTION, *CONFIGS]
    code = [Path(__file__), ROOT / 'lab/training_common/production_guard.py',
            ROOT / 'lab/training_common/engine/config.py', ROOT / 'lab/training_common/engine/data.py',
            ROOT / 'lab/training_common/train.py']
    with generation_guard('e_bike.integrate_v4_parking_eval:' + rel(batch),
                          list(dict.fromkeys(media)), output,
                          {'batch': rel(batch), 'val': 24, 'test': 48},
                          procedural_inputs=list(dict.fromkeys(procedural)),
                          code_paths=code, inplace_paths=output, root=ROOT) as run:
        if run.reused:
            return
        _integrate(batch, rows, original, qa_path, manifest_path, active_records)


def _validate(rows, original, active_records):
    """Check copied content and exact cross-split identity; no model evaluation."""
    occupied = defaultdict(list)
    for record in active_records:
        image = ROOT / record['image']
        occupied[image.stat().st_size].append(image)
    indices = set()
    for row in rows:
        index = row['source_record_index']
        if not isinstance(index, int) or index < 0 or index >= len(original['records']) or index in indices:
            raise ValueError('Missing or duplicate source record index')
        indices.add(index)
        parent = original['records'][index]
        if (row['source_image'] != parent['image'] or row['source_label'] != parent['label'] or
                row['image_size'] != parent['image_size'] or row['objects'] != parent['objects'] or
                row['tone'] != parent['tone'] or row['background_parent'] != parent['background_parent'] or
                row['background_image'] != parent['background_image'] or
                row.get('other_actors', []) != parent.get('other_actors', []) or
                row['background_source_video'] != parent['background_source_video'] or
                row['background_source_pts_seconds'] != parent['background_source_pts_seconds']):
            raise ValueError('Selected record differs from its recorded parent')
        image, parent_image = ROOT / row['image'], ROOT / row['source_image']
        label, parent_label = ROOT / row['label'], ROOT / row['source_label']
        if not filecmp.cmp(image, parent_image, shallow=False) or not filecmp.cmp(label, parent_label, shallow=False):
            raise ValueError('Selected image or label differs from its recorded parent')
        if not any(obj['class'] == 'fallen' for obj in row['objects']):
            raise ValueError('Expected fallen-bike positive')
        if any(filecmp.cmp(image, previous, shallow=False) for previous in occupied[image.stat().st_size]):
            raise ValueError('Exact output image already occurs in active train/val or another selection')
        occupied[image.stat().st_size].append(image)


def _integrate(batch, rows, original, qa_path, manifest_path, active_records):
    _validate(rows, original, active_records)
    source = json.loads(SOURCE.read_text(encoding='utf8'))
    distribution = json.loads(DISTRIBUTION.read_text(encoding='utf8'))
    assigned = Counter(row.get('split') for row in source['samples']
                       if 'e_bike' in row.get('selected_tasks', []))
    if assigned != {'train': 930, 'val': 4}:
        raise ValueError('Expected existing 930/4 baseline and no selected test records')
    ids = {row['id'] for row in source['samples']}
    images = {row.get('image') for row in source['samples']}
    for row in rows:
        stem = Path(row['image']).stem
        parent_id, sample_id = 'parking_eval_parent_' + stem, 'parking_eval_' + row['split'] + '_' + stem
        if row['image'] in images or parent_id in ids or sample_id in ids:
            raise ValueError('Selected record is already registered: ' + row['image'])
        images.add(row['image'])
        ids.update((parent_id, sample_id))
        source['samples'].append({'id': parent_id, 'image': row['source_image'],
                                  'parent_ids': [], 'split': 'unassigned', 'existence': 'present'})
        task = {'state': 'complete_positive', 'label': row['label'], 'review_evidence': rel(qa_path),
                'label_review': 'Inherited recorded parent boxes and generated mask/box QA; representative AI review, no per-image human acceptance.'}
        if row['tone'] in ('mild', 'moderate'):
            task.update({'strata': {'conditions': ['low_light', 'low_saturation']},
                         'strata_review': 'confirmed', 'strata_review_evidence': rel(manifest_path),
                         'strata_scope': 'relative synthetic HSV gains; illumination not physically measured'})
        foreground = [obj['foreground_source'] for obj in row['objects'] if obj.get('foreground_source')]
        source['samples'].append({
            'id': sample_id, 'image': row['image'], 'split': row['split'],
            'source_group': row['background_source_video'], 'source_kind': 'derived_image',
            'generation_method': 'composite', 'parent_ids': [parent_id], 'image_size': row['image_size'],
            'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
            'review_evidence': rel(qa_path), 'parent_video': row['background_source_video'],
            'source_time_seconds': row['background_source_pts_seconds'], 'physical_assets': None,
            'source_assets': [item['path'] for item in foreground] +
                             [actor['source'] for actor in row.get('other_actors', [])],
            'image_processing': {'resize': 'none; copy source composite without resampling',
                                 'compression': 'PNG copy', 'blur': 'none added', 'tone_profile': row['tone']},
            'selected_tasks': ['e_bike'], 'tasks': {'e_bike': task},
            'provenance': {'original_batch_manifest': row['source_manifest'],
                           'original_batch_record_index': row['source_record_index'],
                           'original_batch_image': row['source_image'],
                           'background_image': row['background_parent'],
                           'background_video': row['background_source_video'],
                           'background_time_seconds': row['background_source_pts_seconds'],
                           'foreground': foreground, 'evaluation_scope': 'shared-source synthetic parking scenes'},
            'unknowns': ['Unrecorded near duplicates and physical asset identities remain unknown.',
                         'Background and foreground sources are reused across development and test.',
                         'All new evaluation images contain fallen bikes; false positives need other cases.']})
    edits = []
    for config in CONFIGS:
        text = config.read_text(encoding='utf8')
        original_config = yaml.safe_load(text)
        overrides = original_config.get('data_overrides', {})
        if 'train' not in overrides or 'val' in overrides or 'test' in overrides:
            raise ValueError('Existing split overrides require explicit reconciliation')
        if original_config.get('test_unavailable') is not True or original_config.get('data') != BASE_DATA:
            raise ValueError('Existing V4 base/test configuration differs')
        text = text.replace('test_unavailable: true', 'test_unavailable: false', 1)
        text = text.replace('当前训练集 930 张、开发集 4 张', '当前训练集 930 张、开发集 28 张、测试集 48 张')
        text = text.replace('930 张训练和 4 张开发候选', '930 张训练、28 张开发及 48 张测试候选')
        text = text.replace('开发集无倒伏正例，当前数据入口未配置测试集。', '开发和测试集包含同源合成倒伏正例。')
        text = text.replace('开发集无倒伏正例，尚无测试集。', '开发和测试集包含同源合成倒伏正例。')
        text = text.replace('当前开发候选仅两个父帧且没有倒伏正例；已接触倒伏实拍单列开发回归，不作为独立测试。',
                            '原开发候选来自两个父帧；新增开发和测试为同源低视角场景合成，包含倒伏正例。')
        text = text.replace('样本复用实拍背景及前景，未知近重复和物理道具身份保留为未知；当前入口未配置测试集。',
                            '开发和测试复用背景视频及车辆前景；未知近重复与物料身份保留为未知。')
        text = text.replace('已知来源和重复范围通过门控；未知近重复及物理道具复用保留为未知。测试集不可用，不能冻结最终测试方案。',
                            '已知来源和重复范围通过门控；开发与测试共享背景和前景素材，不代表独立场次。')
        text += ('\n  val:\n'
                 f'  - data: {BASE_DATA}\n    split: val\n'
                 f'  - images: {rel(batch / "images/val")}\n'
                 f'    labels: {rel(batch / "labels/val")}\n'
                 '  test:\n'
                 f'  - images: {rel(batch / "images/test")}\n'
                 f'    labels: {rel(batch / "labels/test")}\n')
        updated = yaml.safe_load(text)
        if (updated['data_overrides']['train'] != overrides['train'] or
                updated['test_unavailable'] is not False or
                set(updated['data_overrides']) != {'train', 'val', 'test'}):
            raise ValueError('Updated configuration would alter existing train source')
        edits.append((config, text))
    distribution['scope'] = ('Candidate train/val/test 930/28/48; newly added evaluation images are '
                             'real-scene composites sharing background and foreground assets. '
                             'No independent-session, per-image human acceptance, training or final-test claim.')
    for evidence in (rel(manifest_path), rel(qa_path)):
        if evidence not in distribution['additional_records']:
            distribution['additional_records'].append(evidence)
    SOURCE.write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    DISTRIBUTION.write_text(json.dumps(distribution, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    for path, text in edits:
        path.write_text(text, encoding='utf8')
    print(json.dumps({'train': 930, 'val': 28, 'test': 48}, ensure_ascii=False))


if __name__ == '__main__':
    main()
