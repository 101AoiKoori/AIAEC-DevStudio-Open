"""Register a checked 640x480 candidate batch in the existing V4 source gate."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402

OWNER = ROOT / 'datasets/exports/e_bike'
SOURCE = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/staged_source_gate.json'
DISTRIBUTION = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/distribution_manifest.json'
CONFIGS = [ROOT / 'lab/features/e_bike/training/config/e_bike_v4_reviewed_858.yaml',
           ROOT / 'lab/features/e_bike/training/config/e_bike_v4.yaml']
BASE_DATA = 'datasets/exports/e_bike/v4_reviewed_858_20261113/data.yaml'


def rel(path):
    return path.resolve().relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    args = parser.parse_args()
    batch = (ROOT / args.batch).resolve()
    if not batch.is_relative_to(OWNER.resolve()) or batch == OWNER.resolve():
        parser.error('batch must be an e-bike export directory')
    manifest_path, qa_path = batch / 'manifest.json', batch / 'qa.json'
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    qa = json.loads(qa_path.read_text(encoding='utf-8'))
    rows = data['records']
    if not rows or qa.get('error_count') != 0 or qa.get('images') != len(rows):
        raise ValueError('Complete batch structural QA is required')
    media = [ROOT / row[key] for row in rows for key in ('image', 'source_image')]
    media += [ROOT / obj['visible_mask'] for row in rows for obj in row['objects']]
    media += [ROOT / row['source_background'][key] for row in rows for key in ('image', 'source_video')]
    media += [ROOT / path for row in rows for path in row['source_background'].get('person_sources', [])]
    media += [ROOT / obj['foreground_source'][key] for row in rows for obj in row['objects']
              for key in ('path', 'source_frame', 'source_video')]
    media.append(ROOT / qa['visual_evidence'])
    labels = [ROOT / row['label'] for row in rows]
    procedural = [SOURCE, DISTRIBUTION, *CONFIGS, manifest_path, qa_path, *labels]
    output = [SOURCE, DISTRIBUTION, *CONFIGS]
    operation = 'e_bike.integrate_v4_masked_train:' + rel(batch)
    with generation_guard(operation, list(dict.fromkeys(media)), output,
                          {'batch': rel(batch), 'records': len(rows)},
                          procedural_inputs=list(dict.fromkeys(procedural)),
                          code_paths=[Path(__file__), ROOT / 'lab/training_common/production_guard.py'],
                          inplace_paths=output, root=ROOT) as run:
        if run.reused:
            return
        _integrate(batch, rows, qa_path, manifest_path)


def _integrate(batch, rows, qa_path, manifest_path):
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    distribution = json.loads(DISTRIBUTION.read_text(encoding='utf-8'))
    known_ids = {row['id'] for row in source['samples']}
    existing_images = {row.get('image') for row in source['samples']}
    val_images = [ROOT / row['image'] for row in source['samples']
                  if row.get('split') == 'val' and 'e_bike' in row.get('selected_tasks', [])]
    if len(val_images) != 4:
        raise ValueError('Expected the existing four development images')
    for row in rows:
        image = ROOT / row['image']
        if row['image'] in existing_images:
            raise ValueError('Candidate is already registered: ' + row['image'])
        for val in val_images:
            if image.stat().st_size == val.stat().st_size and image.read_bytes() == val.read_bytes():
                raise ValueError('Exact output image also occurs in development: ' + row['image'])
        stem = Path(row['image']).stem
        parent_id, sample_id = 'resume_parent_' + stem, 'resume_train_' + stem
        if parent_id in known_ids or sample_id in known_ids:
            raise ValueError('Candidate ID already exists: ' + stem)
        known_ids.update((parent_id, sample_id))
        source['samples'].append({'id': parent_id, 'image': row['source_image'],
                                  'parent_ids': [], 'split': 'unassigned', 'existence': 'present'})
        bg = row['source_background']
        task = {'state': 'complete_positive' if row['objects'] else 'complete_negative',
                'label': row['label'], 'review_evidence': rel(qa_path),
                'label_review': 'Generated visible masks checked against labels; representative AI image review, no per-image human acceptance.'}
        if row['tone_profile'] in ('mild', 'moderate'):
            task.update({'strata': {'conditions': ['low_light', 'low_saturation']},
                         'strata_review': 'confirmed', 'strata_review_evidence': rel(manifest_path),
                         'strata_scope': 'relative synthetic HSV gains; illumination not physically measured'})
        source['samples'].append({
            'id': sample_id, 'image': row['image'], 'split': 'train',
            'source_group': bg['source_video'], 'source_kind': 'derived_image',
            'generation_method': 'composite', 'parent_ids': [parent_id], 'image_size': row['image_size'],
            'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
            'review_evidence': rel(qa_path), 'parent_video': bg['source_video'],
            'source_time_seconds': bg['seconds'], 'physical_assets': None,
            'source_assets': [obj['foreground_source']['path'] for obj in row['objects']],
            'image_processing': {'resize': '960x480 to 640x480 crop', 'compression': 'PNG',
                                 'blur': 'none added', 'crop_xyxy': row['crop_xyxy'],
                                 'tone_profile': row['tone_profile']},
            'selected_tasks': ['e_bike'], 'tasks': {'e_bike': task},
            'provenance': {'original_batch_manifest': rel(manifest_path),
                           'original_batch_image': row['source_image'],
                           'background_image': bg['image'], 'background_video': bg['source_video'],
                           'background_time_seconds': bg['seconds'],
                           'foreground': [obj['foreground_source'] for obj in row['objects']]},
            'unknowns': ['Unrecorded near duplicates and physical asset identities remain unknown.',
                         'Composite geometry and contact are visually fitted, not physically measured.']})
    new_count = sum(row.get('split') == 'train' and 'e_bike' in row.get('selected_tasks', [])
                    for row in source['samples'])
    edits = []
    for config in CONFIGS:
        text = config.read_text(encoding='utf-8')
        if 'data_overrides:' in text:
            raise ValueError('Existing data overrides require explicit reconciliation')
        text = text.replace('858 张训练和 4 张开发候选', f'{new_count} 张训练和 4 张开发候选')
        text = text.replace('当前训练集 858 张、开发集 4 张', f'当前训练集 {new_count} 张、开发集 4 张')
        text += ('\n# 新批次使用已有场景视频及前景裁片，来源关系见现有 V4 来源清单。\n')
        text += ('\ndata_overrides:\n  train:\n'
                 f'  - data: {BASE_DATA}\n    split: train\n'
                 f'  - images: {rel(batch / "images/train")}\n'
                 f'    labels: {rel(batch / "labels/train")}\n')
        edits.append((config, text))
    distribution['scope'] = (f'Candidate train/val {new_count}/4; no test. '
                             'New source images are real-scene composites sharing existing background and foreground parents. '
                             'No independent acquisition, per-image human acceptance, training or final-test claim.')
    for evidence in (rel(manifest_path), rel(qa_path)):
        if evidence not in distribution['additional_records']:
            distribution['additional_records'].append(evidence)
    SOURCE.write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    DISTRIBUTION.write_text(json.dumps(distribution, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for path, text in edits:
        path.write_text(text, encoding='utf-8')
    print(json.dumps({'added': len(rows), 'train_total': new_count, 'val': 4}, ensure_ascii=False))


if __name__ == '__main__':
    main()
