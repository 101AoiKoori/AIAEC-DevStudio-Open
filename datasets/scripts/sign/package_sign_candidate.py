"""Package a planned sign candidate batch without asserting label approval."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir


def prepare(source, plan_path):
    source = Path(source).resolve()
    plan_path = Path(plan_path).resolve()
    load_task_config('sign')
    if source != sample_dir('sign').resolve() or not plan_path.is_relative_to(ROOT):
        raise ValueError('Source and plan must be inside the workspace')
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    if manifest.get('status') != 'complete_candidates' or plan.get('model_key') != 'sign':
        raise ValueError('Expected a completed sign candidate batch and its plan')
    samples = manifest.get('samples')
    if not isinstance(samples, list) or not samples or len(samples) != plan.get('budget'):
        raise ValueError('Batch manifest does not match its declared budget')
    files, labels, seen = [], [], set()
    for row in samples:
        image, label = (source / row['image']).resolve(), (source / row['label']).resolve()
        if not image.is_relative_to(source / 'images') or not label.is_relative_to(source / 'labels_candidate'):
            raise ValueError('Batch image or label path is outside its source directory')
        if image.stem != label.stem or image.stem in seen or not image.is_file() or not label.is_file():
            raise ValueError('Missing or duplicate candidate image and label')
        seen.add(image.stem)
        files.append(image)
        labels.append(label)
    return manifest, plan, files, labels, manifest_path


def require_usable_labels(manifest):
    """Honor explicit incompleteness while retaining computed candidate labels."""
    for row in manifest['samples']:
        if row.get('scope_complete') is False or row.get('review_status') in ('rejected', 'incomplete', 'excluded'):
            raise ValueError('Sign sample has rejected or incomplete labels: ' + str(row.get('id')))


def package(source, plan_path, output):
    source, plan_path, output = (Path(p).resolve() for p in (source, plan_path, output))
    manifest, plan, images, labels, manifest_path = prepare(source, plan_path)
    split = plan.get('target_split')
    if output != split_dir('sign', split).resolve():
        raise ValueError('Package output must match the declared sign train/test/val directory')
    require_usable_labels(manifest)
    config = [manifest_path, plan_path, *labels,
              ROOT/'datasets/scripts/sign/config/source.yaml',
              ROOT/'datasets/scripts/sign/config/dataset.yaml']
    code = [Path(__file__), ROOT / 'lab/training_common/production_guard.py']
    with generation_guard('sign:package_candidates:' + output.relative_to(ROOT).as_posix(),
                          images + config, [output], {'source_batch': source.relative_to(ROOT).as_posix(),
                          'count': len(images), 'label_status': 'candidate'},
                          code_paths=code, procedural_inputs=config, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output already exists; refusing replacement')
        (output / 'images').mkdir(parents=True)
        (output / 'labels').mkdir(parents=True)
        records = []
        for row, image, label in zip(manifest['samples'], images, labels):
            shutil.copy2(image, output / 'images' / image.name)
            shutil.copy2(label, output / 'labels' / label.name)
            records.append({'id': row['id'], 'state_id': row['state_id'],
                            'image': 'images/' + image.name, 'label': 'labels/' + label.name,
                            'sources': row['sources'], 'actual': row['actual'], 'label_status': 'candidate'})
        positive = sum(item['actual']['bbox_xyxy'] is not None for item in records)
        metadata = {'status': 'complete_candidates', 'task': 'sign', 'purpose': 'planned_label_package', 'split': split,
                    'source_batch': source.relative_to(ROOT).as_posix(),
                    'source_manifest': manifest_path.relative_to(ROOT).as_posix(),
                    'production_plan': plan_path.relative_to(ROOT).as_posix(),
                    'counts': {'images': len(records), 'candidate_positive_instances': positive,
                               'candidate_empty_labels': len(records) - positive},
                    'source_groups': {'background': sorted({x['actual']['background_video'] for x in records if x['actual'].get('background_video')}),
                                      'foreground': sorted({x['actual']['foreground_video'] for x in records if x['actual'].get('foreground_video')})},
                    'label_semantics': 'class 0 visible sign fixture; empty labels are sign-absence candidates',
                    'samples': records}
        (output / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = package(ROOT / args.source, ROOT / args.plan, ROOT / args.output)
    print(json.dumps({'status': result['status'], 'counts': result['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
