"""Package declared YOLO image/label batches as an unverified train-only ZIP.

The JSON spec has model_key, dataset (YOLO header with names and optional
kpt_shape), and batches. Each batch has id, images, labels, evidence (file
paths), source_kind, generation_method, label_status and a dataset header
matching the package header. Use either images/labels directories or records
containing explicit image/label paths. Optional notes,
lineage and failures are copied verbatim. Paths are workspace-relative; image
and label directories are scanned one level and matched by stem. Evidence is
preserved as supplied, not interpreted as independent acquisition or approval.
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _path(root, value, file=False):
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError('expected a workspace-relative path')
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('path escapes workspace: ' + value)
    if file and not path.is_file():
        raise ValueError('missing file: ' + value)
    return path


def _text(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('missing ' + key)
    return value


def prepare(root, spec_path):
    root = Path(root).resolve()
    spec_path = _path(root, str(spec_path), file=True)
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    model = _text(spec, 'model_key')
    header = spec.get('dataset')
    if not isinstance(header, dict) or not header.get('names') or any(k in header for k in ('path', 'train', 'val', 'test')):
        raise ValueError('dataset must contain names and no path or split fields')
    batches = spec.get('batches')
    if not isinstance(batches, list) or not batches:
        raise ValueError('at least one batch is required')
    entries, evidence, seen_ids = [], set(), set()
    for batch in batches:
        batch_id = _text(batch, 'id')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', batch_id) or batch_id.casefold() in seen_ids:
            raise ValueError('batch IDs must be unique and contain only letters, digits, underscore or hyphen')
        seen_ids.add(batch_id.casefold())
        for key in ('source_kind', 'generation_method', 'label_status'):
            _text(batch, key)
        if batch.get('dataset') != header:
            raise ValueError('batch dataset class/keypoint contract differs from package: ' + batch_id)
        batch_evidence = batch.get('evidence')
        if not isinstance(batch_evidence, list) or not batch_evidence:
            raise ValueError('each batch must name existing evidence files')
        evidence.update(_path(root, item, file=True) for item in batch_evidence)
        rows = batch.get('records')
        if rows is not None:
            if 'images' in batch or 'labels' in batch or not isinstance(rows, list) or not rows:
                raise ValueError('records must be nonempty and cannot be combined with directories')
            pairs = [(_path(root, _text(row, 'image'), file=True),
                      _path(root, _text(row, 'label'), file=True)) for row in rows]
        else:
            images, labels = (_path(root, _text(batch, key)) for key in ('images', 'labels'))
            if not images.is_dir() or not labels.is_dir() or not images.is_relative_to(root/'datasets') or not labels.is_relative_to(root/'datasets'):
                raise ValueError('image and label directories must exist inside datasets')
            files = sorted((p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name.casefold())
            if not files:
                raise ValueError('no images in batch ' + batch_id)
            stems, pairs = set(), []
            for image in files:
                image = image.resolve()
                if not image.is_relative_to(images) or not image.is_file() or image.stem.casefold() in stems:
                    raise ValueError('duplicate or invalid image in batch ' + batch_id)
                stems.add(image.stem.casefold())
                label = (labels / (image.stem + '.txt')).resolve()
                if not label.is_relative_to(labels) or not label.is_file():
                    raise ValueError('missing label for ' + str(image.relative_to(root)))
                pairs.append((image, label))
        for index, (image, label) in enumerate(pairs):
            if (not image.is_relative_to(root/'datasets') or not label.is_relative_to(root/'datasets')
                    or image.suffix.lower() not in IMAGE_SUFFIXES or label.suffix.lower() != '.txt'):
                raise ValueError('records must reference dataset images and text labels')
            target = batch_id + '__' + str(index).zfill(8) + '__' + image.stem
            entries.append((batch_id, image, label, target))
    target_names = [target.casefold() for _, _, _, target in entries]
    if len(target_names) != len(set(target_names)):
        raise ValueError('batch-prefixed file names collide')
    return spec, spec_path, entries, sorted(evidence)


def package(spec_path, output, root=ROOT):
    root = Path(root).resolve()
    spec, spec_file, entries, evidence = prepare(root, spec_path)
    output = _path(root, str(output))
    if not output.is_relative_to(root/'datasets/exports') or output.suffix.lower() != '.zip':
        raise ValueError('output must be a ZIP under datasets/exports')
    media = [row[1] for row in entries]
    procedural = [spec_file, *evidence, *(row[2] for row in entries)]
    operation = 'candidate_data.package:' + output.relative_to(root).as_posix()
    code = [root/'lab/scripts/data/package_candidate_data.py']
    params = {'model_key': spec['model_key'], 'format': 'yolo_candidate_train_zip', 'images': len(entries)}
    with generation_guard(operation, media + procedural, [output], params,
                          procedural_inputs=procedural, code_paths=code, root=root) as run:
        if run.reused:
            return {'status': 'reused', 'output': output.relative_to(root).as_posix(), 'images': len(entries)}
        if output.exists():
            raise ValueError('output already exists; refusing replacement')
        output.parent.mkdir(parents=True, exist_ok=True)
        records = []
        # Claim the filename exclusively before opening the archive, so a
        # concurrent creator's file is never removed by our failure cleanup.
        owned = output.open('xb')
        try:
            with owned, zipfile.ZipFile(owned, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for batch_id, image, label, target in entries:
                    image_out = 'images/train/' + target + image.suffix.lower()
                    label_out = 'labels/train/' + target + '.txt'
                    archive.write(image, image_out)
                    archive.write(label, label_out)
                    rows = sum(bool(line.strip()) for line in label.read_text(encoding='utf-8-sig').splitlines())
                    records.append({'batch_id': batch_id, 'source_image': image.relative_to(root).as_posix(),
                                    'source_label': label.relative_to(root).as_posix(), 'image': image_out,
                                    'label': label_out, 'nonempty_label_rows': rows})
                for path in evidence:
                    archive.write(path, 'evidence/' + path.relative_to(root).as_posix())
                header = {'path': '.', 'train': 'images/train', **spec['dataset']}
                archive.writestr('data.yaml', json.dumps(header, ensure_ascii=False, indent=2) + '\n')
                archive.writestr('package_spec.json', json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
                reuse = {}
                for row in records:
                    reuse.setdefault(row['source_image'], []).append(row['batch_id'])
                metadata = {'status': 'candidate_not_verified', 'model_key': spec['model_key'],
                            'scope': 'Declared inputs only; no completeness or quality claim.',
                            'validation_and_test': 'not_included', 'model_performance': 'unknown',
                            'images': len(records), 'nonempty_label_rows': sum(r['nonempty_label_rows'] for r in records),
                            'reused_source_images': {path: batches for path, batches in reuse.items() if len(batches) > 1},
                            'source_refs': run.source_refs, 'samples': records}
                archive.writestr('package_manifest.json', json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        except BaseException:
            # Remove only the new incomplete archive owned by this invocation.
            if output.is_file():
                output.unlink()
            raise
    return {'status': 'candidate_not_verified', 'output': output.relative_to(root).as_posix(),
            'images': len(records), 'nonempty_label_rows': metadata['nonempty_label_rows']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='workspace-relative JSON specification')
    parser.add_argument('--output', required=True, help='new ZIP under datasets/exports')
    args = parser.parse_args()
    print(json.dumps(package(args.spec, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
