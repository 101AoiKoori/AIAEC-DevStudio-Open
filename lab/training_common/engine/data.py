"""Resolve source datasets and stage a writable, traceable run-local dataset."""
from pathlib import Path
import json
import random
import shutil
import yaml
from .config import read_yaml, resolve

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def dataset_header(root, path):
    spec = read_yaml(resolve(root, path))
    if 'download' in spec:
        raise ValueError('Dataset download directives are forbidden')
    names = spec.get('names')
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError('Class IDs must be consecutive integers')
        names = [str(names[i]) for i in range(len(names))]
    if not isinstance(names, list) or not names:
        raise ValueError('Dataset needs nonempty class names')
    result = {'names': names}
    if 'kpt_shape' in spec:
        shape = spec['kpt_shape']
        if not isinstance(shape, list) or len(shape) != 2 or shape[1] != 3:
            raise ValueError('Pose requires kpt_shape [N, 3]')
        result['kpt_shape'] = shape
        if 'flip_idx' in spec:
            result['flip_idx'] = spec['flip_idx']
    return result


def _source_directory(root, data_path, spec, split):
    value = spec.get(split)
    if not isinstance(value, str):
        raise ValueError('Source YAML split must identify one image directory: ' + split)
    base = data_path.parent
    if spec.get('path'):
        candidate = Path(spec['path'])
        base = candidate if candidate.is_absolute() else (root / candidate)
        if not base.exists() and not candidate.is_absolute():
            base = data_path.parent / candidate
    images = (base / value).resolve()
    if not images.is_relative_to(root):
        raise ValueError('Dataset source escapes root: ' + str(images))
    if 'images' not in images.parts:
        raise ValueError('Cannot infer labels; provide explicit images/labels data_overrides')
    parts = list(images.parts)
    index = len(parts) - 1 - parts[::-1].index('images')
    parts[index] = 'labels'
    return images, Path(*parts)


def resolve_sources(root, config, split):
    """Return source specifications with absolute paths contained within root."""
    root = Path(root).resolve()
    header = dataset_header(root, config['data'])
    overrides = config.get('data_overrides', {})
    if not isinstance(overrides, dict) or set(overrides) - {'train', 'val', 'test'}:
        raise ValueError('data_overrides accepts train/val/test only')
    sources = overrides.get(split, [{'data': config['data'], 'split': split}])
    if not isinstance(sources, list) or not sources:
        raise ValueError('Each split requires a nonempty source list')
    result = []
    for source in sources:
        if not isinstance(source, dict) or set(source) - {'data', 'split', 'images', 'labels', 'source_group'}:
            raise ValueError('Invalid split source fields')
        if 'data' in source:
            path = resolve(root, source['data'])
            source_header = dataset_header(root, source['data'])
            if source_header != header:
                raise ValueError('Sources disagree on class/keypoint definitions: ' + str(path))
            images, labels = _source_directory(root, path, read_yaml(path), source.get('split', split))
            group = source.get('source_group', source['data'])
        else:
            images, labels = resolve(root, source['images']), resolve(root, source['labels'])
            group = source.get('source_group', source['images'])
        if not images.is_dir() or not labels.is_dir():
            raise ValueError('Source images/labels directory missing: ' + str(images))
        result.append({'images': images, 'labels': labels, 'source_group': str(group)})
    return result


def records(root, config, split):
    """Enumerate paired images and labels without executing or altering sources."""
    root = Path(root).resolve()
    header = dataset_header(root, config['data'])
    keypoints = header.get('kpt_shape', [0])[0]
    if config['task'] == 'pose' and not keypoints:
        raise ValueError('Pose dataset has no keypoint definition')
    from .reporting import legacy
    manifest_path = resolve(root, config['data']).parent / 'workspace_sources.json'
    provenance = {}
    if manifest_path.is_file():
        provenance = {row['image']: row for row in json.loads(manifest_path.read_text(encoding='utf-8'))['records']}
    output, seen = [], set()
    for source in resolve_sources(root, config, split):
        images = sorted(p for p in source['images'].rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        stems = set()
        for image in images:
            relative = image.relative_to(source['images'])
            label = source['labels'] / relative.with_suffix('.txt')
            if relative.with_suffix('') in stems:
                raise ValueError('Duplicate image stem in source: ' + str(image))
            stems.add(relative.with_suffix(''))
            if not label.is_file():
                raise ValueError('Missing label: ' + str(label))
            if image.resolve() in seen:
                raise ValueError('Duplicate image in split: ' + str(image))
            seen.add(image.resolve())
            legacy().check_label(label, len(header['names']), keypoints)
            rows = [line.split() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
            image_key = image.relative_to(root).as_posix()
            known = provenance.get(image_key, {})
            output.append({'image': image_key,
                           'label': label.relative_to(root).as_posix(), 'split': split,
                           'source_group': known.get('source_group', source['source_group']),
                           'workspace_image': known.get('workspace_image', image_key),
                           'workspace_label': known.get('workspace_label', label.relative_to(root).as_posix()),
                           'classes': sorted({int(float(row[0])) for row in rows}),
                           'instances': len(rows)})
        orphan = {p.relative_to(source['labels']).with_suffix('') for p in source['labels'].rglob('*.txt')} - stems
        if orphan:
            raise ValueError('Labels without images: ' + str(sorted(orphan)[:5]))
    if not output:
        raise ValueError('Empty dataset split: ' + split)
    return output


def select_records(all_records, limit, seed):
    """Seeded coverage-first sample, retaining all category definitions in YAML."""
    if limit is None or limit >= len(all_records):
        return list(all_records)
    order = list(range(len(all_records)))
    random.Random(seed).shuffle(order)
    remaining = {c for row in all_records for c in row['classes']}
    selected = []
    while remaining and len(selected) < limit:
        index = max(order, key=lambda i: len(remaining.intersection(all_records[i]['classes'])))
        if not remaining.intersection(all_records[index]['classes']):
            break
        selected.append(index); order.remove(index)
        remaining.difference_update(all_records[index]['classes'])
    negative = next((i for i in order if not all_records[i]['classes']), None)
    if negative is not None and len(selected) < limit:
        selected.append(negative); order.remove(negative)
    selected.extend(order[:limit-len(selected)])
    return [all_records[index] for index in selected]


def inspect_dataset(root, config):
    header = dataset_header(root, config['data'])
    if config.get('class_names') is not None:
        names = config['class_names']
        if isinstance(names, dict) and set(names) == set(range(len(names))):
            names = [names[index] for index in range(len(names))]
        if not isinstance(names, list) or len(names) != len(header['names']):
            raise ValueError('class_names must preserve the dataset class count and order')
        header['names'] = [str(value) for value in names]
    splits = {split: records(root, config, split) for split in ('train', 'val', 'test')}
    counts = {split: len(items) for split, items in splits.items()}
    overlap = {}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap[left + '_' + right] = len({r['image'] for r in splits[left]} & {r['image'] for r in splits[right]})
    return header, splits, {'available_images': counts, 'exact_path_overlap': overlap,
                            'source_notes': config.get('source_notes', []),
                            'source_independence': 'Not inferred from distinct paths or random seeds.'}


def stage_dataset(root, config, run_dir):
    """Copy selected inputs into runs so framework caches never touch sources."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    header, splits, report = inspect_dataset(root, config)
    destination = run_dir / 'data'
    destination.mkdir()
    selected = {}
    manifests = []
    for split, items in splits.items():
        limit = config.get('train_limit') if split == 'train' else config.get('eval_limit')
        chosen = select_records(items, limit, config.get('seed', 42))
        selected[split] = len(chosen)
        for kind in ('images', 'labels'):
            (destination / kind / split).mkdir(parents=True)
        for index, record in enumerate(chosen):
            source_image, source_label = root / record['image'], root / record['label']
            name = f'{split}_{index:06d}_{source_image.stem}'
            image = destination / 'images' / split / (name + source_image.suffix.lower())
            label = destination / 'labels' / split / (name + '.txt')
            shutil.copy2(source_image, image)
            shutil.copy2(source_label, label)
            manifests.append({**record, 'run_image': image.relative_to(run_dir).as_posix(),
                              'run_label': label.relative_to(run_dir).as_posix()})
    spec = {**header, 'path': str(destination), **{split: 'images/' + split for split in splits}}
    path = destination / 'data.yaml'
    path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (destination / 'samples.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding='utf-8')
    report['selected_images'] = selected
    report['selection_policy'] = 'Seeded class-coverage-first; all declared class IDs retained.'
    (destination / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return path, report


def relocate_staged_data(run_dir):
    """Refresh runtime YAML and discard only generated label caches after a move."""
    run_dir = Path(run_dir).resolve()
    path = run_dir / 'data/data.yaml'
    spec = read_yaml(path); spec['path'] = str(run_dir / 'data')
    path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding='utf-8')
    for cache in (run_dir / 'data/labels').glob('*.cache'):
        cache.unlink()
    return path
