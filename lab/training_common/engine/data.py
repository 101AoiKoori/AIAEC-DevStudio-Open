"""Resolve source datasets and stage a writable, traceable run-local dataset."""
from pathlib import Path
from copy import deepcopy
import filecmp
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
    manifest_model = None
    evidence_map = None
    evidence_path = manifest_path.parent / 'source_records/evidence_map.json'
    if evidence_path.is_file():
        evidence_map = json.loads(evidence_path.read_text(encoding='utf-8'))
        if not isinstance(evidence_map, dict):
            raise ValueError('Source evidence map must be a mapping')
    if manifest_path.is_file():
        source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest_model = source_manifest.get('model_key')
        for row in source_manifest['records']:
            if row['image'] in provenance:
                raise ValueError('Duplicate image in source manifest: ' + str(row['image']))
            provenance[row['image']] = row
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
            record = {'image': image_key,
                           'label': label.relative_to(root).as_posix(), 'split': split,
                           'source_group': known.get('source_group', known.get('source_id', source['source_group'])),
                           'workspace_image': known.get('workspace_image', image_key),
                           'workspace_label': known.get('workspace_label', label.relative_to(root).as_posix()),
                           'classes': sorted({int(float(row[0])) for row in rows}),
                           'instances': len(rows)}
            if str(config.get('version', '')).lower() == 'v4':
                model_key = config.get('model_key')
                # Refuse to attach an audited claim to a different split,
                # label, or task if a manifest has been copied incorrectly.
                nested = isinstance(known.get('selected_tasks'), list) and model_key in known['selected_tasks']
                flat = manifest_model == model_key and known.get('label_status') == 'reviewed'
                matching = (known.get('split') == split and
                            known.get('label') == label.relative_to(root).as_posix() and
                            (nested or flat))
                is_flat = flat and not nested
                task = known if is_flat else (known.get('tasks', {}).get(model_key, {})
                                               if isinstance(known.get('tasks'), dict) else {})
                task = task if isinstance(task, dict) else {}
                source_evidence = known.get('source_review_evidence') if is_flat else known.get('review_evidence')
                evidence_ok = (evidence_exists(root, source_evidence, evidence_map, evidence_path.parent) and
                               evidence_exists(root, task.get('review_evidence'), evidence_map, evidence_path.parent))
                record['reviewed_strata'] = reviewed_strata(known if matching and evidence_ok else {}, model_key,
                                                          flat=is_flat, image_path=image)
                basis = record['reviewed_strata'].get('image_domain_basis')
                if basis is not None:
                    basis['source_review_evidence'] = (source_evidence if evidence_map is None else
                                                       evidence_map[source_evidence])
                if 'conditions' in record['reviewed_strata']:
                    condition_evidence = task['review_evidence']
                    record['reviewed_strata']['conditions_review_evidence'] = (
                        condition_evidence if evidence_map is None else evidence_map[condition_evidence])
            output.append(record)
        orphan = {p.relative_to(source['labels']).with_suffix('') for p in source['labels'].rglob('*.txt')} - stems
        if orphan:
            raise ValueError('Labels without images: ' + str(sorted(orphan)[:5]))
    if not output:
        raise ValueError('Empty dataset split: ' + split)
    return output


def evidence_exists(root, recorded, mapping, package_evidence_dir):
    """Resolve evidence inside the workspace or its packaged evidence map."""
    if not isinstance(recorded, str) or not recorded.strip():
        return False
    value = recorded if mapping is None else mapping.get(recorded)
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        path = resolve(root, value)
        return path.is_file() and (mapping is None or path.is_relative_to(package_evidence_dir.resolve()))
    except (ValueError, OSError):
        return False


def reviewed_image_domain(provenance, image_path):
    """Classify only explicitly reviewed origin/processing; unknown is None."""
    method, kind = provenance.get('generation_method'), provenance.get('source_kind')
    processing = provenance.get('image_processing')
    processing = processing if isinstance(processing, dict) else {}
    if method == 'blender_render' and kind == 'blender_scene':
        return 'blender_render', {'generation_method': method}
    if method == 'synthetic_render' and kind == 'synthetic_scene':
        return 'other_synthetic', {'generation_method': method}
    if method in ('composite', 'augmentation') and kind == 'derived_image':
        if processing.get('origin_review') != 'confirmed':
            return None, None
        if method == 'composite' and processing.get('background_origin') == 'direct_camera':
            return 'real_scene_composite', {'generation_method': method, 'origin_review': 'confirmed',
                                            'background_origin': 'direct_camera'}
        if method == 'augmentation' and processing.get('parent_origin') == 'direct_camera':
            return 'real_augmentation', {'generation_method': method, 'origin_review': 'confirmed',
                                         'parent_origin': 'direct_camera'}
        if method == 'augmentation' and processing.get('parent_origin') == 'synthetic':
            return 'other_synthetic', {'generation_method': method, 'origin_review': 'confirmed',
                                       'parent_origin': 'synthetic'}
        return None, None
    direct = ((method == 'camera_capture' and kind in ('photo', 'task_point')) or
              (method == 'provided_video_frame' and kind in ('frame_sequence', 'provided_subset', 'video_frame')))
    if not direct or processing.get('resolution_review') != 'confirmed' or image_path is None:
        return None, None
    source_size, output_size = processing.get('source_size'), processing.get('output_size')
    if any(not isinstance(size, list) or len(size) != 2 or
           any(type(value) is not int or value <= 0 for value in size) for size in (source_size, output_size)):
        return None, None
    from PIL import Image
    try:
        with Image.open(image_path) as image:
            if list(image.size) != output_size:
                return None, None
    except (OSError, ValueError):
        return None, None
    resampling, clarity = processing.get('resampling'), processing.get('clarity')
    details = {'generation_method': method, 'resolution_review': 'confirmed',
               'source_size': source_size, 'output_size': output_size,
               'resampling': resampling, 'clarity': clarity}
    if resampling == 'none' and clarity == 'original' and source_size == output_size:
        return 'direct_real_native_clarity', details
    if resampling == 'resampled' and clarity == 'modified':
        return 'direct_real_resampled', details
    return None, None


def reviewed_strata(provenance, model_key, flat=False, image_path=None):
    """Snapshot only provenance and task attributes explicitly marked reviewed.

    Optional task ``strata`` fields are considered only when ``strata_review``
    is confirmed. Missing or unreviewed fields remain unknown at evaluation.
    The source gate and builder, not this function, establish file provenance.
    """
    if not isinstance(provenance, dict):
        return {'supervision': 'unreviewed'}
    tasks = provenance.get('tasks')
    task = tasks.get(model_key, {}) if isinstance(tasks, dict) else {}
    if flat:
        task = {'state': provenance.get('review_state'), 'review_evidence': provenance.get('review_evidence'),
                'strata': provenance.get('strata'), 'strata_review': provenance.get('strata_review')}
    if not isinstance(task, dict):
        task = {}
    state = task.get('state')
    source_confirmed = (bool(provenance.get('source_review_evidence')) and provenance.get('label_status') == 'reviewed'
                        if flat else provenance.get('lineage_review') == 'confirmed' and
                        provenance.get('near_duplicate_review') == 'confirmed' and bool(provenance.get('review_evidence')))
    confirmed = source_confirmed and bool(task.get('review_evidence')) and state in ('complete_positive', 'complete_negative')
    if not confirmed:
        return {'supervision': 'unreviewed'}
    result = {'supervision': 'complete', 'label_state': state}
    for key in ('source_group', 'source_kind', 'generation_method'):
        value = provenance.get('source_id') if flat and key == 'source_group' else provenance.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value
    domain, basis = reviewed_image_domain(provenance, image_path)
    if domain is not None:
        result['image_domain'] = domain
        result['image_domain_basis'] = basis
    attributes = task.get('strata')
    if task.get('strata_review') == 'confirmed' and isinstance(attributes, dict):
        difficulty = attributes.get('difficulty')
        if isinstance(difficulty, str) and difficulty.strip():
            result['difficulty'] = difficulty
        for key in ('conditions', 'target_states'):
            values = attributes.get(key)
            if (isinstance(values, list) and all(isinstance(x, str) and x.strip() for x in values)
                    and len(values) == len(set(values))):
                result[key] = list(values)
    return result


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
    missing_test = config.get('test_unavailable', False)
    if missing_test:
        spec = read_yaml(resolve(root, config['data']))
        overrides = config.get('data_overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError('data_overrides must be a mapping')
        if spec.get('test') is not None or 'test' in overrides:
            raise ValueError('test_unavailable requires no test path or test override')
    splits = {split: ([] if split == 'test' and missing_test else records(root, config, split))
              for split in ('train', 'val', 'test')}
    counts = {split: len(items) for split, items in splits.items()}
    overlap = {}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap[left + '_' + right] = len({r['image'] for r in splits[left]} & {r['image'] for r in splits[right]})
    return header, splits, {'available_images': counts, 'exact_path_overlap': overlap,
                            'test_status': 'unavailable' if missing_test else 'available',
                            'source_notes': config.get('source_notes', []),
                            'source_independence': 'Not inferred from distinct paths or random seeds.'}


def stage_dataset(root, config, run_dir):
    """Copy selected inputs into runs so framework caches never touch sources."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    header, splits, report = inspect_dataset(root, config)
    if str(config.get('version', '')).lower() == 'v4':
        for split, items in splits.items():
            for item in items:
                review = item.get('reviewed_strata', {})
                state = review.get('label_state')
                if (review.get('supervision') != 'complete' or
                        (state == 'complete_positive' and not item['instances']) or
                        (state == 'complete_negative' and item['instances'] != 0) or
                        state not in ('complete_positive', 'complete_negative')):
                    raise ValueError('V4 requires complete reviewed supervision and matching provenance: ' + item['image'])
    destination = run_dir / 'data'
    destination.mkdir()
    selected = {}
    manifests = []
    copied_evidence = {}
    def snapshot_evidence(source_ref):
        if source_ref not in copied_evidence:
            evidence = resolve(root, source_ref)
            if not evidence.is_file():
                raise ValueError('Reviewed stratum evidence is missing: ' + source_ref)
            target_dir = destination / 'review_evidence'
            target_dir.mkdir(exist_ok=True)
            target = target_dir / (f'{len(copied_evidence):06d}_' + evidence.name)
            shutil.copy2(evidence, target)
            if not filecmp.cmp(evidence, target, shallow=False):
                raise RuntimeError('Reviewed stratum evidence changed while copying: ' + source_ref)
            copied_evidence[source_ref] = target.relative_to(run_dir).as_posix()
        return copied_evidence[source_ref]
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
            record = deepcopy(record)
            review = record.get('reviewed_strata', {})
            basis = review.get('image_domain_basis')
            if basis is not None:
                basis['source_review_evidence_snapshot'] = snapshot_evidence(basis['source_review_evidence'])
            if 'conditions_review_evidence' in review:
                review['conditions_review_evidence_snapshot'] = snapshot_evidence(review['conditions_review_evidence'])
            manifests.append({**record, 'run_image': image.relative_to(run_dir).as_posix(),
                              'run_label': label.relative_to(run_dir).as_posix()})
    spec = {**header, 'path': str(destination),
            **{split: 'images/' + split for split in splits if not (split == 'test' and config.get('test_unavailable'))}}
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
