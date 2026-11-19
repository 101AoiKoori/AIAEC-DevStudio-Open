"""Resolve source datasets and stage a writable, traceable run-local dataset."""
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import chain, combinations
import filecmp
import hashlib
import json
import random
import shutil
import yaml
from .config import VALIDATING_MODES, read_yaml, resolve

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def dataset_header(root, path):
    return _dataset_header(read_yaml(resolve(root, path)))


def _dataset_header(spec):
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


def _source_manifest_path(root, config):
    return resolve(root, config['data']).parent / 'workspace_sources.json'


class _LabelSnapshot:
    """Expose one read of a label to the existing path-based validator."""

    def __init__(self, path):
        self.path = path
        self.contents = path.read_bytes()

    def read_text(self, encoding):
        return self.contents.decode(encoding)

    def __str__(self):
        return str(self.path)


def records(root, config, split, *, training_mode=None):
    """Enumerate paired images and labels without executing or altering sources."""
    root = Path(root).resolve()
    header = dataset_header(root, config['data'])
    keypoints = header.get('kpt_shape', [0])[0]
    if config['task'] == 'pose' and not keypoints:
        raise ValueError('Pose dataset has no keypoint definition')
    from .reporting import legacy
    manifest_path = _source_manifest_path(root, config)
    provenance = {}
    manifest_only = False
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
        mode = source_manifest.get('selection_mode')
        if mode not in (None, 'manifest_only'):
            raise ValueError('Unsupported source manifest selection_mode: ' + str(mode))
        manifest_only = mode == 'manifest_only'
        for row in source_manifest['records']:
            if training_mode and row.get('split') != split:
                continue
            if row['image'] in provenance:
                raise ValueError('Duplicate image in source manifest: ' + str(row['image']))
            provenance[row['image']] = row
    output, seen = [], set()
    for source in resolve_sources(root, config, split):
        images = sorted(p for p in source['images'].rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if manifest_only:
            images = [p for p in images if p.relative_to(root).as_posix() in provenance]
        stems = set()
        for image in images:
            relative = image.relative_to(source['images'])
            label = source['labels'] / relative.with_suffix('.txt')
            if manifest_only and (not image.resolve().is_relative_to(source['images'].resolve()) or
                                  not label.resolve().is_relative_to(source['labels'].resolve())):
                raise ValueError('Manifest pair escapes the declared source: ' + str(image))
            if relative.with_suffix('') in stems:
                raise ValueError('Duplicate image stem in source: ' + str(image))
            stems.add(relative.with_suffix(''))
            if not label.is_file():
                raise ValueError('Missing label: ' + str(label))
            if image.resolve() in seen:
                raise ValueError('Duplicate image in split: ' + str(image))
            seen.add(image.resolve())
            label_input = _LabelSnapshot(label) if training_mode else label
            legacy().check_label(label_input, len(header['names']), keypoints)
            rows = [line.split() for line in label_input.read_text(encoding='utf-8').splitlines() if line.strip()]
            counts = Counter(int(float(row[0])) for row in rows)
            image_key = image.relative_to(root).as_posix()
            known = provenance.get(image_key, {})
            if manifest_only and (known.get('split') != split or
                                  known.get('label') != label.relative_to(root).as_posix()):
                raise ValueError('Manifest split or paired label mismatch: ' + image_key)
            record = {'image': image_key,
                           'label': label.relative_to(root).as_posix(), 'split': split,
                           'source_group': known.get('source_group', known.get('source_id', source['source_group'])),
                           'workspace_image': known.get('workspace_image', image_key),
                           'workspace_label': known.get('workspace_label', label.relative_to(root).as_posix()),
                           'classes': sorted(counts),
                           'instances': len(rows)}
            if training_mode:
                record['label_sha256'] = hashlib.sha256(label_input.contents).hexdigest()
                record['boxes_by_class'] = dict(sorted(counts.items()))
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
                contact = known.get('development_contact')
                if matching and evidence_ok and isinstance(contact, list):
                    record['development_contact'] = sorted({value for value in contact if isinstance(value, str) and value})
                if training_mode in VALIDATING_MODES and record['reviewed_strata']['supervision'] == 'complete':
                    record['lineage'] = _lineage(known, is_flat)
                    if split == 'val' and keypoints:
                        record['visible_keypoints'] = sum(float(v) > 0 for row in rows for v in row[7::3])
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
        if orphan and not manifest_only:
            raise ValueError('Labels without images: ' + str(sorted(orphan)[:5]))
    if manifest_only:
        expected = {key for key, row in provenance.items() if row.get('split') == split}
        if {row['image'] for row in output} != expected:
            raise ValueError('Manifest image is missing or outside the declared split sources: ' + split)
    if not output:
        raise ValueError('Empty dataset split: ' + split)
    return output


def _lineage(row, flat):
    lineage = {'source_group': row.get('source_group') or row.get('source_id'),
               'lineage_group': row.get('near_duplicate_group'), 'parent_video': row.get('parent_video'),
               'mother_scene': row.get('mother_scene'), 'episode_group': row.get('episode_group'),
               'segment_id': row.get('fragment' if flat else 'segment_id')}
    for field in ('parent_ids', 'physical_assets', 'source_assets'):
        value = row.get('parent_samples' if flat and field == 'parent_ids' else field)
        if isinstance(value, list):
            lineage[field] = [entry for entry in value if isinstance(entry, str) and entry.strip()]
    parent = row.get('parent_sample')
    if flat and not lineage.get('parent_ids') and isinstance(parent, str) and parent.strip():
        lineage['parent_ids'] = [parent]
    return {key: value for key, value in lineage.items()
            if (isinstance(value, str) and value.strip()) or (isinstance(value, list) and value)}


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
    confirmed = bool(task.get('review_evidence')) and state in ('complete_positive', 'complete_negative')
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


def _select_splits(splits, config, mode):
    validation_limit = None if mode else config.get('eval_limit')
    seed = config.get('seed', 42)
    return {split: select_records(items, config.get('train_limit') if split == 'train' else validation_limit, seed)
            for split, items in splits.items()}


def _label_counts(splits, names):
    report = {}
    for split, items in splits.items():
        images, boxes = Counter(), Counter()
        for item in items:
            images.update(item['classes'])
            boxes.update(item['boxes_by_class'])
        classes = [{'class_id': index, 'class_name': name, 'images': images[index], 'boxes': boxes[index]}
                   for index, name in enumerate(names)]
        report[split] = {'classes': classes, 'negative_images': sum(item['instances'] == 0 for item in items)}
    return report


def _inspect_splits(root, config, mode=None, *, validation_only=False):
    header = dataset_header(root, config['data'])
    if config['task'] == 'pose' and config.get('keypoint_names'):
        expected = [len(config['keypoint_names']), 3]
        if header.get('kpt_shape') != expected:
            raise ValueError(f'data={config["data"]!r} kpt_shape={header.get("kpt_shape")!r}; '
                             f'training config keypoint_names requires kpt_shape={expected}')
    if config.get('class_names') is not None:
        names = config['class_names']
        if isinstance(names, dict) and set(names) == set(range(len(names))):
            names = [names[index] for index in range(len(names))]
        if not isinstance(names, list) or len(names) != len(header['names']):
            raise ValueError('class_names must preserve the dataset class count and order')
        names = [str(value) for value in names]
        if mode and names != header['names']:
            raise ValueError('Training policy class_names must match the dataset class names and order')
        header['names'] = names
    missing_test = not mode and not validation_only and config.get('test_unavailable', False)
    if missing_test:
        spec = read_yaml(resolve(root, config['data']))
        overrides = config.get('data_overrides', {})
        if not isinstance(overrides, dict):
            raise ValueError('data_overrides must be a mapping')
        if spec.get('test') is not None or 'test' in overrides:
            raise ValueError('test_unavailable requires no test path or test override')
    requested = ('train',) if mode == 'fixed' else ('train', 'val') if mode or validation_only else ('train', 'val', 'test')
    splits = {split: ([] if split == 'test' and missing_test else records(root, config, split, training_mode=mode))
              for split in requested}
    counts = {split: len(items) for split, items in splits.items()}
    overlap = {}
    for left, right in combinations(requested, 2):
        overlap[left + '_' + right] = len({r['image'] for r in splits[left]} & {r['image'] for r in splits[right]})
    test_status = 'unavailable' if missing_test else 'available'
    if mode or validation_only:
        test_status = 'not_used_in_training'
    return header, splits, {'available_images': counts, 'exact_path_overlap': overlap,
                            'test_status': test_status, 'source_notes': config.get('source_notes', []),
                            'source_independence': 'Not inferred from distinct paths or random seeds.'}


def inspect_dataset(root, config):
    """Inspect the configured datasets for workflows outside training."""
    return _inspect_splits(root, config)


def _require_v4_review(splits):
    for items in splits.values():
        for item in items:
            review = item.get('reviewed_strata', {})
            state = review.get('label_state')
            expected = 'complete_positive' if item['instances'] else 'complete_negative'
            if review.get('supervision') != 'complete' or state != expected:
                raise ValueError('V4 requires complete reviewed supervision and matching provenance: '
                                 f'{item["image"]}; current={state or "unknown"}, required={expected}; '
                                 'check the paired label and task evidence in the dataset source manifest')


def _check_validation_images(root, splits):
    from PIL import Image, ImageOps

    identities = {}
    for split, items in splits.items():
        for row in items:
            path = Path(root) / row['image']
            try:
                file_digest = _file_sha256(path)
                with Image.open(path) as image:
                    pixels = ImageOps.exif_transpose(image).convert('RGB')
                    pixel_digest = hashlib.sha256(pixels.tobytes()).digest()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise ValueError('Cannot read image for validation source check: ' + row['image']) from exc
            row['image_sha256'] = file_digest
            keys = (('path', row['image']), ('workspace_image', row['workspace_image']), ('file', file_digest),
                    ('pixels', pixels.size, pixel_digest))
            for key in keys:
                previous = identities.get(key)
                if split == 'val' and previous is not None:
                    raise ValueError(f'Validation repeats an image ({key[0]}): '
                                     f'{row["image"]} and {previous}')
                identities.setdefault(key, row['image'])


def _validation_source_reuse(splits, declared):
    observed = defaultdict(lambda: defaultdict(int))
    unknown_groups = {split: 0 for split in splits}
    for split, rows in splits.items():
        for row in rows:
            lineage = row['lineage']
            if not lineage.get('source_group'):
                unknown_groups[split] += 1
            for relation, value in lineage.items():
                scope = lineage.get('source_group') if relation in ('episode_group', 'segment_id') else None
                # Episode and segment IDs are local to a source group; without
                # that scope equal strings do not establish a shared source.
                if relation in ('episode_group', 'segment_id') and not scope:
                    continue
                for source in set(value if isinstance(value, list) else [value]):
                    observed[relation, source, scope][split] += 1
    declared_keys = {(row['relation'], row['source'], row.get('source_group')) for row in declared
                     if {'train', 'val'} <= set(row.get('splits', []))}
    reused = []
    for (relation, source, scope), counts in sorted(observed.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or '')):
        if len(counts) != 2:
            continue
        reused.append({'relation': relation, 'source': source, 'source_group': scope, 'counts': dict(counts),
                       'declared_in_manifest': (relation, source, scope) in declared_keys})
    return reused, unknown_groups


def preflight_dataset(root, config):
    """Check the training splits and their evidence without staging or running a model."""
    if 'training_policy' not in config:
        return inspect_dataset(root, config)
    mode = config['training_policy']['mode']
    if mode not in ('fixed', *VALIDATING_MODES) or str(config.get('version', '')).lower() != 'v4':
        raise ValueError('Training data policy requires V4 and a supported training policy mode')
    if not resolve(root, config['data']).is_file():
        raise ValueError(f'Training data YAML missing (training config key data={config["data"]!r}); '
                         'point data to the actual exported dataset entry')
    manifest_path = _source_manifest_path(root, config)
    if not manifest_path.is_file():
        raise ValueError('V4 requires complete reviewed supervision; missing source evidence manifest: '
                         + str(manifest_path))
    header, splits, report = _inspect_splits(root, config, mode)
    _require_v4_review(splits)
    report['training_mode'] = mode
    report['available_label_counts'] = _label_counts(splits, header['names'])
    selected = _select_splits(splits, config, mode)
    report['selected_label_counts'] = _label_counts(selected, header['names'])
    report['class_coverage_warnings'] = [
        {'split': split, 'missing_classes': missing,
         'message': 'Selected split has no positive examples for: ' + ', '.join(missing)}
        for split, counts in report['selected_label_counts'].items()
        if (missing := [row['class_name'] for row in counts['classes'] if not row['boxes']])
    ]
    if mode in VALIDATING_MODES:
        if not any(row['instances'] for row in splits['val']):
            raise ValueError(f'training_policy.mode={mode} requires validation ground-truth instances for model '
                             f'selection; val has none in data={config["data"]!r}. '
                             'Use a labeled val split, or choose training_policy.mode=fixed for a train-only run.')
        if config['task'] == 'pose' and not any(row['visible_keypoints'] for row in splits['val']):
            raise ValueError('Pose validation needs labeled visible keypoints')
        _check_validation_images(root, splits)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        declared = manifest.get('source_reuse', [])
        report['source_reuse'], report['source_group_unknown_counts'] = _validation_source_reuse(splits, declared)
        if manifest.get('exposure_warnings'):
            report['exposure_warnings'] = manifest['exposure_warnings']
        report['development_contact_counts'] = {
            split: sum(bool(row.get('development_contact')) for row in rows) for split, rows in splits.items()}
        report['evaluation_scope'] = ('same_source_diagnostic' if report['source_reuse'] or
                                      report['development_contact_counts']['val'] else 'independence_not_established')
    return header, splits, report


def stage_dataset(root, config, run_dir, *, training=True, validation_only=False):
    """Copy selected inputs into runs so framework caches never touch sources."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if training:
        header, splits, report = preflight_dataset(root, config)
    else:
        header, splits, report = _inspect_splits(root, config, validation_only=validation_only)
    mode = report.get('training_mode')
    if not mode and str(config.get('version', '')).lower() == 'v4':
        _require_v4_review(splits)
    chosen_splits = _select_splits(splits, config, mode)
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
    for split, chosen in chosen_splits.items():
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
            if training:
                label_hash = _file_sha256(label)
                if mode and label_hash != record['label_sha256']:
                    raise ValueError('Training label changed between preflight and staging: ' + record['label'])
                image_hash = _file_sha256(image)
                if 'image_sha256' in record and image_hash != record['image_sha256']:
                    raise ValueError('Training image changed between preflight and staging: ' + record['image'])
                record['label_sha256'] = label_hash
                record['image_sha256'] = image_hash
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
    if mode == 'fixed':
        # Ultralytics expects a validation key in data YAML even when unused.
        spec['val'] = None
    path = destination / 'data.yaml'
    path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (destination / 'samples.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding='utf-8')
    report['selected_images'] = selected
    report['selection_policy'] = 'Seeded class-coverage-first; all declared class IDs retained.'
    (destination / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return path, report


def _file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _staged_signature(spec, samples):
    header = {key: value for key, value in spec.items() if key != 'path'}
    encoded = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(encoded)
    digest.update(b'\x00')
    digest.update(samples)
    return digest.hexdigest()


def staged_dataset_signature(run_dir):
    """Sign the recorded inputs without re-reading images after initial staging."""
    data = Path(run_dir) / 'data'
    return _staged_signature(read_yaml(data / 'data.yaml'), (data / 'samples.json').read_bytes())


def _resume_dataset_splits(spec, config):
    mode = config.get('training_policy', {}).get('mode')
    required = {'train'}
    if mode != 'fixed':
        required.add('val')
    if mode is None and not config.get('test_unavailable'):
        required.add('test')
    if 'val' not in spec:
        raise ValueError('Staged resume dataset YAML is missing the val key')
    for split in ('train', 'val', 'test'):
        expected = 'images/' + split if split in required else None
        if spec.get(split) != expected:
            raise ValueError('Staged resume dataset has an unexpected path or split: ' + split)
    return required


def _check_resume_dataset_header(spec, config):
    if set(spec) - {'path', 'names', 'train', 'val', 'test', 'kpt_shape', 'flip_idx'}:
        raise ValueError('Staged resume dataset YAML has unsupported fields')
    header = _dataset_header(spec)
    expected_names = config.get('class_names')
    if expected_names is not None:
        expected = _dataset_header({'names': expected_names})['names']
        if list(map(str, header['names'])) != list(map(str, expected)):
            raise ValueError('Frozen dataset class order differs from recorded training configuration')
    shape = header.get('kpt_shape')
    if config['task'] == 'pose':
        names = config.get('keypoint_names')
        if shape is None or type(shape[0]) is not int or shape[0] < 1 or (names and len(names) != shape[0]):
            raise ValueError('Frozen dataset keypoints differ from recorded pose training configuration')
    elif shape is not None or 'flip_idx' in spec:
        raise ValueError('Frozen detection dataset has unexpected pose metadata')


def validate_staged_resume_data(run_dir, config):
    """Validate a run's copied inputs without reading its former workspace sources."""
    run_dir = Path(run_dir).resolve()
    data = run_dir / 'data'
    files = set()
    for entry in chain((data,), data.rglob('*')):
        if entry.is_symlink() or entry.is_junction():
            raise ValueError('Staged resume dataset must not contain symlinks or junctions: ' + str(entry))
        if entry.is_file():
            files.add(entry)
    spec = read_yaml(data / 'data.yaml')
    required = _resume_dataset_splits(spec, config)
    _check_resume_dataset_header(spec, config)
    samples = (data / 'samples.json').read_bytes()
    records = json.loads(samples.decode('utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Staged resume sample manifest is empty or invalid')

    manifest_images, manifest_labels, counts = set(), set(), Counter()
    for row in records:
        split = row['split']
        if split not in required:
            raise ValueError('Unexpected staged resume sample split: ' + split)
        image, label = resolve(run_dir, row['run_image']), resolve(run_dir, row['run_label'])
        images = data / 'images' / split
        labels = data / 'labels' / split
        if (not image.is_relative_to(images) or not label.is_relative_to(labels) or
                image not in files or label not in files or image.suffix.lower() not in EXTENSIONS or
                label.suffix.lower() != '.txt' or image in manifest_images or label in manifest_labels or
                image.relative_to(images).with_suffix('') != label.relative_to(labels).with_suffix('')):
            raise ValueError('Missing, duplicated, misplaced or unpaired staged resume image/label: ' + row['run_image'])
        for path, digest_field in ((image, 'image_sha256'), (label, 'label_sha256')):
            if digest_field in row and _file_sha256(path) != row[digest_field]:
                raise ValueError(f'Staged resume {digest_field} mismatch: {path.relative_to(run_dir).as_posix()}')
        manifest_images.add(image)
        manifest_labels.add(label)
        counts[split] += 1
    if any(not counts[split] for split in required):
        raise ValueError('Each required staged resume split needs at least one sample')
    actual_images = {path for path in files if path.is_relative_to(data / 'images') and path.suffix.lower() in EXTENSIONS}
    actual_labels = {path for path in files if path.is_relative_to(data / 'labels') and path.suffix.lower() == '.txt'}
    if manifest_images != actual_images or manifest_labels != actual_labels:
        raise ValueError('Staged resume images/labels disagree with the sample manifest')
    return _staged_signature(spec, samples)


def relocate_staged_data(run_dir):
    """Refresh runtime YAML and discard only generated label caches after a move."""
    run_dir = Path(run_dir).resolve()
    path = run_dir / 'data/data.yaml'
    spec = read_yaml(path); spec['path'] = str(run_dir / 'data')
    path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding='utf-8')
    for cache in (run_dir / 'data/labels').glob('*.cache'):
        cache.unlink()
    return path
