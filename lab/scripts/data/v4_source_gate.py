"""Build a traceable car-capture review queue and check proposed V4 selections.

The queue is not a labelled dataset. A passing audit only checks recorded evidence
and relationships; it cannot establish semantic correctness or independent capture.
"""
import argparse
import ast
import csv
from collections import Counter, defaultdict
import filecmp
import json
import math
from pathlib import Path
import re
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from train import check_label
from engine.data import dataset_header
from engine.config import read_yaml
BASE = 'datasets/annotated/car_real_capture_20261112'
INVENTORY = f'{BASE}/inventory.json'
SEGMENTS = f'{BASE}/segments_v1/segments.json'
SELECTION = f'{BASE}/segments_v1/selection.json'
ANNOTATIONS = [f'{BASE}/field_eval_v1/annotations_root.json',
               f'{BASE}/field_eval_v1/annotations_frames.json']
REVIEW_SUMMARY = f'{BASE}/field_eval_v1/review_v2/review_summary.json'
EXPOSURE = 'runs/car_real_capture_20261112_v3_eval/selected_final/exposure_history.json'
DUPLICATES = 'runs/car_real_capture_20261112_v3_eval/selected_final/duplicate_review/report.json'
SITE_SELECTIONS = ('datasets/annotated/site_capture_20261112/field_eval_v1/photo_selection.csv',
                   'datasets/annotated/site_capture_20261112/field_eval_v1/video_selection.csv')
DEVELOPMENT_EXPOSURE = 'datasets/annotated/site_capture_20261112/v4_development_exposure.json'
MODELS = ('person', 'e_bike', 'fire_sticker', 'building_name_paper', 'sign',
          'waste_bin_state', 'waste_sticker', 'meter', 'plate_detection')
SPLITS = {'train', 'val', 'test'}
DERIVED = {'augmentation', 'composite', 'crop'}
GENERATION_METHODS = {'camera_capture', 'provided_video_frame', *DERIVED,
                      'blender_render', 'synthetic_render'}
SOURCE_METHODS = {
    'task_point': {'camera_capture'}, 'photo': {'camera_capture'},
    'frame_sequence': {'provided_video_frame'}, 'provided_subset': {'provided_video_frame'},
    'video_frame': {'provided_video_frame'},
    'derived_image': DERIVED,
    'blender_scene': {'blender_render'}, 'synthetic_scene': {'synthetic_render'},
}
VIDEO_KINDS = {'frame_sequence', 'provided_subset', 'video_frame'}
SITE_SEQUENCE = 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene'
SITE_VIDEO = 'Archive/Assets/captures/site_capture_20261112/videos/full_scene.mp4'
PERSON_SEQUENCE = 'datasets/annotated/site_capture_20261112/sequence_v2/person_standees'
PERSON_VIDEO = 'Archive/Assets/captures/site_capture_20261112/videos/person_standees.mp4'
OWNERS = {'fire_sticker': 'fire', 'building_name_paper': 'fire',
          'waste_bin_state': 'waste_bin', 'waste_sticker': 'waste_bin',
          'plate_detection': 'plate'}


def workspace_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'路径不在工作空间内: {value}')
    return path


def read_json(root, path):
    return json.loads(workspace_path(root, path).read_text(encoding='utf-8-sig'))


def site_sequence_times(root):
    """Read decoder presentation times; frame numbers are one-based PNG names."""
    log = workspace_path(root, SITE_SEQUENCE + '/ffmpeg.log')
    times = {}
    pattern = re.compile(r'\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:([0-9.]+)')
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        match = pattern.search(line)
        if match:
            frame = int(match.group(1)) + 1
            if frame in times:
                raise ValueError('Duplicate frame index in decoder log')
            value = float(match.group(2))
            if not math.isfinite(value):
                raise ValueError('Invalid frame presentation time')
            times[frame] = value
    if not times:
        raise ValueError('Decoder log has no frame presentation times')
    return times


def person_sequence_times(root):
    """Use the probe's display-order timestamps for the person video frames."""
    data = read_json(root, PERSON_SEQUENCE + '/frames_probe.json')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Person sequence probe has no frames')
    times = {index: float(frame['best_effort_timestamp_time']) for index, frame in enumerate(frames, 1)}
    if not all(math.isfinite(value) for value in times.values()):
        raise ValueError('Invalid person sequence presentation time')
    return times


def development_exposure(root, *, missing_paths=None):
    """Load exposure claims; optionally retain missing historical evidence as unknown."""
    ledger = read_json(root, DEVELOPMENT_EXPOSURE)
    if (ledger.get('schema_version') != 1 or not isinstance(ledger.get('items'), list) or
            not ledger['items']):
        raise ValueError('Invalid development exposure ledger')
    images, video_frames, evidence, seen = set(), [], set(), set()
    for item in ledger['items']:
        if not isinstance(item, dict):
            raise ValueError('Invalid development exposure item')
        image = item.get('image')
        roles, reasons, sources = item.get('roles'), item.get('reasons'), item.get('evidence')
        if (not isinstance(image, str) or not image.strip() or
                not isinstance(roles, list) or not roles or
                not isinstance(reasons, list) or not reasons or
                not isinstance(sources, list) or not sources or
                any(not isinstance(x, str) or not x.strip() for x in roles + reasons + sources)):
            raise ValueError('Incomplete development exposure evidence: ' + str(image))
        if set(roles) - {'regression_image', 'direct_parent_image', 'direct_background',
                         'direct_sticker_parent', 'direct_foreground_source'}:
            raise ValueError('Unsupported development exposure role: ' + image)
        if workspace_path(root, image).relative_to(root).as_posix() != image:
            raise ValueError('Noncanonical development exposure image: ' + image)
        if not workspace_path(root, image).is_file():
            if missing_paths is None:
                raise ValueError('Incomplete development exposure evidence: ' + image)
            missing_paths.append({'path': image, 'kind': 'exposure_image'})
        if image in seen:
            raise ValueError('Duplicate development exposure image: ' + image)
        seen.add(image)
        for source in sources:
            evidence_path = workspace_path(root, source)
            if evidence_path.relative_to(root).as_posix() != source:
                raise ValueError('Development exposure evidence is missing: ' + source)
            if not evidence_path.is_file():
                if missing_paths is None:
                    raise ValueError('Development exposure evidence is missing: ' + source)
                missing_paths.append({'path': source, 'kind': 'exposure_evidence'})
            else:
                evidence.add(source)
        # Reused source artwork and physical foregrounds are recorded for
        # lineage; using one in a regression composite alone does not make
        # that material a vehicle diagnostic image. Exact regression images,
        # their whole-scene parents/backgrounds, and contacted site-video
        # foreground frames are restricted. Split reuse still needs review.
        restricted = (bool(set(roles) & {'regression_image', 'direct_parent_image',
                                         'direct_background', 'direct_sticker_parent'}) or
                      ('direct_foreground_source' in roles and
                       (image.startswith(SITE_SEQUENCE + '/') or image.startswith(PERSON_SEQUENCE + '/'))))
        if restricted:
            images.add(image)
        video, seconds = item.get('parent_video'), item.get('source_time_seconds')
        if seconds is not None:
            if (not isinstance(video, str) or not video.strip() or
                    not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds)):
                raise ValueError('Invalid development exposure video time: ' + image)
            if workspace_path(root, video).relative_to(root).as_posix() != video:
                raise ValueError('Noncanonical development exposure video: ' + image)
            if not workspace_path(root, video).is_file():
                if missing_paths is None:
                    raise ValueError('Invalid development exposure video time: ' + image)
                missing_paths.append({'path': video, 'kind': 'exposure_parent_video'})
            if restricted:
                video_frames.append({'video': video, 'seconds': seconds, 'image': image})
    return images, video_frames, sorted(evidence)


def development_duplicate_pairs(root, *, missing_paths=None):
    """Read explicitly confirmed near duplicates and their review evidence."""
    ledger = read_json(root, DEVELOPMENT_EXPOSURE)
    items = ledger.get('near_duplicate_pairs', [])
    if not isinstance(items, list):
        raise ValueError('Invalid development duplicate list')
    pairs, evidence, seen = [], set(), set()
    for item in items:
        if not isinstance(item, dict) or item.get('review_status') != 'confirmed':
            raise ValueError('Development duplicate must be confirmed')
        paths = [item.get('a'), item.get('b'), item.get('evidence')]
        if any(not isinstance(value, str) or not value.strip() for value in paths):
            raise ValueError('Development duplicate requires two paths and evidence')
        for value in paths:
            path = workspace_path(root, value)
            if path.relative_to(root).as_posix() != value:
                raise ValueError('Development duplicate path or evidence missing: ' + value)
            if not path.is_file():
                if missing_paths is None:
                    raise ValueError('Development duplicate path or evidence missing: ' + value)
                missing_paths.append({'path': value, 'kind': 'duplicate_evidence' if value == paths[2] else 'duplicate_image'})
        pair = tuple(sorted(paths[:2]))
        if pair[0] == pair[1] or pair in seen:
            raise ValueError('Duplicate or self-referential near-duplicate pair')
        seen.add(pair)
        pairs.append(list(pair))
        if workspace_path(root, paths[2]).is_file():
            evidence.add(paths[2])
    return pairs, sorted(evidence)


def unavailable_history(root):
    """Return lifecycle-excluded historical claims and their policy dependencies."""
    from production_guard import _index
    root = Path(root).resolve()
    core = _index(root).core
    core.load_lifecycle_rules()
    if core._LIFECYCLE_WARNINGS:
        raise ValueError('lifecycle evidence unavailable: ' + str(core._LIFECYCLE_WARNINGS))
    ledger = read_json(root, DEVELOPMENT_EXPOSURE)
    claims = []

    def claim(value, role, index):
        if not isinstance(value, str) or not value:
            return
        reason = core.lifecycle_reason(value)
        if reason:
            claims.append({'path': value, 'role': role, 'reason': reason,
                           'source': DEVELOPMENT_EXPOSURE, 'record_index': index,
                           'status': 'lifecycle_excluded_not_comparable'})

    for index, item in enumerate(ledger.get('items', [])):
        for role in item.get('roles', []):
            claim(item.get('image'), role, index)
        claim(item.get('parent_video'), 'parent_video', index)
        for value in item.get('evidence', []):
            claim(value, 'exposure_evidence', index)
    for index, item in enumerate(ledger.get('near_duplicate_pairs', [])):
        for key in ('a', 'b', 'evidence'):
            claim(item.get(key), 'near_duplicate_' + key, index)
    dependencies = [root / core.LIFECYCLE_POLICY]
    dependencies.extend(root / row['path'] for row in core._POLICY_METADATA.get('sources', []))
    return claims, dependencies


def queue_inputs(root):
    """Declare the files consumed by the source queue, without rebuilding it."""
    root = Path(root).resolve()
    unavailable, lifecycle_inputs = unavailable_history(root)
    excluded = {workspace_path(root, item['path']) for item in unavailable}
    procedural = {workspace_path(root, name) for name in (
        INVENTORY, SEGMENTS, SELECTION, *ANNOTATIONS, REVIEW_SUMMARY, EXPOSURE,
        DUPLICATES, *SITE_SELECTIONS, DEVELOPMENT_EXPOSURE, 'config/perception_tasks.json')}
    media = set()
    visited = set()
    media_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff',
                      '.mp4', '.mov', '.avi', '.mkv', '.npz'}

    def collect(value):
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and value and '\n' not in value:
            if value in visited:
                return
            visited.add(value)
            try:
                if not (root / value).is_file():
                    return
                path = workspace_path(root, value)
                if path not in excluded:
                    (media if path.suffix.lower() in media_suffixes else procedural).add(path)
            except (OSError, ValueError):
                pass

    # These existing manifests own source identity. Do not invent new source rows.
    for path in tuple(procedural):
        if path.suffix == '.json':
            collect(json.loads(path.read_text(encoding='utf-8-sig')))
        elif path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                collect(list(csv.DictReader(stream)))
    procedural.update(lifecycle_inputs)
    for name in MODELS:
        config_path = root / 'lab/features' / OWNERS.get(name, name) / 'training/config' / (name + '_v3.yaml')
        config = read_yaml(config_path)
        procedural.add(config_path)
        data = workspace_path(root, config['data'])
        if data.is_file():
            procedural.add(data)
    code = [root / 'lab/scripts/data/v4_source_gate.py', root / 'lab/training_common/engine/config.py',
            root / 'lab/training_common/engine/data.py',
            root / 'lab/training_common/train.py',
            root / 'lab/features/meter/training/scripts/generate_meter_synthetic.py']
    return sorted(media), sorted(procedural), code


def build_queue(root, *, inputs=None):
    """Reuse a database-recorded queue until its actual dependencies change."""
    from production_guard import generation_guard
    root = Path(root).resolve()
    media, procedural, code = inputs if inputs is not None else queue_inputs(root)
    with generation_guard('source_queue:car_capture', media, [], {'version': 1},
                          procedural_inputs=procedural, code_paths=code, root=root) as run:
        if run.reused:
            return run.value
        value = _build_queue(root)
        run.set_result(value)
        return value


def _build_queue(root):
    inventory = read_json(root, INVENTORY)
    segments = read_json(root, SEGMENTS)['segments']
    selection = read_json(root, SELECTION)
    exposed = {x['image'] for x in read_json(root, EXPOSURE)['items']}
    exposed.update(x['path'] for x in selection['items'])
    exposed_video_frames = []
    missing_exposure_paths = []
    unavailable, _ = unavailable_history(root)
    development_images, development_frames, development_evidence = development_exposure(root, missing_paths=missing_exposure_paths)
    development_pairs, duplicate_evidence = development_duplicate_pairs(root, missing_paths=missing_exposure_paths)
    exposed.update(development_images)
    exposed_video_frames.extend(development_frames)
    for filename in SITE_SELECTIONS:
        with workspace_path(root, filename).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['disposition'] not in ('selected', 'near_duplicate'):
                    continue
                exposed.add(row['image'])
                if row.get('source_video') and row.get('video_seconds'):
                    exposed_video_frames.append({'video': row['source_video'],
                                                 'seconds': float(row['video_seconds']),
                                                 'image': row['image']})
    annotations = []
    for filename in ANNOTATIONS:
        for row in read_json(root, filename):
            annotations.append((filename, row))
            exposed.add(row['image'])
            exposed.update(row.get('checked_frames', []))
    pairs = [p for p in read_json(root, DUPLICATES)['candidate_pairs']
             if p['review_status'] == 'visually_confirmed_near_same_static_composition']
    by_image = defaultdict(dict)
    for filename, row in annotations:
        by_image[row['image']][row['model_key']] = {
            'state': 'candidate', 'source_status': row['review_status'],
            'source_record': filename, 'label': None, 'review_evidence': None,
        }
    records = []
    confirmations = read_json(root, REVIEW_SUMMARY)['human_confirmations']
    for confirmation in confirmations:
        candidate = by_image.get(confirmation['image'], {}).get(confirmation['model_key'])
        if candidate is not None:
            candidate.setdefault('instance_evidence', []).append({
                'candidate_xyxy': confirmation['xyxy'], 'class_id': confirmation['class_id'],
                'class_name': confirmation['class_name'], 'confirmed_fact': 'physical_class_only',
                'source': confirmation['source'], 'box_and_image_completeness': 'candidate',
            })
    for item in inventory['items']:
        frame = item.get('filename_frame_id')
        segment = next((s for s in segments if frame is not None and
                        s['start_frame'] <= frame <= s['end_frame']), None)
        records.append({
            'id': item['path'], 'image': item['path'], 'source_group': item['source_group'],
            'image_size': [item['width'], item['height']],
            'source_kind': item['role'], 'generation_method': item['generation_method'],
            'parent_video': item.get('reported_parent_video'),
            'time_alignment': item.get('video_alignment_status', item.get('relation_to_video')),
            'segment_id': segment['segment_id'] if segment else None,
            'episode_group': segment['episode_group'] if segment else None,
            'parent_ids': [item['duplicate_of']] if item.get('duplicate_of') else [],
            'physical_assets': None, 'source_assets': None, 'mother_scene': None,
            'image_processing': None, 'split': 'unassigned',
            'lineage_review': 'pending', 'near_duplicate_review': 'pending',
            'review_evidence': None, 'selected_tasks': [], 'tasks': by_image.get(item['path'], {}),
        })
    canonical, missing_canonical = {}, {}
    for name in MODELS:
        config = read_yaml(root / 'lab/features' / OWNERS.get(name, name) /
                           'training/config' / (name + '_v3.yaml'))
        source = workspace_path(root, config['data'])
        if source.is_file():
            canonical[name] = dataset_header(root, config['data'])
        else:
            canonical[name] = None
            missing_canonical[name] = config['data']
    requirements = read_json(root, 'config/perception_tasks.json')
    waste = next(task['waste_items'] for task in requirements['tasks'] if task['id'] == 'waste_bin')
    if canonical['waste_sticker'] is not None and canonical['waste_sticker']['names'] != list(waste):
        raise ValueError('垃圾贴纸当前类别次序与需求配置不一致')
    tree = ast.parse((root / 'lab/features/meter/training/scripts/generate_meter_synthetic.py').read_text(encoding='utf8'))
    points = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'KEYPOINTS' for target in node.targets))
    if canonical['meter'] is not None:
        canonical['meter']['keypoint_names'] = list(points)
    return {'version': 1, 'purpose': 'review_queue_not_training_export',
            'inputs': list(dict.fromkeys([INVENTORY, SEGMENTS, SELECTION, *ANNOTATIONS, REVIEW_SUMMARY, EXPOSURE,
                                         DUPLICATES, *SITE_SELECTIONS, DEVELOPMENT_EXPOSURE,
                                         *development_evidence, *duplicate_evidence])),
            'confirmed_person_scope': {'source': 'config/perception_tasks.json',
                                       'scope': 'near frontal recognizable standees only; white backs are hard negatives'},
            'limitations': ['Only supplied inventories and known review evidence are represented.',
                            'Some vehicle-camera frame to video time alignment is not verified.',
                            'Unreviewed frames are not negative labels.',
                            'Known near duplicates do not constitute a full-library search.'],
            'models': {}, 'canonical_models': canonical, 'canonical_models_unavailable': missing_canonical,
            'missing_exposure_paths': sorted({(item['path'], item['kind']) for item in missing_exposure_paths}),
            'unavailable_exposure_paths': unavailable,
            'exposure_status': 'incomplete' if missing_exposure_paths or unavailable else 'complete',
            'exposed_images': sorted(exposed),
            'exposed_video_frames': exposed_video_frames,
            'near_duplicate_pairs': [[p['a'], p['b']] for p in pairs] + development_pairs, 'samples': records}


def _source_reuse(records, lineage_groups):
    """Summarize cross-split relationships already recorded in the manifest."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        split, key = row.get('split'), row.get('id')
        if split not in SPLITS or key not in lineage_groups:
            continue
        origins = [('lineage_group', lineage_groups[key], None)]
        for field in ('source_group', 'parent_video', 'mother_scene', 'episode_group', 'segment_id'):
            value = row.get(field)
            if isinstance(value, str) and value:
                scope = row.get('source_group') if field in ('episode_group', 'segment_id') else None
                origins.append((field, value, scope if isinstance(scope, str) else None))
        for field in ('parent_ids', 'physical_assets', 'source_assets'):
            values = row.get(field)
            if isinstance(values, list):
                origins.extend((field, value, None) for value in values if isinstance(value, str) and value)
        for origin in origins:
            groups[origin][split].append(key)
    report = []
    for (relation, source, scope), splits in sorted(groups.items(), key=lambda item: (
            item[0][0], item[0][1], item[0][2] or '')):
        if len(splits) > 1:
            item = {'relation': relation, 'source': source, 'splits': sorted(splits),
                    'sample_ids': {split: sorted(set(keys)) for split, keys in sorted(splits.items())}}
            if scope is not None:
                item['source_group'] = scope
            report.append(item)
    return report


def audit(root, data, known=None):
    errors = []
    exposure_warnings = []
    if known is not None and (known.get('missing_exposure_paths') or known.get('unavailable_exposure_paths')):
        exposure_warnings.append('部分历史诊断接触或近重复证据缺失，独立性未证实')
    site_times = None
    person_times = None

    def check_sequence(row, key):
        """Verify source and presentation time for known full-sequence paths."""
        nonlocal site_times, person_times
        name = row.get('image', '')
        if not isinstance(name, str):
            return
        if name.startswith(SITE_SEQUENCE + '/'):
            prefix, video, label = SITE_SEQUENCE, SITE_VIDEO, '全场'
        elif name.startswith(PERSON_SEQUENCE + '/'):
            prefix, video, label = PERSON_SEQUENCE, PERSON_VIDEO, '人物'
        else:
            return
        match = re.fullmatch(r'frame_(\d+)\.png', name[len(prefix) + 1:])
        if (not match or row.get('parent_video') != video or row.get('source_kind') not in VIDEO_KINDS or
                row.get('generation_method') != 'provided_video_frame'):
            errors.append(f'{key}: {label}序列帧来源声明无效')
            return
        try:
            if prefix == SITE_SEQUENCE:
                if site_times is None:
                    site_times = site_sequence_times(root)
                expected_time = site_times[int(match.group(1))]
                error_label = '解码日志实际帧时间'
            else:
                if person_times is None:
                    person_times = person_sequence_times(root)
                expected_time = person_times[int(match.group(1))]
                error_label = '人物序列探测时间'
            actual_time = row.get('source_time_seconds')
            if (not isinstance(actual_time, (int, float)) or isinstance(actual_time, bool) or
                    not math.isfinite(actual_time) or abs(actual_time - expected_time) > .001):
                errors.append(f'{key}: 源时间与{error_label}不符')
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            errors.append(f'{key}: 无法核对序列帧时间: {exc}')

    records = data['samples']
    by_id = {}
    for row in records:
        key = row.get('id')
        if not key or key in by_id:
            errors.append(f'缺少或重复样本 ID: {key}')
        else:
            by_id[key] = row
    links = {key: set() for key in by_id}
    for row in records:
        key = row.get('id')
        if key not in links:
            continue
        for parent in row.get('parent_ids', []):
            if parent not in by_id:
                errors.append(f'{key}: 父样本不在清单: {parent}')
            else:
                links[key].add(parent)
                links[parent].add(key)
    images = defaultdict(list)
    for key, row in by_id.items():
        images[row.get('image')].append(key)
    for keys in images.values():
        for key in keys[1:]:
            links[key].add(keys[0])
            links[keys[0]].add(key)
    for pair in data.get('near_duplicate_pairs', []):
        if len(pair) != 2 or any(x not in links for x in pair):
            errors.append(f'近重复关系无法解析: {pair}')
            continue
        a, b = pair
        links[a].add(b)
        links[b].add(a)
    exposed_images = set(data.get('exposed_images', []))
    if known is not None:
        exposed_images.update(known.get('exposed_images', []))
    direct_exposure = set(exposed_images)
    known_components = []
    if known is not None:
        known_links = defaultdict(set)
        for row in known['samples']:
            for parent in row.get('parent_ids', []):
                known_links[row['id']].add(parent)
                known_links[parent].add(row['id'])
        for a, b in known.get('near_duplicate_pairs', []):
            known_links[a].add(b)
            known_links[b].add(a)
        seen_known = set()
        for first in known_links:
            if first in seen_known:
                continue
            component, pending = set(), [first]
            while pending:
                member = pending.pop()
                if member in component:
                    continue
                component.add(member)
                pending.extend(known_links[member] - component)
            seen_known.update(component)
            known_components.append(component)
            if component & exposed_images:
                exposed_images.update(component)
            represented = [key for image in component for key in images.get(image, [])]
            for key in represented[1:]:
                links[key].add(represented[0])
                links[represented[0]].add(key)
    exposed = {key for key, row in by_id.items() if row.get('image') in exposed_images or key in exposed_images}
    # A parent may be unassigned while a descendant is selected.  Compare the
    # bytes of all records connected to a selected image before propagating
    # exposure, so a renamed copy of a diagnostic parent cannot bypass it.
    exposed_files = defaultdict(list)
    exact_exposure_files = defaultdict(list)
    development_contact = defaultdict(set)
    known_files = defaultdict(list)
    related_files = defaultdict(list)
    # Exact copies are related even when no external inventory is supplied.
    # Missing unselected records are checked only if the closure reaches them.
    for key, row in by_id.items():
        try:
            name = row.get('image')
            if not isinstance(name, str) or not name:
                continue
            path = workspace_path(root, name)
            if path.relative_to(root).as_posix() != name or not path.is_file():
                continue
            bucket = related_files[path.stat().st_size]
            for prior, prior_key in bucket:
                if filecmp.cmp(prior, path, shallow=False):
                    links[key].add(prior_key)
                    links[prior_key].add(key)
                    break
            else:
                bucket.append((path, key))
        except (OSError, ValueError):
            continue
    unavailable_paths = {item['path'] for item in (known or {}).get('unavailable_exposure_paths', [])}
    for image in sorted(exposed_images - unavailable_paths):
        path = workspace_path(root, image)
        if not path.is_file():
            exposure_warnings.append(f'诊断接触来源缺失: {image}')
            continue
        size = path.stat().st_size
        exposed_files[size].append(path)
        if image in direct_exposure:
            exact_exposure_files[size].append(path)
    if known is not None:
        for item in known['samples']:
            if (not item.get('image') or item['image'] in unavailable_paths or
                    item.get('source_kind') not in SOURCE_METHODS):
                continue
            path = workspace_path(root, item['image'])
            if path.is_file():
                known_files[path.stat().st_size].append((path, item))
        component_files = defaultdict(list)
        component_representative = {}
        known_by_id = {item['id']: item for item in known['samples']}
        for index, component in enumerate(known_components):
            represented = [key for member in component for key in images.get(member, [])]
            if represented:
                component_representative[index] = represented[0]
            paths = set(component)
            paths.update(known_by_id[member].get('image') for member in component if member in known_by_id)
            for name in paths:
                if not isinstance(name, str) or not name or name in unavailable_paths:
                    continue
                try:
                    path = workspace_path(root, name)
                    if path.is_file():
                        component_files[path.stat().st_size].append((path, index))
                except (OSError, ValueError):
                    continue
        relevant, pending = set(), [key for key, row in by_id.items() if row.get('split') in SPLITS]
        while pending:
            key = pending.pop()
            if key in relevant:
                continue
            relevant.add(key)
            row = by_id[key]
            name = row.get('image')
            try:
                if not isinstance(name, str) or not name:
                    raise ValueError('缺少图片路径')
                path = workspace_path(root, name)
                if path.relative_to(root).as_posix() != name or not path.is_file():
                    raise ValueError('图片路径不规范或文件不存在')
                size = path.stat().st_size
                for prior, index in component_files[size]:
                    if filecmp.cmp(prior, path, shallow=False):
                        representative = component_representative.setdefault(index, key)
                        links[key].add(representative)
                        links[representative].add(key)
                if any(filecmp.cmp(prior, path, shallow=False)
                       for prior in exposed_files[size]):
                    exposed.add(key)
                if row.get('split') not in SPLITS:
                    check_sequence(row, key)
                observations = [row]
                observations.extend(original for prior, original in known_files[size]
                                    if filecmp.cmp(prior, path, shallow=False))
                for observation in observations:
                    if observation.get('source_kind') not in VIDEO_KINDS:
                        continue
                    video = observation.get('parent_video')
                    if (not isinstance(video, str) or not video or
                            workspace_path(root, video).relative_to(root).as_posix() != video or
                            not workspace_path(root, video).is_file()):
                        errors.append(f'{key}: 关联视频父样本来源路径无效')
                        continue
                    same_video = [frame for frame in known.get('exposed_video_frames', [])
                                  if frame['video'] == video]
                    seconds = observation.get('source_time_seconds')
                    if same_video:
                        development_contact[key].add('shared_diagnostic_video')
                    if same_video and (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or
                                       not math.isfinite(seconds)):
                        errors.append(f'{key}: 关联视频父样本须记录源时间')
                    elif same_video and any(abs(seconds - frame['seconds']) <= .25 + 1e-6
                                            for frame in same_video):
                        development_contact[key].add('near_diagnostic_frame')
                    if observation is not row and row.get('source_kind') in VIDEO_KINDS:
                        declared = row.get('source_time_seconds')
                        if (not isinstance(declared, (int, float)) or isinstance(declared, bool) or
                                not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or
                                not math.isfinite(declared) or not math.isfinite(seconds) or
                                abs(declared - seconds) > .001):
                            errors.append(f'{key}: 关联视频帧源时间与已登记原图不符')
            except (OSError, ValueError) as exc:
                errors.append(f'{key}: 关联父样本或近重复图片无法核查: {exc}')
            pending.extend(links[key] - relevant)
    visited = set()
    lineage_groups = {}
    for first in links:
        if first in visited:
            continue
        group, pending = set(), [first]
        while pending:
            key = pending.pop()
            if key in group:
                continue
            group.add(key)
            pending.extend(links[key] - group)
        visited.update(group)
        contacts = {reason for key in group for reason in development_contact.get(key, ())}
        if group & exposed:
            contacts.add('known_diagnostic_lineage')
        for key in group:
            if by_id[key].get('split') in SPLITS:
                lineage_groups[key] = min(group)
                if contacts:
                    development_contact[key].update(contacts)
    counts = Counter()
    selected_images = defaultdict(list)
    label_owners = {}
    for row in records:
        key, split = row.get('id'), row.get('split')
        if split not in SPLITS | {'unassigned', 'regression', 'exclude'}:
            errors.append(f'{key}: 非法集合 {split}')
            continue
        counts[split] += 1
        if split not in SPLITS:
            continue
        for field in ('image', 'source_group', 'source_kind', 'generation_method'):
            if not row.get(field):
                errors.append(f'{key}: 缺少 {field}')
        for field in ('image', 'parent_video'):
            value = row.get(field)
            if isinstance(value, str) and value:
                try:
                    canonical = workspace_path(root, value).relative_to(root).as_posix()
                    if value != canonical:
                        errors.append(f'{key}: {field} 须使用规范工作空间相对路径')
                except (ValueError, OSError) as exc:
                    errors.append(f'{key}: {field} 路径无效: {exc}')
        check_sequence(row, key)
        image_path = workspace_path(root, row['image'])
        decoded_size = None
        if not image_path.is_file():
            errors.append(f'{key}: 图片不存在')
        else:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                decoded_size = [width, height]
                if min(width, height) > 720 or max(width, height) > 1280:
                    errors.append(f'{key}: 图片尺寸超过 720p 横竖边界: {width}x{height}')
                if row.get('image_size') is not None and row['image_size'] != [width, height]:
                    errors.append(f'{key}: 记录图片尺寸与实际解码尺寸不符')
            except (OSError, ValueError, OverflowError, Image.DecompressionBombError) as exc:
                errors.append(f'{key}: 图片解码失败: {exc}')
            image_size = image_path.stat().st_size
            for prior, prior_split in selected_images[image_size]:
                if filecmp.cmp(prior, image_path, shallow=False):
                    errors.append(f'{key}: 所选图片字节相同（{prior_split}/{split}）: {prior}')
            if row['image'] in direct_exposure or key in direct_exposure:
                errors.append(f'{key}: 所选输出是已接触诊断图: {row["image"]}')
            for prior in exact_exposure_files[image_size]:
                if filecmp.cmp(prior, image_path, shallow=False):
                    errors.append(f'{key}: 图片与已接触诊断图字节相同: {prior}')
                    break
            for prior, original in known_files[image_size]:
                if not filecmp.cmp(prior, image_path, shallow=False):
                    continue
                declared, observed = row.get('source_kind'), original['source_kind']
                if declared != observed and not (declared in VIDEO_KINDS and observed in VIDEO_KINDS):
                    errors.append(f'{key}: 声明来源种类与已登记原图不符: {prior}')
                for field in ('source_group', 'generation_method'):
                    if row.get(field) != original.get(field):
                        errors.append(f'{key}: {field} 与已登记原图不符: {prior}')
                if observed in VIDEO_KINDS:
                    for field in ('parent_video', 'segment_id', 'episode_group'):
                        if row.get(field) != original.get(field):
                            errors.append(f'{key}: {field} 与已登记序列不符: {prior}')
            selected_images[image_size].append((image_path, split))
        kind, method = row.get('source_kind'), row.get('generation_method')
        processing = row.get('image_processing')
        if isinstance(processing, dict):
            resolution_review = processing.get('resolution_review')
            if resolution_review not in (None, 'pending', 'confirmed'):
                errors.append(f'{key}: 分辨率审核状态无效')
            if resolution_review == 'confirmed':
                source_size, output_size = processing.get('source_size'), processing.get('output_size')
                valid_size = lambda value: (isinstance(value, list) and len(value) == 2 and
                                            all(type(n) is int and n > 0 for n in value))
                resampling, clarity = processing.get('resampling'), processing.get('clarity')
                if (not valid_size(source_size) or not valid_size(output_size) or
                        output_size != decoded_size or resampling not in ('none', 'resampled', 'unknown') or
                        clarity not in ('original', 'modified', 'unknown') or
                        (resampling == 'none' and clarity == 'original' and source_size != output_size) or
                        (resampling == 'resampled' and clarity == 'original')):
                    errors.append(f'{key}: 已确认的分辨率或清晰度记录不一致')
            origin_review = processing.get('origin_review')
            if origin_review not in (None, 'pending', 'confirmed'):
                errors.append(f'{key}: 来源类别审核状态无效')
            if origin_review == 'confirmed' and method in ('composite', 'augmentation'):
                field = 'background_origin' if method == 'composite' else 'parent_origin'
                if processing.get(field) not in ('direct_camera', 'synthetic', 'unknown'):
                    errors.append(f'{key}: 已确认的 {field} 无效')
        if method not in GENERATION_METHODS:
            errors.append(f'{key}: 生成方式不在允许范围')
        if kind not in SOURCE_METHODS or method not in SOURCE_METHODS[kind]:
            errors.append(f'{key}: 来源种类与生成方式不一致或不受支持')
        if (row.get('lineage_review') != 'confirmed' or
                row.get('near_duplicate_review') != 'confirmed' or not row.get('review_evidence') or
                not workspace_path(root, row['review_evidence']).is_file()):
            errors.append(f'{key}: 来源、近重复或审核证据未确认')
        if kind in VIDEO_KINDS:
            if (not row.get('segment_id') or not row.get('episode_group') or not row.get('parent_video') or
                    not workspace_path(root, row['parent_video']).is_file()):
                errors.append(f'{key}: 序列帧缺少片段、事件或有效来源视频')
        if kind == 'provided_subset' and not row.get('parent_ids'):
            errors.append(f'{key}: 提供的序列副本缺少父帧')
        if method in DERIVED and not row.get('parent_ids'):
            errors.append(f'{key}: 派生图片缺少父样本')
        if method in DERIVED | {'blender_render', 'synthetic_render'}:
            processing = row.get('image_processing')
            if (not isinstance(processing, dict) or
                    any(name not in processing or processing[name] in (None, '', [], {})
                        for name in ('resize', 'compression', 'blur'))):
                errors.append(f'{key}: 缺少缩放、压缩或模糊参数记录')
        if method in {'blender_render', 'synthetic_render'} and (
                not isinstance(row.get('mother_scene'), str) or not row['mother_scene'].strip() or
                not row.get('source_assets')):
            errors.append(f'{key}: 渲染图缺少母场景或来源素材')
        for field in ('physical_assets', 'source_assets'):
            value = row.get(field)
            if value is not None and (not isinstance(value, list) or
                                      any(not isinstance(x, str) or not x.strip() for x in value)):
                errors.append(f'{key}: {field} 须为素材 ID 列表或 null')
        tasks = row.get('tasks', {})
        selected_tasks = row.get('selected_tasks', [])
        if not isinstance(selected_tasks, list) or not selected_tasks or len(selected_tasks) != len(set(selected_tasks)):
            errors.append(f'{key}: 未提供任务标签')
            continue
        for task in selected_tasks:
            review = tasks.get(task, {})
            if task not in MODELS:
                errors.append(f'{key}: 未知模型 {task}')
                continue
            strata = review.get('strata')
            strata_review = review.get('strata_review')
            if strata_review not in (None, 'pending', 'confirmed'):
                errors.append(f'{key}/{task}: 分层审核状态无效')
            if strata is not None:
                if not isinstance(strata, dict) or set(strata) - {'difficulty', 'conditions', 'target_states'}:
                    errors.append(f'{key}/{task}: 分层字段无效')
                else:
                    difficulty = strata.get('difficulty')
                    if difficulty is not None and (not isinstance(difficulty, str) or not difficulty.strip()):
                        errors.append(f'{key}/{task}: 难度须为非空字符串')
                    for field in ('conditions', 'target_states'):
                        values = strata.get(field)
                        if values is not None and (not isinstance(values, list) or
                                len(values) > 64 or
                                any(not isinstance(value, str) or not value.strip() for value in values) or
                                len(values) != len(set(values))):
                            errors.append(f'{key}/{task}: {field} 须为不超过 64 项且无重复的非空字符串列表')
            if review.get('state') not in ('complete_positive', 'complete_negative'):
                errors.append(f'{key}/{task}: 不是完整审核状态')
                continue
            if (not review.get('review_evidence') or not review.get('label') or
                    not workspace_path(root, review['review_evidence']).is_file()):
                errors.append(f'{key}/{task}: 缺少审核证据或标签路径')
                continue
            label = workspace_path(root, review['label'])
            if not label.is_file():
                errors.append(f'{key}/{task}: 标签文件不存在')
                continue
            if label in label_owners and label_owners[label] != (key, task):
                errors.append(f'{key}/{task}: 标签路径被其他图像或任务复用')
            label_owners[label] = (key, task)
            lines = [x for x in label.read_text(encoding='utf-8').splitlines() if x.strip()]
            if bool(lines) != (review['state'] == 'complete_positive'):
                errors.append(f'{key}/{task}: 正负状态与标签内容不一致')
            model = data.get('models', {}).get(task, {})
            try:
                header = dataset_header(root, model['data_yaml'])
                names = header['names']
                points = header.get('kpt_shape', [0])[0]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'{key}/{task}: 类别表无法读取: {exc}')
                continue
            if points != (8 if task == 'meter' else 0):
                errors.append(f'{key}/{task}: 关键点规格错误')
                continue
            if known is not None:
                expected = known['canonical_models'][task]
                if expected is None:
                    errors.append(f'{key}/{task}: 当前类别契约来源缺失')
                    continue
                if names != expected['names']:
                    errors.append(f'{key}/{task}: 类别名称或顺序与当前契约不一致')
                if task == 'meter' and model.get('keypoint_names') != expected['keypoint_names']:
                    errors.append(f'{key}/{task}: 八点名称或顺序与当前契约不一致')
            try:
                check_label(label, len(names), points)
            except ValueError as exc:
                errors.append(f'{key}/{task}: 标签格式无效: {exc}')
    assigned = sum(counts[s] for s in SPLITS)
    return {'status': 'failed' if errors else ('passed' if assigned else 'pending'), 'counts': dict(counts),
            'assigned': assigned, 'errors': errors, 'lineage_groups': lineage_groups,
            'source_reuse': _source_reuse(records, lineage_groups),
            'development_contact': {key: sorted(development_contact[key]) for key in sorted(lineage_groups)
                                    if development_contact.get(key)},
            'exposure_warnings': exposure_warnings,
            'scope': 'Recorded relationships and review states only; no semantic truth or unseen-source claim.'}


def provenance(data, model, audit_result):
    """Translate audited task records into the V4 packager's provenance schema."""
    if model not in MODELS:
        raise ValueError('未知模型: ' + model)
    if audit_result.get('status') != 'passed' or not isinstance(audit_result.get('lineage_groups'), dict):
        raise ValueError('来源导出需要通过的完整来源审核结果')
    records = []
    for row in data['samples']:
        if row.get('split') not in SPLITS or model not in row.get('selected_tasks', []):
            continue
        task = row['tasks'][model]
        parents = row.get('parent_ids', [])
        assets = row.get('physical_assets') or []
        records.append({'image': row['image'], 'label': task['label'], 'split': row['split'],
                        'source_id': row['source_group'], 'parent_sample': parents[0] if parents else None,
                        'parent_samples': parents, 'physical_asset': assets[0] if assets else None,
                        'physical_assets': row.get('physical_assets'), 'source_assets': row.get('source_assets'),
                        'mother_scene': row.get('mother_scene'), 'image_processing': row.get('image_processing'),
                        'image_size': row.get('image_size'),
                        'parent_video': row.get('parent_video'), 'source_time_seconds': row.get('source_time_seconds'),
                        'time_alignment': row.get('time_alignment'),
                        'source_kind': row.get('source_kind'), 'fragment': row.get('segment_id'),
                        'episode_group': row.get('episode_group'), 'generation_method': row['generation_method'],
                        'near_duplicate_group': audit_result['lineage_groups'][row['id']], 'label_status': 'reviewed',
                        'review_state': task['state'], 'review_evidence': task['review_evidence'],
                        'source_review_evidence': row['review_evidence'],
                        'strata': task.get('strata'), 'strata_review': task.get('strata_review'),
                        'development_contact': audit_result.get('development_contact', {}).get(row['id'], [])})
    if not records:
        raise ValueError('该模型没有通过门控的选定样本')
    return {'schema_version': 1, 'model_key': model, 'records': records,
            'source_reuse': audit_result.get('source_reuse', []),
            'exposure_warnings': audit_result.get('exposure_warnings', []),
            'scope': 'Selected records; review is not an independent accuracy or delivery claim.'}


def _selection_files(root, data):
    """Collect existing explicit file references used by selection audit/export."""
    files = set()
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value and '\n' not in value:
            try:
                path = workspace_path(root, value)
                if path.is_file():
                    files.add(path)
            except (OSError, ValueError):
                pass
    visit(data)
    # audit reads decoder timing evidence for every referenced sequence row,
    # including unassigned parents. A missing log must not silently hit a receipt.
    for row in data.get('samples', []):
        image = row.get('image', '')
        if isinstance(image, str):
            image = image.replace('\\', '/').removeprefix('./')
            if image.startswith(SITE_SEQUENCE + '/'):
                files.add(workspace_path(root, SITE_SEQUENCE + '/ffmpeg.log'))
            if image.startswith(PERSON_SEQUENCE + '/'):
                files.add(workspace_path(root, PERSON_SEQUENCE + '/frames_probe.json'))
    return sorted(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    initial = sub.add_parser('bootstrap')
    initial.add_argument('--output', required=True)
    check = sub.add_parser('audit')
    check.add_argument('--manifest', required=True)
    export = sub.add_parser('export-provenance')
    export.add_argument('--manifest', required=True)
    export.add_argument('--model', required=True, choices=MODELS)
    export.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'bootstrap':
            from production_guard import generation_guard
            out = workspace_path(ROOT, args.output)
            base = workspace_path(ROOT, 'datasets/annotated')
            if not out.is_relative_to(base) or out == base:
                raise ValueError('输出须在 datasets/annotated 下')
            media, procedural, code = queue_inputs(ROOT)
            with generation_guard('source_bootstrap:' + args.output, media, [out],
                                  {'command': args.command}, procedural_inputs=procedural,
                                  code_paths=code, root=ROOT) as run:
                if run.reused:
                    print('SOURCE_REUSED', args.output)
                    return 0
                data = build_queue(ROOT)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open('w', encoding='utf-8') as stream:
                    json.dump(data, stream, ensure_ascii=False, indent=2)
                    stream.write('\n')
                print(f'已建立 {len(data["samples"])} 项待审核来源记录: {args.output}')
        else:
            data = read_json(ROOT, args.manifest)
            if args.command == 'export-provenance':
                from production_guard import generation_guard
                output = workspace_path(ROOT, args.output)
                if not output.is_relative_to(workspace_path(ROOT, 'datasets/annotated')):
                    raise ValueError('来源清单输出须在 datasets/annotated 下')
                media, procedural, code = queue_inputs(ROOT)
                dependencies = set(procedural) | set(_selection_files(ROOT, data)) | {workspace_path(ROOT, args.manifest)}
                with generation_guard('source_export:' + args.output, media, [output],
                                      {'model': args.model}, procedural_inputs=sorted(dependencies),
                                      code_paths=code, root=ROOT) as run:
                    if run.reused:
                        print('SOURCE_REUSED', args.output)
                        return 0
                    known = build_queue(ROOT)
                    result = audit(ROOT, data, known)
                    if result['status'] != 'passed' or not result['assigned']:
                        raise ValueError('来源审核未通过: ' + json.dumps(result, ensure_ascii=False))
                    value = provenance(data, args.model, result)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('w', encoding='utf8') as stream:
                        json.dump(value, stream, ensure_ascii=False, indent=2)
                        stream.write('\n')
            else:
                result = audit(ROOT, data, build_queue(ROOT))
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if result['status'] != 'passed' or not result['assigned']:
                    return 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'失败: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
