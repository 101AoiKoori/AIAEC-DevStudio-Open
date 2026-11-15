"""Build and query a distributed, local media inventory.

Examples, from the workspace root::

    python lab/scripts/media/media_index.py build
    python lab/scripts/media/media_index.py build --source datasets/exports/waste_bin
    python lab/scripts/media/media_index.py query --contains waste_sticker --limit 10
    python lab/scripts/media/media_index.py query --contains full_scene --with-evidence

The catalog records observed files separately from unverified claims in existing
manifests. It does not assign annotations, split membership or provenance from a
directory name, and it is not a source gate or a dataset approval mechanism.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time
from urllib.parse import quote
import uuid


def io_path(path):
    """Use Windows extended paths only for I/O, never for catalog identities."""
    path = Path(path)
    if os.name != 'nt':
        return path
    value = str(path.absolute())
    if value.startswith('\\\\?\\'):
        return path
    if value.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + value[2:])
    return Path('\\\\?\\' + value)

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / 'datasets/annotated/media_index'
DEFAULT_SOURCES = ('datasets/raw', 'datasets/annotated', 'datasets/exports',
                   'datasets/assets',
                   'training_packages', 'Archive',
                   'lab', 'refs', 'docs', 'ws/src', 'runs', 'tmp', 'config')
ROOT_FILE_SOURCES = ('.', 'datasets')
PROJECT_ASSET_DIRS = ('datasets/exports/waste_bin/.cache/assets',)
LIFECYCLE_POLICY = 'lab/scripts/media/media_index_lifecycle.json'
_DYNAMIC_LIFECYCLE = []
_LIFECYCLE_RECORDS = []
_LIFECYCLE_PATTERNS = []
_LIFECYCLE_WARNINGS = []
_POLICY_METADATA = {}


def path_key(value: str):
    normalized = value.replace('\\', '/').strip('/')
    return normalized.lower() if os.name == 'nt' else normalized
IMAGES = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp',
          '.ppm', '.pgm', '.heic', '.heif', '.avif'}
VIDEOS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm', '.mpeg', '.mpg',
          '.mts', '.m2ts', '.3gp'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'site-packages',
             'portable_env', 'weights', 'checkpoints', 'bundles',
             'build', 'devel', 'install'}
MANIFEST_NAMES = {'manifest.json', 'manifest.jsonl', 'workspace_sources.json',
                  'local_index.json', 'inventory.json', 'sources.json', 'samples.jsonl',
                  'source_inventory.csv', 'photo_review.csv', 'inventory.csv', 'assets.json'}
MANIFEST_HINTS = ('manifest', 'inventory', 'provenance', 'source', 'handoff',
                  'migration', 'lineage', 'metadata', 'relation', 'selection',
                  'audit', 'label', 'annotation', 'sample', 'distribution', 'evaluation')
TASK_CONFIG_NAMES = {'person', 'e_bike', 'fire', 'building_name_paper', 'sign',
                     'waste_bin_state', 'waste_sticker', 'meter'}
TASK_PLAN_KEYS = ('production_plan', 'render_plan', 'direct_real_plan', 'blender_plan')
BATCH_CONTEXT_KEYS = ('scope', 'purpose', 'status', 'dataset_version', 'data_version',
                      'task_version', 'model_key', 'data_distribution', 'evaluation',
                      'evaluation_scope', 'conclusion_scope', 'result_evidence', 'metrics',
                      'reuse_relations', 'shared_source_groups_by_split',
                      'shared_physical_assets', 'known_near_duplicate_scope', 'index_handling')
PATH_KEYS = ('image', 'image_path', 'path', 'video', 'video_path', 'package_image',
             'workspace_image', 'file', 'filename', 'current_path', 'source')
REFERENCE_KEYS = ('parent_image', 'source_image', 'background_parent', 'foreground_source',
                  'source_video', 'parent_video', 'parent_render', 'parent_ids',
                  'parent_sample', 'parent_samples', 'source_assets', 'equivalent_upstream_image',
                  'historical_intermediate_image', 'source')
EVIDENCE_KEYS = ('source_image', 'source_video', 'parent_image', 'parent_video',
                 'parent_render', 'source_manifest', 'source_manifests', 'source_group',
                 'source_kind', 'generation_method', 'label', 'label_path', 'split',
                 'label_state', 'review_status', 'review_state', 'status', 'image_size',
                 'original_path', 'historical_path', 'source_assets', 'source_records',
                 'label_review_status', 'content_review', 'relation_to_video', 'parent_ids',
                 'source_frame', 'source_fps', 'seconds_approx', 'capture_session',
                 'independence', 'annotations_status', 'labels', 'tasks', 'role', 'device',
                 'reuse_relations', 'reused_assets', 'shared_assets', 'shared_sources',
                 'evaluation_scope', 'conclusion_scope', 'result_evidence',
                 'evaluation_evidence', 'metrics', 'test_kind', 'parent_sample',
                 'parent_samples', 'physical_asset', 'physical_assets', 'mother_scene',
                 'image_processing', 'episode_group', 'near_duplicate_group', 'label_status',
                 'source_id', 'review_evidence', 'source_review_evidence', 'strata',
                 'observed_content', 'review_flag', 'background_view',
                 'review_method', 'observed_tasks', 'observed_tasks_scope', 'view', 'view_evidence', 'surface_role', 'camera_angles', 'angle_basis',
                 'camera_yaw', 'camera_pitch', 'camera_roll', 'angle_evidence', 'use_status',
                 'regions', 'region_review_status', 'uncertainties', 'annotation_scope', 'review_scope', 'classification_basis', 'task_observations', 'task_observation_scope', 'frame_number', 'source_frame_sequence_index',
                 'observed_instances', 'rejected_instances', 'instances', 'instances_scope', 'source_review', 'proposal_model',
                 'keypoints', 'keypoints_scope', 'dial_bbox', 'bbox_status',
                 'prior_review_evidence', 'prior_review_record_index', 'prior_person_assessment', 'review_evidence_record_index',
                 'known_occupied_objects', 'known_occupied_scope', 'task_use_status', 'background_observations', 'local_count_observations',
                 'ground_state', 'known_occupied_source', 'occupancy_scope', 'visible_ground_boundary_status', 'occlusion_geometry_status', 'visible_ground_patches', 'projection_reference', 'placement_approval',
                 'rejected_previous_regions', 'instance_exhaustive', 'task_review', 'placement_review_status',
                 'purpose', 'historical_split_claim', 'source_record_index', 'source_label_claim', 'source_review_assessment',
                 'background_placement_plane', 'background_target_presence',
                 'background_use', 'background_review_state',
                 'mean_luma_0_255', 'mean_saturation_0_255',
                 'original_source', 'source_relocation_basis', 'name', 'cls', 'mode', 'w', 'h', 'ver',
                 'background_parent', 'foreground_source', 'background_source_group',
                 'foreground_source_group', 'source_batch', 'source_time_seconds',
                 'equivalent_upstream_image', 'equivalent_upstream_label', 'equivalent_upstream_basis',
                 'matching_basis', 'intermediate_record', 'source_record_scope',
                 'historical_intermediate_image', 'historical_intermediate_label', 'source_relation', 'dataset_version',
                 'data_version', 'task_version')


def lifecycle_reason(path: str):
    value = path_key(path)
    for prefix, reason in _DYNAMIC_LIFECYCLE:
        prefix = path_key(prefix)
        if value == prefix or value.startswith(prefix + '/'):
            return reason
    parts = value.split('/')
    if (any(x.get('pattern') == 'training_packages/.<task>-previous-*' for x in _LIFECYCLE_PATTERNS)
            and len(parts) > 1 and parts[0] == 'training_packages' and parts[1].startswith('.') and '-previous-' in parts[1]):
        return 'superseded_training_package_copy'
    return None


def policy_identity(value):
    """Return lifecycle policy identity without its current filesystem path."""
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key != 'path'}


def load_lifecycle_rules():
    _DYNAMIC_LIFECYCLE.clear()
    _LIFECYCLE_RECORDS.clear()
    _LIFECYCLE_PATTERNS.clear()
    _LIFECYCLE_WARNINGS.clear()
    _POLICY_METADATA.clear()
    policy_path = ROOT / LIFECYCLE_POLICY
    policy = json.loads(policy_path.read_text(encoding='utf8'))
    if not isinstance(policy, dict) or policy.get('schema_version') != 1:
        raise ValueError('invalid lifecycle policy')
    _POLICY_METADATA.update({'path': LIFECYCLE_POLICY, 'schema_version': policy['schema_version'],
                             'revision': policy.get('policy_revision'),
                             'bytes': io_path(policy_path).stat().st_size,
                             'modified_ns': io_path(policy_path).stat().st_mtime_ns, 'sources': []})
    for row in policy.get('rules', []):
        prefix = relative(inside(row['path']))
        if row.get('scope') != 'media_subtree' or row.get('state') != 'inactive':
            raise ValueError('unsupported lifecycle rule: ' + prefix)
        record = {'path': prefix, 'state': row['state'], 'reason': row.get('reason'),
                  'scope': row['scope'], 'evidence': row.get('evidence', []), 'policy': LIFECYCLE_POLICY}
        _LIFECYCLE_RECORDS.append(record)
        _DYNAMIC_LIFECYCLE.append((prefix, str(row.get('reason') or row['state'])))
    _LIFECYCLE_PATTERNS.extend(policy.get('excluded_patterns', []))
    for spec in policy.get('source_rules', []):
        source = spec['path']
        path = inside(source)
        if not path.is_file():
            _LIFECYCLE_WARNINGS.append({'source': source, 'error': 'lifecycle source is missing'})
            continue
        stat = io_path(path).stat()
        _POLICY_METADATA['sources'].append({'path': source, 'bytes': stat.st_size,
                                            'modified_ns': stat.st_mtime_ns})
        data = json.loads(path.read_text(encoding='utf8'))
        for part in spec['items']:
            data = data.get(part) if isinstance(data, dict) else None
        if not isinstance(data, list):
            raise ValueError('invalid lifecycle source: ' + source)
        for row in data:
            if not isinstance(row, dict) or not isinstance(row.get('path'), str):
                continue
            prefix = relative(inside(row['path']))
            state = str(row.get(spec['status_field'], 'unrecorded'))
            evidence = [source]
            for field in spec.get('evidence_fields', []):
                value = row.get(field)
                if isinstance(value, str):
                    evidence.append(value)
                elif isinstance(value, list):
                    evidence.extend(x for x in value if isinstance(x, str))
            record = {'path': prefix, 'state': state, 'reason': row.get(spec['reason_field']),
                      'scope': spec['scope'], 'evidence': evidence, 'policy': LIFECYCLE_POLICY}
            _LIFECYCLE_RECORDS.append(record)
            if state in spec['exclude_states']:
                _DYNAMIC_LIFECYCLE.append((prefix, state))


def structured_claims(records, include_references=False):
    """Collect explicit values only. Empty lists mean no value was recorded."""
    fields = {'generation_methods': ('generation_method',),
              'sources_and_parents': ('source_image', 'source_video', 'parent_image',
                                      'parent_video', 'parent_render', 'parent_ids', 'source_assets',
                                      'parent_sample', 'parent_samples', 'mother_scene', 'original_source',
                                      'background_parent', 'foreground_source', 'equivalent_upstream_image',
                                      'historical_intermediate_image'),
              'source_groups': ('source_group', 'source_id', 'episode_group', 'near_duplicate_group'),
              'reuse_relations': ('reuse_relations', 'reused_assets', 'shared_assets',
                                  'shared_sources', 'relation_to_video', 'physical_asset', 'physical_assets',
                                  'equivalent_upstream_image', 'equivalent_upstream_label',
                                  'equivalent_upstream_basis', 'matching_basis', 'source_relation'),
              'declared_splits': ('split',),
              'label_states': ('label_state', 'label_review_status', 'annotations_status', 'label_status'),
              'evaluation_scope': ('evaluation_scope', 'conclusion_scope', 'test_kind'),
              'result_evidence': ('result_evidence', 'evaluation_evidence', 'metrics')}
    result = {}
    for field, keys in fields.items():
        values = []
        for record in records:
            if not include_references and record.get('relation') == 'referenced_as':
                continue
            for key in keys:
                value = record['claims'].get(key)
                if value is None or value == '':
                    continue
                for element in value if isinstance(value, list) else [value]:
                    if element not in values:
                        values.append(element)
        result[field] = values
    split_claims = []
    v4_splits = []
    for record in records:
        if record.get('relation') == 'referenced_as' or record['claims'].get('split') is None:
            continue
        claims = record['claims']
        context = record.get('manifest_context', {})
        version = next((obj[key] for obj in (claims, context) for key in ('dataset_version', 'data_version', 'task_version')
                        if obj.get(key) is not None), None)
        item = {'value': claims['split'], 'dataset_version': version,
                'manifest': record['manifest'], 'record_index': record['record_index']}
        if item not in split_claims:
            split_claims.append(item)
        if isinstance(version, str) and version.casefold() == 'v4' and claims['split'] not in v4_splits:
            v4_splits.append(claims['split'])
    result['split_claims'] = split_claims
    result['v4_declared_splits'] = v4_splits
    return result


def claim_matches(evidence, args):
    exact_fields = {'background_view', 'background_placement_plane',
                    'background_target_presence', 'background_use',
                    'background_review_state'}
    for option, field in (('observed_task', 'observed_tasks'),):
        requested = getattr(args, option, None)
        if requested:
            def matches(value):
                if not isinstance(value, list):
                    return False
                return any(isinstance(item, str) and item.casefold() == requested.casefold()
                           if option == 'observed_task' else
                           isinstance(item, dict) and str(item.get('role', '')).casefold() == requested.casefold()
                           for item in value)
            if not any(matches(record['claims'].get(field)) for record in evidence):
                return False
    role, task = getattr(args, 'region_role', None), getattr(args, 'region_task', None)
    if role or task:
        def region_matches(region):
            if not isinstance(region, dict):
                return False
            tasks = region.get('task_candidates', [])
            return ((not role or str(region.get('role', '')).casefold() == role.casefold()) and
                    (not task or (isinstance(tasks, list) and any(isinstance(t, str) and t.casefold() == task.casefold() for t in tasks))))
        if not any(any(region_matches(r) for r in record['claims'].get('regions', []))
                   for record in evidence if isinstance(record['claims'].get('regions'), list)):
            return False
    patch_use = getattr(args, 'ground_patch_use', None)
    if patch_use and not any(
            isinstance(patch, dict) and str(patch.get('use_status', '')).casefold() == patch_use.casefold()
            for record in evidence for patch in (record['claims'].get('visible_ground_patches')
                                                 if isinstance(record['claims'].get('visible_ground_patches'), list) else [])):
        return False
    for option, fields in (('generation_method', ('generation_method',)),
                           ('source_group', ('source_group', 'source_id', 'episode_group', 'near_duplicate_group')),
                           ('observed_content', ('observed_content',)),
                           ('review_flag', ('review_flag',)),
                           ('background_view', ('background_view',)),
                           ('background_placement_plane', ('background_placement_plane',)),
                           ('background_target_presence', ('background_target_presence',)),
                           ('background_use', ('background_use',)),
                           ('background_review_state', ('background_review_state',)),
                           ('split', ('split',)),
                           ('evaluation_scope', ('evaluation_scope', 'conclusion_scope', 'test_kind'))):
        requested = getattr(args, option, None)
        if requested and not any((str(record['claims'][field]).casefold() == requested.casefold()
                                  if field in exact_fields else
                                  requested.casefold() in json.dumps(record['claims'][field], ensure_ascii=False).casefold())
                                 for record in evidence if option != 'split' or record.get('relation') != 'referenced_as'
                                 for field in fields if record['claims'].get(field) is not None):
            return False
    return True


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def inside(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    if not candidate.is_relative_to(ROOT) or candidate == ROOT:
        raise ValueError('path must be inside the workspace and below its root: ' + value)
    return candidate


def source_allowed(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    if path_key(relative(path)) in {path_key(p) for p in PROJECT_ASSET_DIRS}:
        return True
    if lifecycle_reason(relative(path)):
        return False
    return not any(part.lower() in SKIP_DIRS or part.startswith('.') for part in parts)


def walk(path: Path):
    errors = []
    def failure(exc):
        errors.append(str(exc))
    for directory, dirs, files in os.walk(path, followlinks=False, onerror=failure):
        d = Path(directory)
        dirs[:] = [x for x in dirs if x.lower() not in SKIP_DIRS and not x.startswith('.')
                   and not (d / x).is_symlink() and (d / x).resolve() != INDEX
                   and not lifecycle_reason(relative(d / x))]
        dirs.sort()
        yield d, files
    if errors:
        raise OSError('; '.join(errors[:10]))


def existing_roots(sources):
    """The first level is a physical inventory partition, not a task assertion."""
    seen = set()
    if tuple(sources) == DEFAULT_SOURCES:
        for value in ROOT_FILE_SOURCES:
            base = ROOT if value == '.' else ROOT / value
            if base.is_dir():
                seen.add(relative(base) + '/@root')
                yield base, True
        for value in PROJECT_ASSET_DIRS:
            base = ROOT / value
            if base.is_dir():
                seen.add(value)
                yield base
    for value in sources:
        base = inside(value)
        if not source_allowed(base):
            raise ValueError('excluded source: ' + value)
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.name.casefold()):
            if child.is_dir() and not child.is_symlink() and source_allowed(child) and child.resolve() != INDEX:
                # Export tasks contain many independent generation batches;
                # checkpoint each batch rather than one giant task partition.
                if relative(base) == 'datasets/exports':
                    for batch in sorted(child.iterdir(), key=lambda p: p.name.casefold()):
                        if batch.is_dir() and not batch.is_symlink() and source_allowed(batch):
                            key = relative(batch)
                            if key not in seen:
                                seen.add(key)
                                yield batch
                    key = relative(child) + '/@root'
                    if key not in seen:
                        seen.add(key)
                        yield child, True
                    continue
                key = relative(child)
                if key not in seen:
                    seen.add(key)
                    yield child
        # Source root files are indexed as a separate partition.
        key = relative(base) + '/@root'
        if key not in seen:
            seen.add(key)
            yield base, True


def image_size(path: Path):
    """Read inexpensive headers; failure is represented as null."""
    try:
        with io_path(path).open('rb') as f:
            head = f.read(32)
            if head[:8] == b'\x89PNG\r\n\x1a\n' and len(head) >= 24:
                return list(struct.unpack('>II', head[16:24]))
            if head[:6] in (b'GIF87a', b'GIF89a') and len(head) >= 10:
                return list(struct.unpack('<HH', head[6:10]))
            if head[:2] == b'BM' and len(head) >= 26:
                w, h = struct.unpack('<ii', head[18:26])
                return [abs(w), abs(h)]
            if head[:2] == b'\xff\xd8':
                f.seek(2)
                while True:
                    marker = f.read(1)
                    if not marker:
                        break
                    if marker != b'\xff':
                        continue
                    code = f.read(1)
                    while code == b'\xff':
                        code = f.read(1)
                    if not code or code in (b'\xd8', b'\xd9'):
                        continue
                    size = f.read(2)
                    if len(size) != 2:
                        break
                    length = int.from_bytes(size, 'big')
                    if length < 2:
                        break
                    if code[0] in {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
                                   0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}:
                        info = f.read(5)
                        if len(info) == 5:
                            h, w = struct.unpack('>HH', info[1:5])
                            return [w, h]
                        break
                    f.seek(length - 2, 1)
    except (OSError, ValueError, struct.error):
        pass
    if Image is not None:
        try:
            with Image.open(io_path(path)) as image:
                return list(image.size)
        except (OSError, ValueError):
            pass
    return None


def video_info(path: Path, ffprobe: str | None):
    if not ffprobe:
        return None
    try:
        proc = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                               '-show_entries', 'stream=width,height:format=duration',
                               '-of', 'json', str(path)], capture_output=True, text=True,
                              timeout=15, check=True)
        info = json.loads(proc.stdout)
        stream = (info.get('streams') or [{}])[0]
        duration = info.get('format', {}).get('duration')
        return {'size': [stream.get('width'), stream.get('height')]
                if stream.get('width') and stream.get('height') else None,
                'duration_seconds': float(duration) if duration is not None else None}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def task_config_source(path: Path):
    """Recognize only the eight task-owned source declarations."""
    parts = path.relative_to(ROOT).parts
    return (len(parts) == 5 and parts[:2] == ('datasets', 'scripts') and
            parts[2] in TASK_CONFIG_NAMES and parts[3:] == ('config', 'source.yaml'))


def task_config_documents(source_path: Path):
    """Read current declarations, without opening referenced media or generators."""
    import yaml
    config_dir = source_path.parent.resolve()
    task = source_path.parts[-3]
    dataset_path = config_dir / 'dataset.yaml'
    for path in (source_path, dataset_path):
        if path.resolve().parent != config_dir or not path.is_file():
            raise ValueError('task configuration must be a present task-local file: ' + relative(path))
        if io_path(path).stat().st_size > 64 * 1024 * 1024:
            raise ValueError('task configuration exceeds 64 MiB: ' + relative(path))
    try:
        source = yaml.safe_load(io_path(source_path).read_text(encoding='utf-8-sig'))
        dataset = yaml.safe_load(io_path(dataset_path).read_text(encoding='utf-8-sig'))
    except yaml.YAMLError as exc:
        raise ValueError('invalid task YAML declaration: ' + str(exc)) from exc
    if not isinstance(source, dict) or not isinstance(dataset, dict):
        raise ValueError('task YAML declarations must be mappings')
    if (dataset.get('task') != task or source.get('task') != task or
            dataset.get('source_config') != 'config/source.yaml'):
        raise ValueError('task YAML files must identify their own task and source configuration')
    paths = [source_path, dataset_path]
    plans = []
    for key in TASK_PLAN_KEYS:
        value = source.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError('task plan reference must be a path: ' + key)
        candidate = (ROOT / value.replace('\\', '/')).resolve()
        if candidate.parent != config_dir or candidate.suffix.lower() != '.json':
            raise ValueError('task plan must be a JSON file in its own config directory: ' + value)
        paths.append(candidate)
        plans.append({'plan_key': key, 'path': relative(candidate), 'present': candidate.is_file()})
    declared_files = []
    for path in dict.fromkeys(paths):
        present = path.is_file()
        stat = io_path(path).stat() if present else None
        declared_files.append({'path': relative(path), 'present': present,
                               'size': stat.st_size if stat else None,
                               'mtime_ns': stat.st_mtime_ns if stat else None})
    return {'schema_version': 1, 'task': task, 'source_config': source,
            'dataset_config': dataset, 'declared_files': declared_files, 'plans': plans}


def task_declared_paths(value, pointer=''):
    """Yield explicit workspace references and JSON pointers, not observations."""
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace('~', '~0').replace('/', '~1')
            yield from task_declared_paths(child, pointer + '/' + escaped)
    elif isinstance(value, list):
        for position, child in enumerate(value):
            yield from task_declared_paths(child, pointer + '/' + str(position))
    elif isinstance(value, str):
        normalized = value.replace('\\', '/')
        if ':' in normalized or '#' in normalized or '\n' in normalized:
            return
        if Path(normalized).parts[:1] not in [('datasets',), ('Archive',), ('lab',), ('runs',), ('tmp',), ('refs',)]:
            return
        candidate = (ROOT / normalized).resolve()
        if candidate.is_relative_to(ROOT):
            name = relative(candidate)
            if not lifecycle_reason(name):
                yield pointer, name


def task_config_evidence(source_path: Path):
    """Index plan declarations separately from observed media and label evidence."""
    snapshot = task_config_documents(source_path)
    config_name = relative(source_path)
    records = [(config_name, {'manifest': config_name, 'record_index': 0,
                              'relation': 'task_config', 'claims': {'task_config': snapshot}})]
    for plan_ref in snapshot['plans']:
        if not plan_ref['present']:
            continue
        path = inside(plan_ref['path'])
        if io_path(path).stat().st_size > 64 * 1024 * 1024:
            raise ValueError('task plan exceeds 64 MiB: ' + plan_ref['path'])
        plan = json.loads(io_path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(plan, dict) or not isinstance(plan.get('strata'), list):
            raise ValueError('referenced task plan must contain strata: ' + plan_ref['path'])
        plan_ref.update({key: plan.get(key) for key in ('budget', 'target_split', 'model_key')})
        for index, state in enumerate(plan['strata']):
            if not isinstance(state, dict):
                raise ValueError('task plan stratum must be a mapping')
            declaration = {'task': snapshot['task'], 'plan_key': plan_ref['plan_key'],
                           'plan_path': plan_ref['path'], 'target_split': plan.get('target_split'),
                           'budget': plan.get('budget'), 'stratum': state}
            record = {'manifest': plan_ref['path'], 'record_index': index,
                      'relation': 'plan_state', 'claims': {'plan_state': declaration}}
            records.append((plan_ref['path'], record))
            # Only explicit workspace source paths become reverse references.
            # A declaration is neither an observation nor an availability claim.
            for pointer, name in task_declared_paths(state):
                records.append((name, {**record, 'relation': 'plan_input_reference',
                                       'reference_field': pointer}))
    return records


def manifest_rows(path: Path):
    # Existing records vary in schema. Only declared matching paths are linked.
    if io_path(path).stat().st_size > 64 * 1024 * 1024:
        return
    if path.suffix.lower() == '.csv':
        with io_path(path).open(encoding='utf-8-sig', newline='') as f:
            yield from ((i, r, {}) for i, r in enumerate(csv.DictReader(f)))
    elif path.suffix.lower() == '.jsonl':
        with io_path(path).open(encoding='utf-8-sig') as f:
            for index, line in enumerate(f):
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield index, row, {}
    else:
        with io_path(path).open(encoding='utf-8-sig') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                # Some historical package snapshots retain JSONL content in a
                # .json file. Accept only if every nonempty line is valid JSON.
                f.seek(0)
                lines = [line for line in f if line.strip()]
                try:
                    parsed = [json.loads(line) for line in lines]
                except json.JSONDecodeError:
                    raise exc
                yield from ((i, row, {}) for i, row in enumerate(parsed) if isinstance(row, dict))
                return
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = next((data[k] for k in ('records', 'items', 'samples', 'files', 'sources')
                         if isinstance(data.get(k), list)), [])
        else:
            rows = []
        context = {key: data[key] for key in ('scope', 'batch_id', 'model_key', 'review_date',
                                              'dataset_version', 'data_version', 'task_version')
                   if isinstance(data, dict) and key in data}
        batch = {key: data[key] for key in BATCH_CONTEXT_KEYS if isinstance(data, dict) and key in data}
        if batch:
            yield None, None, batch
        yield from ((i, r, context) for i, r in enumerate(rows) if isinstance(r, dict))


def path_candidates(value: str, manifest: Path, dependencies=None):
    """Accept only exact workspace or manifest-relative paths, never basename joins."""
    try:
        value_path = Path(value.replace('\\', '/'))
        if value_path.is_absolute():
            candidates = [Path(os.path.abspath(value_path))]
        else:
            workspace = Path(os.path.abspath(ROOT / value_path))
            local = Path(os.path.abspath(manifest.parent / value_path))
            anchored = value_path.parts and value_path.parts[0].lower() in {
                'datasets', 'training_packages', 'archive', 'lab', 'runs', 'tmp', 'refs'}
            if anchored and workspace.is_relative_to(ROOT):
                yield relative(workspace)
                return
            candidates = [workspace, local] if anchored else [local, workspace]
        candidates = [p for p in candidates if p.is_relative_to(ROOT)]
        if dependencies is not None and len(candidates) > 1:
            for candidate in candidates:
                state = candidate.is_file()
                key = relative(candidate)
                if key in dependencies and dependencies[key] != state:
                    dependencies[key] = None
                else:
                    dependencies[key] = state
        existing = [p for p in candidates if p.is_file()]
        # One exact resolved path, preferring the declared convention. Missing
        # paths remain searchable as claims but are never treated as observed.
        if existing or candidates:
            yield relative((existing or candidates)[0])
    except (ValueError, OSError):
        return


def manifest_evidence(directory: Path, root_only: bool, dependencies=None):
    evidence = defaultdict(list)
    warnings = []
    dirs = [(directory, [p.name for p in directory.iterdir() if p.is_file()])] if root_only else walk(directory)
    for folder, files in dirs:
        for name in files:
            low = name.lower()
            candidate = folder / name
            if task_config_source(candidate):
                try:
                    for claimed_path, record in task_config_evidence(candidate):
                        evidence[claimed_path].append(record)
                except (OSError, ValueError, UnicodeError) as exc:
                    warnings.append({'manifest': relative(candidate), 'error': str(exc)})
                continue
            if low not in MANIFEST_NAMES and not (Path(low).suffix in {'.json', '.jsonl', '.csv'} and
                                                   any(hint in low for hint in MANIFEST_HINTS)):
                continue
            manifest = folder / name
            try:
                if io_path(manifest).stat().st_size > 64 * 1024 * 1024:
                    warnings.append({'manifest': relative(manifest), 'error': 'manifest exceeds 64 MiB; not parsed'})
                    continue
                for index, row, context in manifest_rows(manifest):
                    if row is None:
                        evidence[relative(manifest)].append({'manifest': relative(manifest),
                                                             'record_index': None,
                                                             'relation': 'batch_context',
                                                             'batch_context': context, 'claims': {}})
                        continue
                    claims = {key: row[key] for key in EVIDENCE_KEYS if row.get(key) is not None}
                    if not claims:
                        continue
                    record = {'manifest': relative(manifest), 'record_index': index,
                              'manifest_context': context, 'relation': 'describes', 'claims': claims}
                    row_seen = set()
                    subjects = [row[k] for k in PATH_KEYS if k != 'source' and isinstance(row.get(k), str) and
                                Path(row[k]).suffix.lower() in IMAGES | VIDEOS]
                    for key in PATH_KEYS:
                        value = row.get(key)
                        if key == 'source' and subjects:
                            continue
                        if isinstance(value, str) and (Path(value).suffix.lower() in IMAGES | VIDEOS or
                                (path_key(relative(manifest.parent)) in {path_key(p) for p in PROJECT_ASSET_DIRS} and Path(value).suffix.lower() == '.npz')):
                            for path in path_candidates(value, manifest, dependencies):
                                identity = (path, 'describes')
                                if not lifecycle_reason(path) and identity not in row_seen:
                                    evidence[path].append(record)
                                    row_seen.add(identity)
                    for key in REFERENCE_KEYS:
                        value = row.get(key)
                        if key == 'source' and not subjects:
                            continue
                        for item in value if isinstance(value, list) else [value]:
                            if not isinstance(item, str):
                                continue
                            declared = item.removeprefix('parent:')
                            if Path(declared).suffix.lower() not in IMAGES | VIDEOS:
                                continue
                            reference = {**record, 'relation': 'referenced_as',
                                         'reference_field': key, 'subject_paths': subjects}
                            for path in path_candidates(declared, manifest, dependencies):
                                identity = (path, 'referenced_as', key)
                                if not lifecycle_reason(path) and identity not in row_seen:
                                    evidence[path].append(reference)
                                    row_seen.add(identity)
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                warnings.append({'manifest': relative(manifest), 'error': str(exc)})
    return evidence, warnings


@contextmanager
def catalog_guard(timeout=60):
    """OS-held catalog lease; it is released automatically if a process exits."""
    INDEX.mkdir(parents=True, exist_ok=True)
    with (INDEX / '.catalog.lock').open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                lock.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('catalog access is busy')
                time.sleep(.05)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, obj):
    def publish():
        temp = path.with_name(path.name + '.tmp')
        with temp.open('w', encoding='utf-8', newline='\n') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(temp, path)
    if path == INDEX / 'catalog.json':
        # A broad evidence query can take over a minute; wait for its snapshot
        # rather than mark a finished partition failed while the reader drains.
        with catalog_guard(timeout=300):
            publish()
    else:
        publish()


def build_partition(directory: Path, root_only: bool, ffprobe, relocation, dependencies=None):
    name = relative(directory) + ('/@root' if root_only else '')
    version = uuid.uuid4().hex
    shard = INDEX / 'shards' / (quote(name, safe='') + '.' + version + '.jsonl')
    temp = shard.with_name(shard.name + '.tmp')
    evidence_shard = INDEX / 'shards' / (quote(name, safe='') + '.' + version + '.evidence.jsonl')
    evidence_temp = evidence_shard.with_name(evidence_shard.name + '.tmp')
    counts = Counter()
    evidence, warnings = manifest_evidence(directory, root_only, dependencies)
    dirs = [(directory, [p.name for p in directory.iterdir() if p.is_file()])] if root_only else walk(directory)
    try:
        # Evidence is a separate reverse index: manifests may refer to media in
        # another partition. A claim does not establish that the media exists.
        with evidence_temp.open('w', encoding='utf8', newline='\n') as f:
            for claimed_path, claims in evidence.items():
                # A video can be the parent of thousands of frames. Emit one
                # claim per line so no single JSON record grows with the batch.
                for claim in claims:
                    f.write(json.dumps({'claimed_path': claimed_path, 'evidence': [claim]},
                                       ensure_ascii=False, separators=(',', ':')) + '\n')
        with temp.open('w', encoding='utf-8', newline='\n') as f:
            for folder, files in dirs:
                for filename in sorted(files):
                    path = folder / filename
                    suffix = path.suffix.lower()
                    array_asset = path_key(relative(directory)) in {path_key(p) for p in PROJECT_ASSET_DIRS} and suffix == '.npz'
                    if (suffix not in IMAGES | VIDEOS and not array_asset) or path.is_symlink():
                        continue
                    try:
                        stat = io_path(path).stat()
                    except OSError as exc:
                        warnings.append({'path': relative(path), 'error': str(exc)})
                        continue
                    key = relative(path)
                    history = []
                    identity = key
                    # Each relationship is an explicitly supplied directory move.
                    for old, new in reversed(relocation):
                        if path_key(identity) == path_key(new) or path_key(identity).startswith(path_key(new) + '/'):
                            identity = old + identity[len(new):]
                            history.append({'path': identity, 'basis': 'explicit_relocation'})
                    record_evidence = evidence.get(key, [])
                    item = {'id': 'workspace:' + identity, 'path': key,
                            'media_type': 'array_asset' if array_asset else ('image' if suffix in IMAGES else 'video'),
                            'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns,
                            'image_size': None, 'duration_seconds': None,
                            'path_history': history, 'recorded_claims': structured_claims(record_evidence),
                            'evidence': record_evidence}
                    if array_asset:
                        pass
                    elif suffix in IMAGES:
                        item['image_size'] = image_size(path)
                    else:
                        info = video_info(path, ffprobe)
                        if info:
                            item['image_size'] = info['size']
                            item['duration_seconds'] = info['duration_seconds']
                    counts[item['media_type']] += 1
                    if item['evidence']:
                        counts['with_manifest_evidence'] += 1
                    f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(evidence_temp, evidence_shard)
        os.replace(temp, shard)
    except BaseException:
        if temp.exists():
            temp.unlink()
        if evidence_temp.exists():
            evidence_temp.unlink()
        if shard.exists():
            shard.unlink()
        if evidence_shard.exists():
            evidence_shard.unlink()
        raise
    return {'source': name, 'source_present': True, 'shard': relative(shard), 'counts': dict(counts),
            'evidence_shard': relative(evidence_shard), 'evidence_claimed_paths': len(evidence),
            'warnings': warnings, 'updated_utc': datetime.now(timezone.utc).isoformat()}


def filter_old_parent_partitions(catalog):
    """Remove excluded media from old broad partitions without rescanning sources."""
    prefixes = [path for path, _ in _DYNAMIC_LIFECYCLE]
    for name, partition in catalog['partitions'].items():
        if name.endswith('/@root') or not any(p.startswith(name + '/') for p in prefixes):
            continue
        old_shard = inside(partition['shard'])
        old_evidence = inside(partition['evidence_shard']) if partition.get('evidence_shard') else None
        version = uuid.uuid4().hex
        new_shard = INDEX / 'shards' / (quote(name, safe='') + '.' + version + '.jsonl')
        new_evidence = INDEX / 'shards' / (quote(name, safe='') + '.' + version + '.evidence.jsonl')
        counts = Counter()
        removed = 0
        try:
            with old_shard.open(encoding='utf8') as source, new_shard.open('w', encoding='utf8', newline='\n') as output:
                for line in source:
                    item = json.loads(line)
                    if lifecycle_reason(item['path']):
                        removed += 1
                        continue
                    original = item.get('evidence', [])
                    retained = [row for row in original if not lifecycle_reason(row['manifest'])]
                    if len(retained) != len(original):
                        item['evidence'] = retained
                        item['recorded_claims'] = structured_claims(retained)
                        line = json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n'
                    counts[item['media_type']] += 1
                    if item.get('evidence'):
                        counts['with_manifest_evidence'] += 1
                    output.write(line)
            evidence_paths = set()
            with new_evidence.open('w', encoding='utf8', newline='\n') as output:
                if old_evidence:
                    with old_evidence.open(encoding='utf8') as source:
                        for line in source:
                            entry = json.loads(line)
                            if lifecycle_reason(entry['claimed_path']):
                                continue
                            original = entry.get('evidence', [])
                            retained = [row for row in original if not lifecycle_reason(row['manifest'])]
                            if not retained:
                                continue
                            if len(retained) != len(original):
                                entry['evidence'] = retained
                                line = json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n'
                            output.write(line)
                            evidence_paths.add(entry['claimed_path'])
        except BaseException:
            new_shard.unlink(missing_ok=True)
            new_evidence.unlink(missing_ok=True)
            raise
        partition['shard'] = relative(new_shard)
        partition['evidence_shard'] = relative(new_evidence)
        partition['evidence_claimed_paths'] = len(evidence_paths)
        partition['counts'] = dict(counts)
        partition['lifecycle_filtered_records'] = removed


def remove_unreferenced_shards(catalog):
    """Discard only index-generated files no longer named by the active catalog."""
    referenced = {value for p in catalog['partitions'].values()
                  for value in (p.get('shard'), p.get('evidence_shard'), p.get('input_shard')) if value}
    warnings = []
    for path in (INDEX / 'shards').glob('*.jsonl'):
        if relative(path) not in referenced:
            try:
                path.unlink()
            except OSError as exc:
                warnings.append({'shard': relative(path), 'error': str(exc)})
    return warnings


def input_lines(directory, root_only, ffprobe, relocation):
    """Stable file observations; unchanged checks do not read media contents."""
    tool_stat = Path(__file__).stat()
    probe_stat = Path(ffprobe).stat() if ffprobe else None
    header = {'schema': 1, 'tool': [tool_stat.st_size, tool_stat.st_mtime_ns],
              'python': list(sys.version_info[:3]), 'image_library': getattr(Image, '__version__', None),
              'ffprobe': [ffprobe, probe_stat.st_size, probe_stat.st_mtime_ns] if probe_stat else None,
              'lifecycle': _POLICY_METADATA, 'relocations': relocation, 'root_only': root_only}
    yield json.dumps(['header', header], ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
    dirs = [(directory, [p.name for p in directory.iterdir() if p.is_file()])] if root_only else walk(directory)
    for folder, files in dirs:
        for name in sorted(files):
            path = folder / name
            if task_config_source(path):
                for declared in task_config_documents(path)['declared_files']:
                    yield json.dumps(['task_config_file', declared], ensure_ascii=False,
                                     sort_keys=True, separators=(',', ':')) + '\n'
                continue
            low = name.lower()
            suffix = path.suffix.lower()
            manifest = low in MANIFEST_NAMES or (suffix in {'.json', '.jsonl', '.csv'} and
                                                 any(hint in low for hint in MANIFEST_HINTS))
            asset = path_key(relative(directory)) in {path_key(p) for p in PROJECT_ASSET_DIRS} and suffix == '.npz'
            if not manifest and suffix not in IMAGES | VIDEOS and not asset:
                continue
            if path.is_symlink() and not manifest:
                continue
            stat = io_path(path).stat()
            yield json.dumps(['file', relative(path), stat.st_size, stat.st_mtime_ns], ensure_ascii=False, separators=(',', ':')) + '\n'


def input_matches(snapshot, directory, root_only, ffprobe, relocation):
    try:
        with snapshot.open(encoding='utf8') as old:
            for line in input_lines(directory, root_only, ffprobe, relocation):
                if old.readline() != line:
                    return False
            for line in old:
                item = json.loads(line)
                if item[0] != 'dependency' or (ROOT / item[1]).is_file() != item[2]:
                    return False
        return True
    except (OSError, ValueError, IndexError, json.JSONDecodeError):
        return False


def build_changed_partition(directory, root_only, ffprobe, relocation):
    temp = INDEX / 'shards' / (uuid.uuid4().hex + '.inputs.jsonl.tmp')
    result = None
    try:
        with temp.open('w', encoding='utf8', newline='\n') as f:
            f.writelines(input_lines(directory, root_only, ffprobe, relocation))
        dependencies = {}
        result = build_partition(directory, root_only, ffprobe, relocation, dependencies)
        if not input_matches(temp, directory, root_only, ffprobe, relocation):
            raise OSError('partition inputs changed while indexing: ' + relative(directory))
        with temp.open('a', encoding='utf8', newline='\n') as f:
            for path, present in sorted(dependencies.items()):
                if present is None or (ROOT / path).is_file() != present:
                    raise OSError('manifest path changed while indexing: ' + path)
                f.write(json.dumps(['dependency', path, present], ensure_ascii=False, separators=(',', ':')) + '\n')
        snapshot = Path(str(inside(result['shard'])) + '.inputs.jsonl')
        os.replace(temp, snapshot)
        result['input_shard'] = relative(snapshot)
        return result
    except BaseException:
        if result:
            inside(result['shard']).unlink(missing_ok=True)
            inside(result['evidence_shard']).unlink(missing_ok=True)
        raise
    finally:
        temp.unlink(missing_ok=True)


def build_locked(args):
    INDEX.joinpath('shards').mkdir(parents=True, exist_ok=True)
    catalog_path = INDEX / 'catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf8')) if catalog_path.exists() else {
        'schema_version': 1, 'scope': 'local filesystem discovery; manifest claims are not verified',
        'partitions': {}}
    if catalog.get('schema_version') != 1 or not isinstance(catalog.get('partitions'), dict):
        raise ValueError('unsupported or invalid existing catalog')
    previous_scope = catalog.get('latest_build_scope', {})
    previous_policy = previous_scope.get('lifecycle_policy_revision')
    previous_relocations = [tuple(x) for x in catalog.get('relocations', [])]
    previous_relocation_sources = list(catalog.get('relocation_sources', []))
    requested_roots = args.scan_root or DEFAULT_SOURCES
    catalog['latest_build_scope'] = {
        'roots': list(requested_roots), 'selected_sources': list(args.source),
        'root_file_sources': list(ROOT_FILE_SOURCES) if tuple(requested_roots) == DEFAULT_SOURCES else [],
        'project_asset_roots': list(PROJECT_ASSET_DIRS) if tuple(requested_roots) == DEFAULT_SOURCES else [],
        'lifecycle_policy': LIFECYCLE_POLICY,
        'lifecycle_policy_revision': dict(_POLICY_METADATA),
        'lifecycle_exclusions': [r for r in _LIFECYCLE_RECORDS if lifecycle_reason(r['path'])] + list(_LIFECYCLE_PATTERNS),
        'lifecycle_warnings': list(_LIFECYCLE_WARNINGS),
        'excluded_directory_names': sorted(SKIP_DIRS),
        'excluded_hidden_directories': True, 'follow_symlinks': False,
        'archives_expanded': False, 'manifest_names': sorted(MANIFEST_NAMES),
        'manifest_filename_hints': list(MANIFEST_HINTS),
        'manifest_size_limit_bytes': 64 * 1024 * 1024,
        'video_probe': bool(shutil.which('ffprobe') and not args.no_video_probe),
        'missing_roots': [value for value in requested_roots if not inside(value).is_dir()],
        'state': 'running', 'failures': [],
    }
    relocation = [tuple(x) for x in catalog.get('relocations', [])]
    requested_relocations = list(args.relocation)
    for file_name in args.relocation_file:
        mapping = inside(file_name)
        data = json.loads(mapping.read_text(encoding='utf8'))
        if not isinstance(data, dict) or not isinstance(data.get('moves'), list):
            raise ValueError('invalid relocation file: ' + file_name)
        for row in data['moves']:
            if not isinstance(row, dict) or not isinstance(row.get('original_path'), str) or not isinstance(row.get('current_path'), str):
                raise ValueError('invalid move in relocation file: ' + file_name)
            requested_relocations.append(row['original_path'] + '=' + row['current_path'])
        if file_name not in catalog.setdefault('relocation_sources', []):
            catalog['relocation_sources'].append(file_name)
    for value in requested_relocations:
        old, sep, new = value.partition('=')
        if not sep:
            raise ValueError('relocation requires OLD=NEW')
        old_rel, new_rel = relative(inside(old)), relative(inside(new))
        pair = (old_rel, new_rel)
        if path_key(old_rel) == path_key(new_rel) or any(path_key(old_rel) == path_key(o) and path_key(new_rel) != path_key(n) or
                                     path_key(new_rel) == path_key(n) and path_key(old_rel) != path_key(o)
                                     for o, n in relocation):
            raise ValueError('ambiguous or identical relocation: ' + value)
        if pair not in relocation:
            relocation.append(pair)
    catalog['relocations'] = relocation
    ffprobe = shutil.which('ffprobe') if not args.no_video_probe else None
    # A completed, identical full build is a read-only operation, including
    # the catalog. The SQLite cache can then retain its exact snapshot too.
    if (not args.rebuild and not args.source and not args.scan_root and
            previous_scope.get('state') == 'complete' and not previous_scope.get('failures') and
            catalog.get('last_full_build', {}).get('state') == 'complete' and
            not catalog.get('last_full_build', {}).get('failures') and
            policy_identity(catalog.get('last_full_build', {}).get('lifecycle_policy_revision')) == policy_identity(_POLICY_METADATA) and
            policy_identity(previous_policy) == policy_identity(_POLICY_METADATA) and not _LIFECYCLE_WARNINGS and
            previous_scope.get('roots') == list(DEFAULT_SOURCES) and
            not previous_scope.get('selected_sources') and
            previous_scope.get('video_probe') == bool(ffprobe) and
            previous_relocations == relocation and
            previous_relocation_sources == catalog.get('relocation_sources', [])):
        entries = list(existing_roots(requested_roots))
        named = [(x[0], x[1]) if isinstance(x, tuple) else (x, False) for x in entries]
        names = {relative(d) + ('/@root' if root_only else '') for d, root_only in named}
        parts = catalog['partitions']
        if names == set(parts) and all(
                parts[name].get('source_present', True) and not parts[name].get('warnings') and
                parts[name].get('input_shard') and inside(parts[name]['shard']).is_file() and
                inside(parts[name]['evidence_shard']).is_file() and
                input_matches(inside(parts[name]['input_shard']), d, root_only, ffprobe, relocation)
                for d, root_only in named for name in [relative(d) + ('/@root' if root_only else '')]):
            print(json.dumps({'state': 'complete', 'partitions': len(names), 'unchanged': True}), flush=True)
            return 0
    if ((args.source or args.scan_root) and not args.rebuild and
            previous_scope.get('state') == 'complete' and not previous_scope.get('failures') and
            policy_identity(previous_policy) == policy_identity(_POLICY_METADATA) and not _LIFECYCLE_WARNINGS and
            previous_relocations == relocation and
            previous_relocation_sources == catalog.get('relocation_sources', [])):
        selected = [relative(inside(x)) for x in args.source]
        def matches(name, source):
            name, source = path_key(name), path_key(source)
            return name == source or name.startswith(source + '/') or source.startswith(name + '/')
        entries = list(existing_roots(requested_roots))
        named = [(x[0], x[1]) if isinstance(x, tuple) else (x, False) for x in entries]
        named = [(d, only) for d, only in named
                 if not selected or any(matches(relative(d), s) for s in selected)]
        matched = {s for s in selected for d, _ in named if matches(relative(d), s)}
        names = {relative(d) + ('/@root' if only else '') for d, only in named}
        def covered(name):
            source = name.removesuffix('/@root')
            key = path_key(source)
            in_roots = any(key == path_key(root) or key.startswith(path_key(root) + '/')
                           for root in requested_roots)
            if tuple(requested_roots) == DEFAULT_SOURCES and source in ROOT_FILE_SOURCES:
                in_roots = True
            return in_roots and (not selected or any(matches(source, s) for s in selected))
        parts = catalog['partitions']
        known_names = {name for name in parts if covered(name)}
        if names and matched == set(selected) and names == known_names and all(
                parts[name].get('source_present', True) and not parts[name].get('warnings') and
                parts[name].get('input_shard') and inside(parts[name]['shard']).is_file() and
                inside(parts[name]['evidence_shard']).is_file() and
                input_matches(inside(parts[name]['input_shard']), d, only, ffprobe, relocation)
                for d, only in named for name in [relative(d) + ('/@root' if only else '')]):
            print(json.dumps({'state': previous_scope['state'], 'partitions': len(names),
                              'unchanged': True, 'scope': 'selected'}), flush=True)
            return 0
    # Preserve a missing partition for historical queries; do not present its
    # contents as current files. No automatic deletion or identity matching.
    # Active catalog never offers retired dataset snapshots as queryable media.
    catalog.pop('retired_partitions', None)
    for name in list(catalog['partitions']):
        reason = lifecycle_reason(name)
        if reason:
            del catalog['partitions'][name]
    for name, partition in catalog['partitions'].items():
        source = name.removesuffix('/@root')
        partition['source_present'] = (ROOT / source).is_dir()
    if policy_identity(previous_policy) != policy_identity(_POLICY_METADATA):
        filter_old_parent_partitions(catalog)
    atomic_json(catalog_path, catalog)
    cleanup_warnings = [] if args.no_clean else remove_unreferenced_shards(catalog)
    if cleanup_warnings:
        catalog['latest_build_scope']['shard_cleanup_warnings'] = cleanup_warnings
        atomic_json(catalog_path, catalog)
    selected = [relative(inside(x)) for x in args.source]
    failures = list(_LIFECYCLE_WARNINGS)
    matched = set()
    for entry in existing_roots(requested_roots):
        directory, root_only = (entry if isinstance(entry, tuple) else (entry, False))
        name = relative(directory)
        matching = [s for s in selected if path_key(name) == path_key(s) or
                    path_key(name).startswith(path_key(s) + '/') or path_key(s).startswith(path_key(name) + '/')]
        if selected and not matching:
            continue
        matched.update(matching)
        try:
            partition_name = name + ('/@root' if root_only else '')
            current = catalog['partitions'].get(partition_name, {})
            snapshot = current.get('input_shard')
            if (not args.rebuild and snapshot and current.get('source_present', True)
                    and not current.get('warnings') and inside(current['shard']).is_file()
                    and inside(current['evidence_shard']).is_file()
                    and input_matches(inside(snapshot), directory, root_only, ffprobe, relocation)):
                print(json.dumps({'source': partition_name, 'unchanged': True}, ensure_ascii=False), flush=True)
                continue
            result = build_changed_partition(directory, root_only, ffprobe, relocation)
            catalog['partitions'][result['source']] = result
            catalog['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(catalog_path, catalog)
            print(json.dumps({'source': result['source'], 'counts': result['counts'],
                              'warnings': len(result['warnings'])}, ensure_ascii=False), flush=True)
        except (OSError, ValueError) as exc:
            failures.append({'source': name, 'error': str(exc)})
            print(json.dumps(failures[-1], ensure_ascii=False), file=sys.stderr, flush=True)
    for source in set(selected) - matched:
        failures.append({'source': source, 'error': 'no matching existing source partition'})
    catalog['latest_build_scope']['state'] = 'partial' if failures else 'complete'
    catalog['latest_build_scope']['failures'] = failures
    is_full = not args.source and not args.scan_root
    if is_full:
        catalog['last_full_build'] = dict(catalog['latest_build_scope'])
    atomic_json(catalog_path, catalog)
    if failures:
        print(json.dumps({'failures': failures}, ensure_ascii=False), file=sys.stderr)
        return 1
    if is_full:
        retire_overlaps(clean_shards=not args.no_clean)
    return 0


def with_build_lock(operation):
    INDEX.mkdir(parents=True, exist_ok=True)
    lock = INDEX / '.build.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError('another builder may be active; inspect ' + relative(lock)) from exc
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump({'pid': os.getpid(), 'host': socket.gethostname(),
                       'started_utc': datetime.now(timezone.utc).isoformat()}, f)
        return operation()
    finally:
        lock.unlink(missing_ok=True)


def build(args):
    return with_build_lock(lambda: build_locked(args))


def retire_overlaps(*, clean_shards=True):
    """Retire old export task partitions only after all current batches exist."""
    catalog_path = INDEX / 'catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf8'))
    partitions = catalog['partitions']
    retired = []
    changed = False
    for name, partition in partitions.items():
        present = (ROOT / name.removesuffix('/@root')).is_dir()
        if partition.get('source_present', True) != present:
            partition['source_present'] = present
            changed = True
    for name, old in list(partitions.items()):
        parts = Path(name).parts
        if len(parts) != 3 or parts[:2] != ('datasets', 'exports') or name.endswith('/@root'):
            continue
        task = ROOT / name
        if not task.is_dir():
            continue
        required = [name + '/@root']
        required.extend(relative(p) for p in task.iterdir()
                        if p.is_dir() and not p.is_symlink() and source_allowed(p))
        if all(p in partitions and partitions[p].get('source_present', True) and
               partitions[p].get('updated_utc', '') >= old.get('updated_utc', '')
               for p in required):
            retired.append(name)
    for name in retired:
        del partitions[name]
    if retired or changed:
        atomic_json(catalog_path, catalog)
    cleanup_warnings = remove_unreferenced_shards(catalog) if clean_shards else []
    print(json.dumps({'retired_partitions': retired,
                      'shard_cleanup_warnings': cleanup_warnings}, ensure_ascii=False))
    return 0


def process_exists(pid: int):
    """Return None when existence cannot be established; never terminate a PID."""
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        kernel.GetExitCodeProcess.restype = ctypes.c_int
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            exit_code = ctypes.c_uint32()
            checked = kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel.CloseHandle(handle)
            return (exit_code.value == 259) if checked else None
        return False if ctypes.get_last_error() == 87 else None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return None


def unlock():
    lock = INDEX / '.build.lock'
    raw = lock.read_text(encoding='utf8')
    record = json.loads(raw)
    pid = record.get('pid')
    if record.get('host') != socket.gethostname() or not isinstance(pid, int) or pid <= 0:
        raise ValueError('lock host/PID cannot be verified; leave it in place')
    state = process_exists(pid)
    if state is not False:
        raise ValueError('lock owner is alive or cannot be checked; leave it in place')
    if lock.read_text(encoding='utf8') != raw:
        raise ValueError('lock changed during verification; leave it in place')
    lock.unlink()
    print(json.dumps({'unlocked': relative(lock), 'previous_pid': pid}, ensure_ascii=False))
    return 0


def query_locked(args):
    catalog = json.loads((INDEX / 'catalog.json').read_text(encoding='utf8'))
    active = list(catalog['partitions'].values())
    partitions = sorted(active, key=lambda p: p.get('updated_utc', ''), reverse=True)
    join = bool(args.with_evidence or args.generation_method or args.source_group or
                any(getattr(args, field, None) for field in (
                    'observed_content', 'review_flag', 'background_view',
                    'background_placement_plane', 'background_target_presence',
                    'background_use', 'background_review_state', 'observed_task', 'region_role', 'region_task', 'ground_patch_use')) or
                args.split or args.evaluation_scope)
    matching = defaultdict(list)
    matching_keys = defaultdict(set)
    if join:
        for partition in active:
            if not args.include_missing and not partition.get('source_present', True):
                continue
            evidence_path = partition.get('evidence_shard')
            if not evidence_path:
                continue
            with inside(evidence_path).open(encoding='utf8') as f:
                for line in f:
                    entry = json.loads(line)
                    entry['evidence'] = [r for r in entry['evidence'] if r.get('relation') != 'batch_context']
                    if not entry['evidence']:
                        continue
                    if claim_matches(entry['evidence'], args):
                        for record in entry['evidence']:
                            signature = (record['manifest'], record['record_index'],
                                         record.get('relation'), record.get('reference_field'))
                            claimed_key = path_key(entry['claimed_path'])
                            if signature not in matching_keys[claimed_key]:
                                matching[claimed_key].append(record)
                                matching_keys[claimed_key].add(signature)
    matches = 0
    seen = set()
    for partition in partitions:
        if lifecycle_reason(partition['source']):
            continue
        if not args.include_missing and not partition.get('source_present', True):
            continue
        if args.partition and args.partition.casefold() not in partition['source'].casefold():
            continue
        with inside(partition['shard']).open(encoding='utf8') as f:
            for line in f:
                item = json.loads(line)
                item['evidence'] = [r for r in item['evidence'] if r.get('relation') != 'batch_context']
                if lifecycle_reason(item['path']):
                    continue
                key = path_key(item['path'])
                if key in seen:
                    continue
                seen.add(key)
                if args.kind and item['media_type'] != args.kind:
                    continue
                if join:
                    local = item.get('evidence', [])
                    joined = list(local)
                    signatures = {(r['manifest'], r['record_index'], r.get('relation'), r.get('reference_field')) for r in local}
                    for record in matching.get(key, []):
                        signature = (record['manifest'], record['record_index'], record.get('relation'), record.get('reference_field'))
                        if signature not in signatures:
                            joined.append(record)
                            signatures.add(signature)
                    item['evidence'] = joined
                    item['recorded_claims'] = structured_claims(item['evidence'])
                else:
                    # Recompute from original evidence so older index snapshots
                    # never present unscoped splits as current V4 membership.
                    item['recorded_claims'] = structured_claims(item.get('evidence', []))
                if args.with_evidence and not item['evidence']:
                    continue
                if not claim_matches(item['evidence'], args):
                    continue
                if args.contains and args.contains.casefold() not in json.dumps(item, ensure_ascii=False).casefold():
                    continue
                if not args.include_missing and not (ROOT / item['path']).is_file():
                    continue
                print(json.dumps(item, ensure_ascii=False))
                matches += 1
                if args.limit and matches >= args.limit:
                    return 0
    return 0


def query_batches_locked(args):
    catalog = json.loads((INDEX / 'catalog.json').read_text(encoding='utf8'))
    matches = 0
    seen = set()
    for partition in catalog['partitions'].values():
        if not args.include_missing and not partition.get('source_present', True):
            continue
        evidence_path = partition.get('evidence_shard')
        if not evidence_path:
            continue
        with inside(evidence_path).open(encoding='utf8') as f:
            for line in f:
                entry = json.loads(line)
                for record in entry['evidence']:
                    if record.get('relation') != 'batch_context':
                        continue
                    signature = (path_key(record['manifest']), record.get('record_index'), record['relation'])
                    if signature in seen:
                        continue
                    seen.add(signature)
                    context = record['batch_context']
                    if args.contains and args.contains.casefold() not in json.dumps(record, ensure_ascii=False).casefold():
                        continue
                    evaluation = context.get('evaluation')
                    evaluation = evaluation if isinstance(evaluation, dict) else {}
                    if args.evaluation_scope:
                        scopes = [context.get('evaluation_scope'), context.get('conclusion_scope'), evaluation.get('data_scope')]
                        if not any(args.evaluation_scope.casefold() in json.dumps(x, ensure_ascii=False).casefold()
                                   for x in scopes if x is not None):
                            continue
                    if args.status:
                        statuses = [context.get('status'), evaluation.get('status')]
                        if not any(args.status.casefold() in str(x).casefold() for x in statuses if x is not None):
                            continue
                    print(json.dumps(record, ensure_ascii=False))
                    matches += 1
                    if args.limit and matches >= args.limit:
                        return 0
    return 0


def query_evidence_locked(args):
    path = relative(inside(args.path)) if args.path else None
    catalog = json.loads((INDEX / 'catalog.json').read_text(encoding='utf8'))
    matches = 0
    seen = set()
    partitions = list(catalog['partitions'].values())
    for partition in partitions:
        if not args.include_missing and not partition.get('source_present', True):
            continue
        evidence_path = partition.get('evidence_shard')
        if not evidence_path:
            continue
        with inside(evidence_path).open(encoding='utf8') as f:
            for line in f:
                item = json.loads(line)
                item['evidence'] = [r for r in item['evidence'] if r.get('relation') != 'batch_context']
                if not item['evidence']:
                    continue
                if lifecycle_reason(item['claimed_path']):
                    continue
                if path and path_key(item['claimed_path']) != path_key(path):
                    continue
                unique = []
                for record in item['evidence']:
                    signature = (path_key(item['claimed_path']), path_key(record['manifest']),
                                 record['record_index'], record.get('relation'), record.get('reference_field'))
                    if signature not in seen:
                        seen.add(signature)
                        unique.append(record)
                item['evidence'] = unique
                if not unique:
                    continue
                if not claim_matches(item['evidence'], args):
                    continue
                item['recorded_claims'] = structured_claims(item['evidence'])
                print(json.dumps(item, ensure_ascii=False))
                matches += 1
                if args.limit and matches >= args.limit:
                    return 0
    return 0


def query_lifecycle(args):
    for row in _LIFECYCLE_RECORDS + _LIFECYCLE_PATTERNS:
        if args.state and str(row.get('state', '')).casefold() != args.state.casefold():
            continue
        if args.contains and args.contains.casefold() not in json.dumps(row, ensure_ascii=False).casefold():
            continue
        print(json.dumps(row, ensure_ascii=False))
    for warning in _LIFECYCLE_WARNINGS:
        print(json.dumps(warning, ensure_ascii=False), file=sys.stderr)
    return 1 if _LIFECYCLE_WARNINGS else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    generate = sub.add_parser('build', help='refresh complete source partitions')
    generate.add_argument('--source', action='append', default=[], help='refresh matching partition(s) only')
    generate.add_argument('--scan-root', action='append', default=[], help='workspace directory; repeat to select other roots')
    generate.add_argument('--relocation', action='append', default=[], help='explicit old=new workspace paths')
    generate.add_argument('--relocation-file', action='append', default=[], help='JSON with moves/original_path/current_path')
    generate.add_argument('--no-video-probe', action='store_true')
    generate.add_argument('--no-clean', action='store_true',
                          help='preserve unreferenced index shards during this build')
    generate.add_argument('--rebuild', action='store_true', help='rebuild selected partitions even when inputs match')
    find = sub.add_parser('query', help='read records without changing the index')
    find.add_argument('--contains')
    find.add_argument('--partition')
    find.add_argument('--kind', choices=('image', 'video', 'array_asset'))
    find.add_argument('--with-evidence', action='store_true')
    find.add_argument('--include-missing', action='store_true', help='include previous unavailable partitions')
    find.add_argument('--limit', type=int, default=20, help='0 means unlimited')
    evidence = sub.add_parser('evidence', help='look up manifest claims across all indexed partitions')
    evidence.add_argument('path', nargs='?', help='workspace-relative media path; omit to search claims')
    evidence.add_argument('--include-missing', action='store_true')
    evidence.add_argument('--limit', type=int, default=20, help='0 means unlimited')
    lifecycle = sub.add_parser('lifecycle', help='query recorded dataset lifecycle states and evidence')
    lifecycle.add_argument('--contains')
    lifecycle.add_argument('--state')
    batches = sub.add_parser('batches', help='query batch-level scope, reuse and evaluation evidence')
    batches.add_argument('--contains')
    batches.add_argument('--evaluation-scope')
    batches.add_argument('--status')
    batches.add_argument('--include-missing', action='store_true')
    batches.add_argument('--limit', type=int, default=20, help='0 means unlimited')
    sub.add_parser('unlock', help='remove a same-host lock only if its PID is confirmed absent')
    sub.add_parser('retire-overlaps', help='drop superseded export task entries only after all batches were indexed')
    for action in (find, evidence):
        action.add_argument('--generation-method', help='filter explicit generation method claims')
        action.add_argument('--source-group', help='filter explicit source group claims')
        action.add_argument('--observed-content', help='filter declared visual content observations')
        action.add_argument('--review-flag', help='filter declared visual review flags')
        action.add_argument('--background-view', help='filter declared camera view')
        action.add_argument('--background-placement-plane', help='filter declared placement plane')
        action.add_argument('--background-target-presence', help='filter declared target presence')
        action.add_argument('--background-use', help='filter declared background use')
        action.add_argument('--background-review-state', help='filter declared background review state')
        action.add_argument('--observed-task', help='filter an exact candidate task in visual observations')
        action.add_argument('--region-role', help='filter an exact role in candidate visual regions')
        action.add_argument('--region-task', help='filter a task candidate in the same visual region')
        action.add_argument('--ground-patch-use', help='filter the exact use status of a visible ground patch')
        action.add_argument('--split', help='filter explicitly recorded split claims')
        action.add_argument('--evaluation-scope', help='filter explicit evaluation scope claims')
    args = parser.parse_args(argv)
    if args.command in ('query', 'evidence', 'batches') and args.limit < 0:
        parser.error('--limit must be nonnegative')
    try:
        load_lifecycle_rules()
        if args.command == 'build':
            return build(args)
        if args.command == 'evidence':
            with catalog_guard():
                return query_evidence_locked(args)
        if args.command == 'unlock':
            return unlock()
        if args.command == 'lifecycle':
            return query_lifecycle(args)
        if args.command == 'batches':
            with catalog_guard():
                return query_batches_locked(args)
        if args.command == 'retire-overlaps':
            return with_build_lock(retire_overlaps)
        with catalog_guard():
            return query_locked(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print('media index:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
