"""Read raw-image visual candidates and their provenance from the media cache.

The result preserves each source declaration and its review scope. Instances
from separate records may refer to the same object and are not deduplicated.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
import media_index as core
import media_index_sqlite as cache


def list_values(value):
    return value if isinstance(value, list) else []


def instance_record(source, item, task=None, task_scope=None):
    result = {'source': source, 'task': item.get('task', task),
              'bbox_xyxy': item.get('bbox_xyxy'), 'bbox_untyped': item.get('bbox'),
              'status': item.get('status'), 'use_status': item.get('use_status'), 'instance': item}
    if task_scope is not None:
        result['task_scope'] = task_scope
    return result


def possible_duplicates(instances, threshold=0.8):
    """Flag highly overlapping same-task boxes; do not merge declarations."""
    result = []
    for i, left in enumerate(instances):
        a = left.get('bbox_xyxy')
        if not left.get('task') or not isinstance(a, list) or len(a) != 4 or not all(isinstance(v, (int, float)) for v in a):
            continue
        for j in range(i + 1, len(instances)):
            right = instances[j]
            b = right.get('bbox_xyxy')
            if left['task'] != right.get('task') or not isinstance(b, list) or len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
                continue
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
            if union and intersection / union >= threshold:
                result.append({'instance_indices': [i, j], 'iou': round(intersection / union, 4),
                               'status': 'possible_duplicate_not_merged'})
    return result


def query_image(path: str) -> dict:
    """Return source claims, regions and instance candidates for one raw image."""
    current = core.inside(path)
    raw = (core.ROOT / 'datasets/raw').resolve()
    if not current.is_relative_to(raw) or not current.is_file():
        raise ValueError('path must be an existing image below datasets/raw')
    rel = core.relative(current)
    if current.suffix.lower() not in core.IMAGES:
        raise ValueError('path must be an image')
    args = types.SimpleNamespace(**{key: None for key in (*cache.FILTER_FIELDS, *cache.NUMERIC_FILTERS)})
    observations, regions, instances, rejected, keypoints, background, ground_patches, counts = [], [], [], [], [], [], [], []
    seen = set()
    with closing(cache.connect(readonly=True)) as db:
        db.execute('BEGIN')
        cache.validated(db, False)
        for _, raw_entry, descriptor in cache.matching_entries(db, args, rel):
            part = json.loads(descriptor)
            if core.lifecycle_reason(part['name']) or not part['source_present']:
                continue
            for entry in json.loads(raw_entry)['evidence']:
                if entry.get('relation') != 'describes':
                    continue
                identity = (entry.get('manifest'), entry.get('record_index'), entry.get('relation'))
                if identity in seen:
                    continue
                seen.add(identity)
                source = {'manifest': entry.get('manifest'), 'record_index': entry.get('record_index')}
                claims = entry.get('claims', {})
                observations.append({'source': source, 'claims': claims})
                for region in list_values(claims.get('regions')):
                    if isinstance(region, dict):
                        regions.append({'source': source, 'role': region.get('role'),
                                        'polygon': region.get('polygon'),
                                        'task_candidates': region.get('task_candidates'),
                                        'use_status': region.get('use_status'),
                                        'region_use': region.get('region_use'),
                                        'geometry_quality': region.get('geometry_quality'),
                                        'region': region})
                for key in ('instances', 'observed_instances'):
                    for item in list_values(claims.get(key)):
                        if isinstance(item, dict):
                            instances.append(instance_record(source, item))
                tasks = claims.get('task_observations')
                for task, value in (tasks.items() if isinstance(tasks, dict) else []):
                    if isinstance(value, dict):
                        for item in list_values(value.get('instances')):
                            if isinstance(item, dict):
                                scope = {k: v for k, v in value.items() if k != 'instances'}
                                instances.append(instance_record(source, item, task, scope))
                for item in list_values(claims.get('rejected_instances')):
                    if isinstance(item, dict):
                        rejected.append(instance_record(source, item))
                for item in list_values(claims.get('background_observations')):
                    if isinstance(item, dict):
                        background.append({'source': source, 'observation': item})
                for patch in list_values(claims.get('visible_ground_patches')):
                    if isinstance(patch, dict):
                        ground_patches.append({'source': source, 'patch': patch})
                if 'keypoints' in claims:
                    keypoints.append({'source': source, 'keypoints': claims['keypoints'],
                                      'scope': claims.get('keypoints_scope'),
                                      'dial_bbox': claims.get('dial_bbox'),
                                      'bbox_status': claims.get('bbox_status')})
                task_counts = claims.get('local_count_observations')
                if isinstance(task_counts, dict):
                    counts.extend({'source': source, 'task': task, 'observation': value}
                                  for task, value in task_counts.items() if isinstance(value, dict))
        db.rollback()
    return {'path': rel, 'observations': observations, 'regions': regions,
            'instances': instances, 'rejected_instances': rejected,
            'keypoints': keypoints, 'background_observations': background, 'visible_ground_patches': ground_patches,
            'local_count_observations': counts,
            'possible_duplicate_instances': possible_duplicates(instances),
            'interpretation': 'Candidates may overlap or describe the same object; inspect source scopes and statuses.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True, help='workspace-relative raw image path')
    args = parser.parse_args()
    print(json.dumps(query_image(args.path), ensure_ascii=False))


if __name__ == '__main__':
    main()
