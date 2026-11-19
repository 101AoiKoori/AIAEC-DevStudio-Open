"""Ingest selected empty Blender route views with source-ground evidence.

The rendered Arena scenes contain no V4 task targets. Ground eligibility is
computed from the saved camera transform and the 4.2 m floor. Building-name
and meter tasks accept the general empty scene; fire and sticker placement
still need task-specific facade or bin surfaces.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .ingest_frame_environment import ingest_frame
from .paths import TASKS, WORKSPACE
from .review.placement import GroundPlacement
from .review.placement_record import encode_placement_record


MANIFESTS = [
    'datasets/annotated/common/sources/aiaec_eight_routes_20261117/manifest.json',
    'datasets/annotated/common/sources/aiaec_eight_routes_indoor_20261117/manifest.json',
    'datasets/annotated/common/sources/aiaec_eight_routes_detailed_20261117/manifest.json',
]
GROUND_TASKS = {'person', 'e_bike', 'sign', 'waste_bin_state'}
GENERAL_SCENE_TASKS = {'building_name_paper', 'meter'}
LENS_MM = {'person': 23, 'e_bike': 25, 'fire': 24, 'building_name_paper': 27,
           'sign': 22, 'waste_bin_state': 25, 'waste_sticker': 24, 'meter': 26}


def floor_mask(record):
    width, height = record['image_size']
    matrix = np.asarray(record['camera_matrix_world'], dtype=np.float64)
    u, v = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
    # Saved cameras use Blender's 36 mm horizontal sensor and AUTO fit.
    local = np.stack(((u / width - .5) * 36 / LENS_MM[record['route']],
                      (.5 - v / height) * 36 / LENS_MM[record['route']],
                      -np.ones_like(u)), axis=-1)
    rays = local @ matrix[:3, :3].T
    origin = matrix[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = (-.0125 - origin[2]) / rays[:, :, 2]
        x = origin[0] + distance * rays[:, :, 0]
        y = origin[1] + distance * rays[:, :, 1]
    return ((distance > 0) & (np.abs(x) < 2.08) & (np.abs(y) < 2.08)).astype(np.uint8)


def selected_records(manifests, *, all_frames=False):
    selected = []
    for manifest in manifests:
        data = json.loads((WORKSPACE / manifest).read_text(encoding='utf-8'))
        for route in LENS_MM:
            records = sorted((row for row in data['records'] if row['route'] == route),
                             key=lambda row: row['frame'])
            if len(records) != 24:
                raise ValueError(f'Expected 24 frame positions for {route}: {manifest}')
            indices = range(24) if all_frames else (0, 3, 6, 9, 12, 15, 18, 21)
            selected.extend((manifest, records[index]) for index in indices)
    return selected


def ingest(manifests=MANIFESTS, *, all_frames=False):
    counts = Counter()
    for manifest, row in selected_records(manifests, all_frames=all_frames):
        ground = floor_mask(row)
        contours, _ = cv2.findContours(ground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        polygon = cv2.convexHull(largest).reshape(-1, 2).tolist() if largest is not None else []
        floor_ready = int(ground.sum()) >= int(ground.size * .12) and len(polygon) >= 3
        placement = encode_placement_record(GroundPlacement(
            'candidate' if floor_ready else 'unknown', 'blender_floor_projection_v1',
            ground.astype(bool) if floor_ready else np.zeros_like(ground, dtype=bool),
            np.zeros_like(ground, dtype=bool),
        ))
        evidence = {'method': 'machine_frame_review_blender_scene_object_inventory',
                    'manifest': manifest, 'record_id': f"{row['scene']}:{row['route']}:{row['frame']}",
                    'source_blend_sha256': row['source_blend_sha256']}
        annotations = {task: {'presence': 'absent', 'completeness': 'complete',
                              'scope': {'kind': 'frame'}, 'instances': [],
                              'split': 'train', 'evidence': evidence}
                       for task in TASKS}
        geometry = {}
        for task in TASKS:
            if task in GROUND_TASKS and floor_ready:
                geometry[task] = {
                    'status': 'candidate', 'ground_polygon': polygon,
                    'placeable_ground': placement,
                    'evidence': {'basis': 'blender_4p2m_floor_and_camera_projection', **evidence},
                }
            elif task in GENERAL_SCENE_TASKS:
                geometry[task] = {
                    'status': 'candidate',
                    'regions': [{'kind': 'full_frame_empty_scene',
                                 'polygon': [[0, 0], [640, 0], [640, 480], [0, 480]]}],
                    'evidence': {'basis': 'blender_empty_scene_full_frame', **evidence},
                }
            else:
                geometry[task] = {
                    'status': 'unknown',
                    'reason_codes': ['task_surface_not_present_in_empty_arena'] if task not in GROUND_TASKS
                                    else ['rendered_floor_area_too_small'],
                    'evidence': evidence,
                }
        ingest_frame(
            row['image'], objects=[], task_annotations=annotations,
            task_geometry=geometry,
            environment={'status': 'observed', 'algorithm_version': 'blender_scene_object_inventory_v1',
                         'evidence': evidence},
            lineage={'source_kind': 'blender_render', 'source_blend': row['source_blend'],
                     'mother_scene': row['source_blend'], 'scene': row['scene'],
                     'route': row['route'], 'render_frame': row['frame'],
                     'parent_group': row['source_blend'],
                     'capture_identity': f"{row['scene']}:{row['route']}:{row['frame']}",
                     'provenance': evidence},
            expected_sha256=row['content_sha256'], origin='machine',
        )
        counts['frames'] += 1
        counts['ground_ready'] += int(floor_ready)
        counts[row['scene']] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', help='Override the three default render manifests')
    parser.add_argument('--all-frames', action='store_true', help='Ingest all 24 rendered frames per route')
    args = parser.parse_args()
    print(json.dumps(ingest(args.manifest or MANIFESTS, all_frames=args.all_frames),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
