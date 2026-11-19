"""Register rendered blank bins and their source-pixel fronts for the sticker task."""

import argparse
from collections import Counter
import json
from pathlib import Path

from datasets.scripts.common.config import source_path
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.ingest_frame_environment import ingest_frame
from datasets.scripts.common.paths import WORKSPACE
from datasets.scripts.waste_sticker.generate_sticker_real_photo_v4 import _polygon
from datasets.scripts.waste_sticker.source_inputs import surface_background_catalog


TASK = 'waste_sticker'
COLORS = {'blue', 'red', 'green', 'gray'}


def ingest(manifest, split='train'):
    manifest = manifest.resolve()
    relative = manifest.relative_to(WORKSPACE).as_posix()
    batch = json.loads(manifest.read_text(encoding='utf-8'))
    records = batch['records']
    if len({row['image'] for row in records}) != len(records):
        raise ValueError('Each rendered background must have a unique image path')
    for row in records:
        colors = {face['bin_color'] for face in row['bins']}
        if (row['source_kind'] != 'blender_render' or row['item_stickers_hidden'] is not True or
                not colors or not colors.issubset(COLORS) or len(colors) != len(row['bins'])):
            raise ValueError('Expected one or more verified bin fronts and hidden item stickers')
        for face in row['bins']:
            _polygon(face['allowed_polygon'], *row['image_size'], quad=True)
    counts = Counter()
    for index, row in enumerate(records):
        evidence = {
            'method': 'machine_frame_review_blender_scene_object_inventory',
            'manifest': relative, 'record_index': index, 'source_blend_sha256': row['source_blend_sha256'],
            'content_sha256': row['content_sha256'], 'coordinate_size': row['image_size'],
            'basis': 'projected bin front vertices with direct visibility ray tests',
        }
        source_ref = f"{row['scene']}:{row['route']}:{row['frame']}"
        objects = []
        faces = []
        for face in row['bins']:
            object_id = f"box:{row['content_sha256']}:{face['bin_id']}"
            objects.append({'object_id': object_id, 'kind': 'waste_bin', 'bin_color': face['bin_color']})
            faces.append({**face, 'object_id': object_id})
        ingest_frame(
            row['image'], objects=objects,
            task_annotations={TASK: {
                'presence': 'absent', 'completeness': 'complete', 'scope': {'kind': 'frame'},
                'instances': [], 'split': split, 'evidence': evidence,
            }},
            task_geometry={TASK: {
                'status': 'candidate', 'bins': faces, 'coordinate_size': row['image_size'],
                'reason_codes': ['rendered_visible_bin_front'], 'evidence': evidence,
            }},
            environment={'status': 'observed', 'primary_subject_task': 'waste_bin_state', 'evidence': evidence},
            lineage={
                'source_kind': row['source_kind'], 'source_blend': row['source_blend'],
                'source_blend_sha256': row['source_blend_sha256'], 'mother_scene': row['source_blend'],
                'parent_group': row['parent_group'], 'scene': row['scene'], 'route': row['route'],
                'render_frame': row['frame'], 'capture_identity': f"{row['source_blend_sha256']}:{source_ref}",
                'provenance': evidence,
            },
            origin='machine', expected_sha256=row['content_sha256'],
        )
        counts[row['route']] += 1
    images = {row['image'] for row in records}
    consumed = surface_background_catalog(images=images, split=split)
    assert all(row['background_kind'] == 'blender_render' and row['label']['presence'] == 'absent'
               for row in consumed.values())
    empty = query_frames(task=TASK, role='empty_negative', split=split, status='eligible')
    return {'frames': len(consumed), 'routes': dict(counts),
            'empty_negative': len(images & {p for row in empty for p in row['paths']})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=source_path(TASK, 'rendered_surface_manifest'))
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    args = parser.parse_args()
    print(json.dumps(ingest(args.manifest, args.split), ensure_ascii=False, indent=2))
