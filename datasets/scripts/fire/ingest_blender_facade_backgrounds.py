"""Ingest flame-free Blender facade renders and their measured paper slots."""

import json

from datasets.scripts.common.ingest_frame_environment import ingest_frame
from datasets.scripts.common.paths import WORKSPACE


MANIFEST = 'datasets/annotated/common/sources/aiaec_fire_c_facade_20261117/manifest.json'


def main():
    batch = json.loads((WORKSPACE / MANIFEST).read_text(encoding='utf-8'))
    if batch['task'] != 'fire' or batch['purpose'] != 'flame_free_facade_background':
        raise ValueError('Expected the flame-free facade render batch')
    count = 0
    for row in batch['records']:
        if row['fire_target_presence'] != 'absent' or len(row['placement_surfaces']) < 1:
            raise ValueError('Facade frame lacks a verified flame-free surface: ' + row['id'])
        evidence = {'method': 'blender_scene_machine_frame_review', 'manifest': MANIFEST,
                    'record_id': row['id'], 'source_blend_sha256': row['source_blend_sha256']}
        regions = []
        for surface in row['placement_surfaces']:
            quad = surface['quad_px']
            if (len(quad) != 4 or any(len(point) != 2 or not 0 <= point[0] <= row['image_size'][0]
                                      or not 0 <= point[1] <= row['image_size'][1] for point in quad)):
                raise ValueError('Facade paper quadrilateral is outside the render: ' + row['id'])
            regions.append({'slot_id': surface['slot'], 'plan_region_xy': quad,
                            'quad_world_m': surface['quad_world_m'],
                            'support_object': surface['support_object'],
                            'visibility': surface['visibility']})
        ingest_frame(
            row['image'], objects=[],
            task_annotations={'fire': {'presence': 'absent', 'completeness': 'complete',
                                       'scope': {'kind': 'frame'}, 'instances': [],
                                       'split': batch['split'], 'evidence': evidence}},
            task_geometry={'fire': {'status': 'candidate', 'regions': regions,
                                    'evidence': {'basis': 'blender_projected_facade_slots', **evidence}}},
            environment={'status': 'observed', 'algorithm_version': 'blender_facade_projection_v1',
                         'evidence': evidence},
            lineage={'source_kind': 'blender_render', 'source_blend': row['source_blend'],
                     'mother_scene': row['source_blend'], 'scene': row['scene'],
                     'parent_group': row['parent_group'],
                     'capture_identity': f"{row['scene']}:{row['id']}",
                     'provenance': evidence},
            expected_sha256=row['content_sha256'], origin='machine',
        )
        count += 1
    print(json.dumps({'fire_composite_backgrounds': count, 'manifest': MANIFEST},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
