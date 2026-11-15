"""Add evidence-backed Blender background fields to the existing meter V4 selection."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

SOURCE = ROOT / 'datasets/exports/meter/v4/meter/source_manifest.json'
BATCH = ROOT / 'datasets/exports/meter/v4_parametric_station_pose_optix_20261113/manifest.json'
SCENE = 'datasets/assets/blender/meter/v4_parametric_station_v2_20261113/station.blend'


def main():
    raw = SOURCE.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    batch_raw = BATCH.read_bytes()
    batch = json.loads(batch_raw.decode('utf-8-sig'))
    if batch.get('source_blend') != SCENE or batch.get('status') != 'complete_synthetic_pose_candidates':
        raise ValueError('unexpected meter Blender source or incomplete batch')
    annotations = []
    for row in data['samples']:
        ref = row.get('review_evidence', '')
        path = (ROOT / ref).resolve()
        expected = (BATCH.parent / 'annotations' / (row['id'] + '.json')).resolve()
        if path != expected or not path.is_file():
            raise ValueError('sample lacks its original batch annotation: ' + row.get('id', ''))
        annotations.append(path)
    inputs = [SOURCE, BATCH, ROOT / SCENE, *annotations]
    with generation_guard('meter.annotate_v4_background_20261113', [], [SOURCE],
                          {'source_scene': SCENE}, code_paths=[Path(__file__)],
                          procedural_inputs=inputs, inplace_paths=[SOURCE]) as run:
        if run.reused:
            print('METER_BACKGROUND_REUSED', json.dumps(run.value))
            return
        if SOURCE.read_bytes() != raw or BATCH.read_bytes() != batch_raw:
            raise ValueError('manifest changed while entering guard')
        counts = {}
        for row, path in zip(data['samples'], annotations):
            annotation = json.loads(path.read_text(encoding='utf-8-sig'))
            if annotation['id'] != row['id'] or annotation.get('annotation_status') != 'synthetic_pose_geometric':
                raise ValueError('sample annotation mismatch: ' + row['id'])
            z = annotation['state']['camera_z_fitted_scene_units']
            if z not in (7.5, 8.5, 10.0):
                raise ValueError('unrecognized measured scene camera position: ' + row['id'])
            group = 'camera_z_' + str(z).replace('.', '_')
            row['background'] = {'kind': 'blender', 'source_group': SCENE, 'view_group': group}
            key = row['split'] + '/' + group
            counts[key] = counts.get(key, 0) + 1
        SOURCE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        run.set_result({'count': len(annotations), 'by_split_view_group': counts})
        print('METER_BACKGROUND_COMPLETE', json.dumps(run.value), flush=True)


if __name__ == '__main__':
    main()
