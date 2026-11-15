"""Export reviewed vehicle-camera plate photos for development regression."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import source_role  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from prepare_plate_v4_mixed_candidates import valid_label  # noqa: E402

REVIEW = ROOT/'datasets/annotated/plate/v4_real_photo_video_candidates_20261112'
OUT = ROOT/'datasets/exports/plate/v4_vehicle_photo_regression_20261112'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def _generate():
    if OUT.exists():
        raise ValueError('Output exists; select a new version rather than overwrite')
    rows = json.loads((REVIEW/'review_manifest.json').read_text(encoding='utf-8'))['samples']
    if len(rows) != 9:
        raise ValueError('Expected nine previously reviewed real photos')
    known = build_queue(ROOT)
    records = []
    (OUT/'images/regression').mkdir(parents=True)
    (OUT/'labels/regression').mkdir(parents=True)
    for row in rows:
        source = ROOT/row['parent_image']
        role, reason = source_role(ROOT, source, known)
        if row['split'] != 'regression' or role != 'regression' or row['state'] != 'candidate_complete_positive':
            raise ValueError(f'Unexpected source role for {source}: {role} {reason}')
        label_text = valid_label(ROOT/row['label'])
        if not label_text or len(label_text.splitlines()) != len(row['candidate_boxes_xyxy']):
            raise ValueError('Review boxes do not match label count: ' + row['id'])
        image_out = OUT/'images/regression'/(row['id']+'.png')
        label_out = OUT/'labels/regression'/(row['id']+'.txt')
        shutil.copy2(source,image_out)
        label_out.write_text(label_text+'\n',encoding='utf-8')
        records.append({'image':rel(image_out),'label':rel(label_out),'split':'regression',
            'parent_image':row['parent_image'],'generation_method':'direct_camera_capture',
            'image_size':row['image_size'],'source_group':row['source_group'],
            'review_evidence':row['review_evidence'],'diagnostic_near_duplicate_evidence':row['exposure_evidence'],
            'physical_asset_appearance':row['physical_asset_appearance'],'physical_identity':row['physical_identity']})
    manifest={'status':'development_regression_only','images':len(records),
              'scope':'Nine previously contacted 640x480 vehicle-camera photos and known diagnostic near-duplicates; '
                      'not training data, a frozen test, or independent capture evidence.', 'records':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':rel(OUT),'images':len(records)},ensure_ascii=False))


def _guard_inputs():
    rows = json.loads((REVIEW / 'review_manifest.json').read_text(encoding='utf-8'))['samples']
    return [ROOT / r['parent_image'] for r in rows], [REVIEW / 'review_manifest.json'] + [ROOT / r['label'] for r in rows], {}


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media, procedural, parameters = _guard_inputs()
    from lab.scripts.data import v4_source_gate as gate
    procedural += [ROOT / value for value in (gate.INVENTORY, gate.SEGMENTS, gate.SELECTION,
        gate.EXPOSURE, gate.DUPLICATES, gate.DEVELOPMENT_EXPOSURE, gate.REVIEW_SUMMARY,
        'config/perception_tasks.json', *gate.ANNOTATIONS, *gate.SITE_SELECTIONS)]
    from lab.features.plate.training.scripts.plate_guard_inputs import source_gate_dependencies
    gate_files, gate_code = source_gate_dependencies(ROOT, gate.build_queue(ROOT))
    procedural += list(gate_files)
    media = sorted(set(media))
    procedural = sorted(set(procedural))
    with generation_guard("plate.prepare_plate_v4_photo_regression", media, [OUT], parameters,
                          code_paths=sorted(gate_code | {Path(__file__), ROOT / 'lab/training_common/camera_tone.py', ROOT / 'lab/features/plate/training/scripts/prepare_plate_v4_mixed_candidates.py'}), procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _generate()
        manifest_path = OUT / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['source_refs'] = run.source_refs
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
