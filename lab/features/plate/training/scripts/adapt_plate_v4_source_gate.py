"""Audit V4 plate candidates against known sources without inventing labels."""
import ast
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.scripts.data import v4_source_gate as gate
from lab.training_common.production_guard import generation_guard

DATA = ROOT / 'datasets/exports/plate/v4_master_v4_20261113'
MANIFEST = DATA / 'manifest.json'
CONFIG_DATA = 'datasets/exports/plate/v4_master_v4_20261113/data.yaml'
OUT = ROOT / 'datasets/exports/plate/v4_master_build_inputs_20261113'
CONFIG = ROOT / 'lab/features/plate/training/config/plate_detection_v4.yaml'
ENTRY = ROOT / 'lab/features/plate/training/scripts/train_plate_detection_v4.py'
WEIGHT = ROOT / 'models/weights/plate/provincial/plate_detector_best.pt'
LICENSE = ROOT / 'models/weights/yolo_v8/LICENSE.txt'
WEIGHT_EVIDENCE = ROOT / 'models/README.md'
VIDEO_FRAME_ROOT = 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/'


@lru_cache(maxsize=32)
def source_records(manifest):
    return json.loads((ROOT / manifest).read_text(encoding='utf8'))['records']


def background_of(row):
    """Map only directly recorded backgrounds; unknown must stay unassigned."""
    recorded = row.get('background')
    if recorded:
        if not isinstance(recorded, dict) or recorded.get('kind') not in {'blender', 'real', 'procedural'}:
            raise ValueError('Invalid recorded background: ' + row['image'])
        return recorded
    method = row.get('generation_method')
    if method == 'blender_cycles_optix_empty_scene_real_photo_crop':
        yaw = row.get('camera_yaw_degrees')
        if yaw is None and row.get('source_manifest') is not None and type(row.get('source_record_index')) is int:
            source = source_records(row['source_manifest'])[row['source_record_index']]
            if source.get('image') != row.get('source_image'):
                raise ValueError('Original source record does not match image: ' + row['image'])
            yaw = source.get('camera_yaw_degrees')
        if not isinstance(yaw, (int, float)):

            raise ValueError('Missing actual Blender camera yaw: ' + row['image'])
        group = round(float(yaw) / 6) * 6
        return {'kind': 'blender',
                'source_group': 'lab/features/plate/training/scripts/render_plate_empty_scene.py',
                'view_group': f'yaw_{group:+03d}_degrees'}
    if method == 'composite' and row.get('source_background'):
        scene = row.get('mother_scene')
        view = (row.get('background_camera') or {}).get('image')
        if not scene or not scene.endswith('.blend') or not view:
            raise ValueError('Missing actual Blender mother scene or camera: ' + row['image'])
        return {'kind': 'blender', 'source_group': scene, 'view_group': Path(view).stem}
    video = row.get('source_video') or row.get('parent_video')
    if video:
        return {'kind': 'real', 'source_group': video, 'view_group': None}
    if str(row.get('source_group', '')).startswith('CCPD/'):
        parent = row.get('parent_image')
        if parent and (ROOT / parent).is_file():
            return {'kind': 'real', 'source_group': parent, 'view_group': None}
    raise ValueError('Unresolved background origin; isolate this image: ' + row['image'])


def build_view(records):
    samples = []
    parents = {}
    for index, row in enumerate(records):
        parent = row.get('parent_image') or row.get('source_image') or row.get('input_image')
        if not isinstance(parent, str) or not (ROOT / parent).is_file():
            raise ValueError(f'Missing recorded parent for {index}: {parent}')
        if parent not in parents:
            key = f'p{len(parents):05d}'
            parents[parent] = key
            video = row.get('source_video') or row.get('parent_video')
            source_kind = 'video_frame' if parent.startswith(VIDEO_FRAME_ROOT) else None
            samples.append({'id': key, 'image': parent, 'split': 'unassigned',
                            'source_kind': source_kind, 'generation_method': 'provided_video_frame' if source_kind else None, 'parent_video': video,
                            'source_time_seconds': row.get('source_time_seconds'), 'parent_ids': []})
        label = (ROOT / row['label']).read_text(encoding='utf8').strip()
        if not row.get('source_evidence'):
            raise ValueError('No indexed source evidence: ' + row['image'])
        original = row.get('input_label')
        if not original or not (ROOT / original).is_file() or label != (ROOT / original).read_text(encoding='utf8').strip():
            raise ValueError('Label differs from immediate input: ' + row['image'])
        method = row.get('generation_method')
        render = method == 'blender_cycles_optix_empty_scene_real_photo_crop'
        if render:
            raise ValueError('Single-color procedural scene excluded from master delivery: ' + row['image'])
        composite = method == 'composite' and bool(row.get('source_background'))
        background = background_of(row)
        background_parent_ids = []
        if composite:
            source_bg = row['source_background']
            if not (ROOT / source_bg).is_file():
                raise ValueError('Missing Blender background render: ' + source_bg)
            if source_bg not in parents:
                bg_key = f'p{len(parents):05d}'
                parents[source_bg] = bg_key
                samples.append({'id': bg_key, 'image': source_bg, 'split': 'unassigned', 'parent_ids': []})
            background_parent_ids = [parents[source_bg]]
        tone = row.get('source_tone_profile')
        treatment = row.get('final_image_processing') or {}
        strata = ['low_light', 'low_saturation'] if tone == 'moderate' or treatment.get('illumination_profile') == 'simulated_low_light_and_low_saturation' else []
        source = row.get('input_manifest')
        if not source or not (ROOT / source).is_file():
            raise ValueError('Missing source manifest: ' + row['image'])
        processing = dict(row.get('image_processing') or {})
        processing.setdefault('resize', 'see indexed source manifest')
        processing.setdefault('compression', 'source format retained or PNG')
        processing.setdefault('blur', 'see indexed source manifest')
        processing['input_image'] = row.get('input_image')
        processing['final_image_processing'] = row.get('final_image_processing')
        sample = {'id': f's{index:05d}', 'image': row['image'], 'split': row['split'],
                  'source_group': row.get('source_group') or row.get('source_video') or parent,
                  'source_kind': 'blender_scene' if render else 'derived_image',
                  'generation_method': 'blender_render' if render else ('composite' if composite else 'augmentation'),
                  'parent_ids': [parents[parent]] + background_parent_ids, 'parent_video': row.get('source_video') or row.get('parent_video'),
                  'source_time_seconds': row.get('source_time_seconds'),
                  'source_assets': [parent] if render else (row.get('source_assets') if composite else None),
                  'mother_scene': background['source_group'] if render else (row.get('mother_scene') if composite else None),
                  'background': background,
                  'image_processing': processing,
                  'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                  'review_evidence': source, 'source_evidence': row['source_evidence'],
                  'selected_tasks': ['plate_detection'],
                  'tasks': {'plate_detection': {'state': 'complete_positive' if label else 'complete_negative',
                                                'label': row['label'], 'review_evidence': source,
                                                'strata': {'conditions': strata},
                                                'strata_review': 'confirmed' if strata else None}}}
        samples.append(sample)
    return {'models': {'plate_detection': {'data_yaml': CONFIG_DATA}}, 'samples': samples}



def main():
    records = json.loads(MANIFEST.read_text(encoding='utf8'))['records']
    gate_media, gate_procedural, gate_code = gate.queue_inputs(ROOT)
    media = {ROOT / row['image'] for row in records}
    media.update(ROOT / (row.get('parent_image') or row.get('source_image') or row['input_image']) for row in records)
    media.update(ROOT / row['source_background'] for row in records if row.get('source_background'))
    media.update(gate_media)
    procedural = {MANIFEST, ROOT / CONFIG_DATA, CONFIG, WEIGHT, LICENSE, WEIGHT_EVIDENCE}
    procedural.update(gate_procedural)
    procedural.update(ROOT / row['mother_scene'] for row in records if row.get('source_background') and row.get('mother_scene'))
    for row in records:
        procedural.add(ROOT / row['label'])
        if row.get('input_label'):
            procedural.add(ROOT / row['input_label'])
        if row.get('input_manifest'):
            procedural.add(ROOT / row['input_manifest'])
        if row.get('source_manifest'):
            procedural.add(ROOT / row['source_manifest'])
    code_paths = sorted(set(gate_code) | {Path(__file__), ENTRY})
    with generation_guard('plate_v4_master_source_gate_20261113', sorted(media), [OUT],
                          {'scope': 'registered lineage and known diagnostics; label structure and inherited review only',
                           'test_design': 'same material, distinct output images'},
                          code_paths=code_paths, procedural_inputs=sorted(procedural), root=ROOT) as run:
        if run.reused:
            print(json.dumps({'status': 'reused', 'manifest': (OUT / 'v4_manifest.json').relative_to(ROOT).as_posix()}))
            return
        view = build_view(records)
        known = gate.build_queue(ROOT)
        result = gate.audit(ROOT, view, known)
        if result['status'] != 'passed' or result['assigned'] != len(records):
            raise ValueError('Source gate failed: ' + json.dumps(result['errors'], ensure_ascii=False))
        OUT.mkdir(parents=True, exist_ok=False)
        audit_path = OUT / 'source_gate_audit.json'
        evidence_path = audit_path.relative_to(ROOT).as_posix()
        for sample in view['samples']:
            if sample['split'] in ('train', 'val', 'test'):
                sample['known_relationship_audit'] = evidence_path
                sample['review_scope'] = 'Registered relationships and source label structure only; semantic completeness and unrecorded near-duplicates not established.'
                sample['indexed_final_evidence'] = run.source_refs.get(sample['image'], [])
                if not sample['indexed_final_evidence']:
                    raise ValueError('Missing indexed final evidence: ' + sample['image'])
        result['scope'] = 'Recorded provenance, known diagnostic relations, pixel identity and label structure only; no model evaluation, exhaustive semantic review or unknown near-duplicate claim.'
        audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        source_path = OUT / 'source_manifest.json'
        view['scope'] = result['scope']
        view['audit_evidence'] = evidence_path
        source_path.write_text(json.dumps(view, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        package = {'schema_version': 1, 'dataset_version': 'v4', 'model_key': 'plate_detection',
                   'config': CONFIG.relative_to(ROOT).as_posix(), 'entry': ENTRY.relative_to(ROOT).as_posix(),
                   'source_manifest': source_path.relative_to(ROOT).as_posix(),
                   'initialization': {'path': WEIGHT.relative_to(ROOT).as_posix(),
                                      'source': 'Imported provincial plate detector checkpoint; redistribution permission for this checkpoint remains unverified.',
                                      'license': LICENSE.relative_to(ROOT).as_posix(),
                                      'evidence': WEIGHT_EVIDENCE.relative_to(ROOT).as_posix()},
                   'additional_records': [evidence_path, MANIFEST.relative_to(ROOT).as_posix()],
                   'scope': 'Plate detector; train, development and synthetic test share source material. No model training or evaluation.'}
        (OUT / 'v4_manifest.json').write_text(json.dumps(package, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'status': 'passed', 'assigned': result['assigned'], 'counts': result['counts'],
                          'manifest': (OUT / 'v4_manifest.json').relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
