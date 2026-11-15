"""Create a source-gated V4 meter split from the completed synthetic Pose batch.

All three splits share the same generated station, face and fitted scene.  The
split is a synthetic development split, not an independent capture evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/data'), str(ROOT / 'lab/training_common')]
from lab.training_common.production_guard import generation_guard
import v4_source_gate as gate
from engine.data import dataset_header

BATCH = ROOT / 'datasets/exports/meter/v4_parametric_station_pose_optix_20261113'
OUTPUT = ROOT / 'datasets/exports/meter/v4/meter'
CONFIG = ROOT / 'lab/features/meter/training/config/meter_v4.yaml'
ENTRY = ROOT / 'lab/features/meter/training/scripts/train_meter_v4.py'
POINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
          'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')


def relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('missing or external workspace input: ' + str(path))
    return path.relative_to(ROOT).as_posix()


def resolve_asset(path):
    """Resolve old source-manifest paths to current Blender asset locations."""
    original = Path(path)
    candidate = (ROOT / original).resolve()
    if candidate.is_file():
        return candidate
    parts = original.parts
    if parts[:3] == ('datasets', 'annotated', 'mixed_scene') and len(parts) > 3:
        moved = ROOT / 'datasets/assets/blender/shared' / parts[3] / Path(*parts[4:])
    elif parts[:3] == ('datasets', 'annotated', 'meter') and len(parts) > 3:
        moved = ROOT / 'datasets/assets/blender/meter' / Path(*parts[3:])
    elif parts[:3] == ('datasets', 'annotated', 'fire') and len(parts) > 3:
        moved = ROOT / 'datasets/assets/blender/fire' / Path(*parts[3:])
    elif parts[:3] == ('datasets', 'annotated', 'waste_bin') and len(parts) > 3:
        moved = ROOT / 'datasets/assets/blender/waste_bin' / Path(*parts[3:])
    else:
        moved = None
    if moved is not None and moved.is_file():
        return moved.resolve()
    raise ValueError('manifest source asset is missing at its recorded and current paths: ' + str(path))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def selected_rows():
    manifest = read_json(BATCH / 'manifest.json')
    if manifest.get('status') != 'complete_synthetic_pose_candidates':
        raise ValueError('render batch is not complete synthetic pose candidates')
    if manifest.get('resolution') != [640, 480] or manifest.get('keypoint_order') != list(POINTS):
        raise ValueError('render batch image or keypoint contract differs')
    rows = [json.loads(line) for line in (BATCH / 'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != manifest.get('completed_count') or len(rows) != manifest.get('requested_count'):
        raise ValueError('batch sample count does not match the completed manifest')
    selected, seen = [], set()
    for row in rows:
        label = row.get('pose_label')
        if not label:
            continue
        name = row.get('id')
        if not isinstance(name, str) or not name.startswith('station_') or not name[8:].isdigit() or name in seen:
            raise ValueError('invalid or duplicate sample ID')
        seen.add(name)
        if (row.get('annotation_status') != 'synthetic_pose_geometric' or
                row.get('image') != 'images/' + name + '.png' or label != 'labels/' + name + '.txt'):
            raise ValueError('invalid synthetic image/label reference: ' + name)
        image, label_path, annotation = BATCH / row['image'], BATCH / label, BATCH / 'annotations' / (name + '.json')
        for path in (image, label_path, annotation):
            relative(path)
        if read_json(annotation) != row:
            raise ValueError('per-frame annotation and JSONL differ: ' + name)
        lines = label_path.read_text(encoding='utf8').splitlines()
        if len(lines) != 1:
            raise ValueError('expected one positive pose object: ' + name)
        values = [float(v) for v in lines[0].split()]
        if len(values) != 29 or values[0] != 0 or not all(math.isfinite(v) for v in values):
            raise ValueError('invalid class, point count or numerical value: ' + name)
        if not all(0 <= v <= 1 for v in values[1:5]) or min(values[3:5]) <= 0:
            raise ValueError('invalid normalized box: ' + name)
        mask = row.get('dial_raster_evidence', {})
        box = mask.get('bbox_xyxy_pixels')
        if mask.get('bbox_status') != 'candidate_from_isolated_raster_and_visible_fraction' or not box:
            raise ValueError('pose box lacks raster evidence: ' + name)
        x1, y1, x2, y2 = box
        expected_box = [(x1+x2)/1280, (y1+y2)/960, (x2-x1)/640, (y2-y1)/480]
        if any(abs(a-b) > 1e-6 for a, b in zip(values[1:5], expected_box)):
            raise ValueError('box differs from raster evidence: ' + name)
        for i, point in enumerate(POINTS):
            x, y, visibility = values[5 + i*3:8 + i*3]
            info = row['keypoint_visibility'][point]
            if visibility not in (0, 1, 2) or visibility != info['visibility']:
                raise ValueError('label visibility differs from evidence: ' + name + '/' + point)
            if visibility == 0:
                if (x, y) != (0, 0) or row['keypoint_ray_evidence'][point]['in_frame']:
                    raise ValueError('outside-point evidence differs: ' + name + '/' + point)
            else:
                projected = row['keypoints_xy_depth_normalized_candidate'][point]
                ray = row['keypoint_ray_evidence'][point]
                if not (ray['in_frame'] and 0 <= x <= 1 and 0 <= y <= 1 and
                        abs(x-projected[0]) < 1e-6 and abs(y-projected[1]) < 1e-6):
                    raise ValueError('keypoint coordinate differs from projection: ' + name + '/' + point)
                if visibility == 2 and (ray['blocked'] or not point.endswith('pivot') or
                        info.get('reason') != 'unblocked_target_cap_and_local_contrast' or
                        info.get('cap_diameter_px', 0) < 3 or info.get('local_contrast_linear', 0) < .08):
                    raise ValueError('visible point lacks required evidence: ' + name + '/' + point)
        state = row['state']
        if state['exposure_ev'] not in (-.5, 0.0) or state['saturation'] not in (.65, 1.0):
            raise ValueError('unexpected tone stratum: ' + name)
        selected.append((row, image, label_path, annotation))
    if len(selected) != manifest.get('pose_label_count') or len(selected) < 30:
        raise ValueError('pose label count differs or is too small for three splits')
    return manifest, selected


def image_metrics(path):
    """Measure normalized sRGB luminance and HSV saturation on a 160x120 thumbnail."""
    with Image.open(path) as image:
        if image.size != (640, 480):
            raise ValueError('unexpected decoded image dimensions: ' + str(path))
        rgb = np.asarray(image.convert('RGB').resize((160, 120)), dtype=np.float32) / 255
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.divide(high-low, high, out=np.zeros_like(high), where=high > 1e-6)
    luminance = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    return {'median_srgb_luma': float(np.median(luminance)),
            'mean_hsv_saturation': float(np.mean(saturation))}


def allocate(selected, seed, metrics):
    groups = defaultdict(list)
    for item in selected:
        state = item[0]['state']
        measured = metrics[item[0]['id']]
        flags = (state['exposure_ev'] == -.5 and measured['median_srgb_luma'] <= .38,
                 state['saturation'] == .65 and measured['mean_hsv_saturation'] <= .08)
        groups[(state['exposure_ev'], state['saturation'], *flags)].append(item)
    rng = random.Random(seed)
    assignments = []
    for stratum, items in sorted(groups.items()):
        rng.shuffle(items)
        for index, item in enumerate(items):
            split = 'val' if index % 10 == 0 else 'test' if index % 10 == 1 else 'train'
            assignments.append((split, stratum, item))
    dark = {split for split, stratum, _ in assignments if stratum[2:] == (True, True)}
    if dark != {'train', 'val', 'test'}:
        raise ValueError('each split needs synthetic reduced-exposure and reduced-saturation samples')
    return assignments


def similarity_summary(assignments):
    """Flag low-resolution cross-split similarity without claiming independent scenes."""
    thumbnails, ids, splits = [], [], []
    for split, _, (row, image, _, _) in assignments:
        with Image.open(image) as picture:
            thumbnails.append(np.asarray(picture.convert('RGB').resize((32, 24)), dtype=np.float32).reshape(-1) / 255)
        ids.append(row['id'])
        splits.append(split)
    pixels = np.stack(thumbnails)
    count = 0
    examples = []
    for index in range(len(ids)):
        difference = np.mean(np.abs(pixels[index + 1:] - pixels[index]), axis=1)
        for offset in np.flatnonzero(difference <= .025):
            other = index + 1 + int(offset)
            if splits[index] != splits[other]:
                count += 1
                if len(examples) < 20:
                    examples.append({'a': ids[index], 'b': ids[other], 'mean_rgb_abs_difference': float(difference[offset])})
    return {'method': '32x24 RGB thumbnail mean absolute difference in normalized sRGB',
            'threshold': .025, 'cross_split_pair_count': count, 'example_pairs_max_20': examples,
            'scope': 'Approximate similarity screen only; same synthetic scene across all splits; not human review.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--license', required=True, help='Workspace file proving the initialization checkpoint license')
    ap.add_argument('--evidence', required=True, help='Workspace evidence for the exact initialization checkpoint')
    ap.add_argument('--additional-evidence', action='append', default=[],
                    help='Additional workspace evidence for initialization provenance; may be repeated')
    ap.add_argument('--initialization-source', required=True, help='Factual source of the V3 checkpoint')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    license_file, evidence = ROOT / args.license, ROOT / args.evidence
    license_ref, evidence_ref = relative(license_file), relative(evidence)
    additional_evidence = [ROOT / value for value in args.additional_evidence]
    additional_refs = [relative(path) for path in additional_evidence]
    if not args.initialization_source.strip():
        raise ValueError('initialization source must be factual and nonempty')
    config = yaml.safe_load(CONFIG.read_text(encoding='utf8'))
    if (config['data'] != OUTPUT.relative_to(ROOT).as_posix() + '/data.yaml' or
            config.get('keypoint_names') != list(POINTS)):
        raise ValueError('existing meter V4 config is not compatible with this output')
    weight = ROOT / config['model']
    weight_ref = relative(weight)
    if not weight.stat().st_size:
        raise ValueError('initialization checkpoint is empty')
    manifest, selected = selected_rows()
    metrics = {row['id']: image_metrics(image) for row, image, _, _ in selected}
    assignments = allocate(selected, args.seed, metrics)
    source_blend = resolve_asset(manifest['source_blend'])
    source_assets = [relative(resolve_asset(p)) for p in manifest['source_assets']]
    relative(source_blend)
    media = {image for _, image, _, _ in selected}
    media.update(ROOT / p for p in source_assets if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'})
    queue_media, queue_procedural, queue_code = gate.queue_inputs(ROOT)
    media.update(ROOT / p for p in queue_media)
    procedural = {BATCH / 'manifest.json', BATCH / 'samples.jsonl', CONFIG, ENTRY,
                  weight, license_file, evidence, source_blend}
    procedural.update(ROOT / p for p in source_assets if Path(p).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'})
    procedural.update(additional_evidence)
    procedural.update(path for _, _, label, annotation in selected for path in (label, annotation))
    procedural.update(ROOT / p for p in queue_procedural)
    code = [Path(__file__), *(ROOT / p for p in queue_code)]
    params = {'seed': args.seed, 'selected': len(selected), 'license': license_ref,
              'evidence': evidence_ref, 'additional_evidence': additional_refs,
              'initialization_source': args.initialization_source}
    with generation_guard('meter.prepare_pose_v4_delivery_20261113', sorted(media), [OUTPUT], params,
                          code_paths=code, procedural_inputs=sorted(procedural), root=ROOT) as run:
        if run.reused:
            print('Reused meter V4 split and delivery manifest:', run.value)
            return
        if OUTPUT.exists():
            raise ValueError('output already exists; refusing to overwrite')
        current_manifest, current_selected = selected_rows()
        if current_manifest != manifest or current_selected != selected:
            raise ValueError('batch changed while entering production guard')
        if any(image_metrics(image) != metrics[row['id']] for row, image, _, _ in selected):
            raise ValueError('batch image measurements changed while entering production guard')
        OUTPUT.mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            (OUTPUT / 'images' / split).mkdir(parents=True)
            (OUTPUT / 'labels' / split).mkdir(parents=True)
        similarity = similarity_summary(assignments)
        data_yaml = OUTPUT / 'data.yaml'
        data_yaml.write_text(yaml.safe_dump({'train': 'images/train', 'val': 'images/val',
            'test': 'images/test', 'kpt_shape': [8, 3], 'names': {0: 'meter'}}, sort_keys=False), encoding='utf8')
        if dataset_header(ROOT, data_yaml.relative_to(ROOT).as_posix())['names'] != ['meter']:
            raise ValueError('unexpected meter dataset header')
        samples = []
        for split, stratum, (row, image, label, annotation) in assignments:
            new_image = OUTPUT / 'images' / split / image.name
            new_label = OUTPUT / 'labels' / split / label.name
            shutil.copy2(image, new_image)
            shutil.copy2(label, new_label)
            conditions = []
            if stratum[2]:
                conditions.append('low_light')
            if stratum[3]:
                conditions.append('low_saturation')
            annotation_ref = relative(annotation)
            samples.append({'id': row['id'], 'image': relative(new_image), 'split': split,
                'source_group': 'station_parametric_v4_shared_scene', 'source_kind': 'blender_scene',
                'generation_method': 'blender_render', 'parent_ids': [], 'physical_assets': [],
                'source_assets': source_assets, 'mother_scene': relative(source_blend), 'image_size': [640, 480],
                'background': {'kind': 'blender', 'source_group': relative(source_blend),
                               'view_group': 'camera_z_' + str(row['state']['camera_z_fitted_scene_units']).replace('.', '_')},
                'image_processing': {'resize': 'none; native 640x480 render', 'compression': 'PNG lossless',
                                     'blur': 'none added'},
                'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                'review_evidence': annotation_ref, 'indexed_source_references': run.source_refs[relative(image)],
                'source_review_scope': ('Automated source, exact-file, and 32x24 thumbnail similarity checks; '
                                        'all splits share one scene; no human acceptance.'),
                'selected_tasks': ['meter'], 'tasks': {'meter': {'state': 'complete_positive',
                    'label': relative(new_label), 'review_evidence': annotation_ref,
                    'reviewer': 'synthetic geometry and automated source gate', 'human_accepted': False,
                    'strata_review': 'confirmed', 'strata': {'conditions': conditions},
                    'image_measurements': metrics[row['id']]}}})
        source = {'version': 1, 'purpose': 'synthetic meter V4 distribution source manifest',
                  'models': {'meter': {'data_yaml': relative(data_yaml), 'keypoint_names': list(POINTS)}},
                  'near_duplicate_pairs': [], 'samples': samples,
                  'cross_split_similarity_screen': similarity,
                  'limitations': ['All splits share one fitted synthetic station, face and scene.',
                                  'Weak or obscured projected landmarks are visibility 1; no human review.',
                                  'Checkpoint initialization license evidence does not establish rights for all V3 training material.',
                                  'Low-light means EV -0.5 and thumbnail median sRGB luma <= 0.38; '
                                  'low-saturation means saturation parameter 0.65 and thumbnail mean HSV '
                                  'saturation <= 0.08. These are synthetic image thresholds, not physical measurements.']}
        source_path = OUTPUT / 'source_manifest.json'
        source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        audit = gate.audit(ROOT, source, gate.build_queue(ROOT))
        if audit.get('status') != 'passed' or audit.get('assigned') != len(samples):
            raise ValueError('V4 source gate rejected split: ' + json.dumps(audit.get('errors', [])[:30], ensure_ascii=False))
        distribution = {'schema_version': 1, 'dataset_version': 'v4', 'model_key': 'meter',
                        'config': relative(CONFIG), 'entry': relative(ENTRY),
                        'source_manifest': relative(source_path),
                        'initialization': {'path': weight_ref, 'source': args.initialization_source,
                                           'license': license_ref, 'evidence': evidence_ref},
                        'additional_records': [relative(BATCH / 'manifest.json'), *additional_refs]}
        (OUTPUT / 'distribution_manifest.json').write_text(
            json.dumps(distribution, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        counts = {split: sum(item[0] == split for item in assignments) for split in ('train', 'val', 'test')}
        run.set_result({'counts': counts, 'excluded_without_box': manifest['completed_count'] - len(samples),
                        'distribution_manifest': relative(OUTPUT / 'distribution_manifest.json')})
        print('METER_V4_DELIVERY_READY', json.dumps(run.value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
