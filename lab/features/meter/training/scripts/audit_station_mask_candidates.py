"""Audit station image/mask pairs and unknown points without declaring pose truth."""
import argparse
import collections
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
EXPORTS = (ROOT / 'datasets/exports/meter').resolve()
POINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
          'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')


def png_size(path):
    with path.open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'invalid PNG: {path}')
    return list(struct.unpack('>II', header[16:24]))


def _execute(args):
    batch = args.batch.resolve()
    if not batch.is_relative_to(EXPORTS) or batch == EXPORTS:
        ap.error('batch must be under datasets/exports/meter')
    output = batch / 'mask_audit.json'
    if output.exists():
        ap.error('audit output exists; refusing to overwrite')
    manifest = json.loads((batch / 'manifest.json').read_text(encoding='utf8'))
    samples = [json.loads(line) for line in (batch / 'samples.jsonl').read_text(encoding='utf8').splitlines()]
    errors, statuses, states = [], collections.Counter(), collections.defaultdict(collections.Counter)
    ids = set()
    for sample in samples:
        sid = sample['id']
        if sid in ids:
            errors.append(f'{sid}: duplicate id')
        ids.add(sid)
        if sample.get('split') != 'unassigned' or sample.get('image_size') != [640, 480]:
            errors.append(f'{sid}: unexpected split or size')
        if sample.get('keypoint_visibility') != dict.fromkeys(POINTS, 'unknown'):
            errors.append(f'{sid}: points not all unknown')
        evidence = sample.get('dial_raster_evidence', {})
        statuses[evidence.get('bbox_status', 'missing')] += 1
        for key in ('visible', 'isolated'):
            entry = evidence.get(key, {})
            relative = Path(entry.get('path', ''))
            path = (batch / relative).resolve()
            if not path.is_relative_to(batch) or not path.is_file() or png_size(path) != [640, 480]:
                errors.append(f'{sid}: bad {key} mask')
        image = (batch / sample['image']).resolve()
        if not image.is_relative_to(batch) or not image.is_file() or png_size(image) != [640, 480]:
            errors.append(f'{sid}: bad image')
        annotation = batch / 'annotations' / (sid + '.json')
        if not annotation.is_file() or json.loads(annotation.read_text(encoding='utf8')) != sample:
            errors.append(f'{sid}: record differs from annotation')
        box = evidence.get('bbox_xyxy_pixels')
        if box is not None and not (0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 480):
            errors.append(f'{sid}: invalid box')
        if not 0 <= evidence.get('visible_fraction_candidate', -1) <= 1:
            errors.append(f'{sid}: invalid overlap')
        for key in ('temperature_bin', 'humidity_bin', 'ground_angle_deg', 'camera_z_fitted_scene_units',
                    'exposure_ev', 'saturation', 'side_light_shift_scene_units'):
            states[key][str(sample['state'][key])] += 1
    if manifest.get('status') != 'complete_candidates' or manifest.get('completed_count') != len(samples):
        errors.append('manifest not complete or count mismatch')
    pose_files = list(batch.rglob('*.txt'))
    if pose_files:
        errors.append('unexpected text label files')
    report = {'status': 'pass' if not errors else 'fail', 'samples': len(samples), 'bbox_statuses': statuses,
              'state_counts': states, 'errors': errors, 'pose_labels': 0,
              'scope': 'file/metadata and synthetic silhouette checks only; no real-world or pose validation'}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('batch', type=Path)
    args = ap.parse_args()
    batch = args.batch.resolve()
    records = [json.loads(line) for line in (batch / 'samples.jsonl').read_text(encoding='utf8').splitlines()]
    media = []
    metadata = [batch / 'manifest.json', batch / 'samples.jsonl']
    for item in records:
        media.append(batch / item['image'])
        metadata.append(batch / 'annotations' / (item['id'] + '.json'))
        media.extend(batch / item['dial_raster_evidence'][kind]['path'] for kind in ('visible', 'isolated'))
    output = batch / 'mask_audit.json'
    with generation_guard('meter.audit_station_mask:' + batch.relative_to(ROOT).as_posix(),
                          media, [output], {}, code_paths=[Path(__file__)],
                          procedural_inputs=metadata, root=ROOT) as run:
        if run.reused:
            print('REUSED', output)
            return
        if [json.loads(line) for line in (batch / 'samples.jsonl').read_text(encoding='utf8').splitlines()] != records:
            raise RuntimeError('samples changed before execution')
        _execute(args)


if __name__ == '__main__':
    main()
