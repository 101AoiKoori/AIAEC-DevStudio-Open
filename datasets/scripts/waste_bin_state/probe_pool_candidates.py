"""Batch-diagnose bin placement on reviewed shared-pool frames without changing quota.

The shared contact mask is a search hint. This probe records exact source-pixel
footprints; task-scope labels and approximate size can qualify a background
at intake. A foreground render's split matters only when formally generating
the composite output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from datasets.scripts.common.frame_environment import DEFAULT_DATABASE
from datasets.scripts.common.paths import WORKSPACE, bin_state_foreground_dir
from datasets.scripts.common.review.placement_record import decode_placement_record
from datasets.scripts.waste_bin_state import composite_bin_real_scene as base
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as grouped


POOL = WORKSPACE / 'datasets/annotated/common/sources/background_pool_manifest.json'
RENDER = bin_state_foreground_dir() / 'manifest.json'
PRIOR_PROBES = (
    WORKSPACE / 'datasets/annotated/waste_bin_state/v5_new_foreground_placement_20261117/manifest.json',
    WORKSPACE / 'datasets/annotated/waste_bin_state/pool_fast_probe_20261117/manifest.json',
)
DEFAULT_OUT = WORKSPACE / 'datasets/annotated/waste_bin_state/pool_batch_probe_20261117'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _foreground(manifest, *, state_id):
    report = _read(manifest)
    matches = [sample for sample in report.get('samples', ())
               if sample.get('plan_state_id') == state_id and sample.get('candidate_label')]
    if report.get('status') != 'complete_candidates' or len(matches) != 1:
        raise ValueError(f'exactly one complete render state is required: {state_id}')
    sample = matches[0]
    if (sample.get('visible_bins') != ['blue'] or sample.get('bin_count') != 1 or
            sample.get('open_pose') == 'upright' or
            [(item.get('bin'), item.get('state'), item.get('label_eligible'))
             for item in sample.get('labels', ())] != [('blue', 'open', True)]):
        raise ValueError('selected render must be one correctly hinged blue open bin')
    image_path = Path(manifest).parent / sample['image']
    mask_path = Path(manifest).parent / sample['mask_visible']
    image = base.read_image(image_path)
    rgba = base.read_image(mask_path, cv2.IMREAD_UNCHANGED)
    if image.shape != (480, 640, 3) or rgba.shape != (480, 640, 4):
        raise ValueError('render image and mask must be 640x480 with a four-channel mask')
    rgb = rgba[:, :, [2, 1, 0]]
    visible = rgba[:, :, 3] >= 191
    for channel, value in enumerate(base.COLORS['blue']):
        visible &= rgb[:, :, channel] >= 191 if value else rgb[:, :, channel] <= 63
    if not visible.any():
        raise ValueError('blue bin pixels are absent from the visible mask')
    x, y, width, height = cv2.boundingRect(visible.astype(np.uint8))
    return {
        'crop': image[y:y + height, x:x + width].copy(),
        'masks': {'blue': visible[y:y + height, x:x + width].copy()},
        'union': visible[y:y + height, x:x + width].astype(np.uint8) * 255,
        'crop_xywh': [x, y, width, height],
        'split': report.get('split'),
        'image': image_path.relative_to(WORKSPACE).as_posix(),
        'mask': mask_path.relative_to(WORKSPACE).as_posix(),
        'mask_sha256': sha256(mask_path.read_bytes()).hexdigest(),
    }


def _overlaps(box, existing):
    return any(min(box[2], old[2]) > max(box[0], old[0]) and
               min(box[3], old[3]) > max(box[1], old[1]) for old in existing)


def probe_positions(ground, masks, existing, *, heights=(90, 75, 110), limit=3):
    """Find distinct footprints, then recheck their entire contact in source pixels."""
    floor = ground.candidate_mask
    if floor.shape != (480, 640) or not floor.any():
        return []
    mask = masks['blue']
    content = getattr(ground, 'content', (0, 0, 640, 480))
    desired_y = content[1] + .79 * (content[3] - content[1])
    proposals = []
    for height in heights:
        width = max(1, round(mask.shape[1] * height / mask.shape[0]))
        if width > 640 or height > 480:
            continue
        _, supports = grouped._group_supports(masks, [0, 0, width, height])
        feet = supports['blue']
        offsets = [point[0] for point in feet]
        lowest = feet[1][1]
        lefts = np.arange(max(0, content[0]), min(640 - width, content[2] - width) + 1, 4)
        if not len(lefts):
            continue
        for foot_y in range(max(lowest, content[1] + lowest), min(465, content[3] - 4) + 1, 4):
            good = np.logical_and.reduce([floor[foot_y, lefts + offset] for offset in offsets])
            top = foot_y - lowest
            for left in lefts[good]:
                left = int(left)
                box = [left, top, left + width, top + height]
                if _overlaps(box, existing):
                    continue
                center_x = left + (offsets[0] + offsets[-1]) / 2
                score = (abs(foot_y - desired_y) + .36 * abs(center_x - 320) +
                         .12 * abs(height - 90))
                proposals.append((score, [left, top, width, height]))
    selected = []
    for _, placement in sorted(proposals, key=lambda item: (item[0], item[1]))[:800]:
        _, supports = grouped._group_supports(masks, placement)
        points = supports['blue']
        contact_width = points[-1][0] - points[0][0]
        if any(ground.contact_status(point, contact_width=contact_width) != 'candidate'
               for point in points):
            continue
        center = points[1]
        if any(abs(center[0] - prior['supports']['blue'][1][0]) < 70 and
               abs(center[1] - prior['supports']['blue'][1][1]) < 40 for prior in selected):
            continue
        selected.append({'placement_xywh': placement, 'supports': supports,
                         'contact_width_px': contact_width})
        if len(selected) == limit:
            break
    return selected


def _prior_indices(paths):
    previous = set()
    for path in paths:
        if path.is_file():
            previous.update(item['index'] for item in _read(path).get('records', ()) if 'index' in item)
    return previous


def _pool_rows(conn, pool, previous):
    seen = set()
    skipped = Counter()
    for index, item in enumerate(pool['records']):
        match = conn.execute('''SELECT f.* FROM frame_paths p JOIN frames f USING(frame_id)
                                WHERE p.path = ?''', (item['image'],)).fetchone()
        if match is None:
            raise ValueError(f'pool image lacks frame identity: {item["image"]}')
        frame_id = match['frame_id']
        if frame_id in seen:
            skipped['duplicate_pool_path'] += 1
            continue
        seen.add(frame_id)
        decisions = list(conn.execute('''SELECT * FROM decisions WHERE frame_id = ?
                                         AND task = 'waste_bin_state' AND role = 'composite_background'
                                         AND current = 1''', (frame_id,)))
        if any(row['status'] == 'eligible' for row in decisions):
            skipped['already_qualified'] += 1
            continue
        split_rows = [row for row in decisions if row['split'] in ('train', 'val', 'test')
                      and row['status'] == 'unknown']
        if len({row['split'] for row in split_rows}) != 1:
            skipped['no_single_split_unknown'] += 1
            continue
        if index in previous:
            skipped['previously_probed'] += 1
            continue
        yield index, item, match, split_rows[0], skipped


def _current_label(conn, frame_id):
    rows = list(conn.execute('''SELECT evidence_id, payload_json FROM observations WHERE frame_id = ?
                                AND task = 'waste_bin_state' AND kind = 'label' AND current = 1''',
                             (frame_id,)))
    labels = [(row, json.loads(row['payload_json'])) for row in rows]
    labels = [(row, payload) for row, payload in labels
              if payload.get('scope', {}).get('kind') == 'frame']
    if not labels:
        return None, 'frame_bin_review_missing'
    complete = [(row, payload) for row, payload in labels
                if payload.get('completeness') == 'complete']
    if not complete:
        # A partial review may still provide known boxes. The geometry probe
        # must run, while production qualification remains unresolved.
        labels.sort(key=lambda pair: (pair[1].get('completeness') == 'partial',
                                      len(pair[1].get('instances') or [])), reverse=True)
        return labels[0], 'complete_target_scope_label_pending'
    signatures = {json.dumps((payload.get('presence'), payload.get('instances'),
                              payload.get('excluded_instances')), sort_keys=True, ensure_ascii=False)
                  for _, payload in complete}
    if len(signatures) != 1:
        return complete[0], 'conflicting_current_bin_labels'
    return complete[0], None


def known_bin_boxes(label, source_size):
    """Conservatively avoid every localized bin without requiring its state label."""
    width, height = source_size
    result = []
    for kind in ('instances', 'excluded_instances'):
        for item in label.get(kind) or []:
            box = item.get('xyxy')
            if (not isinstance(box, list) or len(box) != 4 or
                    any(type(value) is not int for value in box) or
                    not 0 <= box[0] < box[2] <= width or
                    not 0 <= box[1] < box[3] <= height):
                continue
            result.append({'bin_id': item.get('object_id'), 'annotation_group': kind,
                           'disposition': item.get('disposition'),
                           'reason': item.get('reason'),
                           'visibility': item.get('visibility'),
                           'source_box_xyxy': box,
                           'bbox_xyxy': grouped.output_box(box, source_size)})
    return result


def relative_scale_cue(placement, known_bins, *, content_height):
    """Compare same-class pixel heights at approximate vertical depth, not metric scale."""
    new_foot_y = placement[1] + placement[3]
    references = []
    for item in known_bins:
        x1, y1, x2, y2 = item['bbox_xyxy']
        if y2 >= new_foot_y:
            continue
        old_height = y2 - y1
        expected = old_height * (1 + (new_foot_y - y2) / content_height)
        references.append({'bin_id': item['bin_id'], 'box_height_px': old_height,
                           'foot_y': y2, 'rough_nearer_height_px': round(expected, 1)})
    if not references:
        return {'status': 'no_comparable_existing_bin', 'basis': 'vertical_footpoint_heuristic'}
    nearest = max(references, key=lambda item: item['rough_nearer_height_px'])
    ratio = nearest['rough_nearer_height_px'] / placement[3]
    return {'status': 'rough_scale_too_small' if ratio > 1.08 else 'no_small_scale_warning',
            'basis': 'vertical_footpoint_heuristic_not_camera_calibration',
            'new_height_px': placement[3], 'new_foot_y': new_foot_y,
            'reference': nearest, 'rough_expected_to_actual_ratio': round(ratio, 3)}


def diagnose(*, pool=POOL, database=DEFAULT_DATABASE, render=RENDER,
             state_id='blue_open_normal', prior_probes=PRIOR_PROBES, max_previews=18):
    """Return a complete read-only candidate matrix and the transient preview inputs."""
    foreground = _foreground(render, state_id=state_id)
    source = _read(pool)
    if source.get('role') != 'shared_background_pool':
        raise ValueError('expected the shared background pool manifest')
    previous = _prior_indices(prior_probes)
    conn = sqlite3.connect(f'file:{Path(database).resolve().as_posix()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    records, skipped, previews = [], Counter(), []
    try:
        conn.execute('BEGIN')
        for index, item, frame, decision, skipped in _pool_rows(conn, source, previous):
            frame_id = frame['frame_id']
            record = {
                'index': index, 'frame_id': frame_id, 'image': item['image'],
                'content_sha256': frame['content_sha256'],
                'source_image_size': [frame['width'], frame['height']],
                'parent_group': frame['parent_group'], 'split': decision['split'],
                'foreground_split': foreground['split'],
                'formal_generation_split_compatible': decision['split'] == foreground['split'],
                'decision_record_id': decision['record_id'],
                'status': 'rejected', 'reason_codes': [], 'preview': None,
                'formal_label_ready': False,
            }
            (label, label_issue) = _current_label(conn, frame_id)
            known = []
            if label_issue:
                record['reason_codes'].append(label_issue)
            if label is not None:
                label_row, label_data = label
                record['label_evidence_id'] = label_row['evidence_id']
                record['label_presence'] = label_data.get('presence')
                record['label_completeness'] = label_data.get('completeness')
                record['excluded_instances'] = [
                    {key: item.get(key) for key in
                     ('object_id', 'disposition', 'reason', 'visibility', 'xyxy')}
                    for item in label_data.get('excluded_instances') or []]
                known = known_bin_boxes(label_data, [frame['width'], frame['height']])
                record['localized_existing_bins'] = known
                if any(item.get('reason') != 'distant_non_target' and
                       item.get('disposition') not in ('non_target', 'background_only')
                       for item in label_data.get('excluded_instances') or []):
                    record['reason_codes'].append('excluded_instance_scope_review_pending')
                if label_issue is None:
                    try:
                        grouped.existing_bin_labels({'label': label_data,
                                                     'image_size': [frame['width'], frame['height']]})
                        record['formal_label_ready'] = True
                    except ValueError as error:
                        record['formal_label_ready'] = False
                        record['reason_codes'].append('generator_target_label_format_pending')
                        record['detail'] = str(error)
                    if 'excluded_instance_scope_review_pending' in record['reason_codes']:
                        record['formal_label_ready'] = False
                else:
                    record['formal_label_ready'] = False
            spatial = conn.execute('''SELECT evidence_id, payload_json FROM observations WHERE frame_id = ?
                                      AND task = '' AND kind = 'geometry' AND scope = 'shared_background'
                                      AND current = 1''', (frame_id,)).fetchone()
            if spatial is None:
                record['reason_codes'].append('shared_spatial_observation_missing')
            if spatial is None:
                records.append(record)
                continue
            record['shared_evidence_id'] = spatial['evidence_id']
            observation = json.loads(spatial['payload_json'])
            try:
                ground = grouped.output_ground(decode_placement_record(observation['placement_record']),
                                               observation['image_size'])
            except (KeyError, ValueError) as error:
                record['reason_codes'].append('shared_contact_mask_unavailable')
                record['detail'] = str(error)
                records.append(record)
                continue
            record['candidate_contact_pixels'] = int(np.count_nonzero(ground.candidate_mask))
            record['placements'] = probe_positions(
                ground, foreground['masks'], [row['bbox_xyxy'] for row in known])
            if not record['placements']:
                record['reason_codes'].append('no_source_verified_nonoverlap_footprint')
                records.append(record)
                continue
            content = getattr(ground, 'content', (0, 0, 640, 480))
            record['relative_scale'] = relative_scale_cue(
                record['placements'][0]['placement_xywh'], known,
                content_height=content[3] - content[1])
            if record['relative_scale']['status'] == 'rough_scale_too_small':
                record['reason_codes'].append('rough_relative_scale_too_small')
            image_bytes = (WORKSPACE / item['image']).read_bytes()
            if sha256(image_bytes).hexdigest() != frame['content_sha256']:
                record['placements'] = []
                record['reason_codes'].append('image_bytes_changed_since_intake')
                records.append(record)
                continue
            record['status'] = 'machine_footprint_candidate'
            record['reason_codes'].append('task_background_qualification_not_yet_written')
            if not record['formal_generation_split_compatible']:
                record['reason_codes'].append('formal_output_requires_matching_foreground_split')
            records.append(record)
            previews.append((record, ground))
    finally:
        conn.close()
    # Sample parent/split groups evenly, with their broadest contact first.
    groups = defaultdict(list)
    for row, ground in previews:
        groups[(row['parent_group'], row['split'])].append((row, ground))
    for rows in groups.values():
        rows.sort(key=lambda item: -item[0]['candidate_contact_pixels'])
    chosen = []
    while len(chosen) < max_previews and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < max_previews:
                chosen.append(groups[key].pop(0))
    return records, skipped, chosen, foreground


def run(*, out=DEFAULT_OUT, max_previews=18, **kwargs):
    out = Path(out).resolve()
    allowed = WORKSPACE / 'datasets/annotated/waste_bin_state'
    if not out.is_relative_to(allowed) or (out.exists() and any(out.iterdir())):
        raise ValueError('output must be a new empty directory under annotated/waste_bin_state')
    records, skipped, previews, foreground = diagnose(max_previews=max_previews, **kwargs)
    out.mkdir(parents=True, exist_ok=True)
    for row, ground in previews:
        raw = (WORKSPACE / row['image']).read_bytes()
        if sha256(raw).hexdigest() != row['content_sha256']:
            raise ValueError(f'preview frame changed since intake: {row["image"]}')
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or list(frame.shape[:2][::-1]) != row['source_image_size']:
            raise ValueError(f'preview frame has unexpected pixels: {row["image"]}')
        frame = grouped.output_frame(frame)
        placement = row['placements'][0]['placement_xywh']
        # _group_composite consumes an output-ground object and a source-size mapping.
        original_size = row['source_image_size']
        background = {'source_image_size': original_size,
                      'output_content_xyxy': grouped.output_content_rectangle(original_size)}
        result, actual, _, supports, scale = grouped._group_composite(
            frame, foreground['crop'], foreground['union'], foreground['masks'],
            placement, background, ground, {})
        path = out / f'pool_{row["index"]:03d}.png'
        ok, encoded = cv2.imencode('.png', result)
        if not ok:
            raise ValueError(f'cannot encode preview for pool index {row["index"]}')
        encoded.tofile(str(path))
        row['preview'] = path.relative_to(WORKSPACE).as_posix()
        row['preview_placement_xywh'] = actual
        row['preview_supports_output_xy'] = supports
        row['preview_scale'] = scale
    counts = Counter(row['status'] for row in records)
    reasons = Counter(reason for row in records for reason in row['reason_codes'])
    by_split = defaultdict(Counter)
    by_parent = defaultdict(Counter)
    reasons_by_split = defaultdict(Counter)
    reasons_by_parent = defaultdict(Counter)
    for row in records:
        by_split[row['split']][row['status']] += 1
        by_parent[row['parent_group']][row['status']] += 1
        for reason in row['reason_codes']:
            reasons_by_split[row['split']][reason] += 1
            reasons_by_parent[row['parent_group']][reason] += 1
    manifest = {
        'schema_version': 1, 'scope': 'ground_footprint_diagnostic_only_no_quota_write',
        'qualification_note': ('machine_footprint_candidate proves only source-pixel contact and '
                               'non-overlap with localized bins; it does not prove suitable scale, '
                               'visual clearance, task labels, or formal split compatibility'),
        'foreground': {key: foreground[key] for key in ('image', 'mask', 'mask_sha256', 'split', 'crop_xywh')},
        'summary': {'considered': len(records), 'statuses': dict(counts),
                    'reason_codes': dict(reasons), 'skipped': dict(skipped),
                    'previews': len(previews),
                    'by_split': {key: dict(value) for key, value in by_split.items()},
                    'by_parent': {key: dict(value) for key, value in by_parent.items()},
                    'reasons_by_split': {key: dict(value) for key, value in reasons_by_split.items()},
                    'reasons_by_parent': {key: dict(value) for key, value in reasons_by_parent.items()}},
        'records': records,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--max-previews', type=int, default=18)
    args = parser.parse_args()
    if args.max_previews < 0 or args.max_previews > 100:
        raise ValueError('max-previews must be between 0 and 100')
    result = run(out=args.out, max_previews=args.max_previews)
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
