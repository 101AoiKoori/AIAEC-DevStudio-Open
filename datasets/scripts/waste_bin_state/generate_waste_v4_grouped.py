"""Compose planned 640x480 bin-state candidates on selected backgrounds.

Sticker generators continue to import the shared helper functions below.
"""
import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

_WORKSPACE = Path(__file__).resolve().parents[3]
_TASK_HELPERS = _WORKSPACE / 'datasets/scripts/waste_bin_state'
sys.path.insert(0, str(_WORKSPACE))
from datasets.scripts.waste_bin_state import composite_bin_real_scene as base
from datasets.scripts.waste_bin_state.bin_render_contract import STATE_CONTRACT

ROOT = base.ROOT
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from camera_tone import apply_tone, source_role  # noqa: E402

sys.path.insert(0, str(ROOT))
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from lab.scripts.data import v4_source_gate as source_gate  # noqa: E402
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import sample_dir, split_dir

BG_MANIFEST = source_path('waste_bin_state', 'background_manifest')
DESTINATION = ROOT / 'datasets/exports/waste_bin_state'
# These source illustrations are too ambiguous in the 28-pixel audit.  Keep
# their boxes as partial candidates until the actual output is reviewed.
UNCERTAIN = {0, 2, 11, 13, 14, 19, 20, 23}
COLORS = tuple(base.COLORS)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError('cannot encode: ' + str(path))
    encoded.tofile(str(path))


def box_line(cls, box):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
        raise ValueError('invalid visible bounding box')
    return f'{cls} {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}'


def ground_placement(crop, masks, backgrounds, rng, readable_single=False, height_range=None):
    """Fit each bin's visible lower edge within a reviewed floor polygon."""
    for attempt in range(300):
        record, background, floor = rng.choice(backgrounds)
        target_height = rng.randint(*height_range) if height_range else (
            rng.randint(165, 170) if readable_single else rng.randint(125, 165))
        scale = min(target_height / crop.shape[0], 600 / crop.shape[1])
        width = max(1, round(crop.shape[1] * scale))
        height = max(1, round(crop.shape[0] * scale))
        polygon = np.asarray(record['ground_polygon_xy'], dtype=np.int32)
        x0, y0, w, h = cv2.boundingRect(polygon)
        if w < width + 12 or h < 20:
            continue
        left = rng.randint(max(0, x0), min(640-width, x0+w-width))
        bottom_min = max(height + 1, y0 + 10, 410)
        bottom_max = min(465, y0 + h - 7)
        if bottom_min > bottom_max:
            continue
        bottom = rng.randint(bottom_min, bottom_max)
        top = bottom-height
        scaled_masks, supports = {}, {}
        accepted = True
        for color, mask in masks.items():
            scaled = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(scaled)
            if not len(ys):
                accepted = False
                break
            lowest = int(ys.max())
            foot_xs = xs[ys >= lowest-2]
            lo, center, hi = (int(round(v)) for v in np.percentile(foot_xs, [10, 50, 90]))
            points = [[left+x, top+lowest] for x in (lo, center, hi)]
            if any(not (0 <= x < 640 and 0 <= y < 480 and floor[y, x]) for x, y in points):
                accepted = False
                break
            scaled_masks[color] = scaled
            supports[color] = points
        if accepted:
            return record, background, floor, left, top, width, height, scaled_masks, supports
    raise ValueError('No floor-valid placement after 300 attempts; review background polygons')


def add_contact_shadows(background, floor, scaled_masks, supports):
    """Add a modest synthetic contact shadow; not a measured light model."""
    shadow = np.zeros(background.shape[:2], dtype=np.uint8)
    for color, points in supports.items():
        _, _, w, h = cv2.boundingRect(scaled_masks[color])
        center = points[1]
        cv2.ellipse(shadow, (center[0], min(479, center[1]+1)),
                    (max(3, round(w*.32)), max(2, round(h*.025))), 0, 0, 360, 100, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 3).astype(np.float32) / 255 * .45
    shadow *= floor.astype(np.float32)
    return np.clip(background.astype(np.float32) * (1-shadow[:, :, None]), 0, 255).astype(np.uint8)


def load_quantity_sources(paths, *, split_counts=None, source_splits=None):
    """Read quantity renders; explicit splits require a matching manifest split."""
    paths = tuple(paths)
    if source_splits is not None:
        source_splits = tuple(source_splits)
        if len(source_splits) != len(paths) or any(s not in ('train', 'val', 'test') for s in source_splits):
            raise ValueError('one explicit split is required for each render source')
    if split_counts is None:
        required = {s: (1, 2, 3, 4) for s in ('train', 'val', 'test')}
    else:
        if not isinstance(split_counts, dict) or not split_counts:
            raise ValueError('split_counts must be a nonempty split-to-bin-count mapping')
        required = {}
        for split, counts in split_counts.items():
            if split not in ('train', 'val', 'test') or not isinstance(counts, (list, tuple)) or not counts or any(type(n) is not int or n not in (1, 2, 3, 4) for n in counts):
                raise ValueError('invalid required split or bin count')
            required[split] = tuple(counts)
    pools = defaultdict(list)
    samples = []
    for source_index, path in enumerate(paths):
        report = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        planned_source = source_splits is not None and report.get('label_contract_id') == STATE_CONTRACT
        if report['status'] != 'complete_candidates' or not (report.get('quantity_plan') or planned_source):
            raise ValueError('Quantity Blender render set is incomplete')
        if source_splits is not None and report.get('split') != source_splits[source_index]:
            raise ValueError('source manifest split does not match the declared split')
        samples.extend((source_index, path, sample) for sample in report['samples'])
    for source_index, path, sample in samples:
        if not sample['candidate_label']:
            continue
        group = sample.get('group')
        split = (source_splits[source_index] if source_splits is not None else
                 'train' if source_index == 0 and group < 8 else
                 'val' if source_index == 1 and 8 <= group <= 12 else
                 'test' if source_index == 2 and 16 <= group <= 20 else None)
        if split is None:
            continue
        image = base.read_image(path / sample['image'])
        mask = base.read_image(path / sample['mask_visible'], cv2.IMREAD_UNCHANGED)
        if image.shape != (480, 640, 3) or mask.shape != (480, 640, 4):
            raise ValueError('Expected 640x480 Blender images and masks')
        rgb = mask[:, :, [2, 1, 0]]
        masks = {}
        for name in sample['visible_bins']:
            hit = mask[:, :, 3] >= 191
            for channel, value in enumerate(base.COLORS[name]):
                hit &= rgb[:, :, channel] >= 191 if value else rgb[:, :, channel] <= 63
            if not hit.any():
                raise ValueError('Missing visible Blender mask: ' + name)
            masks[name] = hit
        if len(masks) != sample['bin_count']:
            raise ValueError('Visible bin count does not match actual render state')
        union = np.logical_or.reduce(list(masks.values()))
        x, y, w, h = cv2.boundingRect(union.astype(np.uint8))
        states = {item['bin']: item['state'] for item in sample['labels']}
        if set(states) != set(masks):
            raise ValueError('Rendered bin and candidate label mismatch')
        source = {'image': image[y:y+h, x:x+w].copy(),
                  'masks': {name: bit[y:y+h, x:x+w].copy() for name, bit in masks.items()},
                  'name': sample['image'], 'source_directory': rel(path), 'group': group, 'trajectory': sample.get('trajectory'),
                  'source_union_xywh': [x, y, w, h],
                  'open_pose': sample['open_pose'], 'states': states,
                  'source_code': ''.join('1' if state == 'open' else '0' for state in
                      ([sample['states'][color] for color in COLORS] if isinstance(sample['states'], dict) else sample['states']))}
        pools[(split, sample['bin_count'])].append(source)
    if any(not pools[(s, n)] for s, counts in required.items() for n in counts):
        raise ValueError('A split lacks a fully eligible render for a required bin count')
    return pools


def load_empty_sources(path):
    report = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if report['status'] != 'complete_candidates' or not report.get('empty_plan'):
        raise ValueError('Empty Blender render set is incomplete')
    pools = defaultdict(list)
    for sample in report['samples']:
        group = sample['group']
        split = 'train' if group < 30 else 'val' if 32 <= group < 35 else 'test' if 40 <= group < 43 else None
        if split is None:
            continue
        if sample['bin_count'] != 0 or sample['visible_bins'] or sample['labels']:
            raise ValueError('Empty source has visible bins or labels')
        image_path = path / sample['image']
        image = base.read_image(image_path)
        if image.shape != (480, 640, 3):
            raise ValueError('Expected 640x480 empty render')
        item = {'id': 'empty_blender_'+str(group), 'image': rel(image_path), 'parent_image': rel(image_path),
                'split': split, 'mother_scene': report['scene_file'], 'trajectory': sample['trajectory'],
                'ground_polygon_status': 'not_applicable_empty_Blender_scene'}
        pools[split].append((item, image, None))
    if any(len(pools[s]) != n for s, n in (('train', 30), ('val', 3), ('test', 3))):
        raise ValueError('Missing required distinct empty render frames')
    return pools


def _workspace_file(value):
    if not isinstance(value, str) or not value:
        raise ValueError('source must be a nonempty workspace path')
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('source file is missing or outside workspace: ' + value)
    return path


def _placement(value):
    if (not isinstance(value, list) or len(value) != 4 or
            any(type(n) is not int for n in value)):
        raise ValueError('placement must be four integer pixels')
    left, top, width, height = value
    if left < 0 or top < 0 or width < 1 or height < 1 or left + width > 640 or top + height > 480:
        raise ValueError('placement outside output frame')


def _plan_region(background, params):
    """Allow a plan to narrow, but never enlarge, the declared ground region."""
    if 'ground_polygon_xy' not in params:
        return background
    polygon = params['ground_polygon_xy']
    if (not isinstance(polygon, list) or len(polygon) < 3 or
            any(not isinstance(p, list) or len(p) != 2 or
                any(type(v) is not int for v in p) or
                not (0 <= p[0] < 640 and 0 <= p[1] < 480) for p in polygon)):
        raise ValueError('ground polygon must contain at least three image points')
    allowed = np.zeros((480, 640), dtype=np.uint8)
    requested = np.zeros_like(allowed)
    cv2.fillPoly(allowed, [np.asarray(background['ground_polygon_xy'], dtype=np.int32)], 1)
    cv2.fillPoly(requested, [np.asarray(polygon, dtype=np.int32)], 1)
    if cv2.contourArea(np.asarray(polygon, dtype=np.int32)) <= 0 or np.any(requested > allowed):
        raise ValueError('planned ground polygon must lie within the background region')
    return {**background, 'ground_polygon_xy': polygon,
            'parent_ground_polygon_xy': background['ground_polygon_xy']}


def _planned_inputs(plan):
    """Resolve and validate source metadata without composing or changing pixels."""
    if plan['model_key'] != 'waste_bin_state' or plan['constraints'] != {
            'resolution': [640, 480], 'label_mode': 'candidate', 'ground_placement': 'polygon'}:
        raise ValueError('unsupported grouped-composite model or constraints')
    frames = json.loads(BG_MANIFEST.read_text(encoding='utf8'))['frames']
    by_id = {row['id']: row for row in frames}
    if len(by_id) != len(frames):
        raise ValueError('duplicate background id')
    media, procedural = set(), {BG_MANIFEST}
    queue_media, queue_procedural, queue_code = source_gate.queue_inputs(ROOT)
    media.update(queue_media)
    procedural.update(queue_procedural)
    jobs, seen = [], []
    for state in plan['strata']:
        if state['conditions'] or state['planned_count'] > 1:
            raise ValueError('unsupported conditions or repeated exact state')
        bg, target, labels, aug, params = (state[key] for key in
                                          ('background', 'target', 'labels', 'augmentation', 'parameters'))
        if set(bg) != {'manifest', 'id'} or bg['manifest'] != rel(BG_MANIFEST) or bg['id'] not in by_id:
            raise ValueError('background must name a known frame in the grounded manifest')
        background = by_id[bg['id']]
        if background['split'] != plan['target_split']:
            raise ValueError('background split does not match target split')
        if (set(target) not in ({'render_manifest', 'render_image', 'bins'},
                                {'render_manifest', 'render_image', 'bins', 'bin_ids'}) or
                not isinstance(target['bins'], dict) or not target['bins']):
            raise ValueError('target must specify a nonempty set of bins and a render source')
        if any(name not in COLORS or value not in ('open', 'closed') for name, value in target['bins'].items()):
            raise ValueError('invalid bin states')
        if 'bin_ids' in target:
            ids = target['bin_ids']
            if (not isinstance(ids, dict) or set(ids) != set(target['bins']) or
                    any(not isinstance(v, str) or not v.strip() for v in ids.values()) or
                    len(set(ids.values())) != len(ids)):
                raise ValueError('bin_ids must give each bin a unique nonempty identity')
        source_manifest = _workspace_file(target['render_manifest'])
        report = json.loads(source_manifest.read_text(encoding='utf8'))
        matching = [row for row in report['samples'] if row['image'] == target['render_image']]
        if report.get('status') != 'complete_candidates' or len(matching) != 1:
            raise ValueError('source render is incomplete or image is not unique')
        sample = matching[0]
        observed = {item['bin']: item['state'] for item in sample['labels']}
        if (len(observed) != len(sample['labels']) or observed != target['bins'] or
                set(sample.get('visible_bins', ())) != set(observed) or
                not sample.get('candidate_label') or not sample.get('mask_visible') or
                not sample.get('mask_state_visible') or any(item.get('label_eligible') is not True for item in sample['labels'])):
            raise ValueError('source render lacks complete candidate labels and state evidence')
        source_image = _workspace_file(rel(source_manifest.parent / sample['image']))
        source_mask = _workspace_file(rel(source_manifest.parent / sample['mask_visible']))
        state_mask = _workspace_file(rel(source_manifest.parent / sample['mask_state_visible']))
        candidate_label = _workspace_file(rel(source_manifest.parent / sample['candidate_label']))
        background_image = _workspace_file(background['image'])
        parent = _workspace_file(background['parent_image'])
        if state['sources'] != [rel(background_image), rel(source_image), rel(source_mask)]:
            raise ValueError('sources must exactly specify background, render and visible mask')
        if labels != {'mode': 'candidate', 'open': 0, 'closed': 1}:
            raise ValueError('unsupported label contract')
        if set(aug) != {'value_gain', 'saturation_gain', 'shadow'} or aug['shadow'] != 'simulated_gaussian':
            raise ValueError('unsupported augmentation')
        for key in ('value_gain', 'saturation_gain'):
            value = aug[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 2:
                raise ValueError('invalid ' + key)
        keys = set(params) - {'ground_polygon_xy'}
        if keys not in ({'placement_xywh'}, {'placements_xywh', 'back_to_front'}):
            raise ValueError('unsupported composite parameters')
        background = _plan_region(background, params)
        if 'placement_xywh' in params:
            _placement(params['placement_xywh'])
        else:
            placements, order = params['placements_xywh'], params['back_to_front']
            if (not isinstance(placements, dict) or set(placements) != set(target['bins']) or
                    not isinstance(order, list) or any(not isinstance(name, str) for name in order) or
                    len(order) != len(placements) or
                    set(order) != set(placements)):
                raise ValueError('independent placements and depth order must cover each bin once')
            for placement in placements.values():
                _placement(placement)
            if any(item.get('visible_fraction') != 1.0 or item.get('in_frame') is not True
                   for item in sample['labels']):
                raise ValueError('independent placement requires fully visible source bins')
        identity = {key: state[key] for key in ('background', 'target', 'labels', 'augmentation', 'parameters', 'sources')}
        if state['planned_count'] and identity in seen:
            raise ValueError('duplicate composite state')
        if state['planned_count']:
            seen.append(identity)
            jobs.append((state, background, sample, source_image, source_mask))
        procedural.update((source_manifest, candidate_label))
        media.update((background_image, parent, source_image, source_mask, state_mask))
        if background.get('video'):
            media.add(_workspace_file(background['video']))
        if report.get('scene_file'):
            procedural.add(_workspace_file(report['scene_file']))
    if len(jobs) != plan['budget']:
        raise ValueError('planned composite count does not equal budget')
    if any(row['parent_image'].startswith(source_gate.SITE_SEQUENCE + '/') for _, row, *_ in jobs):
        procedural.add(ROOT / source_gate.SITE_SEQUENCE / 'ffmpeg.log')
    if any(row['parent_image'].startswith(source_gate.PERSON_SEQUENCE + '/') for _, row, *_ in jobs):
        procedural.add(ROOT / source_gate.PERSON_SEQUENCE / 'frames_probe.json')
    return jobs, sorted(media), sorted(procedural), queue_code


def _independent_composite(frame, foreground, masks, state_rgba, floor, job):
    """Place complete source bins separately; reject lost state evidence."""
    if state_rgba.shape != (*foreground.shape[:2], 4):
        raise ValueError('invalid state mask dimensions')
    state_rgb = state_rgba[:, :, [2, 1, 0]]
    layers, supports = {}, {}
    for color, mask in masks.items():
        evidence = state_rgba[:, :, 3] >= 191
        for channel, value in enumerate(base.COLORS[color]):
            evidence &= state_rgb[:, :, channel] >= 191 if value else state_rgb[:, :, channel] <= 63
        if not evidence.any() or np.any(evidence & ~mask):
            raise ValueError('state evidence must be present within the source bin mask')
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        left, top, width, height = job['parameters']['placements_xywh'][color]
        if abs(width / w - height / h) > max(1 / w, 1 / h):
            raise ValueError('placement stretches the render crop')
        scaled = cv2.resize(mask[y:y+h, x:x+w].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        state = cv2.resize(evidence[y:y+h, x:x+w].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.nonzero(scaled)
        if not len(ys) or not state.any():
            raise ValueError('resized bin or state evidence is empty')
        lowest = int(ys.max())
        points = [[left + int(round(v)), top + lowest] for v in np.percentile(xs[ys >= lowest-2], [10, 50, 90])]
        if any(not floor[py, px] for px, py in points):
            raise ValueError('planned bin support falls outside ground polygon')
        supports[color] = points
        full = np.zeros((480, 640), dtype=bool)
        full[top:top+height, left:left+width] = scaled
        full_state = np.zeros_like(full)
        full_state[top:top+height, left:left+width] = state
        patch = cv2.resize(foreground[y:y+h, x:x+w], (width, height), interpolation=cv2.INTER_AREA)
        layers[color] = (full, full_state, patch, (left, top, width, height), scaled)
    visible, covered = {}, np.zeros((480, 640), dtype=bool)
    for color in reversed(job['parameters']['back_to_front']):
        mask, state, *_ = layers[color]
        if np.any(state & covered):
            raise ValueError('independent placement obscures state evidence')
        visible[color] = mask & ~covered
        if not visible[color].any():
            raise ValueError('independent placement hides a bin')
        covered |= mask
    scaled_masks = {color: layer[4].astype(np.uint8) for color, layer in layers.items()}
    frame = add_contact_shadows(frame, floor, scaled_masks, supports)
    for color in job['parameters']['back_to_front']:
        _, _, patch, (left, top, width, height), scaled = layers[color]
        region = frame[top:top+height, left:left+width]
        region[scaled] = patch[scaled]
    boxes = []
    for color in job['target']['bins']:
        x, y, w, h = cv2.boundingRect(visible[color].astype(np.uint8))
        boxes.append({'bin': color, 'state': job['target']['bins'][color], 'bbox_xyxy': [x, y, x+w, y+h]})
    return frame, boxes, supports


def _generate(args, run, jobs):
    out = (ROOT / args.out).resolve()
    if out not in {sample_dir('waste_bin_state').resolve(), split_dir('waste_bin_state', run.plan['target_split']).resolve()}:
        raise ValueError('output must be the current waste_bin_state sample or declared split directory')
    if out.exists() and any(out.iterdir()):
        raise ValueError('output is not empty')
    known = build_queue(ROOT)
    split = run.plan['target_split']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(parents=True)
    (out / 'labels').mkdir(parents=True)
    report = {'status': 'running', 'split': split, 'model_key': 'waste_bin_state', 'records': []}
    try:
        for index, (job, background_record, source, image_path, mask_path) in enumerate(jobs):
            exposure_role, exposure_reason = source_role(ROOT, ROOT / background_record['parent_image'], known)
            if exposure_role == 'regression':
                raise ValueError('background is restricted to regression: ' + background_record['parent_image'])
            foreground = base.read_image(image_path)
            mask_rgba = base.read_image(mask_path, cv2.IMREAD_UNCHANGED)
            frame = base.read_image(ROOT / background_record['image'])
            if frame.shape != (480, 640, 3) or mask_rgba.shape != (*foreground.shape[:2], 4):
                raise ValueError('invalid frame, render or mask dimensions')
            rgb = mask_rgba[:, :, [2, 1, 0]]
            masks = {}
            for color in job['target']['bins']:
                mask = mask_rgba[:, :, 3] >= 191
                for channel, value in enumerate(base.COLORS[color]):
                    mask &= rgb[:, :, channel] >= 191 if value else rgb[:, :, channel] <= 63
                if not mask.any():
                    raise ValueError('bin mask is empty: ' + color)
                masks[color] = mask
            floor = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(floor, [np.asarray(background_record['ground_polygon_xy'], dtype=np.int32)], 1)
            floor = cv2.erode(floor, np.ones((11, 11), dtype=np.uint8))
            if 'placements_xywh' in job['parameters']:
                state_path = _workspace_file(rel(_workspace_file(job['target']['render_manifest']).parent / source['mask_state_visible']))
                state_mask = base.read_image(state_path, cv2.IMREAD_UNCHANGED)
                frame, boxes, supports = _independent_composite(frame, foreground, masks, state_mask, floor, job)
            else:
                union = np.logical_or.reduce(list(masks.values())).astype(np.uint8) * 255
                crop_x, crop_y, crop_w, crop_h = cv2.boundingRect(union)
                crop = foreground[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                cropped = {name: mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for name, mask in masks.items()}
                left, top, width, height = job['parameters']['placement_xywh']
                if abs(width / crop_w - height / crop_h) > max(1 / crop_w, 1 / crop_h):
                    raise ValueError('placement stretches the render crop')
                scaled, supports = {}, {}
                for color, mask in cropped.items():
                    scaled[color] = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    ys, xs = np.nonzero(scaled[color])
                    if not len(ys):
                        raise ValueError('resized bin mask is empty')
                    lowest = int(ys.max())
                    feet = xs[ys >= lowest-2]
                    points = [[left + int(round(v)), top + lowest] for v in np.percentile(feet, [10, 50, 90])]
                    if any(not floor[y, x] for x, y in points):
                        raise ValueError('planned bin support falls outside ground polygon')
                    supports[color] = points
                frame = add_contact_shadows(frame, floor, scaled, supports)
                alpha = cv2.resize(union[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w], (width, height),
                                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                patch = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
                region = frame[top:top+height, left:left+width].astype(np.float32)
                frame[top:top+height, left:left+width] = np.clip(patch * alpha[:, :, None] + region * (1-alpha[:, :, None]), 0, 255).astype(np.uint8)
                boxes = []
                for color, mask in scaled.items():
                    x, y, w, h = cv2.boundingRect(mask)
                    boxes.append({'bin': color, 'state': job['target']['bins'][color],
                                  'bbox_xyxy': [left+x, top+y, left+x+w, top+y+h]})
            aug = job['augmentation']
            frame = apply_tone(frame, value_gain=aug['value_gain'], saturation_gain=aug['saturation_gain'])
            lines = []
            for box in boxes:
                if 'bin_ids' in job['target']:
                    box['bin_id'] = job['target']['bin_ids'][box['bin']]
                cls = 0 if box['state'] == 'open' else 1
                lines.append(box_line(cls, box['bbox_xyxy']))
            stem = f'planned_{index:06d}'
            image_out = out / 'images' / (stem + '.png')
            label_out = out / 'labels' / (stem + '.txt')
            write_png(image_out, frame)
            label_out.write_text('\n'.join(lines) + '\n', encoding='utf8')
            actual = {'background': background_record, 'background_exposure_role': exposure_role,
                      'background_exposure_reason': exposure_reason,
                      'render_image': rel(image_path), 'render_mask': rel(mask_path),
                      'render_manifest': job['target']['render_manifest'],
                      'source_state_mask': rel(_workspace_file(job['target']['render_manifest']).parent /
                                               source['mask_state_visible']),
                      'source_candidate_label': source['candidate_label'],
                      'ground_support_xy': supports, 'bins': boxes, 'augmentation': aug}
            if 'placement_xywh' in job['parameters']:
                actual['placement_xywh'] = job['parameters']['placement_xywh']
            else:
                actual['placements_xywh'] = job['parameters']['placements_xywh']
                actual['back_to_front'] = job['parameters']['back_to_front']
            run.record_sample(job['id'], output=image_out, sources=job['sources'], parameters=job['parameters'], actual=actual)
            report['records'].append({'state_id': job['id'], 'image': rel(image_out), 'label': rel(label_out),
                                      'status': 'candidate', 'generation_method': 'real_scene_composite', 'actual': actual})
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report['status'] = 'failed_or_interrupted'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    plan_path = _workspace_file(args.plan)
    declaration = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    from lab.training_common.production_guard import generation_guard
    from datasets.scripts.common.bayesian import plan_production
    planned = plan_production(declaration, root=ROOT)
    jobs, media, procedural, queue_code = _planned_inputs(planned)
    code = list(queue_code) + [Path(__file__).resolve(), Path(base.__file__).resolve(),
                               ROOT / 'lab/training_common/camera_tone.py',
                               ROOT / 'lab/training_common/bayesian_allocation.py',
                               ROOT / 'datasets/scripts/common/bayesian.py',
                               ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py']
    configs = [ROOT/'datasets/scripts/waste_bin_state/config'/name for name in ('dataset.yaml', 'source.yaml')]
    with generation_guard('waste_bin.grouped:' + str(out.relative_to(ROOT)), media, [out], vars(args),
                          code_paths=code, procedural_inputs=[*procedural, plan_path, *configs], root=ROOT,
                          production_plan=declaration, sample_output_paths=[out / 'images']) as run:
        if run.plan != planned:
            raise ValueError('plan changed before guarded execution')
        if run.reused:
            return
        _generate(args, run, jobs)


if __name__ == '__main__':
    main()
