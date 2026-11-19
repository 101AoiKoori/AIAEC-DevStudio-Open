"""Compose planned 640x480 bin-state candidates on selected backgrounds.

Sticker generators continue to import the shared helper functions below.
"""
import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

_WORKSPACE = Path(__file__).resolve().parents[3]
_TASK_HELPERS = _WORKSPACE / 'datasets/scripts/waste_bin_state'
sys.path.insert(0, str(_WORKSPACE))
from datasets.scripts.waste_bin_state import composite_bin_real_scene as base
from datasets.scripts.waste_bin_state.bin_render_contract import MIN_MOUTH_THICKNESS_PX, STATE_CONTRACT, mouth_thickness

ROOT = base.ROOT
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from camera_tone import apply_tone, source_role, workspace_path  # noqa: E402

sys.path.insert(0, str(ROOT))
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from lab.scripts.data import v4_source_gate as source_gate  # noqa: E402
from datasets.scripts.common.config import source_path
from datasets.scripts.common.frame_environment import frame_history, image_digest, query_frames
from datasets.scripts.common.paths import bin_state_full_composites_dir, bin_state_sample_composites_dir, split_dir
from datasets.scripts.common.review.placement_record import decode_placement_record, contact_status_in_view
from datasets.scripts.common.review.relative_depth import ground_stripe_scale_ratio
from datasets.scripts.waste_bin_state.qualified_pool import qualified_backgrounds

BG_MANIFEST = source_path('waste_bin_state', 'background_manifest')
DESTINATION = ROOT / 'datasets/exports/waste_bin_state'
# These source illustrations are too ambiguous in the 28-pixel audit.  Keep
# their boxes as partial candidates until the actual output is reviewed.
UNCERTAIN = {0, 2, 11, 13, 14, 19, 20, 23}
COLORS = tuple(base.COLORS)
SHADOW_MODEL = {
    'method': 'heuristic_soft_contact', 'light_direction': 'unmeasured',
    'support': 'candidate_ground_mask',
    'cast_offset_x_width': -0.05, 'cast_offset_y_height': 0.035,
    'cast_radius_x_width': 0.55, 'cast_radius_y_height': 0.07,
    'cast_stamp': 150, 'cast_blur_sigma': 6, 'cast_gain': 0.24,
    'contact_stamp': 130, 'contact_blur_sigma': 2, 'contact_gain': 0.20,
    'opacity_cap': 0.18,
}


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


def output_content_rectangle(source_size):
    width, height = source_size
    if width * 3 == height * 4:
        return [0, 0, 640, 480]
    if width * 9 == height * 16:
        return [0, 60, 640, 420]
    raise ValueError('bin background needs a 4:3 or 16:9 source frame')


def max_output_height_from_source_fraction(fraction, source_size):
    """Apply a source-relative height limit after fitting its content to output."""
    content = output_content_rectangle(source_size)
    return fraction * (content[3] - content[1])


def output_box(box, source_size):
    width, height = source_size
    dx0, dy0, dx1, dy1 = output_content_rectangle(source_size)
    x1, y1, x2, y2 = box
    mapped = [round(dx0 + x1 * (dx1-dx0) / width), round(dy0 + y1 * (dy1-dy0) / height),
              round(dx0 + x2 * (dx1-dx0) / width), round(dy0 + y2 * (dy1-dy0) / height)]
    box_line(0, mapped)
    return mapped


def output_polygon(polygon, source_size):
    width, height = source_size
    dx0, dy0, dx1, dy1 = output_content_rectangle(source_size)
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError('bin ground polygon is incomplete')
    mapped = []
    for point in polygon:
        if (not isinstance(point, list) or len(point) != 2 or
                not 0 <= point[0] < width or not 0 <= point[1] < height):
            raise ValueError('bin ground polygon exceeds source pixels')
        mapped.append([min(dx1 - 1, round(dx0 + point[0] * (dx1-dx0) / width)),
                       min(dy1 - 1, round(dy0 + point[1] * (dy1-dy0) / height))])
    return mapped


def existing_bin_labels(review):
    """Map complete source-frame bin states into the 640x480 output view."""
    label = review['label']
    width, height = review['image_size']
    if (label.get('completeness') != 'complete' or
            label.get('scope', {}).get('kind') != 'frame'):
        raise ValueError('existing bin labels require a complete supported source frame')
    output_content_rectangle(review['image_size'])
    instances = label.get('instances') or []
    if label.get('presence') == 'absent':
        if instances:
            raise ValueError('empty bin background contains target instances')
        return []
    if label.get('presence') != 'present' or not instances:
        raise ValueError('positive bin background needs complete visible instances')
    result, identities = [], set()
    for item in instances:
        state = item.get('lid_state', item.get('state'))
        color = item.get('color', item.get('bin_color'))
        box, object_id = item.get('xyxy'), item.get('object_id')
        if (('lid_state' in item and 'state' in item and item['lid_state'] != item['state']) or
                ('color' in item and 'bin_color' in item and item['color'] != item['bin_color']) or
                state not in ('open', 'closed') or not isinstance(color, str) or not color.strip() or
                item.get('kind') not in (None, 'waste_bin') or
                item.get('visibility') not in (None, 'visible', 'partial') or
                item.get('class_name') not in (None, state) or
                item.get('class_id') != (0 if state == 'open' else 1) or
                not isinstance(object_id, str) or not object_id or object_id in identities or
                not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or
                not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError('existing bin needs unique identity, color, lid state and source-pixel box')
        identities.add(object_id)
        mapped = output_box(box, review['image_size'])
        result.append({'bin_id': object_id, 'bin': color, 'state': state,
                       'source_box_xyxy': box, 'bbox_xyxy': mapped})
    return result


class OutputGround:
    """Output mask for search, exact source-mask lookup for each contact."""
    def __init__(self, source, source_size):
        self.source = source
        self.source_size = source_size
        self.content = output_content_rectangle(source_size)
        x0, y0, x1, y1 = self.content
        self.candidate_mask = np.zeros((480, 640), dtype=bool)
        self.candidate_mask[y0:y1, x0:x1] = cv2.resize(
            source.candidate_mask.astype(np.uint8), (x1-x0, y1-y0),
            interpolation=cv2.INTER_NEAREST).astype(bool)

    def contact_status(self, foot_xy, *, contact_width=0):
        width, height = self.source_size
        return contact_status_in_view(self.source, foot_xy, contact_width=contact_width,
                                      crop_xyxy=[0, 0, width, height], content_xyxy=self.content)


def output_ground(assessment, source_size):
    """Search in the output view and recheck contacts against saved source pixels."""
    width, height = source_size
    if assessment.candidate_mask.shape != (height, width):
        raise ValueError('bin contact mask differs from source dimensions')
    return assessment if (width, height) == (640, 480) else OutputGround(assessment, source_size)


def output_frame(frame):
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('invalid bin background frame dimensions')
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = output_content_rectangle([width, height])
    if frame.shape == (480, 640, 3):
        return frame
    output = np.zeros((480, 640, 3), dtype=frame.dtype)
    output[y0:y1, x0:x1] = cv2.resize(frame, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
    return output


def reject_existing_bin_overlap(boxes, existing):
    if any(min(new[2], old[2]) > max(new[0], old[0]) and
           min(new[3], old[3]) > max(new[1], old[1])
           for new in boxes for old in (item['bbox_xyxy'] for item in existing)):
        raise ValueError('new bin obscures a labelled background bin or lid state')


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
        bottom_min = max(height + 1, y0 + 10)
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
        if accepted and record.get('existing_bins'):
            new_boxes = []
            for scaled in scaled_masks.values():
                x, y, w, h = cv2.boundingRect(scaled)
                new_boxes.append([left+x, top+y, left+x+w, top+y+h])
            try:
                reject_existing_bin_overlap(new_boxes, record['existing_bins'])
            except ValueError:
                continue
        if accepted:
            return record, background, floor, left, top, width, height, scaled_masks, supports
    raise ValueError('No floor-valid placement after 300 attempts; review background polygons')


def add_contact_shadows(background, floor, scaled_masks, supports):
    """Cast a short soft shadow from each observed lower edge onto usable ground."""
    model = SHADOW_MODEL
    cast = np.zeros(background.shape[:2], dtype=np.uint8)
    contact = np.zeros_like(cast)
    for color, points in supports.items():
        _, _, w, h = cv2.boundingRect(scaled_masks[color])
        foot_x, foot_y = points[1]
        cv2.ellipse(cast, (foot_x - max(1, round(w*abs(model['cast_offset_x_width']))),
                           min(479, foot_y + max(2, round(h*model['cast_offset_y_height'])))),
                    (max(5, round(w*model['cast_radius_x_width'])),
                     max(3, round(h*model['cast_radius_y_height']))),
                    0, 0, 360, model['cast_stamp'], -1)
        cv2.line(contact, (points[0][0], min(479, foot_y+1)),
                 (points[2][0], min(479, foot_y+1)), model['contact_stamp'], max(2, round(h*.03)))
    soft = cv2.GaussianBlur(cast, (0, 0), model['cast_blur_sigma']).astype(np.float32) / 255 * model['cast_gain']
    core = cv2.GaussianBlur(contact, (0, 0), model['contact_blur_sigma']).astype(np.float32) / 255 * model['contact_gain']
    shadow = np.clip((soft + core) * floor.astype(np.float32), 0, model['opacity_cap'])
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


def eligible_ground_backgrounds(source, frames, split):
    """Read task-qualified ground masks once; legacy polygons only bound the search."""
    database = _workspace_file(source['databases']['frame_environment'])
    rows = query_frames(task='waste_bin_state', role='composite_background', split=split,
                        database=database)
    eligible = {}
    for background in frames:
        matched = [row for row in rows if background['image'] in row['paths']]
        if not matched:
            continue
        if len(matched) != 1:
            raise ValueError('background has conflicting stored role records: ' + background['id'])
        row = matched[0]
        label, geometry = row.get('label') or {}, row.get('geometry') or {}
        placement = geometry.get('placeable_ground')
        size = row['image_size']
        if (geometry.get('coordinate_size') != size or
                row.get('scope', {}).get('kind') != 'frame' or
                row.get('parent_video') != background.get('video') or
                (row.get('parent_image') or background['image']) != background.get('parent_image') or
                image_digest(background['image']) != row['content_sha256'] or
                label.get('presence') not in ('absent', 'present') or
                label.get('completeness') != 'complete' or
                not isinstance(placement, dict) or placement.get('status') != 'candidate' or
                not geometry.get('ground_polygon')):
            raise ValueError('stored bin background lacks matching labels or ground evidence: '
                             + background['id'])
        existing_bin_labels(row)
        assessment = decode_placement_record(placement)
        if assessment.candidate_mask.shape != (size[1], size[0]):
            raise ValueError('stored bin contact mask has wrong pixel coordinates: ' + background['id'])
        eligible[background['id']] = (row, output_ground(assessment, size))
    return database, eligible


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


def current_scale_observation(database, frame_id, source_size):
    """Read the shared cue separately from the task's contact qualification."""
    for row in frame_history(frame_id, task='waste_bin_state', database=database):
        if (not row['current'] or row['task'] or row['kind'] != 'geometry' or
                row['scope'] != 'shared_background' or row['evidence_key'] != 'shared_background_space'):
            continue
        payload = row['payload']
        cues = payload.get('perspective_cues') or {}
        if payload.get('image_size') != list(source_size) or cues.get('coordinate_space') != 'source_image_pixels':
            return None
        return {'evidence_id': row['evidence_id'], 'algorithm_version': row['algorithm_version'],
                'scale': cues.get('scale') or {}, 'local_stripe_scale': cues.get('local_stripe_scale') or {}}
    return None


def _source_point(point, background):
    """Invert the background letterbox so the cue sees native-image pixels."""
    x, y = point
    x0, y0, x1, y1 = background['output_content_xyxy']
    width, height = background['source_image_size']
    return [(x - x0) * width / (x1 - x0), (y - y0) * height / (y1 - y0)]


def _scale_at_foot(foot, background, floor):
    observation = background.get('perspective_scale_observation')
    if not observation:
        return None
    x, y = foot
    target = _source_point(foot, background)
    trace = {'evidence_id': observation['evidence_id'], 'algorithm_version': observation['algorithm_version'],
             'coordinate_space': 'source_image_pixels', 'foot_xy': target}
    horizon = observation['scale']
    if horizon.get('status') == 'estimated' and horizon.get('basis') == 'ground_supported_vanishing_line':
        line = np.asarray(horizon.get('horizon_line_px'), dtype=float)
        ground_rows = np.flatnonzero(floor[:, x])
        if line.shape == (3,) and np.isfinite(line).all() and line[1] > 0 and len(ground_rows):
            far_y, near_y = int(ground_rows[0]), int(ground_rows[-1])
            reference = _source_point((x, near_y), background)
            target_distance = float(line @ [*target, 1])
            reference_distance = float(line @ [*reference, 1])
            if far_y <= y <= near_y and min(target_distance, reference_distance) > 0:
                return {**trace, 'scale_basis': horizon['basis'], 'ratio': target_distance / reference_distance,
                        'reference_xy': reference, 'horizon_line_px': line.tolist(),
                        'reference_y_range_px': [_source_point((x, far_y), background)[1], reference[1]]}
    stripe = observation['local_stripe_scale']
    if stripe.get('status') == 'estimated' and stripe.get('basis') == 'ground_supported_stripe_width_change':
        anchors = stripe['center_anchors_px']
        reference = anchors[-1]
        ratio = ground_stripe_scale_ratio(stripe, target, reference)
        if ratio is not None:
            return {**trace, 'scale_basis': stripe['basis'], 'ratio': ratio, 'reference_xy': reference,
                    'center_anchors_px': anchors, 'anchor_widths_px': stripe['anchor_widths_px'],
                    'width_error_sensitivity': stripe.get('width_error_sensitivity')}
    return None


def _planned_inputs(plan):
    """Resolve and validate source metadata without composing or changing pixels."""
    if plan['model_key'] != 'waste_bin_state' or plan['constraints'] != {
            'resolution': [640, 480], 'label_mode': 'candidate', 'ground_placement': 'stored_contact_mask'}:
        raise ValueError('unsupported grouped-composite model or constraints')
    frames = json.loads(BG_MANIFEST.read_text(encoding='utf8'))['frames']
    from datasets.scripts.common.config import load_task_config
    _, source_config = load_task_config('waste_bin_state')
    pool_manifest = source_config['databases'][source_config['current_selection']['qualified_pool']['database']]
    pool_ids = set()
    for state in plan['strata']:
        background = state['background']
        if background.get('manifest') == pool_manifest:
            if not background.get('id', '').startswith('pool_'):
                raise ValueError('pool background must use its frame-derived ID')
            pool_ids.add(background['id'][len('pool_'):])
    if pool_ids:
        frames.extend(qualified_backgrounds(source_config, plan['target_split'], frame_ids=sorted(pool_ids)))
    by_id = {row['id']: row for row in frames}
    if len(by_id) != len(frames):
        raise ValueError('duplicate background id')
    environment_path, eligible = eligible_ground_backgrounds(source_config, frames, plan['target_split'])
    media, procedural = set(), {BG_MANIFEST, environment_path}
    queue_media, queue_procedural, queue_code = source_gate.queue_inputs(ROOT)
    media.update(queue_media)
    procedural.update(queue_procedural)
    jobs, seen, scale_observations = [], set(), {}

    @lru_cache(maxsize=32)
    def render_report(name):
        manifest = _workspace_file(name)
        report = json.loads(manifest.read_text(encoding='utf8'))
        return manifest, report

    for state in plan['strata']:
        if state['conditions'] or state['planned_count'] > 1:
            raise ValueError('unsupported conditions or repeated exact state')
        bg, target, labels, aug, params = (state[key] for key in
                                          ('background', 'target', 'labels', 'augmentation', 'parameters'))
        if (set(bg) != {'manifest', 'id', 'frame_id', 'record_id'} or
                bg['id'] not in by_id or bg['id'] not in eligible or
                bg['manifest'] != by_id[bg['id']].get('manifest', rel(BG_MANIFEST))):
            raise ValueError('background must name a known legacy or current pool frame')
        qualification, assessment = eligible[bg['id']]
        if (bg['frame_id'] != qualification['frame_id'] or
                bg['record_id'] != qualification['record_id']):
            raise ValueError('planned bin background review has changed: ' + bg['id'])
        if bg['frame_id'] not in scale_observations:
            scale_observations[bg['frame_id']] = current_scale_observation(
                environment_path, bg['frame_id'], qualification['image_size'])
        background = {**by_id[bg['id']],
                      'ground_polygon_xy': output_polygon(qualification['geometry']['ground_polygon'],
                                                          qualification['image_size']),
                      'ground_polygon_coordinate_space': 'output_image_pixels',
                      'output_image_size': [640, 480],
                      'output_content_xyxy': output_content_rectangle(qualification['image_size']),
                      'ground_contact_coordinate_space': 'source_image_pixels',
                      'source_image_size': qualification['image_size'],
                      'perspective_scale_observation': scale_observations[bg['frame_id']],
                      'max_apparent_height_fraction': qualification['geometry'].get('max_apparent_height_fraction'),
                      'tested_height_range_output_px': (qualification['geometry'].get('validated_placement') or {})
                          .get('tested_height_range_output_px'),
                      'existing_bins': existing_bin_labels(qualification),
                      'frame_environment': {'frame_id': bg['frame_id'], 'record_id': bg['record_id'],
                                            'evidence_ids': [entry['evidence_id'] for entry in qualification['evidence']]}}
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
        source_manifest, report = render_report(target['render_manifest'])
        if report.get('split') != plan['target_split']:
            raise ValueError('source render split differs from the output split')
        matching = [row for row in report['samples'] if row['image'] == target['render_image']]
        if report.get('status') != 'complete_candidates' or len(matching) != 1:
            raise ValueError('source render is incomplete or image is not unique')
        sample = matching[0]
        observed = {item['bin']: item['state'] for item in sample['labels']}
        if 'open' in observed.values() and sample.get('open_pose') == 'upright':
            raise ValueError('the source uses invalid upright open-bin geometry')
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
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
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
        max_fraction = qualification['geometry'].get('max_apparent_height_fraction')
        if max_fraction is not None:
            if (isinstance(max_fraction, bool) or not isinstance(max_fraction, (int, float))
                    or not 0 < max_fraction <= 1):
                raise ValueError('invalid stored apparent-height limit')
            heights = ([params['placement_xywh'][3]] if 'placement_xywh' in params
                       else [item[3] for item in params['placements_xywh'].values()])
            limit_output_px = max_output_height_from_source_fraction(max_fraction, qualification['image_size'])
            if any(height > limit_output_px + 1e-6 for height in heights):
                raise ValueError('planned bin exceeds stored apparent-height limit')
        validated = qualification['geometry'].get('validated_placement') or {}
        tested_range = validated.get('tested_height_range_output_px')
        if tested_range is not None and 'placement_xywh' in params:
            if (not isinstance(tested_range, list) or len(tested_range) != 2 or
                    any(type(value) is not int for value in tested_range) or
                    tested_range[0] < 1 or tested_range[0] > tested_range[1]):
                raise ValueError('invalid stored group-height range')
            if not tested_range[0] <= params['placement_xywh'][3] <= tested_range[1]:
                raise ValueError('planned group exceeds stored tested-height range')
        identity = json.dumps({key: state[key] for key in ('background', 'target', 'labels', 'augmentation',
                                                            'parameters', 'sources')}, sort_keys=True, allow_nan=False)
        if state['planned_count'] and identity in seen:
            raise ValueError('duplicate composite state')
        if state['planned_count']:
            seen.add(identity)
            jobs.append((state, background, sample, source_image, source_mask, assessment))
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


def _group_supports(masks, placement):
    left, top, width, height = placement
    scaled, supports = {}, {}
    for color, mask in masks.items():
        raster = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(raster)
        if not len(ys):
            raise ValueError('resized bin mask is empty: ' + color)
        lowest = int(ys.max())
        feet = xs[ys >= lowest - 2]
        supports[color] = [[left + int(round(x)), top + lowest] for x in np.percentile(feet, [10, 50, 90])]
        scaled[color] = raster
    return scaled, supports


def _scaled_group_layout(masks, placement, background, assessment, state_masks):
    """Apply an observed ratio to the planned reference height, fixing one real bin foot."""
    scaled, supports = _group_supports(masks, placement)
    x0, _, width, height = placement
    center_x = x0 + width / 2
    # Preserve the group perspective already present in the render.
    anchor = min(supports, key=lambda name: abs(supports[name][1][0] - center_x))
    foot = supports[anchor][1]
    floor = assessment.candidate_mask
    cue = _scale_at_foot(foot, background, floor)
    trace = {'scale_basis': 'planned_approximation', 'ratio': None, 'reference_height_px': height,
             'applied_height_px': height, 'anchor_bin': anchor,
             'foot_xy': _source_point(foot, background), 'coordinate_space': 'source_image_pixels'}
    if cue is None:
        observation = background.get('perspective_scale_observation')
        return (placement, scaled, supports,
                {**trace, 'reason': 'no_applicable_observed_ratio' if observation else 'no_shared_observation',
                 'evidence_id': observation['evidence_id'] if observation else None})
    ratio = cue['ratio']
    new_height = max(1, round(height * ratio))
    crop_height, crop_width = next(iter(masks.values())).shape
    new_width = max(1, round(new_height * crop_width / crop_height))
    try:
        if not (0 < new_width <= 640 and 0 < new_height <= 480):
            raise ValueError('scaled_size_outside_output')
        tested_range = background.get('tested_height_range_output_px')
        if tested_range and not tested_range[0] <= new_height <= tested_range[1]:
            raise ValueError('scaled_height_outside_tested_range')
        fraction = background.get('max_apparent_height_fraction')
        if fraction and new_height > max_output_height_from_source_fraction(fraction, background['source_image_size']):
            raise ValueError('scaled_height_exceeds_background_limit')
        adjusted_masks, offsets = _group_supports(masks, [0, 0, new_width, new_height])
        left, top = foot[0] - offsets[anchor][1][0], foot[1] - offsets[anchor][1][1]
        adjusted = [left, top, new_width, new_height]
        _placement(adjusted)
        adjusted_supports = {name: [[left + x, top + y] for x, y in points]
                             for name, points in offsets.items()}
        polygon = np.asarray(background['ground_polygon_xy'], dtype=np.float32)
        for points in adjusted_supports.values():
            if any(not floor[y, x] or cv2.pointPolygonTest(polygon, (x, y), False) < 0
                   for x, y in points):
                raise ValueError('scaled_support_outside_usable_ground')
            if assessment.contact_status(points[1], contact_width=points[2][0] - points[0][0]) != 'candidate':
                raise ValueError('scaled_contact_outside_usable_ground')
        for evidence, opening in state_masks.values():
            raster = cv2.resize(evidence, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            if int(np.count_nonzero(raster)) < 24 or (opening and mouth_thickness(raster) < MIN_MOUTH_THICKNESS_PX):
                raise ValueError('scaled_lid_state_evidence_not_resolved')
        new_boxes = []
        for mask in adjusted_masks.values():
            x, y, w, h = cv2.boundingRect(mask)
            new_boxes.append([left + x, top + y, left + x + w, top + y + h])
        reject_existing_bin_overlap(new_boxes, background.get('existing_bins', ()))
    except ValueError as error:
        return placement, scaled, supports, {**trace, 'reason': str(error), 'rejected_observation': cue}
    return adjusted, adjusted_masks, adjusted_supports, {**trace, **cue, 'applied_height_px': new_height}


def _group_composite(frame, crop, union, cropped, placement, background, assessment, state_masks):
    placement, scaled, supports, scale = _scaled_group_layout(
        cropped, placement, background, assessment, state_masks)
    left, top, width, height = placement
    floor = assessment.candidate_mask.astype(np.uint8)
    if any(not floor[y, x] for points in supports.values() for x, y in points):
        raise ValueError('planned bin support falls outside ground polygon')
    frame = add_contact_shadows(frame, floor, scaled, supports)
    alpha = cv2.resize(union, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    patch = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    region = frame[top:top + height, left:left + width].astype(np.float32)
    frame[top:top + height, left:left + width] = np.clip(
        patch * alpha[:, :, None] + region * (1 - alpha[:, :, None]), 0, 255).astype(np.uint8)
    return frame, placement, scaled, supports, scale


def exposure_checker(known):
    """Index the snapshot by file size before comparing each distinct parent image."""
    named = set(known['exposed_images'])
    unavailable = {str(item['path']) for item in known.get('unavailable_exposure_paths', ())}
    by_size = defaultdict(list)
    for item in named - unavailable:
        prior = workspace_path(ROOT, item)
        if prior.is_file():
            by_size[prior.stat().st_size].append(item)

    @lru_cache(maxsize=1024)
    def for_image(parent_image):
        source = ROOT / parent_image
        candidates = [parent_image] if parent_image in named else by_size[source.stat().st_size]
        return source_role(ROOT, source, {**known, 'exposed_images': candidates})

    return for_image


def _generate(args, run, jobs, working=None):
    out = (ROOT / args.out).resolve()
    split = run.plan['target_split']
    allowed = {bin_state_sample_composites_dir().resolve(), bin_state_full_composites_dir(split).resolve(),
               split_dir('waste_bin_state', split).resolve()}
    if out not in allowed:
        raise ValueError('output must be the annotated bin composites or declared split directory')
    if out.exists() and any(out.iterdir()):
        raise ValueError('output is not empty')
    exposure = exposure_checker(build_queue(ROOT))
    @lru_cache(maxsize=96)
    def read_pixels(path, mode=cv2.IMREAD_COLOR):
        return base.read_image(working.path(path) if working else path, mode)

    @lru_cache(maxsize=48)
    def read_background(path):
        return output_frame(base.read_image(working.path(path) if working else path))

    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(parents=True)
    (out / 'labels').mkdir(parents=True)
    report = {'status': 'running', 'split': split, 'model_key': 'waste_bin_state', 'records': []}
    try:
        for index, (job, background_record, source, image_path, mask_path, assessment) in enumerate(jobs):
            exposure_role, exposure_reason = exposure(background_record['parent_image'])
            if exposure_role == 'regression':
                raise ValueError('background is restricted to regression: ' + background_record['parent_image'])
            foreground = read_pixels(image_path)
            mask_rgba = read_pixels(mask_path, cv2.IMREAD_UNCHANGED)
            background_path = ROOT / background_record['image']
            frame = read_background(background_path).copy()
            if mask_rgba.shape != (*foreground.shape[:2], 4):
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
            floor = assessment.candidate_mask.astype(np.uint8)
            if 'placements_xywh' in job['parameters']:
                state_path = _workspace_file(rel(_workspace_file(job['target']['render_manifest']).parent / source['mask_state_visible']))
                state_mask = read_pixels(state_path, cv2.IMREAD_UNCHANGED)
                frame, boxes, supports = _independent_composite(frame, foreground, masks, state_mask, floor, job)
                scale = {'scale_basis': 'planned_approximation', 'ratio': None,
                         'reason': 'explicit_independent_placement_and_depth_order'}
            else:
                union = np.logical_or.reduce(list(masks.values())).astype(np.uint8) * 255
                crop_x, crop_y, crop_w, crop_h = cv2.boundingRect(union)
                crop = foreground[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                cropped = {name: mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for name, mask in masks.items()}
                left, top, width, height = job['parameters']['placement_xywh']
                if abs(width / crop_w - height / crop_h) > max(1 / crop_w, 1 / crop_h):
                    raise ValueError('placement stretches the render crop')
                state_masks = {}
                observation = background_record.get('perspective_scale_observation') or {}
                if any(observation.get(key, {}).get('status') == 'estimated'
                       for key in ('scale', 'local_stripe_scale')):
                    state_path = _workspace_file(rel(_workspace_file(job['target']['render_manifest']).parent / source['mask_state_visible']))
                    state_rgba = read_pixels(state_path, cv2.IMREAD_UNCHANGED)
                    if state_rgba.shape != mask_rgba.shape:
                        raise ValueError('invalid group state mask dimensions')
                    state_rgba = state_rgba[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    state_rgb = state_rgba[:, :, [2, 1, 0]]
                    for color, state in job['target']['bins'].items():
                        visible = state_rgba[:, :, 3] >= 191
                        for channel, value in enumerate(base.COLORS[color]):
                            visible &= state_rgb[:, :, channel] >= 191 if value else state_rgb[:, :, channel] <= 63
                        state_masks[color] = (visible.astype(np.uint8), state == 'open')
                frame, placement, scaled, supports, scale = _group_composite(
                    frame, crop, union[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w], cropped,
                    job['parameters']['placement_xywh'], background_record, assessment, state_masks)
                left, top, width, height = placement
                boxes = []
                for color, mask in scaled.items():
                    x, y, w, h = cv2.boundingRect(mask)
                    boxes.append({'bin': color, 'state': job['target']['bins'][color],
                                  'bbox_xyxy': [left+x, top+y, left+x+w, top+y+h]})
            retained = background_record.get('existing_bins', [])
            reject_existing_bin_overlap([box['bbox_xyxy'] for box in boxes], retained)
            for color, points in supports.items():
                if assessment.contact_status(points[1], contact_width=points[2][0]-points[0][0]) != 'candidate':
                    raise ValueError('planned bin contact is outside stored usable ground: ' + color)
            aug = job['augmentation']
            frame = apply_tone(frame, value_gain=aug['value_gain'], saturation_gain=aug['saturation_gain'])
            lines = []
            for box in boxes:
                if 'bin_ids' in job['target']:
                    box['bin_id'] = job['target']['bin_ids'][box['bin']]
                cls = 0 if box['state'] == 'open' else 1
                lines.append(box_line(cls, box['bbox_xyxy']))
            for box in retained:
                lines.append(box_line(0 if box['state'] == 'open' else 1, box['bbox_xyxy']))
            stem = f'planned_{index:06d}'
            image_out = out / 'images' / (stem + '.png')
            label_out = out / 'labels' / (stem + '.txt')
            write_png(image_out, frame)
            label_out.write_text('\n'.join(lines) + '\n', encoding='utf8')
            actual = {'background': background_record,
                      'working_background': working.actual(background_path) if working else rel(background_path),
                      'working_render_image': working.actual(image_path) if working else rel(image_path),
                      'working_render_mask': working.actual(mask_path) if working else rel(mask_path),
                      'background_exposure_role': exposure_role,
                      'background_exposure_reason': exposure_reason,
                      'render_image': rel(image_path), 'render_mask': rel(mask_path),
                      'render_manifest': job['target']['render_manifest'],
                      'source_state_mask': rel(_workspace_file(job['target']['render_manifest']).parent /
                                               source['mask_state_visible']),
                      'source_candidate_label': source['candidate_label'],
                      'ground_support_xy': supports, 'bins': boxes,
                      'retained_background_bins': retained, 'augmentation': aug,
                      'scale_basis': scale['scale_basis'], 'scale': scale,
                      'shadow_model': dict(SHADOW_MODEL)}
            if 'placement_xywh' in job['parameters']:
                actual['placement_xywh'] = placement
            else:
                actual['placements_xywh'] = job['parameters']['placements_xywh']
                actual['back_to_front'] = job['parameters']['back_to_front']
            run.record_sample(job['id'], output=image_out, sources=job['sources'], parameters=job['parameters'], actual=actual)
            method = ('blender_background_composite' if '/backgrounds/rendered/' in background_record['image']
                      else 'real_scene_composite')
            report['records'].append({'state_id': job['id'], 'image': rel(image_out), 'label': rel(label_out),
                                      'status': 'candidate', 'generation_method': method, 'actual': actual})
            if (index + 1) % 200 == 0:
                print(json.dumps({'split': split, 'generated': index + 1, 'total': len(jobs)}), flush=True)
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
    from datasets.scripts.common.working_inputs import CODE as WORKING_INPUTS_CODE, WorkingInputs
    planned = plan_production(declaration, root=ROOT)
    jobs, media, procedural, queue_code = _planned_inputs(planned)
    working = WorkingInputs('waste_bin_state', backgrounds=[job[1]['image'] for job in jobs],
                            targets=[value for job in jobs for value in job[3:5]], split=declaration['target_split'])
    code = list(queue_code) + [Path(__file__).resolve(), Path(base.__file__).resolve(), WORKING_INPUTS_CODE,
                               ROOT / 'datasets/scripts/waste_bin_state/qualified_pool.py',
                               ROOT / 'datasets/scripts/common/frame_environment.py',
                               ROOT / 'datasets/scripts/common/review/placement_record.py',
                               ROOT / 'datasets/scripts/common/review/relative_depth.py',
                               ROOT / 'lab/training_common/camera_tone.py',
                               ROOT / 'lab/training_common/bayesian_allocation.py',
                               ROOT / 'datasets/scripts/common/bayesian.py',
                               ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py']
    configs = [ROOT/'datasets/scripts/waste_bin_state/config'/name for name in ('dataset.yaml', 'source.yaml')]
    with generation_guard('waste_bin.grouped:' + str(out.relative_to(ROOT)), [*media, *working.media], [out], vars(args),
                          code_paths=code, procedural_inputs=[*procedural, plan_path, *configs, *working.evidence], root=ROOT,
                          production_plan=declaration, sample_output_paths=[out / 'images']) as run:
        if run.plan != planned:
            raise ValueError('plan changed before guarded execution')
        if run.reused:
            return
        _generate(args, run, jobs, working)


if __name__ == '__main__':
    main()
