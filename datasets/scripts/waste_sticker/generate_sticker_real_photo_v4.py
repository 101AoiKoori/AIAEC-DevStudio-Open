"""Composite digital item stickers on qualified photographed or rendered bin surfaces.

Each plan state names a photograph, declared placement regions, artworks and
tone parameters. Qualified plans use current frame-database geometry and labels.
Older candidate plans still use their declared geometry manifest.
"""
import argparse
from collections import Counter
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'datasets/scripts/waste_sticker'))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common import images as base
from datasets.scripts.waste_sticker import generate_stickers as artwork_generator
from lab.training_common.camera_tone import apply_tone
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import SPLITS, sticker_surface_sample_dir
from datasets.scripts.common.working_inputs import CODE as WORKING_INPUTS_CODE, WorkingInputs
from datasets.scripts.waste_sticker.source_inputs import real_source_catalog, artwork_catalog, surface_background_catalog

GEOMETRY = 'datasets/annotated/waste_bin/bin_reference_geometry_manifest.jsonl'
TOOLKIT = source_path('waste_sticker', 'artwork_toolkit').relative_to(ROOT).as_posix()
CONSTRAINTS = {'max_long_edge': 1280, 'max_short_edge': 720,
               'region_geometry': 'axis_aligned_candidate_only',
               'labels': 'inserted_papers_only_background_unknown'}

# Separate contract: preserve the older four-rectangle candidate entry unchanged.
SURFACE_CONSTRAINTS = {'max_long_edge': 1280, 'max_short_edge': 720,
                       'region_geometry': 'explicit_bin_surface_quad',
                       'labels': 'inserted_visible_candidates_background_unknown'}
RECORDED_SURFACE_CONSTRAINTS = {**SURFACE_CONSTRAINTS, 'labels': 'inserted_visible_candidates_background_absence_recorded'}
COMPLETE_SURFACE_CONSTRAINTS = {**SURFACE_CONSTRAINTS, 'labels': 'all_visible_stickers_from_complete_background_and_inserted'}
BIN_COLORS = {'blue', 'green', 'red', 'gray'}


def load_geometry(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def validate_plan(plan, names, geometry):
    if plan['model_key'] != 'waste_sticker' or plan['target_split'] != 'train' or plan['constraints'] != CONSTRAINTS:
        raise ValueError('unsupported task, split or real-photo constraints')
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 distinct sticker artworks')
    identities = set()
    for state in plan['strata']:
        p = state['parameters']
        if set(p) != {'photo', 'record_index', 'class_pool', 'size_fraction', 'gain_range', 'saturation_range'}:
            raise ValueError('unsupported or missing real-photo parameters')
        index = p['record_index']
        if type(index) is not int or not 0 <= index < len(geometry):
            raise ValueError('invalid geometry record index')
        record = geometry[index]
        if record['path'] != p['photo'] or not (ROOT / p['photo']).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('photo differs from geometry record')
        ids = p['class_pool']
        regions = [x for x in record.get('regions', []) if x.get('role') == 'bin_front_blank_candidate']
        if (not isinstance(ids, list) or not 4 <= len(ids) <= 24 or len(set(ids)) != len(ids) or
                any(type(x) is not int or not 0 <= x < 24 for x in ids) or
                len(regions) != 4):
            raise ValueError('a pool of distinct artworks and four candidate regions are required')
        for region in regions:
            points = region.get('polygon')
            if (region.get('status') != 'visually_blank_candidate_not_calibrated' or
                    not isinstance(points, list) or len(points) != 4):
                raise ValueError('unsupported candidate region')
            xs, ys = [v[0] for v in points], [v[1] for v in points]
            corners = {(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))}
            if set(map(tuple, points)) != corners or min(xs) >= max(xs) or min(ys) >= max(ys):
                raise ValueError('unsupported nonrectangular candidate region')
        for key in ('size_fraction', 'gain_range', 'saturation_range'):
            values = p[key]
            if (not isinstance(values, list) or len(values) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                    or not 0 < values[0] < values[1] <= 1.0):
                raise ValueError('invalid non-degenerate sampling range: ' + key)
        sources = [p['photo']] + [f'{TOOLKIT}/SourcePic/{names[c]}.png' for c in ids]
        if (state['background'] != {'kind': 'camera_photo', 'image': p['photo'], 'geometry_manifest': GEOMETRY, 'record_index': index}
                or state['target'] != {'kind': 'four_distinct_stickers', 'class_pool': ids}
                or state['labels'] != {'format': 'yolo_detection', 'status': 'inserted_only_candidate'}
                or state['augmentation'] != {'gain_range': p['gain_range'], 'saturation_range': p['saturation_range'],
                                              'placement': 'seeded_uniform_within_candidate_regions'}
                or state['sources'] != sources):
            raise ValueError('state does not match declared photo, labels, tone and sources')
        identity = json.dumps(p, sort_keys=True, ensure_ascii=False)
        if identity in identities or state['max_count'] > 2000:
            raise ValueError('duplicate state or state exceeds bounded production limit')
        identities.add(identity)
    return plan


def make_image(background, boxes, ids, icons, gain, saturation):
    image = background.copy()
    for box, cls in zip(boxes, ids):
        x1, y1, x2, y2 = box
        width, height = x2-x1, y2-y1
        rgb, alpha = icons[cls]
        paper = np.full((height, width, 3), 240, np.uint8)
        margin = max(1, round(min(width, height) * .10))
        inner_w, inner_h = width - 2*margin, height - 2*margin
        factor = min(inner_w/rgb.shape[1], inner_h/rgb.shape[0])
        iw, ih = max(1, round(rgb.shape[1]*factor)), max(1, round(rgb.shape[0]*factor))
        ink = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
        a = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_AREA)[:, :, None]
        ox, oy = (width-iw)//2, (height-ih)//2
        patch = paper[oy:oy+ih, ox:ox+iw].astype(np.float32)
        paper[oy:oy+ih, ox:ox+iw] = np.clip(ink.astype(np.float32)*a+patch*(1-a), 0, 255).astype(np.uint8)
        image[y1:y2, x1:x2] = paper
    return apply_tone(image, value_gain=gain, saturation_gain=saturation)


def _produce(out, run, names, geometry, icons, seed, working=None):
    plan = validate_plan(run.plan, names, geometry)
    images, labels = out/'images/train', out/'labels/train'
    images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
    backgrounds, records, used_seeds = {}, [], set()
    seed_rng = random.Random(seed)
    for state in plan['strata']:
        if not state['planned_count']:
            continue
        p = state['parameters']; source = p['photo']
        if source not in backgrounds:
            image = base.read_image(working.path(source) if working else ROOT/source)
            expected = geometry[p['record_index']]['image_size']
            if [image.shape[1], image.shape[0]] != expected:
                raise ValueError('photo dimensions differ from geometry manifest')
            factor = min(1280/image.shape[1], 720/image.shape[0], 1.0)
            size = (round(image.shape[1]*factor), round(image.shape[0]*factor))
            backgrounds[source] = (cv2.resize(image, size, interpolation=cv2.INTER_AREA), size, expected)
        background, (width, height), (raw_width, raw_height) = backgrounds[source]
        regions = [x for x in geometry[p['record_index']]['regions'] if x.get('role') == 'bin_front_blank_candidate']
        for _ in range(state['planned_count']):
            for _ in range(16):
                item_seed = seed_rng.getrandbits(62)
                if item_seed not in used_seeds:
                    used_seeds.add(item_seed); break
            else:
                raise RuntimeError('could not allocate a distinct generation seed')
            rng = random.Random(item_seed)
            ids = rng.sample(p['class_pool'], 4)
            raw_boxes = []
            for region in regions:
                xs = [v[0] for v in region['polygon']]; ys = [v[1] for v in region['polygon']]
                left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                side = max(1, round(min(right-left, bottom-top)*rng.uniform(*p['size_fraction'])))
                x = rng.randint(left, right-side); y = rng.randint(top, bottom-side)
                raw_boxes.append([x, y, x+side, y+side])
            boxes = [[round(x1*width/raw_width), round(y1*height/raw_height),
                      round(x2*width/raw_width), round(y2*height/raw_height)] for x1,y1,x2,y2 in raw_boxes]
            if any(not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) for x1,y1,x2,y2 in boxes):
                raise ValueError('scaled paper box is empty or outside the image')
            gain, saturation = rng.uniform(*p['gain_range']), rng.uniform(*p['saturation_range'])
            image = make_image(background, boxes, ids, icons, gain, saturation)
            stem = f'train_{len(records):06d}'
            image_path, label_path = images/(stem+'.png'), labels/(stem+'.txt')
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise RuntimeError('could not encode generated image')
            encoded.tofile(str(image_path))
            lines = []
            for cls, (x1,y1,x2,y2) in zip(ids, boxes):
                lines.append(f'{cls} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
            label_path.write_text('\n'.join(lines)+'\n', encoding='utf8')
            actual = {'seed':item_seed, 'source_photo': source,
                      'working_background': working.actual(source) if working else source,
                      'working_artworks': [working.actual(f'{TOOLKIT}/SourcePic/{names[c]}.png') for c in ids] if working else [],
                      'geometry_manifest': GEOMETRY, 'geometry_record_index': p['record_index'],
                      'class_ids': ids, 'raw_boxes_xyxy': raw_boxes, 'output_boxes_xyxy': boxes, 'output_size': [width,height],
                      'gain': gain, 'saturation': saturation,
                      'used_files': [source] + [f'{TOOLKIT}/SourcePic/{names[c]}.png' for c in ids]}
            records.append({'id':stem,'state_id':state['id'],'image':image_path.relative_to(ROOT).as_posix(),
                            'label':label_path.relative_to(ROOT).as_posix(),'generation_method':'real_photo_composite',
                            'label_status':'candidate_inserted_only_background_unknown','actual':actual})
            run.record_sample(state['id'], output=image_path, sources=state['sources'], parameters=p, actual=actual)
    (out/'classes.json').write_text(json.dumps(names,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (out/'manifest.json').write_text(json.dumps({'status':'candidate_background_targets_unknown','plan':plan,
        'scope':'Real photographs plus inserted digital artwork; repeated composites are not independent captures.',
        'records':records},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'images':len(records),'inserted_instances':len(records)*4},ensure_ascii=False))


def _polygon(points, width, height, *, quad=False):
    """Validate an in-frame, ordered convex polygon in original-image pixels."""
    if (not isinstance(points, list) or len(points) < 3 or (quad and len(points) != 4)
            or any(not isinstance(p, list) or len(p) != 2 or
                   any(type(v) is not int for v in p) for p in points)):
        raise ValueError('polygon requires integer pixel coordinates')
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise ValueError('polygon outside source image')
    contour = np.asarray(points, dtype=np.int32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1:
        raise ValueError('polygon must be nondegenerate and convex')
    return contour


def _mask(points, width, height, *, quad=False):
    result = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(result, _polygon(points, width, height, quad=quad), 1)
    return result.astype(bool)


def surface_output(stored, size, planned_crop=None):
    width, height = size
    scope = stored['label']['scope'] if stored else {'kind': 'frame'}
    crop = [0, 0, width, height]
    if scope['kind'] != 'frame':
        bounds = scope.get('crop_xyxy')
        if (not isinstance(bounds, list) or len(bounds) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)):
            raise ValueError('scoped background needs reviewed rectangular output bounds')
        crop = [math.ceil(bounds[0]), math.ceil(bounds[1]), math.floor(bounds[2]), math.floor(bounds[3])]
    if planned_crop is not None:
        if (not stored or not isinstance(planned_crop, list) or len(planned_crop) != 4 or
                any(type(value) is not int for value in planned_crop) or
                not (crop[0] <= planned_crop[0] < planned_crop[2] <= crop[2] and
                     crop[1] <= planned_crop[1] < planned_crop[3] <= crop[3])):
            raise ValueError('planned crop must be inside the reviewed label scope')
        crop = planned_crop
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('reviewed output crop is outside the source image or empty')
    width, height = x1 - x0, y1 - y0
    factor = min(1280 / width, 720 / height, 1.0)
    return crop, (max(1, round(width * factor)), max(1, round(height * factor)))


def surface_record(parameters, geometry, background_roles):
    if background_roles is not None:
        stored = background_roles[parameters['photo']]
        return {'path': parameters['photo'], 'image_size': stored['geometry']['coordinate_size'],
                'bins': stored['geometry']['bins']}
    index = parameters.get('record_index')
    if type(index) is not int or not 0 <= index < len(geometry):
        raise ValueError('invalid geometry record index')
    return geometry[index]


def validate_surface_plan(plan, names, geometry, geometry_ref=GEOMETRY, *, background_roles=None):
    """Validate explicit source-pixel surfaces and one planned paper per bin.

    Each geometry record supplies ``bins`` with bin_id, bin_color,
    allowed_polygon, exclude_polygons and occluder_polygons. Placements give an
    ordered top-left, top-right, bottom-right, bottom-left paper quad and an
    explicit identifiability decision. Background scope follows the plan contract.
    """
    if (plan['model_key'] != 'waste_sticker' or plan['target_split'] not in SPLITS
            or plan['constraints'] not in (SURFACE_CONSTRAINTS, RECORDED_SURFACE_CONSTRAINTS,
                                          COMPLETE_SURFACE_CONSTRAINTS)):
        raise ValueError('unsupported surface task, split or constraints')
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 distinct sticker artworks')
    identities = set()
    for state in plan['strata']:
        p = state['parameters']
        required = {'photo', 'placements', 'gain', 'saturation'}
        if p.keys() - {'blur_sigma', 'record_index', 'output_crop_xyxy'} != required:
            raise ValueError('unsupported surface parameters')
        stored = background_roles[p['photo']] if background_roles is not None else None
        record = surface_record(p, geometry, background_roles)
        if (record['path'] != p['photo'] or
                not (ROOT / p['photo']).resolve().is_relative_to(ROOT.resolve())):
            raise ValueError('photo differs from geometry record')
        size = record.get('image_size')
        if (not isinstance(size, list) or len(size) != 2 or
                any(type(v) is not int or v <= 0 for v in size)):
            raise ValueError('invalid source image size')
        width, height = size
        crop, output_size = surface_output(stored, size, p.get('output_crop_xyxy'))
        x0, y0, x1, y1 = crop
        bins = record.get('bins')
        if not isinstance(bins, list) or not bins:
            raise ValueError('source must provide explicit bin surfaces')
        by_id = {}
        for item in bins:
            if (not isinstance(item, dict) or not isinstance(item.get('bin_id'), str)
                    or not item['bin_id'].strip() or item['bin_id'] in by_id
                    or item.get('bin_color') not in BIN_COLORS):
                raise ValueError('invalid or duplicate bin surface')
            _polygon(item.get('allowed_polygon'), width, height)
            for key in ('exclude_polygons', 'occluder_polygons'):
                if not isinstance(item.get(key), list):
                    raise ValueError('surface polygons must be lists')
                for polygon in item[key]:
                    _polygon(polygon, width, height)
            by_id[item['bin_id']] = item
        placements = p['placements']
        if not isinstance(placements, list) or not placements:
            raise ValueError('at least one placement is required')
        seen_bins, used, occupied = set(), set(), np.zeros((height, width), dtype=bool)
        for placement in placements:
            if (not isinstance(placement, dict) or set(placement) != {'bin_id', 'class_id', 'paper_quad', 'identifiability'}
                    or placement['bin_id'] not in by_id or placement['bin_id'] in seen_bins
                    or type(placement['class_id']) is not int or not 0 <= placement['class_id'] < 24
                    or placement['identifiability'] not in ('identifiable', 'unknown')):
                raise ValueError('invalid placement, bin identity or class')
            bin_info = by_id[placement['bin_id']]
            quad = _polygon(placement['paper_quad'], width, height, quad=True)
            if cv2.contourArea(quad, oriented=True) <= 0:
                raise ValueError('paper quad must follow top-left, top-right, bottom-right, bottom-left order')
            if np.any(quad < [x0, y0]) or np.any(quad >= [x1, y1]):
                raise ValueError('added paper lies outside the reviewed output crop')
            paper = _mask(placement['paper_quad'], width, height, quad=True)
            allowed = _mask(bin_info['allowed_polygon'], width, height)
            if np.any(paper & ~allowed) or np.any(paper & occupied):
                raise ValueError('paper must fit its surface and not overlap another paper')
            for polygon in bin_info['exclude_polygons']:
                if np.any(paper & _mask(polygon, width, height)):
                    raise ValueError('paper overlaps an excluded surface')
            occlusion = np.zeros_like(paper)
            for polygon in bin_info['occluder_polygons']:
                occlusion |= _mask(polygon, width, height)
            if placement['identifiability'] == 'identifiable' and not np.any(paper & ~occlusion):
                raise ValueError('fully occluded paper cannot be identifiable')
            if stored and (placement['identifiability'] != 'identifiable' or not cv2.resize(
                    (paper & ~occlusion)[y0:y1, x0:x1].astype(np.uint8), output_size,
                    interpolation=cv2.INTER_NEAREST).any()):
                raise ValueError('qualified composite requires a visible label for each added paper')
            occupied |= paper
            seen_bins.add(placement['bin_id'])
            used.add(placement['class_id'])
        for key in ('gain', 'saturation'):
            value = p[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('tone must be finite and in (0, 1]')
        if 'blur_sigma' in p and (type(p['blur_sigma']) not in (int, float) or
                                   not math.isfinite(p['blur_sigma']) or not 0 <= p['blur_sigma'] <= 2):
            raise ValueError('blur sigma must be finite and in [0, 2]')
        sources = [p['photo']] + [f'{TOOLKIT}/SourcePic/{names[c]}.png' for c in sorted(used)]
        augmentation = {'gain': p['gain'], 'saturation': p['saturation'], 'placement': 'explicit_surface_quads'}
        if 'blur_sigma' in p:
            augmentation['blur_sigma'] = p['blur_sigma']
        background_kind = stored.get('background_kind', 'camera_photo') if stored else 'camera_photo'
        background = {'kind': background_kind, 'image': p['photo']}
        if 'record_index' in p:
            legacy_ref = state['background'].get('geometry_manifest') if stored else geometry_ref
            background.update(geometry_manifest=legacy_ref, record_index=p['record_index'])
        elif stored:
            background['record_id'] = stored['record_id']
        if (state['background'] != background or (stored and stored['split'] != plan['target_split'])
                or state['target'] != {'kind': 'explicit_bin_stickers', 'placements': placements}
                or state['labels'] != {'format': 'yolo_detection', 'status': plan['constraints']['labels']}
                or state['augmentation'] != augmentation
                or state['sources'] != sources):
            raise ValueError('surface declarations do not match the parameters')
        identity = json.dumps(p, sort_keys=True, ensure_ascii=False)
        planned_count = state.get('planned_count', 0)
        if (identity in identities or type(state['max_count']) is not int or state['max_count'] > 1
                or state['max_count'] < 0 or type(planned_count) is not int
                or not 0 <= planned_count <= state['max_count']):
            raise ValueError('explicit surface states may produce at most one image')
        identities.add(identity)
        if stored:
            retained_background_stickers(stored, placements, size, output_size, crop=crop)
    return plan


def retained_background_stickers(stored, placements, source_size, output_size, *, crop=None):
    """Clip and transform source labels in the reviewed output scope."""
    label = stored['label']
    width, height = source_size
    out_width, out_height = output_size
    left, top, right, bottom = crop or (0, 0, width, height)
    scale_x, scale_y = out_width / (right - left), out_height / (bottom - top)
    instances = label.get('instances') or []
    if label['presence'] == 'present' and not instances:
        raise ValueError('positive sticker background has no recorded instances')
    if label['presence'] == 'absent' and instances:
        raise ValueError('empty sticker background has recorded instances')
    paper_masks = [_mask(item['paper_quad'], width, height, quad=True) for item in placements]
    retained = []
    for item in instances:
        cls, box = item.get('class_id'), item.get('xyxy')
        if (type(cls) is not int or not 0 <= cls < 24 or
                not isinstance(box, list) or len(box) != 4 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in box)):
            raise ValueError('existing sticker requires a class and source-pixel box')
        bx0, by0, bx1, by1 = box
        if not 0 <= bx0 < bx1 <= width or not 0 <= by0 < by1 <= height:
            raise ValueError('existing sticker box lies outside the source image')
        x1, y1, x2, y2 = max(bx0, left), max(by0, top), min(bx1, right), min(by1, bottom)
        if x1 >= x2 or y1 >= y2:
            continue
        x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
        if any(mask[y1:y2, x1:x2].any() for mask in paper_masks):
            raise ValueError('added paper obscures an existing sticker; visible label is unresolved')
        scaled = [math.floor((x1-left)*scale_x), math.floor((y1-top)*scale_y),
                  math.ceil((x2-left)*scale_x), math.ceil((y2-top)*scale_y)]
        retained.append({'class_id': cls, 'source_box_xyxy': box, 'output_box_xyxy': scaled,
                         'object_id': item.get('object_id')})
    return retained


def _produce_surface(out, run, names, geometry, icons, seed, geometry_ref=GEOMETRY, formal=False, background_roles=None, working=None):
    """Write computed candidate boxes and preserve recorded background scope."""
    plan = validate_surface_plan(run.plan, names, geometry, geometry_ref, background_roles=background_roles)
    images, labels = out / 'images', out / ('labels' if formal else 'labels_candidate')
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    records, counts = [], Counter()
    for state in plan['strata']:
        p = state['parameters']
        record = surface_record(p, geometry, background_roles)
        stored = background_roles[p['photo']] if background_roles is not None else None
        by_id = {item['bin_id']: item for item in record['bins']}
        background = base.read_image(working.path(p['photo']) if working else ROOT / p['photo'])
        height, width = background.shape[:2]
        if [width, height] != record['image_size']:
            raise ValueError('photo dimensions differ from reviewed source coordinates')
        crop, (out_width, out_height) = surface_output(stored, (width, height), p.get('output_crop_xyxy'))
        left, top, right, bottom = crop
        retained = (retained_background_stickers(stored, p['placements'], (width, height),
                                                 (out_width, out_height), crop=crop) if stored else [])
        for _ in range(state['planned_count']):
            frame = background.copy()
            instances, lines = [], []
            for item in retained:
                x1, y1, x2, y2 = item['output_box_xyxy']
                cls = item['class_id']
                lines.append(f'{cls} {(x1+x2)/(2*out_width):.8f} {(y1+y2)/(2*out_height):.8f} {(x2-x1)/out_width:.8f} {(y2-y1)/out_height:.8f}')
                counts[names[cls]] += 1
            for placement in p['placements']:
                bin_info = by_id[placement['bin_id']]
                cls = placement['class_id']
                rgb, alpha = icons[cls]
                paper = np.full((128, 128, 3), 240, dtype=np.uint8)
                scale = min(102 / rgb.shape[1], 102 / rgb.shape[0])
                iw, ih = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
                ink = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
                a = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_AREA)[:, :, None]
                x, y = (128-iw)//2, (128-ih)//2
                paper[y:y+ih, x:x+iw] = np.clip(ink.astype(np.float32)*a + paper[y:y+ih, x:x+iw]*(1-a), 0, 255).astype(np.uint8)
                source_quad = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])
                transform = cv2.getPerspectiveTransform(source_quad, np.float32(placement['paper_quad']))
                warped = cv2.warpPerspective(paper, transform, (width, height))
                full = _mask(placement['paper_quad'], width, height, quad=True)
                visible = full.copy()
                for polygon in bin_info['occluder_polygons']:
                    visible &= ~_mask(polygon, width, height)
                frame[visible] = warped[visible]
                visible_out = cv2.resize(visible[top:bottom, left:right].astype(np.uint8),
                                         (out_width, out_height), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.where(visible_out)
                box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None
                identifiable = placement['identifiability'] == 'identifiable' and box is not None
                if identifiable:
                    x1, y1, x2, y2 = box
                    lines.append(f'{cls} {(x1+x2)/(2*out_width):.8f} {(y1+y2)/(2*out_height):.8f} {(x2-x1)/out_width:.8f} {(y2-y1)/out_height:.8f}')
                    counts[names[cls]] += 1
                instances.append({'bin_id': placement['bin_id'], 'bin_color': bin_info['bin_color'],
                                  'class_id': cls if identifiable else None,
                                  'planned_class_id': cls, 'identifiability': 'identifiable' if identifiable else 'unknown',
                                  'source_paper_quad': placement['paper_quad'], 'visible_bbox_xyxy': box})
            frame = cv2.resize(frame[top:bottom, left:right], (out_width, out_height), interpolation=cv2.INTER_AREA)
            frame = apply_tone(frame, value_gain=p['gain'], saturation_gain=p['saturation'])
            if p.get('blur_sigma', 0):
                frame = cv2.GaussianBlur(frame, (0, 0), p['blur_sigma'])
            stem = f"{plan['target_split']}_{len(records):06d}"
            image_path, label_path = images / (stem+'.png'), labels / (stem+'.txt')
            ok, encoded = cv2.imencode('.png', frame)
            if not ok:
                raise RuntimeError('could not encode generated image')
            encoded.tofile(str(image_path))
            label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf8')
            actual = {'source_photo': p['photo'],
                      'working_background': working.actual(p['photo']) if working else p['photo'],
                      'working_artworks': [working.actual(f'{TOOLKIT}/SourcePic/{names[item["class_id"]]}.png')
                                           for item in p['placements']] if working else [],
                      'background_role': stored, 'geometry_manifest': geometry_ref,
                      'geometry_record_index': p.get('record_index') if geometry_ref else None,
                      'source_size': [width, height], 'output_crop_xyxy': crop,
                      'output_size': [out_width, out_height],
                      'instances': instances, 'retained_background_instances': retained,
                      'gain': p['gain'], 'saturation': p['saturation'],
                      'blur_sigma': p.get('blur_sigma', 0), 'used_files': state['sources']}
            rendered_background = stored and stored.get('background_kind') == 'blender_render'
            records.append({'id': stem, 'state_id': state['id'], 'image': image_path.relative_to(ROOT).as_posix(),
                            'label_candidate': label_path.relative_to(ROOT).as_posix(),
                            'generation_method': 'blender_background_composite' if rendered_background else 'real_photo_composite',
                            'label_status': plan['constraints']['labels'], 'actual': actual})
            run.record_sample(state['id'], output=image_path, sources=state['sources'], parameters=p, actual=actual)
    (out/'classes.json').write_text(json.dumps(names, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    status = ('complete_background_labels_preserved' if background_roles
              else 'candidate_background_targets_unknown')
    (out/'manifest.json').write_text(json.dumps({'status': status, 'seed': seed,
        'plan': plan, 'counts_by_class': {name: counts[name] for name in names}, 'records': records},
        ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    from datasets.scripts.common.config import load_task_config
    from datasets.scripts.common.paths import sample_dir, split_dir
    _, source_config = load_task_config('waste_sticker')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--geometry-manifest', help='workspace-relative geometry JSONL for legacy candidate plans')
    args = parser.parse_args()
    plan_path, out = (ROOT/args.plan).resolve(), (ROOT/args.out).resolve()
    if not plan_path.is_relative_to(ROOT.resolve()) or not plan_path.is_file():
        parser.error('plan must be an existing workspace file')
    formal = out not in {sample_dir('waste_sticker').resolve(), sticker_surface_sample_dir().resolve()}
    toolkit = source_path('waste_sticker', 'artwork_toolkit')
    names = [x['name'] for x in yaml.safe_load((toolkit/'configs/classes.yaml').read_text(encoding='utf8'))['items']]
    artworks = artwork_catalog(source_config)
    if any(artworks[i] != (toolkit/'SourcePic'/f'{name}.png').relative_to(ROOT).as_posix() for i, name in enumerate(names)):
        raise ValueError('toolkit class order differs from configured artwork records')
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    surface_mode = declaration.get('constraints') in (SURFACE_CONSTRAINTS, RECORDED_SURFACE_CONSTRAINTS,
                                                      COMPLETE_SURFACE_CONSTRAINTS)
    recorded_mode = declaration.get('constraints') in (RECORDED_SURFACE_CONSTRAINTS,
                                                       COMPLETE_SURFACE_CONSTRAINTS)
    target_split = declaration['target_split']
    if target_split not in SPLITS:
        parser.error('plan must declare a train, val or test split')
    selected_photos = {state['parameters']['photo'] for state in declaration['strata']}
    geometry_path = geometry_ref = geometry = background_roles = None
    if recorded_mode:
        background_roles = surface_background_catalog(source_config, images=selected_photos, split=target_split)
    else:
        geometry_path = (ROOT / (args.geometry_manifest or source_config['real_photo_geometry_manifest'])).resolve()
        if not geometry_path.is_relative_to(ROOT.resolve()) or not geometry_path.is_file():
            parser.error('geometry manifest must be an existing workspace file')
        geometry_ref = geometry_path.relative_to(ROOT.resolve()).as_posix()
        geometry = load_geometry(geometry_path)
        designated = {row['image'] for row in real_source_catalog(source_config)}
        video_root = source_config['real_video_frames_root'] + '/'
        if any(photo.startswith(video_root) and photo not in designated for photo in selected_photos):
            raise ValueError('sticker frame lies outside source.yaml interval')
    if formal and (not recorded_mode or out != split_dir('waste_sticker', declaration['target_split']).resolve()):
        parser.error('formal output requires the declared split and complete stored background labels')
    if not surface_mode and geometry_ref != GEOMETRY:
        parser.error('legacy mode requires its original geometry manifest')
    def validate(candidate, candidate_names, candidate_geometry):
        if surface_mode:
            return validate_surface_plan(candidate, candidate_names, candidate_geometry, geometry_ref,
                                         background_roles=background_roles)
        return validate_plan(candidate, candidate_names, candidate_geometry)
    validate(declaration, names, geometry)
    working = WorkingInputs('waste_sticker', backgrounds=[s['parameters']['photo'] for s in declaration['strata']],
                            targets=artworks.values(), split=declaration['target_split'])
    icon_paths = [working.path(artworks[i]) for i in range(len(names))]
    icon_dirs = {path.parent for path in icon_paths}
    if len(icon_dirs) != 1 or any(path.name != f'{name}.png' for path, name in zip(icon_paths, names)):
        raise ValueError('Sticker generation requires all ordered icons in one directory')
    icon_dir = icon_dirs.pop()
    media = sorted({ROOT/p for state in declaration['strata'] for p in state['sources']} | working.media)
    procedural = [plan_path, *working.evidence, *[ROOT/p for p in source_config['records']], toolkit/'configs/classes.yaml', toolkit/'configs/gen_default.yaml',
                  ROOT/'datasets/scripts/waste_sticker/config/dataset.yaml',
                  ROOT/'datasets/scripts/waste_sticker/config/source.yaml']
    if background_roles:
        procedural.append(ROOT/source_config['databases']['frame_environment'])
        media = sorted(set(media) | {ROOT / row['source_video'] for row in background_roles.values()
                                   if row['source_video'] and (ROOT / row['source_video']).is_file()})
    else:
        procedural.extend((geometry_path, ROOT/source_config['real_annotations']))
    code = [Path(__file__), WORKING_INPUTS_CODE, Path(artwork_generator.__file__), Path(base.__file__),
            ROOT/'datasets/scripts/common/frame_environment.py',
            ROOT/'lab/training_common/camera_tone.py', ROOT/'datasets/scripts/common/config.py',
            ROOT/'datasets/scripts/common/paths.py', ROOT/'datasets/scripts/waste_sticker/source_inputs.py', toolkit/'tools/gen_dataset.py', toolkit/'tools/matting.py']
    with generation_guard('waste_sticker_real_photo:' + out.relative_to(ROOT).as_posix(), media, [out], vars(args),
                          code_paths=code, procedural_inputs=procedural, root=ROOT, production_plan=declaration,
                          sample_output_paths=[out/'images']) as run:
        if run.reused:
            print('Existing complete output reused:',out)
            return
        validate(run.plan, names, geometry)
        cache = out/'.asset_cache'
        artwork_generator.init_worker(toolkit, cache, out, 1280, 720, False, icon_dir=icon_dir)
        icons = []
        for name in names:
            with np.load(cache/(name+'.npz')) as asset:
                icons.append((asset['rgb'].copy(),asset['alpha'].astype(np.float32)))
        if surface_mode:
            _produce_surface(out, run, names, geometry, icons, args.seed, geometry_ref, formal, background_roles, working)
        else:
            _produce(out, run, names, geometry, icons, args.seed, working)


if __name__ == '__main__':
    main()
