"""Create source-grouped person/vehicle candidates on real scene ground crops.

The 0–5.97 second ground crops were visually checked for existing foreground
targets and wall reflections. They all come from one recorded video and are
training candidates, never an independent validation/test set. The 2D yaw
and pitch are bounded planar projections, not novel measured 3D viewpoints.
"""
import argparse
import csv
import json
import math
import random
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/e_bike/training/scripts'))
import preview_person_occlusions as assets

VIDEO = 'Archive/Assets/场景/综合视频/空场与场景物料_001.mp4'
BACKGROUND_ROOT = ROOT / 'datasets/raw/person/teammate_submission_20261110/dataset'
PEOPLE_ROOT = ROOT / 'datasets/raw/person/team_assets'
VEHICLE_SELECTION = ROOT / 'datasets/annotated/e_bike/multiview_masks_refined_20261110_v2/selection.json'
GROUND_CROP = (480, 600, 1440, 1080)
CONTEXT_CROP = (0, 360, 1440, 1080)
SIGN_CROP = (0, 120, 1920, 1080)
WIDTH, HEIGHT = 960, 480
PERSON_CLASSES = {'community': 0, 'noncommunity': 1}
VEHICLE_CLASSES = {'upright': 0, 'fallen': 1}


def path_in_workspace(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('source path outside workspace')
    return path


def read_color(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot decode image {path}')
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'cannot encode {path}')
    encoded.tofile(str(path))


def scenarios(index):
    """20-image cycle yields 25/30/30/15 percent scene types exactly."""
    cases = (('single', 1, 0, 'none'), ('multi', 2, 0, 'none'),
             ('occlusion', 1, 1, 'vehicle_front'), ('no_person', 0, 1, 'none'),
             ('single', 1, 0, 'none'), ('multi', 2, 0, 'none'),
             ('occlusion', 1, 1, 'person_front'), ('occlusion', 2, 2, 'vehicle_front'),
             ('no_person', 0, 1, 'none'), ('multi', 3, 0, 'none'),
             ('single', 1, 0, 'none'), ('single', 1, 0, 'none'),
             ('single', 1, 0, 'none'), ('multi', 2, 0, 'none'),
             ('multi', 3, 0, 'none'), ('multi', 2, 0, 'none'),
             ('occlusion', 1, 1, 'vehicle_front'), ('occlusion', 2, 2, 'person_front'),
             ('occlusion', 3, 2, 'vehicle_front'), ('no_person', 0, 0, 'none'))
    if index < 0:
        raise ValueError('negative index')
    kind, people, bikes, relation = cases[index % len(cases)]
    return {'kind': kind, 'people': people, 'vehicles': bikes, 'relation': relation}


def scenario_occurrence(index):
    """Within each scene type, cycle labels instead of correlating class to type."""
    current = scenarios(index)['kind']
    per_cycle = {'single': 5, 'multi': 6, 'occlusion': 6, 'no_person': 3}[current]
    return (index // 20) * per_cycle + sum(
        scenarios(previous)['kind'] == current for previous in range(index % 20))


def truncated_gauss(rng, mean, std, low, high):
    """Sample inside bounds by rejection, never clamping values to the edges."""
    if not (math.isfinite(mean) and math.isfinite(std) and math.isfinite(low)
            and math.isfinite(high) and std > 0 and low < high and low <= mean <= high):
        raise ValueError('invalid truncated Gaussian parameters')
    for _ in range(100):
        value = rng.gauss(mean, std)
        if low <= value <= high:
            return value
    raise RuntimeError('truncated Gaussian sampling exceeded retry limit')


def lighting_level(occurrence):
    if occurrence < 0:
        raise ValueError('invalid occurrence')
    return ('dim', 'bright', 'normal')[occurrence % 3]


def background_kind_for(index, plan):
    if index % 3 == 0:
        return 'ground_only'
    if index % 3 == 2 and plan['people'] <= 1 and plan['vehicles'] <= 1:
        return 'sign_context'
    return 'track_context'


def illumination_stratum(index):
    """Separate each feasible scene/background/class/relation combination."""
    plan = scenarios(index)
    occurrence = scenario_occurrence(index)
    person = occurrence % 2 if plan['people'] else None
    vehicle = (occurrence // 2) % 2 if plan['vehicles'] else None
    return (plan['kind'], background_kind_for(index, plan), plan['relation'], person, vehicle)


@lru_cache(maxsize=1)
def illumination_period():
    """240 samples cover the periods of scene, background, and class plans."""
    positions = {}
    for index in range(240):
        positions.setdefault(illumination_stratum(index), []).append(index)
    return positions


def stratified_lighting_level(index):
    """Cycle light inside each stratum without scene/class confounding."""
    key = illumination_stratum(index)
    slots = illumination_period()[key]
    local_index = slots.index(index % 240)
    occurrence = (index // 240) * len(slots) + local_index
    offset = sum(ord(char) for char in repr(key)) % 3
    return lighting_level(occurrence + offset)


def project_corners(width, height, yaw, pitch, roll):
    """Project a flat image with small yaw/pitch/roll, without a reflection."""
    if width < 2 or height < 2 or abs(yaw) > 15 or abs(pitch) > 8 or abs(roll) > 4:
        raise ValueError('invalid image or projection angles')
    yr, pr, rr = map(math.radians, (yaw, pitch, roll))
    result = []
    for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        depth = 1.0 + 0.35 * (x * math.sin(yr) + y * math.sin(pr))
        px, py = x * math.cos(yr) / depth, y * math.cos(pr) / depth
        px, py = px * width, py * height
        result.append((px * math.cos(rr) - py * math.sin(rr),
                       px * math.sin(rr) + py * math.cos(rr)))
    pts = np.asarray(result, dtype=np.float32)
    first, second = pts[1] - pts[0], pts[2] - pts[1]
    cross = first[0]*second[1] - first[1]*second[0]
    if cross <= 0:
        raise ValueError('projection reflects or folds the foreground')
    return pts


def planar_warp(image, yaw, pitch, roll):
    height, width = image.shape[:2]
    src = np.float32(((0, 0), (width-1, 0), (width-1, height-1), (0, height-1)))
    dst = project_corners(width, height, yaw, pitch, roll)
    minimum, maximum = dst.min(axis=0), dst.max(axis=0)
    dst -= minimum
    out_width, out_height = (int(math.ceil(x)) + 2 for x in maximum - minimum)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (out_width, out_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3] > 8
    y, x = np.nonzero(alpha)
    if not len(x):
        raise ValueError('empty projected sprite')
    return warped[y.min():y.max()+1, x.min():x.max()+1]


def alpha_place(canvas, rgba, center_x, foot_y):
    """Composite an entirely in-frame sprite and return its binary occupancy."""
    height, width = canvas.shape[:2]
    sh, sw = rgba.shape[:2]
    left, top = round(center_x - sw / 2), round(foot_y - sh)
    right, bottom = left + sw, top + sh
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError('sprite outside frame')
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    canvas[top:bottom, left:right] = np.clip(
        rgba[:, :, :3] * alpha + canvas[top:bottom, left:right] * (1-alpha), 0, 255).astype(np.uint8)
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[top:bottom, left:right] = rgba[:, :, 3] > 127
    if occupancy.sum() < 100:
        raise ValueError('sprite too small')
    return occupancy


def visible_masks(masks):
    """Input masks are ordered back to front, output same-order visible masks."""
    if not masks:
        return []
    if any(mask.shape != masks[0].shape for mask in masks):
        raise ValueError('mask shape mismatch')
    front = np.zeros(masks[0].shape, dtype=bool)
    result = [None] * len(masks)
    for i in range(len(masks)-1, -1, -1):
        mask = np.asarray(masks[i], dtype=bool)
        result[i] = mask & ~front
        front |= mask
    return result


def has_requested_overlap(masks, kinds, relation):
    """Check actual pixel overlap with the requested back-to-front ordering."""
    if len(masks) != len(kinds) or relation not in ('vehicle_front', 'person_front'):
        raise ValueError('bad overlap inputs')
    behind, front = ('person', 'vehicle') if relation == 'vehicle_front' else ('vehicle', 'person')
    return any(kinds[i] == behind and kinds[j] == front and np.any(masks[i] & masks[j])
               for i in range(len(masks)) for j in range(i+1, len(masks)))


def yolo_row(class_id, bbox, width, height):
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('out-of-frame bbox')
    return f'{class_id} {(x0+x1)/(2*width):.6f} {(y0+y1)/(2*height):.6f} {(x1-x0)/width:.6f} {(y1-y0)/height:.6f}'


def exposure(image, gain, gamma):
    if not 0.78 <= gain <= 1.20 or not 0.92 <= gamma <= 1.10:
        raise ValueError('exposure outside bounded range')
    x = image.astype(np.float32) / 255
    return np.clip(np.power(np.clip(x * gain, 0, 1), gamma) * 255, 0, 255).astype(np.uint8)


def eligible_backgrounds():
    with (BACKGROUND_ROOT / 'reports/backgrounds_manifest.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    chosen = []
    for row in rows:
        if 0 <= int(row['frame']) <= 179:
            path = (BACKGROUND_ROOT / 'backgrounds' / row['file']).resolve()
            if not path.is_relative_to((BACKGROUND_ROOT / 'backgrounds').resolve()) or not path.is_file():
                raise ValueError('missing background or unsafe filename')
            chosen.append({'path': path, 'frame': int(row['frame']), 'seconds': float(row['t'])})
    if len(chosen) != 50:
        raise ValueError('source changed: expected 50 visually reviewed crop frames')
    sign = BACKGROUND_ROOT / 'backgrounds/bg_00137_f00720.jpg'
    if not sign.is_file():
        raise ValueError('missing visually reviewed sign-context background')
    return chosen


def background_choice(index, plan, backgrounds, rng):
    """Separate complete person/vehicle supervision from person-only context."""
    kind = background_kind_for(index, plan)
    if kind == 'ground_only':
        kind, crop, supervised = 'ground_only', GROUND_CROP, True
        background = rng.choice(backgrounds)
    elif kind == 'sign_context':
        kind, crop, supervised = 'sign_context', SIGN_CROP, False
        background = {'path': BACKGROUND_ROOT / 'backgrounds/bg_00137_f00720.jpg',
                      'frame': 720, 'seconds': 24.0}
    else:
        kind, crop, supervised = 'track_context', CONTEXT_CROP, False
        background = rng.choice(backgrounds)
    return background, kind, crop, supervised


def sample_person_anchors(rng, count, background_kind):
    if background_kind == 'ground_only':
        left, right = 130, 830
    elif background_kind == 'track_context':
        left, right = 410, 820
    elif background_kind == 'sign_context':
        if count > 1:
            raise ValueError('sign context accepts at most one person')
        left, right = 430, 490
    else:
        raise ValueError('unknown background kind')
    positions = []
    for _ in range(count):
        for _ in range(100):
            x = rng.uniform(left, right)
            if all(abs(x-previous) >= 90 for previous in positions):
                positions.append(x)
                break
        else:
            raise ValueError('cannot place distinct person anchors')
    return positions


def validate_actor_placement(actor, background_kind):
    sh, sw = actor['sprite'].shape[:2]
    left = actor['foot_x'] - sw/2
    if background_kind == 'track_context' and left < 340:
        raise ValueError('actor would overlap existing vehicles or barrier')
    if background_kind == 'sign_context':
        if left < 380:
            raise ValueError('actor would overlap existing sign')
        boundary_y = 260 + max(0, actor['foot_x'] - 265) * .6875
        if actor['foot_y'] < boundary_y + 8:
            raise ValueError('actor footpoint is not on visible ground')
    if actor['foot_y'] > HEIGHT - 5 or actor['foot_y'] - sh < 0:
        raise ValueError('actor vertical placement outside frame')


def load_sources():
    people = {}
    for cls, folder, expected in (('community', '社区人员', 16), ('noncommunity', '非社区人员', 2)):
        paths = sorted((PEOPLE_ROOT / folder).glob('*.png'))
        if len(paths) != expected:
            raise ValueError(f'expected {expected} {cls} source images')
        people[cls] = [(p, assets.load_rgba(p)) for p in paths]
    selection = json.loads(VEHICLE_SELECTION.read_text(encoding='utf8'))
    if selection.get('split') is not None:
        raise ValueError('vehicle sources must remain unsplit')
    vehicles = {'upright': [], 'fallen': []}
    for item in selection['vehicles']:
        if item['class'] not in vehicles:
            raise ValueError('unknown vehicle class')
        for key in ('path', 'source_frame', 'source_video'):
            if not path_in_workspace(item[key]).is_file():
                raise ValueError(f'missing {key}')
        vehicles[item['class']].append((item, assets.load_rgba(path_in_workspace(item['path']))))
    if any(len(items) < 2 for items in vehicles.values()):
        raise ValueError('need at least two source masks for each vehicle pose')
    return people, vehicles


def random_actor(rng, kind, category, image, source, foot_x, foot_y, person_height):
    angles = {'yaw': truncated_gauss(rng, 0, 5, -12, 12),
              'pitch': truncated_gauss(rng, 0, 2.5, -6, 6),
              'roll': truncated_gauss(rng, 0, 1, -3, 3)}
    projected = planar_warp(image, **angles)
    if kind == 'person':
        target_h = round(person_height)
        sprite = assets.resized(projected, height=target_h)
    elif category == 'upright':
        target_h = round(person_height * rng.uniform(.68, .87))
        sprite = assets.resized(projected, height=target_h)
    else:
        target_w = round(person_height * rng.uniform(.85, 1.12))
        sprite = assets.resized(projected, width=target_w)
    relative_exposure = truncated_gauss(rng, 1.0, .035, .90, 1.08)
    sprite = sprite.copy()
    sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.float32) * relative_exposure, 0, 255).astype(np.uint8)
    alpha = sprite[:, :, 3] > 127
    if not alpha.any() or float(cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_BGR2GRAY)[alpha].mean()) < 30:
        raise ValueError('foreground is too dark')
    return {'kind': kind, 'category': category, 'source': source, 'sprite': sprite,
            'foot_x': float(foot_x), 'foot_y': float(foot_y), 'angles_degrees': angles,
            'relative_exposure': relative_exposure, 'person_reference_height_px': person_height,
            'sprite_width_px': int(sprite.shape[1]), 'sprite_height_px': int(sprite.shape[0])}


def draw_shadow(canvas, actor):
    sh, sw = actor['sprite'].shape[:2]
    layer = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(layer, (round(actor['foot_x']), round(actor['foot_y'])),
                (max(3, round(sw*.28)), max(2, round(sh*.018))), 0, 0, 360, 255, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), max(2, sh*.015)).astype(np.float32) / 255
    factor = (1 - .22*layer)[:, :, None]
    canvas[:] = (canvas.astype(np.float32) * factor).astype(np.uint8)


def generate_one(index, rng, backgrounds, people, vehicles):
    plan = scenarios(index)
    occurrence = scenario_occurrence(index)
    background, background_kind, crop, vehicle_supervised = background_choice(index, plan, backgrounds, rng)
    original = read_color(background['path'])
    if original.shape[:2] != (1080, 1920):
        raise ValueError('unexpected source dimensions')
    x0, y0, x1, y1 = crop
    base = original[y0:y1, x0:x1].copy()
    if base.shape[:2] != (HEIGHT, WIDTH):
        base = cv2.resize(base, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    level = stratified_lighting_level(index)
    if level == 'dim':
        exposure_gain = truncated_gauss(rng, .86, .025, .82, .89)
    elif level == 'bright':
        exposure_gain = truncated_gauss(rng, 1.125, .018, 1.10, 1.15)
    else:
        exposure_gain = truncated_gauss(rng, 1, .045, .90, 1.08)
    gamma = truncated_gauss(rng, 1, .025, .95, 1.06)
    actors = []
    anchors = []
    person_x = sample_person_anchors(rng, plan['people'], background_kind)
    for i in range(plan['people']):
        cls = 'community' if (occurrence + i) % 2 == 0 else 'noncommunity'
        source, image = rng.choice(people[cls])
        foot_x = person_x[i]
        if background_kind == 'sign_context':
            foot_y = HEIGHT * truncated_gauss(rng, .95, .012, .93, .975)
        else:
            foot_y = HEIGHT * truncated_gauss(rng, .75, .07, .60, .90)
        height = 110 + (foot_y/HEIGHT-.60)*290 + truncated_gauss(rng, 0, 5, -12, 12)
        actor = random_actor(rng, 'person', cls, image, source.relative_to(ROOT).as_posix(), foot_x, foot_y, height)
        actors.append(actor)
        anchors.append((foot_x, foot_y, height))
    for i in range(plan['vehicles']):
        cls = 'upright' if (occurrence // 2 + i) % 2 == 0 else 'fallen'
        source, image = rng.choice(vehicles[cls])
        if not anchors:
            foot_x = sample_person_anchors(rng, 1, background_kind)[0]
            if background_kind == 'sign_context':
                foot_y = HEIGHT * truncated_gauss(rng, .95, .012, .93, .975)
            else:
                foot_y = HEIGHT * truncated_gauss(rng, .75, .07, .60, .90)
            height = 110 + (foot_y/HEIGHT-.60)*290 + truncated_gauss(rng, 0, 5, -12, 12)
            actors.append(random_actor(rng, 'vehicle', cls, image, source, foot_x, foot_y, height))
            continue
        px, py, height = anchors[i % len(anchors)]
        sign = -1 if i % 2 else 1
        if plan['relation'] == 'separate':
            foot_x, foot_y = px + sign * rng.uniform(150, 190), py + rng.uniform(-5, 5)
        elif plan['relation'] == 'vehicle_front':
            foot_x, foot_y = px + sign * rng.uniform(15, 35), py + rng.uniform(5, 12)
        else:
            foot_x, foot_y = px + sign * rng.uniform(15, 35), py - rng.uniform(5, 12)
        actors.append(random_actor(rng, 'vehicle', cls, image, source, foot_x, foot_y, height))
    actors.sort(key=lambda a: a['foot_y'])
    for actor in actors:
        validate_actor_placement(actor, background_kind)
    canvas = base.copy()
    for actor in actors:
        draw_shadow(canvas, actor)
    masks = [alpha_place(canvas, actor['sprite'], actor['foot_x'], actor['foot_y']) for actor in actors]
    if plan['kind'] == 'occlusion' and not has_requested_overlap(
            masks, [actor['kind'] for actor in actors], plan['relation']):
        raise ValueError('requested foreground order has no actual pixel overlap')
    visible = visible_masks(masks)
    canvas = exposure(canvas, exposure_gain, gamma)
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    if not 35 <= float(gray.mean()) <= 220:
        raise ValueError('scene exposure outside usable range')
    metadata = []
    for actor, full, shown in zip(actors, masks, visible):
        fraction = shown.sum() / full.sum()
        if fraction < .55 or shown.sum() < 120:
            raise ValueError('occluded target is too ambiguous; resample scene')
        visible_luma = float(gray[shown].mean())
        if not 30 <= visible_luma <= 240:
            raise ValueError('visible target exposure outside usable range')
        box = assets.box(shown)
        meta = {k: actor[k] for k in ('kind', 'category', 'source', 'foot_x', 'foot_y',
                                      'angles_degrees', 'relative_exposure', 'person_reference_height_px',
                                      'sprite_width_px', 'sprite_height_px')}
        meta.update({'visible_bbox_xyxy': box, 'visible_fraction': float(fraction),
                     'visible_luma_mean': visible_luma,
                     'bbox_convention': 'xyxy pixels, right/bottom exclusive'})
        metadata.append(meta)
    return canvas, masks, visible, metadata, {'image': background['path'].relative_to(ROOT).as_posix(),
            'source_video': VIDEO, 'source_group': 'scene_video_49s_shared',
            'frame': background['frame'], 'seconds': background['seconds'], 'crop_xyxy': list(crop),
            'kind': background_kind, 'resize_to': [WIDTH, HEIGHT],
            'person_supervision': 'AI_visual_review_no_obvious_background_person',
            'vehicle_supervision': 'allowed_ground_crop' if vehicle_supervised else 'disabled_unlabelled_existing_or_reflected_vehicles',
            'background_review': 'all 50 crop thumbnails checked' if background_kind != 'sign_context' else 'single sign scene checked at full resolution'}, \
        {'level': level, 'stratum': list(illumination_stratum(index)),
         'scene_gain': exposure_gain, 'gamma': gamma,
         'method': 'bounded global exposure and per-sprite gain'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--seed', type=int, default=7901)
    args = ap.parse_args()
    output = path_in_workspace(args.output)
    if not output.is_relative_to(ROOT / 'datasets/exports/person') or output.exists():
        ap.error('output must be a new directory under datasets/exports/person')
    if not 1 <= args.count <= 100000:
        ap.error('count must be 1..100000')
    backgrounds = eligible_backgrounds()
    people, vehicles = load_sources()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('person/images', 'person/labels', 'vehicle/images', 'vehicle/labels', 'masks/full', 'masks/visible'):
        (output / folder).mkdir(parents=True)
    summary = {'status': 'running', 'requested': args.count, 'completed': 0, 'seed': args.seed,
               'source_group': 'scene_video_49s_shared', 'split': 'train_candidate',
               'source_video': VIDEO,
               'background_modes': {'ground_only': {'crop_xyxy': list(GROUND_CROP), 'vehicle_supervision': True},
                                    'track_context': {'crop_xyxy': list(CONTEXT_CROP), 'vehicle_supervision': False},
                                    'sign_context': {'crop_xyxy': list(SIGN_CROP), 'vehicle_supervision': False}},
               'class_names': {'person': PERSON_CLASSES, 'vehicle': VEHICLE_CLASSES},
               'limits': 'one real-video source; planar projection, not measured 3D; source masks need visual review',
               'illumination': 'bounded global exposure with small per-sprite variation',
               'distribution': {'method': 'bounded Gaussian via rejection, no clipping',
                                'yaw_degrees': [0, 5, -12, 12], 'pitch_degrees': [0, 2.5, -6, 6],
                                'roll_degrees': [0, 1, -3, 3], 'sprite_gain': [1, .035, .90, 1.08],
                                'scene_gain_dim': [.86, .025, .82, .89],
                                'scene_gain_bright': [1.125, .018, 1.10, 1.15],
                                'scene_gain_normal': [1, .045, .90, 1.08],
                                'gamma': [1, .025, .95, 1.06],
                                'format': '[mean, standard deviation, minimum, maximum]',
                                'allocation': 'scene exact 25/30/30/15; light balanced within each scene/background/class/relation stratum'}}
    def save_summary():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save_summary()
    try:
        with (output/'manifest.jsonl').open('w', encoding='utf8') as records:
            for index in range(args.count):
                rng = random.Random(args.seed + index*1009)
                for attempt in range(1, 101):
                    try:
                        image, full, visible, objects, bg, light = generate_one(index, rng, backgrounds, people, vehicles)
                        break
                    except ValueError:
                        if attempt == 100:
                            raise RuntimeError(f'scene {index} could not meet visibility constraints')
                stem = f'scene_{index+1:06d}'
                image_path = output/'person/images'/(stem+'.png')
                write_image(image_path, image)
                vehicle_supervised = bg['vehicle_supervision'] == 'allowed_ground_crop'
                if vehicle_supervised:
                    shutil.copyfile(image_path, output/'vehicle/images'/(stem+'.png'))
                rows = {'person': [], 'vehicle': []}
                for n, (meta, fm, vm) in enumerate(zip(objects, full, visible)):
                    fm_path, vm_path = f'masks/full/{stem}_{n:02d}.png', f'masks/visible/{stem}_{n:02d}.png'
                    write_image(output/fm_path, fm.astype(np.uint8)*255)
                    write_image(output/vm_path, vm.astype(np.uint8)*255)
                    meta.update({'full_mask': fm_path, 'visible_mask': vm_path})
                    classes = PERSON_CLASSES if meta['kind'] == 'person' else VEHICLE_CLASSES
                    rows[meta['kind']].append(yolo_row(classes[meta['category']], meta['visible_bbox_xyxy'], WIDTH, HEIGHT))
                for kind in ('person', 'vehicle'):
                    if kind == 'vehicle' and not vehicle_supervised:
                        continue
                    text = '\n'.join(rows[kind]) + ('\n' if rows[kind] else '')
                    (output/kind/'labels'/(stem+'.txt')).write_text(text, encoding='utf8')
                record = {'image': stem+'.png', 'split': 'train_candidate', 'attempt': attempt,
                          'scenario': scenarios(index), 'background': bg, 'lighting': light, 'objects': objects,
                          'person_image': f'person/images/{stem}.png', 'person_label': f'person/labels/{stem}.txt',
                          'vehicle_image': f'vehicle/images/{stem}.png' if vehicle_supervised else None,
                          'vehicle_label': f'vehicle/labels/{stem}.txt' if vehicle_supervised else None,
                          'status': 'AI_synthetic_candidate_requires_visual_review'}
                records.write(json.dumps(record, ensure_ascii=False)+'\n'); records.flush()
                summary['completed'] = index+1; save_summary()
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'; summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save_summary()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
