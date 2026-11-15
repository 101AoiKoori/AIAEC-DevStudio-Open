"""Generate bounded bike scenes on reviewed ground, keeping source and masks."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASK_HELPERS = ROOT / 'datasets/scripts/e_bike'
sys.path[:0] = [str(ROOT), str(ROOT/'lab/training_common')]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from datasets.scripts.e_bike import synthesize_vehicle_scene as bike
from datasets.scripts.common.perspective import planar_warp  # noqa: E402
from datasets.scripts.common.composite import alpha_place, visible_masks  # noqa: E402

from datasets.scripts.common.config import source_path
BACKGROUND = source_path('e_bike', 'ground_background_manifest')
SEED = 20261118
TONES = {'original':(1.,1.), 'mild':(.9,.85), 'moderate':(.7,.6)}
# Manually fitted ground zones on reviewed 640x480 crops, not physical calibration.
ZONES = {
    'frame_005751': {'poly':[(0,243),(99,243),(405,479),(0,479)], 'x':(65,180),'y':(380,455)},
    'frame_002850': {'poly':[(0,280),(640,270),(640,479),(0,479)], 'x':(75,215),'y':(365,410)},
    'frame_007000': {'poly':[(0,0),(640,0),(640,479),(0,479)], 'x':(435,565),'y':(365,440)},
    'frame_007045': {'poly':[(0,190),(640,100),(640,479),(0,479)], 'x':(380,560),'y':(365,445)},
}
WHITE_BACKS = [(14,278,88,426),(135,276,205,416)]  # approx boxes on background_002850


def rel(p):
    return p.relative_to(ROOT).as_posix()


def grounded(actor, zone):
    alpha = actor['sprite'][:,:,3] > 127
    yy, xx = np.nonzero(alpha)
    if not len(xx):
        return False
    top = round(actor['foot_y']-alpha.shape[0])
    left = round(actor['foot_x']-alpha.shape[1]/2)
    # Check the actual silhouette contact point, not its transparent canvas edge.
    foot_y = top+int(yy.max())
    footprint_x = left+int(np.median(xx[yy >= yy.max()-2]))
    poly = np.asarray(zone['poly'],np.int32)
    return cv2.pointPolygonTest(poly,(float(footprint_x),float(foot_y)),False) >= 0


def _produce():
    raise RuntimeError('e-bike grounded generation is disabled: explicit planned-state execution is not implemented')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=48)
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    if not out.is_relative_to((ROOT/'datasets/exports/e_bike').resolve()) or out.exists():
        raise ValueError('Output must be a new e-bike export directory')
    if not 1 <= args.count <= 5000:
        raise ValueError('count must be in 1..5000')
    backgrounds = json.loads(BACKGROUND.read_text(encoding='utf-8'))['backgrounds']
    if set(b['id'] for b in backgrounds) != set(ZONES):
        raise ValueError('Background set differs from the fitted ground zones')
    known = build_queue(ROOT)
    for bg in backgrounds:
        if bg['task_state'] != 'candidate_complete_negative' or source_role(ROOT, ROOT/bg['parent_image'],known)[0] == 'regression':
            raise ValueError('Background is not eligible: '+bg['id'])
    people, vehicles = bike.sources(include_people=True)
    scene = bike.scene
    for folder in ('images/train','labels/train','masks'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    records, counts, skipped, negative_states = [], Counter(), Counter(), set()
    for index in range(args.count):
        bg = backgrounds[index%len(backgrounds)]
        zone = ZONES[bg['id']]
        slot = (index//4)%4
        scenario = ('upright','fallen','mixed','negative')[slot]
        # Negatives must be genuine source frames; repeated tone variants retain one parent.
        classes = {'upright':['upright'],'fallen':['fallen'],'mixed':['upright','fallen'],'negative':[]}[scenario]
        tone = tuple(TONES)[(index//16)%3]
        if not classes:
            key = (bg['parent_image'],tone)
            if key in negative_states:
                skipped['repeated_negative_parent_tone'] += 1
                continue
            negative_states.add(key)
        with_person = bool(classes) and (index//48)%2 == 1 and bg['id'] != 'frame_002850'
        rng = random.Random(SEED+index)
        result = None
        for attempt in range(35):
            frame = read_image(ROOT/bg['image'])
            if frame.shape != (480,640,3):
                raise ValueError('Background dimensions changed')
            try:
                ref_height = rng.uniform(105,155)  # fitted pixel band, not measured optics
                center = rng.uniform(*zone['x'])
                foot = rng.uniform(*zone['y'])
                actors = []
                for j,category in enumerate(classes):
                    source,sprite = rng.choice(vehicles[category])
                    x = center + (j-(len(classes)-1)/2)*ref_height*.46
                    actor = scene.random_actor(rng,'vehicle',category,sprite,source,x,foot+j*2,ref_height)
                    if not grounded(actor,zone):
                        raise ValueError('bike contact outside ground zone')
                    actors.append(actor)
                if with_person:
                    category = rng.choice(('community','noncommunity'))
                    source,sprite = rng.choice(people[category])
                    # Person stands slightly in front or behind one bike.
                    bike_actor = actors[0]
                    actor = scene.random_actor(rng,'person',category,sprite,rel(source),
                        bike_actor['foot_x']+rng.uniform(-.25,.25)*ref_height,
                        foot+rng.choice((-5,5)),ref_height*rng.uniform(.85,1.05))
                    if not grounded(actor,zone):
                        raise ValueError('person contact outside ground zone')
                    actors.append(actor)
                actors.sort(key=lambda a:a['foot_y'])
                for actor in actors:
                    scene.draw_shadow(frame,actor)
                full = [alpha_place(frame,a['sprite'],a['foot_x'],a['foot_y']) for a in actors]
                visible = visible_masks(full)
                for actor,mask,shown in zip(actors,full,visible):
                    if actor['kind']=='vehicle' and (mask.sum()<120 or shown.sum()/mask.sum()<.7 or shown.sum()<120):
                        raise ValueError('vehicle is too occluded')
                person_overlap = any(np.any(full[i]&full[j]) for i,a in enumerate(actors) for j,b in enumerate(actors)
                                     if a['kind']=='vehicle' and b['kind']=='person')
                if with_person and not person_overlap:
                    raise ValueError('requested person/bike overlap absent')
                result = frame,actors,full,visible,person_overlap,attempt+1
                break
            except ValueError:
                continue
        if result is None:
            skipped['placement_or_visibility'] += 1
            continue
        frame,actors,full,visible,person_overlap,attempts = result
        value,saturation = TONES[tone]
        if tone != 'original':
            frame = apply_tone(frame,value_gain=value,saturation_gain=saturation)
        stem = f'bike_{index:06d}'
        image_path,label_path=out/'images/train'/(stem+'.png'),out/'labels/train'/(stem+'.txt')
        write_image(image_path,frame)
        labels,objects=[],[]
        whiteback_overlap=False
        for j,(actor,mask,shown) in enumerate(zip(actors,full,visible)):
            if actor['kind'] != 'vehicle':
                continue
            y,x=np.nonzero(shown);box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
            labels.append(scene.yolo_row({'upright':0,'fallen':1}[actor['category']],box,640,480))
            mask_path=out/'masks'/f'{stem}_{j:02d}.png'
            write_image(mask_path,shown.astype(np.uint8)*255)
            if bg['id']=='frame_002850':
                whiteback_overlap |= any(np.any(mask[t:b,l:r]) for l,t,r,b in WHITE_BACKS)
            objects.append({'class':actor['category'],'bbox_xyxy':box,'visible_mask':rel(mask_path),
                'visible_fraction':float(shown.sum()/mask.sum()),'foreground_source':actor['source'],
                'foot_xy':[actor['foot_x'],actor['foot_y']], 'angles_degrees':actor['angles_degrees']})
            counts['instances_'+actor['category']]+=1
        label_path.write_text(('\n'.join(labels)+'\n') if labels else '',encoding='utf-8')
        records.append({'image':rel(image_path),'label':rel(label_path),'split':'train_candidate',
            'background':bg,'ground_zone':zone,'scenario':scenario,'objects':objects,
            'other_actors':[{'kind':a['kind'],'source':a['source'],'foot_xy':[a['foot_x'],a['foot_y']]}
                            for a in actors if a['kind']!='vehicle'],
            'person_bike_mask_overlap':person_overlap,'white_back_bbox_overlap_candidate':whiteback_overlap,
            'tone':tone,'tone_parameters':{'value_gain':value,'saturation_gain':saturation},
            'attempts':attempts,'image_size':[640,480],'generation_method':'real_scene_composite',
            'label_review':'visible_mask_candidate_batch_QA_pending'})
        counts['images_'+scenario]+=1;counts['tone_'+tone]+=1
        if person_overlap:counts['person_bike_overlap']+=1
        if whiteback_overlap:counts['white_back_overlap_candidate']+=1
    manifest={'status':'draft_QA_pending','seed':SEED,'requested':args.count,'counts':dict(counts),
        'skipped':dict(skipped),'background_manifest':rel(BACKGROUND),
        'scope':'Four background frames from one site video; shared vehicle and standee assets. '
                'Ground polygons and target scale are pixel fits, not measured geometry. '
                'No independent collection, test split or model result.', 'records':records}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':rel(out),'counts':dict(counts),'skipped':dict(skipped)},ensure_ascii=False))


GROUND_CONSTRAINTS = {'ground_contact': True, 'minimum_visible_pixels': 120,
                      'minimum_visible_fraction': .7, 'allow_truncation': False}
SAFE_CONSTRAINTS = dict(GROUND_CONSTRAINTS, reviewed_safe_regions=True)
GROUND_LABELS = {'format': 'yolo_visible_box', 'classes': {'upright': 0, 'fallen': 1}}


def ground_zone(background):
    """Return the declared 640x480 ground polygon, or a legacy fitted zone."""
    zone = background.get('ground_zone')
    if zone is None:
        zone = ZONES.get(background['id'])
    if not isinstance(zone, dict) or not isinstance(zone.get('poly'), (list, tuple)):
        raise ValueError('background requires a ground polygon')
    points = zone['poly']
    if not 3 <= len(points) <= 100 or any(
            not isinstance(point, (list, tuple)) or len(point) != 2 or
            any(type(value) is not int for value in point) or
            not 0 <= point[0] <= 640 or not 0 <= point[1] <= 480 for point in points):
        raise ValueError('ground polygon must use 640x480 integer pixel coordinates')
    if cv2.contourArea(np.asarray(points, dtype=np.int32)) <= 0:
        raise ValueError('ground polygon must enclose an area')
    return zone


def validate_backgrounds(backgrounds):
    if not isinstance(backgrounds, list) or not backgrounds:
        raise ValueError('background manifest requires a nonempty backgrounds list')
    ids = set()
    for bg in backgrounds:
        if not isinstance(bg, dict) or not isinstance(bg.get('id'), str) or not bg['id'] or bg['id'] in ids:
            raise ValueError('background ids must be unique nonempty strings')
        ids.add(bg['id'])
        for key in ('image', 'parent_image', 'source_video'):
            value = bg.get(key)
            if not isinstance(value, str) or not value or not (ROOT / value).resolve().is_relative_to(ROOT):
                raise ValueError('background paths must be workspace-relative')
            if Path(value).is_absolute():
                raise ValueError('background paths must be workspace-relative')
        ground_zone(bg)


def validate_ground_plan(plan, backgrounds, vehicles, known):
    top = {'schema_version', 'model_key', 'target_split', 'purpose', 'budget', 'uncertainty_weight',
           'independence_basis', 'constraints', 'strata'}
    fields = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight', 'failure_cost',
              'conditions', 'background', 'target', 'labels', 'augmentation', 'parameters', 'sources'}
    if (not isinstance(plan, dict) or set(plan) not in (top, top | {'sampling'}, top | {'allocation_evidence'}) or
            plan['model_key'] != 'e_bike' or plan['target_split'] not in ('train', 'test', 'val')):
        raise ValueError('unsupported plan, model or split')
    if 'sampling' in plan and (not isinstance(plan['sampling'], dict) or not plan['sampling']):
        raise ValueError('sampling metadata must be a nonempty object')
    if 'allocation_evidence' in plan and (not isinstance(plan['allocation_evidence'], dict) or not plan['allocation_evidence']):
        raise ValueError('allocation evidence must be a nonempty object')
    if (plan['constraints'] not in (GROUND_CONSTRAINTS, SAFE_CONSTRAINTS, dict(GROUND_CONSTRAINTS, allow_truncation=True)) or
            type(plan['constraints'].get('ground_contact')) is not bool or
            type(plan['constraints'].get('allow_truncation')) is not bool):
        raise ValueError('unsupported constraints')
    if type(plan['budget']) is not int or not 1 <= plan['budget'] <= 5000:
        raise ValueError('budget must be between 1 and 5000')
    if not isinstance(plan['strata'], list) or not plan['strata']:
        raise ValueError('a nonempty state plan is required')
    validate_backgrounds(backgrounds)
    seen = set()
    background_roles = {}
    for state in plan['strata']:
        if not isinstance(state, dict) or set(state) != fields or state['conditions'] != []:
            raise ValueError('unsupported state fields or conditions')
        if type(state['max_count']) is not int or state['max_count'] > 1:
            raise ValueError('each exact state can generate at most one image')
        bg = state['background']
        if not isinstance(bg, dict) or set(bg) != {'id', 'image'}:
            raise ValueError('unsupported background')
        matches = [row for row in backgrounds if row['id'] == bg['id'] and row['image'] == bg['image']]
        if len(matches) != 1:
            raise ValueError('background does not match one fitted ground zone')
        row = matches[0]
        if plan['constraints'] == SAFE_CONSTRAINTS:
            safe = row.get('safe_regions')
            if not isinstance(safe, list) or not safe or any(
                    not isinstance(poly, list) or len(poly) < 3 or any(
                        not isinstance(pt, list) or len(pt) != 2 or any(type(v) is not int for v in pt) or
                        not 0 <= pt[0] <= 640 or not 0 <= pt[1] <= 480 for pt in poly) or
                    not cv2.isContourConvex(np.asarray(poly, dtype=np.int32)) for poly in safe):
                raise ValueError('safe plan requires reviewed safe polygons')
            boxes = row.get('forbidden_boxes')
            if not isinstance(boxes, list) or any(
                    not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or
                    not 0 <= box[0] < box[2] <= 640 or not 0 <= box[1] < box[3] <= 480 for box in boxes):
                raise ValueError('safe plan requires explicit obstacle boxes')
        parent = row['parent_image']
        if parent not in background_roles:
            background_roles[parent] = source_role(ROOT, ROOT / parent, known)
        if row['task_state'] != 'candidate_complete_negative' or background_roles[parent][0] == 'regression':
            raise ValueError('background is not eligible for training')
        target = state['target']
        if not isinstance(target, dict) or set(target) not in ({'vehicles'}, {'vehicles', 'people'}) or not isinstance(target['vehicles'], list) or len(target['vehicles']) > 7:
            raise ValueError('only zero to seven explicitly declared vehicles are supported')
        if state['labels'] != GROUND_LABELS:
            raise ValueError('unsupported labels')
        augmentation = state['augmentation']
        if not isinstance(augmentation, dict) or set(augmentation) != {'tone'} or augmentation['tone'] not in TONES:
            raise ValueError('unsupported tone')
        params = state['parameters']
        base_params = {'actors', 'value_gain', 'saturation_gain'} | ({'people'} if 'people' in target else set())
        if not isinstance(params, dict) or set(params) not in (base_params, base_params | {'sample_seed'}):
            raise ValueError('unsupported generation parameters')
        if 'sample_seed' in params and (type(params['sample_seed']) is not int or
                                        not 0 <= params['sample_seed'] < 2**64):
            raise ValueError('sample_seed must be an unsigned 64-bit integer')
        if (params['value_gain'], params['saturation_gain']) != TONES[augmentation['tone']]:
            raise ValueError('parameters differ from declared tone')
        if not isinstance(params['actors'], list) or len(params['actors']) != len(target['vehicles']):
            raise ValueError('actor count differs from vehicle count')
        used = {row['image'], row['parent_image'], row['source_video']}
        for target_item, actor in zip(target['vehicles'], params['actors']):
            if not isinstance(target_item, dict) or set(target_item) != {'class', 'source'} or target_item['class'] not in ('upright', 'fallen'):
                raise ValueError('unsupported vehicle target')
            keys = {'foot_x', 'foot_y', 'size_px', 'yaw', 'pitch', 'roll', 'relative_exposure'}
            if not isinstance(actor, dict) or set(actor) != keys:
                raise ValueError('actor must explicitly provide position, size, angles and exposure')
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in actor.values()):
                raise ValueError('actor parameters must be finite numbers')
            if (type(actor['size_px']) is not int or not 1 <= actor['size_px'] <= 640 or
                    not -12 <= actor['yaw'] <= 12 or not -6 <= actor['pitch'] <= 6 or
                    not -3 <= actor['roll'] <= 3 or not .9 <= actor['relative_exposure'] <= 1.08):
                raise ValueError('actor parameters outside supported range')
            matches = [entry for entry, _ in vehicles[target_item['class']] if entry['path'] == target_item['source']]
            if len(matches) != 1:
                raise ValueError('vehicle source missing, duplicated or inconsistent with class')
            used.update(matches[0][key] for key in ('path', 'source_frame', 'source_video', 'source_image') if key in matches[0])
        if 'people' in target:
            used.update(validate_people(target['people'], params['people']))
        if state['sources'] != sorted(used):
            raise ValueError('sources must list the sorted actual background and vehicle sources')
        if any(not isinstance(source, str) or not source or Path(source).is_absolute() or
               not (ROOT / source).resolve().is_relative_to(ROOT) for source in state['sources']):
            raise ValueError('sources must be workspace-relative paths')
        physical = {name: state[name] for name in ('background', 'target', 'labels', 'augmentation')}
        physical['parameters'] = {key: value for key, value in params.items() if key != 'sample_seed'}
        signature = json.dumps(physical, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if signature in seen:
            raise ValueError('duplicate exact generation states are not allowed')
        seen.add(signature)


def validate_people(targets, placements):
    """People are unlabelled occluders for this task, selected from its YAML root."""
    from datasets.scripts.common.config import source_path
    root = source_path('e_bike', 'person_occluder_artwork_root')
    if not isinstance(targets, list) or not isinstance(placements, list) or len(targets) != len(placements) or len(targets) > 7:
        raise ValueError('people and placements must be matching bounded lists')
    used = set()
    for target, spec in zip(targets, placements):
        if not isinstance(target, dict) or set(target) != {'source'}:
            raise ValueError('each person occluder must declare its source')
        path = (ROOT / target['source']).resolve()
        if Path(target['source']).is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('person occluder is outside its configured source root')
        keys = {'foot_x', 'foot_y', 'size_px', 'yaw', 'pitch', 'roll', 'relative_exposure'}
        if not isinstance(spec, dict) or set(spec) != keys or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in spec.values()):
            raise ValueError('person occluder needs finite explicit geometry')
        if (type(spec['size_px']) is not int or not 2 <= spec['size_px'] <= 480 or
                not -12 <= spec['yaw'] <= 12 or not -6 <= spec['pitch'] <= 6 or
                not -3 <= spec['roll'] <= 3 or not .9 <= spec['relative_exposure'] <= 1.08):
            raise ValueError('person occluder geometry is outside supported range')
        used.add(target['source'])
    return used


def place_actor(canvas, sprite, x, y, allow_truncation):
    """Clip a declared edge case and keep its exact visible occupancy for labels."""
    if not allow_truncation:
        return alpha_place(canvas, sprite, x, y)
    height, width = canvas.shape[:2]
    sh, sw = sprite.shape[:2]
    left, top = round(x-sw/2), round(y-sh)
    x1,y1,x2,y2 = max(0,left), max(0,top), min(width,left+sw), min(height,top+sh)
    if x1 >= x2 or y1 >= y2:
        raise ValueError('truncated actor is entirely outside the image')
    patch = sprite[y1-top:y2-top, x1-left:x2-left]
    alpha = patch[:,:,3:4].astype(np.float32)/255
    canvas[y1:y2,x1:x2] = np.clip(patch[:,:,:3]*alpha + canvas[y1:y2,x1:x2]*(1-alpha), 0, 255).astype(np.uint8)
    mask = np.zeros((height,width), bool)
    mask[y1:y2,x1:x2] = patch[:,:,3] > 127
    return mask


def render_ground_state(state, backgrounds, vehicles, allow_truncation=False):
    bg = next(row for row in backgrounds if row['id'] == state['background']['id'])
    frame = read_image(ROOT / bg['image'])
    if frame.shape != (480, 640, 3):
        raise ValueError('background dimensions changed')
    actors = []
    for target, spec in zip(state['target']['vehicles'], state['parameters']['actors']):
        source, sprite = next(item for item in vehicles[target['class']] if item[0]['path'] == target['source'])
        sprite = planar_warp(sprite, spec['yaw'], spec['pitch'], spec['roll'])
        sprite = bike.scene.assets.resized(sprite, **({'height': spec['size_px']} if target['class'] == 'upright' else {'width': spec['size_px']}))
        sprite = sprite.copy()
        sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.float32) * spec['relative_exposure'], 0, 255).astype(np.uint8)
        actor = {'kind': 'vehicle', 'category': target['class'], 'source': source,
                 'sprite': sprite, 'foot_x': spec['foot_x'], 'foot_y': spec['foot_y'],
                 'angles_degrees': {key: spec[key] for key in ('yaw', 'pitch', 'roll')}}
        if not allow_truncation:
            bike.scene.validate_actor_placement(actor, 'ground_only')
        if not grounded(actor, ground_zone(bg)):
            raise ValueError('vehicle contact is outside the fitted ground zone')
        if 'safe_regions' in bg:
            validate_safe_placement(actor, bg)
        actors.append(actor)
    for target, spec in zip(state['target'].get('people', []), state['parameters'].get('people', [])):
        sprite = bike.scene.assets.load_rgba(ROOT / target['source'])
        sprite = planar_warp(sprite, spec['yaw'], spec['pitch'], spec['roll'])
        sprite = bike.scene.assets.resized(sprite, height=spec['size_px'])
        sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.float32)*spec['relative_exposure'], 0, 255).astype(np.uint8)
        actor = {'kind': 'person', 'category': 'occluder', 'source': target,
                 'sprite': sprite, 'foot_x': spec['foot_x'], 'foot_y': spec['foot_y'],
                 'angles_degrees': {key: spec[key] for key in ('yaw','pitch','roll')}}
        if not grounded(actor, ground_zone(bg)):
            raise ValueError('person occluder contact is outside the ground zone')
        actors.append(actor)
    actors.sort(key=lambda actor: actor['foot_y'])
    for actor in actors:
        bike.scene.draw_shadow(frame, actor)
    masks = [place_actor(frame, actor['sprite'], actor['foot_x'], actor['foot_y'], allow_truncation) for actor in actors]
    visible = visible_masks(masks)
    objects, target_masks = [], []
    for actor, mask, shown in zip(actors, masks, visible):
        if actor['kind'] != 'vehicle':
            continue
        original_pixels = int(np.count_nonzero(actor['sprite'][:,:,3] > 127))
        if mask.sum() < 120 or shown.sum() < 120 or not original_pixels or shown.sum() / original_pixels < .7:
            raise ValueError('vehicle visibility below supported constraint')
        target_masks.append(shown)
        yy, xx = np.nonzero(shown)
        objects.append({'class': actor['category'], 'bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1],
                        'visible_fraction': float(shown.sum()/original_pixels), 'truncated_pixels': original_pixels-int(mask.sum()),
                        'foreground_source': actor['source'],
                        'foot_xy': [actor['foot_x'], actor['foot_y']], 'angles_degrees': actor['angles_degrees']})
    p = state['parameters']
    if state['augmentation']['tone'] != 'original':
        frame = apply_tone(frame, value_gain=p['value_gain'], saturation_gain=p['saturation_gain'])
    return frame, objects, target_masks, bg


def validate_safe_placement(actor, background):
    """Keep the complete sprite canvas inside a reviewed area and clear of obstacles."""
    height, width = actor['sprite'].shape[:2]
    left, top = round(actor['foot_x'] - width/2), round(actor['foot_y'] - height)
    right, bottom = left + width, top + height
    points = [(left, top), (right, top), (right, bottom), (left, bottom)]
    if not any(all(cv2.pointPolygonTest(np.asarray(poly, np.int32), (float(x), float(y)), False) >= 0
                   for x, y in points) for poly in background['safe_regions']):
        raise ValueError('vehicle canvas outside reviewed safe region')
    if any(left < r and l < right and top < b and t < bottom
           for l, t, r, b in background['forbidden_boxes']):
        raise ValueError('vehicle canvas overlaps a recorded obstacle')


def main():
    from lab.training_common.production_guard import generation_guard
    from datasets.scripts.common.config import load_task_config
    from datasets.scripts.common.paths import sample_dir, split_dir
    from datasets.scripts.e_bike.e_bike_queue_inputs import queue_input_report
    from lab.scripts.data import v4_source_gate as gate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan', required=True)
    _, source_config = load_task_config('e_bike')
    parser.add_argument('--background-manifest', default=source_config['ground_background_manifest'],
                        help='Workspace-relative JSON with backgrounds and 640x480 ground polygons')
    opts = parser.parse_args()
    out, plan_path, background_path = ((ROOT / value).resolve() for value in
                                       (opts.output, opts.plan, opts.background_manifest))
    selection_path = (ROOT / source_config['ground_vehicle_selection']).resolve()
    if (not plan_path.is_relative_to(ROOT) or not background_path.is_relative_to(ROOT)):
        parser.error('plan and output must be in the permitted workspace directories')
    if not selection_path.is_relative_to(ROOT) or not selection_path.is_file():
        parser.error('vehicle selection must be an existing workspace file')
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    if out not in {sample_dir('e_bike').resolve(), split_dir('e_bike', plan['target_split']).resolve()}:
        raise ValueError('output must be the sample or declared split')
    backgrounds = json.loads(background_path.read_text(encoding='utf-8'))['backgrounds']
    validate_backgrounds(backgrounds)
    if background_path == BACKGROUND and set(row['id'] for row in backgrounds) != set(ZONES):
        raise ValueError('background set differs from fitted ground zones')
    _, vehicles = bike.sources(selection=selection_path, include_people=False)
    known = build_queue(ROOT)
    validate_ground_plan(plan, backgrounds, vehicles, known)
    background_roles = {row['id']: source_role(ROOT, ROOT / row['parent_image'], known)
                        for row in backgrounds}
    media = {(ROOT / source).resolve() for state in plan['strata'] for source in state['sources']}
    qm, qp, missing_inputs = queue_input_report(ROOT, known, media)
    media.update(qm)
    procedural = set(qp) | {plan_path, background_path, selection_path,
                            ROOT/'datasets/scripts/e_bike/config/dataset.yaml',
                            ROOT/'datasets/scripts/e_bike/config/source.yaml'}
    history = {'exposure_status': 'incomplete' if missing_inputs or known.get('missing_exposure_paths') or known.get('unavailable_exposure_paths')
               or known.get('exposure_status') == 'incomplete' else 'complete',
               'missing_input_paths': missing_inputs,
               'missing_exposure_paths': known.get('missing_exposure_paths', []),
               'unavailable_exposure_paths': known.get('unavailable_exposure_paths', [])}
    code = {Path(__file__).resolve(), Path(bike.__file__).resolve(), Path(bike.scene.__file__).resolve(),
            Path(bike.scene.assets.__file__).resolve(), Path(gate.__file__).resolve(),
            TASK_HELPERS / 'e_bike_queue_inputs.py', ROOT / 'lab/training_common/camera_tone.py',
            ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py',
            ROOT / 'datasets/scripts/common/perspective.py', ROOT / 'datasets/scripts/common/composite.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'lab/training_common/production_guard.py'}
    if 'sampling' in plan:
        code.add(TASK_HELPERS / 'plan_grounded_distribution.py')
    if 'allocation_evidence' in plan:
        code.add(TASK_HELPERS / 'plan_safe_distribution.py')
    with generation_guard('e_bike.compose_v4_grounded_train:' + rel(out), sorted(media), [out],
                          {'plan': rel(plan_path), 'background_manifest': rel(background_path), 'historical_exposure': history,
                           'sampling': plan.get('sampling'), 'allocation_evidence': plan.get('allocation_evidence')}, procedural_inputs=sorted(procedural),
                          code_paths=sorted(code), root=ROOT, production_plan=plan,
                          sample_output_paths=[out / 'images'], auxiliary_output_paths=[out / 'masks']) as run:
        if run.reused:
            return
        if out.exists() and any(out.iterdir()):
            raise ValueError('output is not empty')
        states = run.plan['strata']
        if any(type(state.get('planned_count')) is not int or state['planned_count'] not in (0, 1) for state in states):
            raise ValueError('allocation exceeds one image per exact state')
        for state in states:
            render_ground_state(state, backgrounds, vehicles, plan['constraints']['allow_truncation'])
        for folder in ('images', 'labels', 'masks'):
            (out / folder).mkdir(parents=True, exist_ok=True)
        records = []
        for index, state in enumerate(item for item in states if item['planned_count']):
            frame, objects, masks, bg = render_ground_state(state, backgrounds, vehicles, plan['constraints']['allow_truncation'])
            stem = f'bike_{index+1:06d}'
            image, label = out / 'images' / (stem + '.png'), out / 'labels' / (stem + '.txt')
            write_image(image, frame)
            lines = []
            for number, (obj, mask) in enumerate(zip(objects, masks)):
                lines.append(bike.scene.yolo_row(GROUND_LABELS['classes'][obj['class']], obj['bbox_xyxy'], 640, 480))
                mask_path = out / 'masks' / f'{stem}_{number:02d}.png'
                write_image(mask_path, mask.astype(np.uint8)*255)
                obj['visible_mask'] = rel(mask_path)
            label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            actual = {'background': bg, 'ground_zone': ground_zone(bg),
                      'historical_source_role': background_roles[bg['id']],
                      'objects': objects, 'parameters': state['parameters'],
                      'augmentation': state['augmentation'], 'media': state['sources']}
            run.record_sample(state['id'], output=image, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            records.append({'image': rel(image), 'label': rel(label), 'state_id': state['id'],
                            'background': bg, 'ground_zone': ground_zone(bg),
                            'historical_source_role': background_roles[bg['id']],
                            'objects': objects, 'augmentation': state['augmentation'],
                            'parameters': state['parameters'], 'sources': state['sources'],
                            'generation_method': 'real_scene_composite', 'label_review': 'visible_mask_candidate'})
        manifest = {'status': 'training_candidates_not_deliverable', 'plan': rel(plan_path),
                    'historical_exposure': history,
                    'sampling': plan.get('sampling'),
                    'background_manifest': rel(background_path), 'records': records,
                    'scope': 'Ground polygons use 640x480 pixels; silhouette-bottom contact is approximate. '
                             'Generated masks and labels are candidates.'}
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'output': rel(out), 'images': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
