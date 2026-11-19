"""Allocate sign candidates by reviewed source and lighting, then draw placements.

Group weights are design priors, not observed failure rates. Generated labels
remain candidates; the placement rectangles are image-space observations.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from datasets.scripts.sign import compose_sign_v4_measured as measured
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.design_context import design_context

ROOT = measured.ROOT
from datasets.scripts.common.bayesian import allocate

MANIFEST = source_path('sign', 'planning_background_manifest').relative_to(ROOT).as_posix()
PHOTO_MANIFEST = source_path('sign', 'planning_photo_manifest').relative_to(ROOT).as_posix()
PHOTO_PARENT = 'Archive/Assets/captures/site_capture_20261112/onboard_camera_images'

# Each rectangle is a conservatively chosen portion of visible road in the
# 640x480 output after the [80,0,560,360] crop. It is not a measured plane.
# The sprite's entire width and bottom strip must fit, not just one point.
FRAMES = (
    ('junction_003_001.jpg', 0, [220, 390, 315, 440],
     '两栋楼之间的可见路面，避开左右建筑；原图目视近似区域。'),
    ('junction_003_009.jpg', 8, [545, 345, 630, 415],
     '画面右侧路面，避开楼宇与垃圾桶；原图目视近似区域。'),
    ('junction_003_015.jpg', 14, [275, 390, 365, 440],
     '两栋楼之间的可见路面；原图目视近似区域。'),
    ('junction_005_011.jpg', 30, [310, 350, 500, 430],
     '两栋楼之间近处路面，避开远处电动车；原图目视近似区域。'),
    ('junction_005_019.jpg', 38, [345, 390, 575, 440],
     '两栋楼之间的近处路面；远处电动车使用保护矩形避让。'),
    ('parking_003_001.jpg', 80, [380, 295, 550, 355],
     '右侧可见路面，避开左侧两辆电动车；原图目视近似区域。'),
    ('parking_003_011.jpg', 90, [535, 330, 630, 400],
     '右侧近处路面，避开后方电动车；原图目视近似区域。'),
    ('parking_003_017.jpg', 96, [390, 420, 580, 465],
     '电动车前方近处路面；原图目视近似区域。'),
)
MASKS = (('train_20.png', 'front'), ('train_14.png', 'oblique'))
PROTECTED = {'junction_005_019.jpg': [[320, 180, 530, 245]]}
PHOTOS = (
    ('vehicle_0003.png', 2, [505, 380, 630, 450], [[215, 135, 480, 340]],
     '路口右侧路面；信号灯与支架由保护矩形避让。'),
    ('vehicle_0010.png', 9, [200, 390, 320, 445], [[110, 120, 390, 205], [105, 195, 135, 345], [375, 195, 405, 335]],
     '信号灯支架之间的近处路面；灯体与支架由保护矩形避让。'),
)


def legacy_background_inputs(source_config):
    data = json.loads((ROOT / MANIFEST).read_text(encoding='utf8'))
    selected = set(source_config['current_selection']['backgrounds'])
    records, frame_inputs = [], []
    for name, index, rect, evidence in FRAMES:
        record = data['records'][index]
        if record['image'] not in selected:
            continue
        if Path(record['image']).name != name:
            raise ValueError('Source manifest record changed: ' + name)
        for path in (record['image'], record['source_video']):
            if not (ROOT / path).is_file():
                raise ValueError('Source missing: ' + path)
        frame_inputs.append({'image': record['image'], 'parent_image': record['image'],
            'parent_video': record['source_video'], 'pts_seconds': record['source_time_seconds'],
            'workspace_image': True, 'crop_xyxy': [80, 0, 560, 360],
            'sign_absence': 'confirmed', 'sign_absence_evidence': '原 640×360 帧目视无可辨牌具；不推断画外。',
            'source_evidence': {'manifest': MANIFEST, 'record_index': index},
            'ground_evidence': evidence, 'support_rectangle_xyxy': rect,
            'protected_rectangles_xyxy': PROTECTED.get(name, []),
            'task_use_status': [{'task': 'sign', 'split': 'train', 'role': 'background',
                                 'status': 'candidate', 'evidence': evidence}]})
        records.append(record)
    photo_rows = [json.loads(line) for line in (ROOT / PHOTO_MANIFEST).read_text(encoding='utf8').splitlines() if line.strip()]
    for name, index, rect, protected, evidence in PHOTOS:
        record = photo_rows[index]
        image = f'{PHOTO_PARENT}/{name}'
        if image not in selected:
            continue
        if record.get('path') != image or not (ROOT / image).is_file():
            raise ValueError('Photo source manifest or path changed: ' + name)
        frame_inputs.append({'image': image, 'parent_image': image, 'pts_seconds': None,
            'workspace_image': True, 'sign_absence': 'confirmed',
            'sign_absence_evidence': '2026-11-14 原尺寸整帧目视未见可辨牌具；为候选观察，未作人工验收。',
            'source_evidence': {'manifest': PHOTO_MANIFEST, 'record_index': index},
            'prior_sign_observation': record.get('task_observations', {}).get('sign', {}),
            'ground_evidence': evidence, 'support_rectangle_xyxy': rect,
            'protected_rectangles_xyxy': protected,
            'source_group': PHOTO_PARENT,
            'task_use_status': [{'task': 'sign', 'split': 'train', 'role': 'background',
                                 'status': 'candidate', 'evidence': evidence}]})
        records.append({'image': image, 'source_group': PHOTO_PARENT})
    if {row['image'] for row in frame_inputs} != selected:
        raise ValueError('source.yaml contains an unavailable planning background')
    return records, frame_inputs


def build(budget, seed, *, shared_pool=False, requested_budget=None):
    _, source_config = load_task_config('sign')
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if requested_budget is not None and (type(requested_budget) is not int or
                                         requested_budget < budget):
        raise ValueError('requested budget must include the executable sign subset')
    original = {'schema_version': 1, 'model_key': 'sign', 'target_split': 'train',
                'constraints': {'image_size': [640, 480], 'label_kind': 'visible_fixture_box'}}
    template = {'labels': {'class_id': 0, 'kind': 'visible_fixture_box'}}
    if shared_pool:
        frame_inputs = measured.pool_plan_inputs(source_config)
        records = [{**frame, 'source_video': frame.get('parent_video')} for frame in frame_inputs]
    else:
        records, frame_inputs = legacy_background_inputs(source_config)
        frame_inputs = measured.bind_background_reviews(frame_inputs, source_config)
    groups = []
    parent_paths = list(dict.fromkeys(row['source_group'] for row in records))
    if not parent_paths:
        raise ValueError('no eligible sign background source is available')
    for j, parent in enumerate(parent_paths):
        for k, (mask, view) in enumerate(MASKS):
            for tone in ('normal', 'dark_low_saturation'):
                groups.append({'id': f'{j}:{k}:{tone}', 'failure_event': '候选牌具漏检或定位失败',
                    'conditions': [parent, view, tone], 'min_count': 0, 'max_count': budget,
                    'prior': {'alpha': 2 if tone == 'dark_low_saturation' else 1, 'beta': 2},
                    'exposure_weight': 1, 'failure_cost': 1.5 if tone == 'dark_low_saturation' else 1})
    if budget >= len(groups):
        for group in groups:
            group['min_count'] = 1
    request = {'schema_version': 1, 'model_key': 'sign', 'target_split': 'train', 'budget': budget,
        'purpose': '按背景父源和光照状态分配候选生成预算', 'independence_basis': '背景来自三个父视频及同批车载照片；牌具来自同一视频',
        'uncertainty_weight': 0, 'strata': groups}
    if shared_pool:
        request['independence_basis'] = f'背景来自共享池 {len(parent_paths)} 个父来源；牌具来自同一视频'
    allocation = allocate(request, ROOT)
    rng = random.Random(seed)
    sprites = {name: measured.base.cutout(name) for name, _ in MASKS}
    mask_data = json.loads((measured.base.ANNOTATED / 'manifest.json').read_text(encoding='utf8'))
    result = {k: original[k] for k in ('schema_version', 'model_key', 'target_split', 'constraints')}
    result.update({'budget': budget, 'sampling_seed': seed, 'uncertainty_weight': 0,
        'design_context': design_context('sign'),
        'purpose': request['purpose'], 'independence_basis': request['independence_basis'],
        'background_inputs': frame_inputs, 'group_allocation': allocation,
        'foreground_inputs': [{'image': (measured.base.ANNOTATED / name).relative_to(ROOT).as_posix(),
            'parent_image': next(row['source'] for row in mask_data['records']
                                 if row.get('index') == int(name[6:-4])),
            'parent_video': measured.base.SIGN_VIDEO, 'view': view,
            'source_manifest': (measured.base.ANNOTATED / 'manifest.json').relative_to(ROOT).as_posix(),
            'task_use_status': [{'task': 'sign', 'split': 'train', 'role': 'target_foreground',
                                 'status': 'candidate', 'evidence': '原 SAM 候选掩膜；原状态未提升。'}]}
            for name, view in MASKS],
        'allocation_explanation': '先按贝叶斯先验/代价分配目标，再按可行布局生成不重复状态；请求与实际分别见 group_allocation 和 placement_outcome，不使用模型反馈。',
        'source_and_geometry_limits': f'{len(records)} 张候选底图、{len(parent_paths)} 个父来源；牌具视面来自同一视频。底图接触区按各帧证据核对，来源数量不限制同源有效布局数。',
        'sampling_description': '先按背景父视频或照片批次×视面×色调分组分配，组内固定种子有放回随机抽背景及参数。', 'strata': []})
    seen, failed, generated_counts, placement_failures = set(), set(), Counter(), []

    def add_from_group(group):
        j, k, tone = group['id'].split(':', 2)
        j, k = int(j), int(k)
        mask, view = MASKS[k]
        sprite = sprites[mask]
        origin = next(row['source'] for row in mask_data['records'] if row.get('index') == int(mask[6:-4]))
        choices = [i for i, row in enumerate(records) if row['source_group'] == parent_paths[j]]
        n = generated_counts[group['id']]
        for attempt in range(10000):
            frame_index = rng.choice(choices)
            frame, record = frame_inputs[frame_index], records[frame_index]
            x1, y1, x2, y2 = frame['support_rectangle_xyxy']
            height = rng.randint(75, 95) if x2-x1 < 100 else rng.randint(105, 155)
            width = max(1, round(sprite.shape[1] * height / sprite.shape[0]))
            margin = max(3, width // 3)
            if x2-x1 < width + 2*margin or y2-y1 < 12:
                continue
            left = rng.randint(x1+margin, x2-width-margin)
            bottom = rng.randint(y1+6, y2-6)
            top = bottom-height
            if any(left < px2 and left+width > px1 and top < py2 and bottom > py1
                   for px1, py1, px2, py2 in frame['protected_rectangles_xyxy']):
                continue
            contact = measured.recorded_contact_status(
                frame, (left + width / 2, bottom),
                contact_width=measured._sign_contact_width(mask, height))
            if contact != 'candidate':
                continue
            key = (frame_index, k, tone, height, left, top)
            if key not in seen:
                seen.add(key)
                break
        else:
            failed.add(group['id'])
            placement_failures.append({'group_id': group['id'], 'attempts': 10000,
                                       'reason': 'no distinct feasible placement found'})
            return False
        ground = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        state = {'id': f'g{j}_view{k}_{tone}_{n:04d}', 'failure_event': groups[0]['failure_event'],
                'conditions': ['placed', view, tone], 'min_count': 1, 'max_count': 1,
                'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                'background': {'image': record['image'],
                               **{key: frame[key] for key in (
                                   'frame_id', 'record_id', 'evidence_ids', 'placement_algorithm')}},
                'target': {'state': 'placed', 'mask': mask},
                'labels': template['labels'], 'augmentation': {'tone': tone},
                'parameters': {'height': height, 'left': left, 'top': top, 'ground_polygon': ground,
                    'occlusion_polygons': [], 'visible_kind': 'fixture',
                    'shadow_strength': round(rng.uniform(.12, .26), 3),
                    'visibility_evidence': '原候选掩膜牌面、立杆及底座完整；生成标签仍为候选。'},
                'sources': list(dict.fromkeys([record['image']] + ([record['source_video']] if record.get('source_video') else []) + [
                    (measured.base.ANNOTATED / mask).relative_to(ROOT).as_posix(), origin, measured.base.SIGN_VIDEO]))}
        result['strata'].append(state)
        generated_counts[group['id']] += 1
        return True

    for group in allocation['strata']:
        for _ in range(group['planned_count']):
            if not add_from_group(group):
                break
    initial_count = len(result['strata'])
    while len(result['strata']) < budget:
        before = len(result['strata'])
        for group in allocation['strata']:
            if len(result['strata']) >= budget:
                break
            if group['id'] not in failed:
                add_from_group(group)
        if len(result['strata']) == before:
            break
    if not result['strata']:
        raise ValueError('No feasible sign placements in the selected sources')
    result['budget'] = len(result['strata'])
    result['placement_outcome'] = {'requested_budget': budget, 'planned_images': len(result['strata']),
        'unfilled_images': budget - len(result['strata']), 'planned_by_group': dict(generated_counts),
        'reallocated_images': len(result['strata']) - initial_count, 'failed_groups': placement_failures}
    if shared_pool:
        result['background_pool'] = source_config['background_pool_manifest']
        result['source_and_geometry_limits'] = '共享池当前训练分配及逐帧放置证据，牌具前景来自同一视频。'
    if requested_budget is not None:
        parent_by_image = {row['image']: row['source_group'] for row in frame_inputs}
        parent_counts = Counter(parent_by_image[state['background']['image']]
                                for state in result['strata'])
        result['quota_gap'] = {
            'requested_budget': requested_budget, 'executable_budget': len(result['strata']),
            'remaining_shortfall': requested_budget - len(result['strata']),
            'qualified_composite_frames': len(frame_inputs),
            'qualified_parent_groups': len(parent_paths),
            'used_parent_groups': len(parent_counts),
            'planned_by_parent_group': dict(sorted(parent_counts.items())),
            'per_parent_group_cap': None,
            'excluded_parent_groups': source_config.get('current_selection', {}).get('excluded_parent_groups', []),
            'qualification_basis': 'current frame_environment role decision and saved contact mask',
        }
    measured._prepare(result, {}, frame_inputs, json.loads(
        (measured.base.ANNOTATED / 'manifest.json').read_text(encoding='utf8')), ROOT)
    return result


def build_sample_recipe():
    """Compile paired visible-frame crops using current ingest and contact evidence."""
    _, source = load_task_config('sign')
    recipe_path = ROOT / source['sample_recipe']
    recipe = json.loads(recipe_path.read_text(encoding='utf8'))
    pool = {row['image']: row for row in measured.pool_plan_inputs(source)}
    masks = recipe['masks']
    views = dict(measured.VIEWS)
    data = json.loads((measured.base.ANNOTATED / 'manifest.json').read_text(encoding='utf8'))
    origins = {f"train_{row['index']}.png": row['source'] for row in data['records'] if 'index' in row}
    frames, states = [], []
    for spec in recipe['backgrounds']:
        if spec['image'] not in pool:
            raise ValueError('Sample background lacks current train eligibility: ' + spec['image'])
        row = dict(pool[spec['image']])
        if spec.get('crop_xyxy'):
            row['crop_xyxy'] = spec['crop_xyxy']
        frames.append(row)
        refs = {'image': row['image'], **{key: row[key] for key in (
            'frame_id', 'record_id', 'evidence_ids', 'placement_algorithm')}}
        sources = list(dict.fromkeys([row['image'], row['parent_image']] +
                                     ([row['parent_video']] if row.get('parent_video') else [])))
        x1, y1, x2, y2 = spec['ground_rectangle']
        for tone in recipe['tones']:
            states.append({'id': f"b{len(frames)}_empty_{tone}", 'conditions': ['negative', 'none', tone],
                'target': {'state': 'negative', 'mask': 'none'},
                'labels': {'class_id': 0, 'kind': 'empty_candidate'},
                'parameters': {'kind': 'empty_background',
                    'scene_evidence': spec.get('negative_evidence', spec['evidence']),
                    'negative_context': spec.get('negative_context', 'empty_scene')}, 'sources': sources,
                'background': refs, 'augmentation': {'tone': tone},
                'scene_evidence': spec.get('negative_evidence', spec['evidence']),
                'negative_context': spec.get('negative_context', 'empty_scene')})
        for mask in masks:
            sprite = (measured.pilot.repaired_back(None, write_artifacts=False) if mask == 'train_0.png'
                      else measured.base.cutout(mask))
            for layout, height, center, bottom in spec['layouts']:
                width = max(1, round(sprite.shape[1] * height / sprite.shape[0]))
                left = round(center - width / 2)
                # The source-alpha base, rather than its padded image bounds, must
                # fit the manually narrowed ground region as well as saved contact.
                foot_width = measured._sign_contact_width(mask, height)
                if not (x1 <= center-foot_width/2 and center+foot_width/2 <= x2 and y1 <= bottom <= y2):
                    raise ValueError(f'Base exceeds sample road region: {spec["image"]}/{mask}/{layout}')
                for tone in recipe['tones']:
                    states.append({'id': f"b{len(frames)}_{views[mask]}_{layout}_{tone}",
                        'conditions': ['placed', views[mask], tone], 'background': refs,
                        'target': {'state': 'placed', 'mask': mask},
                        'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                        'augmentation': {'tone': tone}, 'layout': layout,
                        'parameters': {'height': height, 'left': left, 'top': bottom-height, 'layout': layout,
                            'ground_polygon': [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                            'occlusion_polygons': [], 'visible_kind': 'identifiable_pole' if layout == 'pole_only' else 'fixture',
                            'visibility_evidence': spec['evidence'] + ' 完整前景放大后由画面边缘自然裁切，保留可辨牌具或杆底座。',
                            'shadow_strength': .18},
                        'sources': list(dict.fromkeys(sources + [(measured.base.ANNOTATED/mask).relative_to(ROOT).as_posix(), origins[mask], measured.base.SIGN_VIDEO]))})
    for state in states:
        state.update(min_count=1, max_count=1, prior={'alpha': 1, 'beta': 1},
                     exposure_weight=1, failure_cost=1, failure_event='牌具漏检或普通杆误报')
        try:
            measured.validate_planned_stratum(state, frames)
        except ValueError as exc:
            raise ValueError(f"Sample state {state['id']}: {exc}") from exc
    result = {'schema_version': 1, 'model_key': 'sign', 'target_split': 'train',
        'constraints': {'image_size': [640,480], 'label_kind': 'visible_fixture_box'},
        'budget': len(states), 'sampling_seed': recipe['seed'], 'uncertainty_weight': 0,
        'purpose': '配对小样：四视面、远近、自然截断、独立色调变化和无牌对照',
        'design_context': design_context('sign'), 'background_inputs': frames,
        'background_pool': source['background_pool_manifest'], 'strata': states,
        'sample_recipe': source['sample_recipe'], 'source_and_geometry_limits': recipe['limits'],
        'independence_basis': '同一牌具视频；三个已接触背景父来源，用于开发训练候选。',
        'foreground_inputs': [{'image': (measured.base.ANNOTATED/mask).relative_to(ROOT).as_posix(),
            'parent_image': origins[mask], 'parent_video': measured.base.SIGN_VIDEO, 'view': views[mask]}
            for mask in masks]}
    measured._prepare(result, {}, frames, data, ROOT)
    return result


def build_background_expansion(split='train', diverse_sample=False):
    """Visit every qualified frame once, reporting actual feasible contacts."""
    import numpy as np
    from datasets.scripts.common.review.placement_record import contact_status_in_view

    _, source = load_task_config('sign')
    settings = source['background_expansion']
    result = build_sample_recipe()
    frames = measured.pool_plan_inputs(source, split=split)
    if diverse_sample and split == 'val':
        # Older qualification may predate this parent-level allocation. Keep
        # those decisions intact, but exclude train parents from this sample.
        train_parents = {row['source_group'] for row in measured.pool_plan_inputs(source, split='train')}
        frames = [row for row in frames if row['source_group'] not in train_parents]
    if diverse_sample:
        groups = {}
        for row in frames:
            image = Path(row['image'])
            key = (row['source_group'], image.parent.as_posix(), image.stem.split('_f')[0]) if '/rendered/' in row['image'] else (row['source_group'],)
            groups.setdefault(key, []).append(row)
        frames = []
        for key, rows in sorted(groups.items()):
            rows.sort(key=lambda row: row['image'])
            cap = settings['sample_frames_per_blender_route'] if len(key) > 1 else settings['sample_frames_per_parent']
            count = min(cap, len(rows))
            selected = [rows[round(i*(len(rows)-1)/max(1,count-1))] for i in range(count)]
            positive = next((row for row in rows if row.get('existing_sign_boxes')), None)
            if positive and not any(row.get('existing_sign_boxes') for row in selected):
                selected[-1] = positive
            frames.extend(selected)
    result.update(target_split=split, background_inputs=frames, strata=[], purpose='逐帧扩源计划：每个可放底图一正一空对照',
                  independence_basis='背景帧按同一物理父来源汇总；帧数不等于独立场景数。')
    origins = {Path(x['image']).name: x['parent_image'] for x in result['foreground_inputs']}
    masks = list(origins)
    rng = random.Random(settings['seed'])
    sprites = {mask: measured.pilot.repaired_back(None, write_artifacts=False) if mask == 'train_0.png'
               else measured.base.cutout(mask) for mask in masks}
    report = []
    for index, row in enumerate(frames):
        mask = masks[index % len(masks)]
        sprite = sprites[mask]
        review, placement = measured._stored_sign_review(row['record_id'], str(ROOT/source['databases']['frame_environment']))
        content = measured.output_content_rectangle(review['image_size'], row.get('crop_xyxy'))
        # Keep feet in the near-ground strip, avoiding wall pixels from broad
        # rectangular envelopes. Test the full lower-base span in saved masks.
        xs = list(range(48, 593, settings['contact_step_px']))
        rng.shuffle(xs)
        chosen = None
        heights = settings['heights_px']
        preferred = index // len(masks) % len(heights)
        for height in heights[preferred:] + heights[:preferred]:
            if height > content[3]-content[1] and content[1] != 0:
                continue  # Do not paste into video letterbox padding.
            width = max(1, round(sprite.shape[1]*height/sprite.shape[0]))
            sy, sx = np.nonzero(sprite[:, :, 3] >= 127)
            base_x = sx[sy >= sy.max() - max(2, round(sprite.shape[0]*.18))]
            contact_width = float(base_x.max()-base_x.min()+1)*height/sprite.shape[0]
            for bottom in range(content[3]-settings['bottom_margin_px'], content[3]-settings['near_strip_px'], -settings['contact_step_px']):
                for center in xs:
                    left = round(center-width/2)
                    if left < 0 or left+width > 640 or (height <= content[3]-content[1] and bottom-height < content[1]):
                        continue
                    if any(left < px2 and left+width > px1 and bottom-height < py2 and bottom > py1
                           for px1,py1,px2,py2 in row['protected_rectangles_xyxy']):
                        continue
                    status = contact_status_in_view(placement, (left+width/2,bottom), contact_width=contact_width,
                        crop_xyxy=row.get('crop_xyxy') or [0,0,*review['image_size']], content_xyxy=content)
                    if status == 'candidate':
                        chosen = height,left,bottom
                        break
                if chosen:
                    break
            if chosen:
                break
        report.append({'image': row['image'], 'frame_id': row['frame_id'], 'record_id': row['record_id'],
                       'parent_group': row['source_group'], 'status': 'planned' if chosen else 'no_feasible_contact',
                       'placement': list(chosen) if chosen else None})
        tone = settings['tones'][index // (len(masks)*len(heights)) % len(settings['tones'])]
        refs = {'image':row['image'], **{key:row[key] for key in ('frame_id','record_id','evidence_ids','placement_algorithm')}}
        sources = list(dict.fromkeys([row['image'],row['parent_image']] + ([row['parent_video']] if row.get('parent_video') else [])))
        common = {'min_count':1,'max_count':1,'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,
                  'failure_event':'牌具漏检或空场误报','background':refs,'augmentation':{'tone':tone}}
        if row['sign_absence'] in ('confirmed', 'existing_labelled'):
            result['strata'].append({**common,'id':f'pool_{index}_empty','conditions':['negative','none',tone],
                'target':{'state':'negative','mask':'none'},'labels':{'class_id':0,'kind':'retained_background_boxes' if row.get('existing_sign_boxes') else 'empty_candidate'},
                'parameters':{'kind':'empty_background'},'sources':sources})
        if not chosen:
            continue
        height,left,bottom = chosen
        result['strata'].append({**common,'id':f'pool_{index}_positive','conditions':['placed',dict(measured.VIEWS)[mask],tone],
            'target':{'state':'placed','mask':mask},'labels':{'class_id':0,'kind':'visible_fixture_box'},
            'sources':list(dict.fromkeys(sources+[(measured.base.ANNOTATED/mask).relative_to(ROOT).as_posix(),origins[mask],measured.base.SIGN_VIDEO])),
            'parameters':{'height':height,'left':left,'top':bottom-height,'layout':'pole_only' if height>=1000 else 'top_clipped' if bottom-height<0 else 'full',
                'ground_polygon':[[0,content[3]-settings['near_strip_px']],[639,content[3]-settings['near_strip_px']],
                                  [639,content[3]-1],[0,content[3]-1]],
                'occlusion_polygons':[],'visible_kind':'identifiable_pole' if height>=1000 else 'fixture','shadow_strength':.18,
                'visibility_evidence':'近地条带内按原帧接地掩膜检查整个底座横向跨度；仍须按成图复核空间与尺度。'}})
    result.update(budget=len(result['strata']), sampling_seed=settings['seed'],
                  background_expansion={'considered':len(frames),'planned_backgrounds':sum(x['status']=='planned' for x in report),
                    'parents':dict(Counter(x['source_group'] for x in frames)), 'records':report})
    data = json.loads((measured.base.ANNOTATED/'manifest.json').read_text(encoding='utf8'))
    measured._prepare(result, {}, frames, data, ROOT)
    if diverse_sample and source.get('sample_refinement'):
        result = refine_paired_sample(result, source, data)
    return result


def refine_paired_sample(plan, source, mask_data, validate=True):
    """Pair tones at fixed geometry, with source-mask constrained varied layouts."""
    import copy
    import numpy as np
    from datasets.scripts.common.review.placement_record import contact_status_in_view

    cfg = source['sample_refinement']
    rng = random.Random(source['background_expansion']['seed'])
    split = plan['target_split']
    pool = measured.pool_plan_inputs(source, split)
    cases = [case for case in cfg['occlusion_cases'] if case['split'] == split]
    frames = plan['background_inputs']
    for case in cases:
        row = next(row for row in pool if row['image'].endswith(case['image_suffix']))
        if not any(frame['image'] == row['image'] for frame in frames):
            frames.append(row)
    origins = {Path(row['image']).name: row['parent_image'] for row in plan['foreground_inputs']}
    masks = list(origins)
    states, outcomes = [], []
    tones = source['background_expansion']['tones']
    for index, row in enumerate(frames):
        review, placement = measured._stored_sign_review(row['record_id'], str(ROOT/source['databases']['frame_environment']))
        w,h = review['image_size']
        # Remove video letterbox before near-frame truncation; remap complete
        # old labels and protected rectangles into the same new view.
        if row.get('crop_xyxy') is None and w*9 == h*16:
            crop_w = h*4//3
            row['crop_xyxy'] = [(w-crop_w)//2,0,(w+crop_w)//2,h]
            old_protected = row['protected_rectangles_xyxy']
            row['protected_rectangles_xyxy'] = [[round((x1/640*w-row['crop_xyxy'][0])/crop_w*640),
                round((y1-60)/360*480),round((x2/640*w-row['crop_xyxy'][0])/crop_w*640),
                round((y2-60)/360*480)] for x1,y1,x2,y2 in old_protected]
            existing = measured.existing_sign_boxes(review,row['crop_xyxy'])
            row.pop('existing_sign_boxes',None)
            if existing: row['existing_sign_boxes'] = existing
            row['sign_absence'] = 'existing_labelled' if existing else 'confirmed'
        content = measured.output_content_rectangle(review['image_size'],row.get('crop_xyxy'))
        case = next((case for case in cases if row['image'].endswith(case['image_suffix'])),None)
        refs = {'image':row['image'],**{key:row[key] for key in ('frame_id','record_id','evidence_ids','placement_algorithm')}}
        sources = list(dict.fromkeys([row['image'],row['parent_image']]+([row['parent_video']] if row.get('parent_video') else [])))
        def append_state(suffix, mask, parameters, tone):
            negative = mask == 'none'
            states.append({'id':f'paired_{index}_{suffix}_{tone}',
                'conditions':['negative' if negative else 'placed','none' if negative else dict(measured.VIEWS)[mask],tone],
                'min_count':1,'max_count':1,'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,
                'failure_event':'牌具漏检或普通杆误报','background':refs,
                'augmentation':{'tone':tone},'target':{'state':'negative' if negative else 'placed','mask':mask},
                'labels':{'class_id':0,'kind':('retained_background_boxes' if row.get('existing_sign_boxes') else 'empty_candidate') if negative else 'visible_fixture_box'},
                'parameters':parameters,'sources':sources if negative else list(dict.fromkeys(sources+[
                    (measured.base.ANNOTATED/mask).relative_to(ROOT).as_posix(),origins[mask],measured.base.SIGN_VIDEO]))})
        for tone in tones if case else [tones[index%len(tones)]]:
            append_state('background','none',{'kind':'empty_background',
                'negative_context':case.get('negative_context','scene_control') if case and not row.get('existing_sign_boxes') else 'scene_control',
                'scene_evidence':case['evidence'] if case else row['sign_absence_evidence']},tone)
        mask = masks[index%len(masks)]
        sprite = measured.pilot.repaired_back(None,write_artifacts=False) if mask=='train_0.png' else measured.base.cutout(mask)
        layout = list(cfg['height_ranges'])[index//len(masks)%len(cfg['height_ranges'])]
        lo,hi = cfg['height_ranges'][layout]
        heights = [rng.randint(lo,hi) for _ in range(7)]
        ys,xs = np.nonzero(sprite[:,:,3]>=127)
        base = xs[ys>=ys.max()-round(sprite.shape[0]*.18)]
        chosen = None
        for height in heights:
            width = round(sprite.shape[1]*height/sprite.shape[0])
            base_span = (base.max()-base.min()+1)*height/sprite.shape[0]
            foot_width = measured._sign_contact_width(mask,height)
            bottoms = list(range(content[3]-20,content[3]-cfg['ground_depth_px'],-11));rng.shuffle(bottoms)
            centers = list(range(45,596,19));rng.shuffle(centers)
            for bottom in bottoms:
                for center in centers:
                    left=round(center-width/2)
                    if (height<480 and (left<0 or left+width>640 or bottom-height<0)) or center-base_span/2<0 or center+base_span/2>639:continue
                    if layout=='partial' and bottom-height>=0:continue
                    if any(left<px2 and left+width>px1 and bottom-height<py2 and bottom>py1 for px1,py1,px2,py2 in row['protected_rectangles_xyxy']):continue
                    if contact_status_in_view(placement,(left+width/2,bottom),contact_width=foot_width,
                        crop_xyxy=row.get('crop_xyxy') or [0,0,w,h],content_xyxy=content)=='candidate':
                        chosen={'height':height,'left':left,'top':bottom-height,'layout':layout,
                            'ground_polygon':[[0,content[3]-cfg['ground_depth_px']],[639,content[3]-cfg['ground_depth_px']],[639,content[3]-1],[0,content[3]-1]],
                            'occlusion_polygons':[],'visible_kind':'identifiable_pole' if layout=='pole_only' else 'fixture',
                            'visibility_evidence':'完整前景按实际画面边缘截断；核对原帧掩膜中的真实下沿接触跨度，完整底座横向留在画面内。','shadow_strength':.18}
                        break
                if chosen:break
            if chosen:break
        outcomes.append({'image':row['image'],'layout':layout,'view':dict(measured.VIEWS)[mask],
                         'status':'planned' if chosen else 'no_feasible_contact'})
        if chosen:
            for repair in cfg.get('placement_repairs',[]):
                if (repair['split']==split and row['image'].endswith(repair['image_suffix']) and
                    repair['layout']==layout and repair['view']==dict(measured.VIEWS)[mask]):
                    width=round(sprite.shape[1]*chosen['height']/sprite.shape[0])
                    chosen['left']=round(repair['center']-width/2)
                    chosen['top']=repair['bottom']-chosen['height']
                    chosen['visibility_evidence'] += ' '+repair['evidence']
            for tone in tones:append_state(layout,mask,copy.deepcopy(chosen),tone)
        if case and not case.get('negative_only'):
            for special_mask in masks[:2]:
                sp=measured.base.cutout(special_mask);height=case['height'];width=round(sp.shape[1]*height/sp.shape[0])
                parameters={'height':height,'left':round(case['center']-width/2),'top':case['bottom']-height,
                    'layout':'occluded','ground_polygon':[[0,300],[639,300],[639,479],[0,479]],
                    'occlusion_polygons':case['polygons'],'visible_kind':'fixture',
                    'visibility_evidence':case['evidence'],'shadow_strength':.18}
                for tone in tones:
                    append_state('occluded_'+special_mask,special_mask,copy.deepcopy(parameters),tone)
    # A failed frame/pose does not veto a required joint cell. Reuse a proven
    # contact on another selected background, checking the new view's footprint.
    frame_by_image = {row['image']:row for row in frames}
    for layout in cfg['height_ranges']:
        for mask in masks:
            if any(state['target']['mask']==mask and state['parameters'].get('layout')==layout for state in states):
                continue
            for template in list(states):
                if template['augmentation']['tone']!='normal' or template['parameters'].get('layout')!=layout:continue
                row=frame_by_image[template['background']['image']]
                p=copy.deepcopy(template['parameters']);old=template['target']['mask']
                old_sprite=measured.pilot.repaired_back(None,write_artifacts=False) if old=='train_0.png' else measured.base.cutout(old)
                sp=measured.pilot.repaired_back(None,write_artifacts=False) if mask=='train_0.png' else measured.base.cutout(mask)
                center=p['left']+round(old_sprite.shape[1]*p['height']/old_sprite.shape[0])/2
                width=round(sp.shape[1]*p['height']/sp.shape[0]);p['left']=round(center-width/2)
                if p['height']<480 and (p['left']<0 or p['left']+width>640):continue
                if any(p['left']<x2 and p['left']+width>x1 and p['top']<y2 and p['top']+p['height']>y1
                       for x1,y1,x2,y2 in row['protected_rectangles_xyxy']):continue
                if measured.recorded_contact_status(row,(p['left']+width/2,p['top']+p['height']),
                    contact_width=measured._sign_contact_width(mask,p['height']))!='candidate':continue
                for tone in tones:
                    state=copy.deepcopy(template);state['id']=f'fill_{layout}_{mask}_{tone}'
                    state['target']['mask']=mask;state['conditions']=['placed',dict(measured.VIEWS)[mask],tone]
                    state['augmentation']['tone']=tone;state['parameters']=copy.deepcopy(p)
                    state['sources']=list(dict.fromkeys([row['image'],row['parent_image']]+([row['parent_video']] if row.get('parent_video') else [])+[
                        (measured.base.ANNOTATED/mask).relative_to(ROOT).as_posix(),origins[mask],measured.base.SIGN_VIDEO]))
                    states.append(state)
                break
    plan.update(strata=states,budget=len(states),background_inputs=frames,
        purpose='困难组合小样：四视面、连续尺度与位置、同布局四色调和实景灯体遮挡',
        refinement_outcome=outcomes)
    if validate:
        measured._prepare(plan,{},frames,mask_data,ROOT)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--sample', action='store_true', help='compile the configured paired sample recipe')
    parser.add_argument('--expand-backgrounds', action='store_true', help='visit every eligible training background and save a separate plan')
    parser.add_argument('--diverse-sample', action='store_true', help='sample across parent sources and Blender routes')
    parser.add_argument('--split', choices=['train','val'], default='train')
    parser.add_argument('--requested-budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1918)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--shared-background-pool', action='store_true', help='compile reviewed shared pool train assignments')
    parser.add_argument('--dry-run', action='store_true', help='validate without writing a plan')
    args = parser.parse_args()
    if not args.dry_run:
        _, source = load_task_config('sign')
        configured = (ROOT / (source['diverse_sample_plans'][args.split] if args.diverse_sample or args.sample else
                              source['background_expansion_plan'] if args.expand_backgrounds else
                              source.get('legacy_production_plan',source['production_plan']))).resolve()
        out = (ROOT / args.output).resolve() if args.output is not None else configured
        if not configured.is_relative_to(ROOT / 'datasets/scripts/sign/config') or out != configured:
            raise ValueError(
                'sign plan output must match datasets/scripts/sign/config/source.yaml '
                f'selected branch: current={out}, required={configured}; omit --output or use --dry-run'
            )
    if args.expand_backgrounds or args.diverse_sample or args.sample:
        value = build_background_expansion(args.split, args.diverse_sample or args.sample)
    else:
        if args.budget is None:
            parser.error('--budget is required unless --sample is selected')
        value = build(args.budget, args.seed, shared_pool=args.shared_background_pool,
                      requested_budget=args.requested_budget)
    if not args.dry_run:
        with out.open('w', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({'budget': value['budget'], 'states': len(value['strata'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
