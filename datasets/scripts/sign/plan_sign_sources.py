"""Allocate sign candidates by reviewed source and lighting, then draw placements.

Group weights are design priors, not observed failure rates. Generated labels
remain candidates; the placement rectangles are image-space observations.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from datasets.scripts.sign import compose_sign_v4_measured as measured
from datasets.scripts.common.config import load_task_config, source_path

ROOT = measured.ROOT
from datasets.scripts.common.bayesian import allocate

MANIFEST = source_path('sign', 'planning_background_manifest').relative_to(ROOT).as_posix()
PHOTO_MANIFEST = source_path('sign', 'planning_photo_manifest').relative_to(ROOT).as_posix()
PHOTO_PARENT = 'datasets/raw/site_capture_20261112/onboard_camera_images'
OUTPUT_PLAN = ROOT / 'datasets/scripts/sign/config/production_plan.json'

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


def build(budget, seed):
    design, source_config = load_task_config('sign')
    if not 16 <= budget <= design['train_target_images']:
        raise ValueError('budget must fit the current sign training design')
    data = json.loads((ROOT / MANIFEST).read_text(encoding='utf8'))
    original = {'schema_version': 1, 'model_key': 'sign', 'target_split': 'train',
                'constraints': {'image_size': [640, 480], 'label_kind': 'visible_fixture_box'}}
    template = {'labels': {'class_id': 0, 'kind': 'visible_fixture_box'}}
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
    groups = []
    parent_paths = list(dict.fromkeys(row['source_group'] for row in records))
    for j, parent in enumerate(parent_paths):
        for k, (mask, view) in enumerate(MASKS):
            for tone in ('normal', 'dark_low_saturation'):
                groups.append({'id': f'{j}:{k}:{tone}', 'failure_event': '候选牌具漏检或定位失败',
                    'conditions': [parent, view, tone], 'min_count': 1, 'max_count': budget,
                    'prior': {'alpha': 2 if tone == 'dark_low_saturation' else 1, 'beta': 2},
                    'exposure_weight': 1, 'failure_cost': 1.5 if tone == 'dark_low_saturation' else 1})
    request = {'schema_version': 1, 'model_key': 'sign', 'target_split': 'train', 'budget': budget,
        'purpose': '按背景父源和光照状态分配候选生成预算', 'independence_basis': '背景来自三个父视频及同批车载照片；牌具来自同一视频',
        'uncertainty_weight': 0, 'strata': groups}
    allocation = allocate(request, ROOT)
    rng = random.Random(seed)
    sprites = {name: measured.base.cutout(name) for name, _ in MASKS}
    mask_data = json.loads((measured.base.ANNOTATED / 'manifest.json').read_text(encoding='utf8'))
    result = {k: original[k] for k in ('schema_version', 'model_key', 'target_split', 'constraints')}
    result.update({'budget': budget, 'sampling_seed': seed, 'uncertainty_weight': 0,
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
        'allocation_explanation': '分组先执行有弹性上下限的贝叶斯先验/代价配额，再生成不重复的单次具体状态；无模型反馈。',
        'source_and_geometry_limits': '八视频帧及两张同批照片的目视近似路面；两种牌具视面来自同一视频；非测量或验收。',
        'sampling_description': '先按背景父视频或照片批次×视面×色调分组分配，组内固定种子有放回随机抽背景及参数。', 'strata': []})
    seen = set()
    for group in allocation['strata']:
        j, k, tone = group['id'].split(':', 2)
        j, k = int(j), int(k)
        mask, view = MASKS[k]
        sprite = sprites[mask]
        origin = next(row['source'] for row in mask_data['records'] if row.get('index') == int(mask[6:-4]))
        choices = [i for i, row in enumerate(records) if row['source_group'] == parent_paths[j]]
        for n in range(group['planned_count']):
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
                key = (frame_index, k, tone, height, left, top)
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise ValueError('Could not draw distinct placements')
            ground = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            state = {'id': f'g{j}_view{k}_{tone}_{n:04d}', 'failure_event': groups[0]['failure_event'],
                'conditions': ['placed', view, tone], 'min_count': 1, 'max_count': 1,
                'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                'background': {'image': record['image']}, 'target': {'state': 'placed', 'mask': mask},
                'labels': template['labels'], 'augmentation': {'tone': tone},
                'parameters': {'height': height, 'left': left, 'top': top, 'ground_polygon': ground,
                    'occlusion_polygons': [], 'visible_kind': 'fixture',
                    'shadow_strength': round(rng.uniform(.12, .26), 3),
                    'visibility_evidence': '原候选掩膜牌面、立杆及底座完整；生成标签仍为候选。'},
                'sources': list(dict.fromkeys([record['image']] + ([record['source_video']] if record.get('source_video') else []) + [
                    (measured.base.ANNOTATED / mask).relative_to(ROOT).as_posix(), origin, measured.base.SIGN_VIDEO]))}
            if n < 2:
                state['parameters'].update({'retain_lower_half': True, 'visible_kind': 'identifiable_pole',
                    'visibility_evidence': '设计裁切：去掉完整牌具上半部，只保留下半部立杆及底座；标签由剩余可见掩膜计算，不声称存在对应实拍。'})
            result['strata'].append(state)
    measured._prepare(result, {}, frame_inputs, json.loads(
        (measured.base.ANNOTATED / 'manifest.json').read_text(encoding='utf8')), ROOT)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--seed', type=int, default=1918)
    parser.add_argument('--output', type=Path, default=OUTPUT_PLAN)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if out != OUTPUT_PLAN.resolve():
        raise ValueError('sign plan must use config/production_plan.json')
    value = build(args.budget, args.seed)
    with out.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'budget': value['budget'], 'groups': [(g['id'], g['planned_count'])
                      for g in value['group_allocation']['strata']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
