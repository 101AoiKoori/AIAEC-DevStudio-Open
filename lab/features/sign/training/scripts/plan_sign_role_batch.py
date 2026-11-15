"""Plan sign compositions from explicit per-image task/split/role relations.

The resulting labels are candidates. This script does not create images or run
the production guard, and its quotas do not indicate independent acquisitions.
"""
import argparse
import json
import random
from pathlib import Path

import cv2

import compose_sign_v4_measured as measured

ROOT = measured.ROOT
SOURCE = ROOT / 'lab/features/sign/training/config/sign_v4_grounded_source_20261114.json'
DISTRIBUTION = ROOT / 'lab/features/sign/training/config/sign_v4_distribution_candidate.json'
DESTINATION = ROOT / 'lab/features/sign/training/config/sign_v4_role_batch_20261114.json'
SEED = 1918


def role_rows(records, role):
    """Match all three relationship fields on one status, never across rows."""
    result = []
    for row in records:
        matching = [item for item in row.get('task_use_status', [])
                    if item.get('task') == 'sign' and item.get('split') == 'train'
                    and item.get('role') == role]
        if any(item.get('status') in ('candidate', 'ready') for item in matching):
            if any(item.get('status') not in ('candidate', 'ready') for item in matching):
                raise ValueError('Conflicting use statuses for ' + row['image'])
            result.append(row)
    return result


def existing(path):
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or resolved.relative_to(ROOT).as_posix() != path or not resolved.is_file():
        raise ValueError('Missing or noncanonical workspace input: ' + path)
    return resolved


def build(source, distribution, seed=SEED):
    backgrounds = role_rows(source['records'], 'background')
    masks = role_rows(source['records'], 'target_foreground')
    # Only the supplied input contains the entry-specific absence observation,
    # crop, and the region for this precise background. Do not infer those from
    # another role, from a file name, or from a parent-video declaration.
    pairs = [(row, entry) for row in backgrounds for entry in source['background_inputs']
             if row['image'] == entry['image'] and entry.get('sign_absence') == 'confirmed'
             and entry.get('sign_absence_evidence')]
    compatible = [(bg, entry, fg) for bg, entry in pairs for fg in masks
                  if Path(fg['image']).name in dict(measured.VIEWS)
                  and fg['image'] == (measured.base.ANNOTATED / Path(fg['image']).name).relative_to(ROOT).as_posix()
                  and bg.get('parent_video') == entry.get('parent_video')]
    if len(compatible) != 1:
        raise ValueError('Expected one explicitly described compatible source pair; found ' + str(len(compatible)))
    bg, entry, fg = compatible[0]
    for path in (bg['image'], bg['parent_video'], fg['image'], fg['parent_image'], fg['parent_video']):
        existing(path)
    sprite = measured.base.cutout(Path(fg['image']).name)
    # Crop validity and image readability are checked without composing images.
    image = measured.base.read_image(existing(bg['image']), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = entry['crop_xyxy']
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]
            and (x2-x1)*3 == (y2-y1)*4):
        raise ValueError('Invalid source crop')
    branches = {b['id']: b for b in distribution['v4_combination_plan']['branches']}
    selected_bg = next(g['id'] for g in distribution['v4_combination_plan']['real_background_source_groups']
                       if g['parent'] == bg['parent_video'])
    selected_fg = next(g['id'] for g in distribution['v4_combination_plan']['foreground_groups']
                       if g['parent'] == fg['parent_video'])
    polygon = source['strata'][0]['parameters']['ground_polygon']
    contour = measured._polygon(polygon, 'Ground polygon')
    rng = random.Random(seed)
    strata = []
    # The 80 pole-only positive subtarget is retained as a gap: a cropped
    # ordinary-looking pole is not evidence that it remains identifiable.
    for branch_id in ('PHOTO-FULL', 'PHOTO-CLIP'):
        branch = branches[branch_id]
        desired = branch['target_images'] - branch.get('pole_only_target', 0)
        seen = set()
        attempts = 0
        while len(seen) < desired:
            attempts += 1
            if attempts > desired * 1000:
                raise ValueError('Insufficient distinct placements for ' + branch_id)
            height = rng.randint(170, 260) if branch_id == 'PHOTO-FULL' else rng.randint(500, 530)
            bottom = rng.randint(345, 377)
            top = bottom - height
            left = rng.randint(40, 170)
            width = max(1, round(sprite.shape[1] * height / sprite.shape[0]))
            tone = rng.choice(('normal', 'dark_low_saturation'))
            identity = (height, left, top, tone)
            if identity in seen or left + width > 640 or cv2.pointPolygonTest(
                    contour, (left + width / 2, float(bottom)), False) < 0:
                continue
            # Clipped examples retain the lower part of the red face as well as
            # the pole/base; no claim is made about pole-only identification.
            if branch_id == 'PHOTO-CLIP' and not (-height * .35 <= top < 0):
                continue
            seen.add(identity)
            n = len(seen)
            evidence = ('原候选掩膜含正面牌面、立杆与底座，完整画幅内；边缘与生成后可辨性仍为候选。'
                        if branch_id == 'PHOTO-FULL' else
                        '原候选掩膜正面牌面上缘裁切，牌面下部及杆、底座留在画幅内；可辨性仍为候选。')
            strata.append({'id': f'{branch_id.lower()}_{n:04d}',
                'failure_event': '可辨牌具漏检或定位失败', 'conditions': ['placed', 'front', tone],
                'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                'exposure_weight': 1, 'failure_cost': 1,
                'background': {'image': bg['image']},
                'target': {'state': 'placed', 'mask': Path(fg['image']).name},
                'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                'augmentation': {'tone': tone},
                'parameters': {'height': height, 'left': left, 'top': top,
                    'ground_polygon': polygon, 'occlusion_polygons': [], 'visible_kind': 'fixture',
                    'visibility_evidence': evidence},
                'sources': list(dict.fromkeys([bg['image'], bg['parent_video'], fg['image'],
                                                fg['parent_image'], fg['parent_video']]))})
    blockers = {
        'DIRECT-FULL': '无已登记 train 直接图像及完整牌具标签的可读入口输入',
        'DIRECT-CLIP': '无已登记 train 直接截断图像及完整标签的可读入口输入',
        'DIRECT-EMPTY': '无已登记 train 直接空牌图像及空标签的可读入口输入',
        'DIRECT-POLE-NEG': '无已登记 train 普通杆无牌直接图像及空标签',
        'PHOTO-CLIP': '牌杆单独可辨的 80 项缺少可辨性证据',
        'PHOTO-EMPTY': '现有显式背景入口不支持跨任务非牌具前景的空标签合成',
        'PHOTO-POLE-NEG': '无普通杆前景掩膜及对应合成入口',
        'BLENDER-FULL': '可供入口读取且已登记用途的 Blender 背景图缺失',
        'BLENDER-CLIP': '可供入口读取且已登记用途的 Blender 背景图缺失',
        'BLENDER-EMPTY': '可供入口读取且已登记用途的无牌 Blender 背景图缺失',
        'BLENDER-POLE-NEG': '可供入口读取的 Blender 背景和普通杆素材缺失',
    }
    reports = []
    for branch in branches.values():
        bid = branch['id']
        count = sum(s['id'].startswith(bid.lower() + '_') for s in strata)
        reports.append({'id': bid, 'target': branch['target_images'], 'schedulable': count,
                        'gap': branch['target_images'] - count,
                        'blocking_condition': blockers.get(bid) if count < branch['target_images'] else None})
    quota_changes = []
    for bid in ('PHOTO-FULL', 'PHOTO-CLIP'):
        branch = branches[bid]
        old = branch['background_foreground_shares'][selected_bg][selected_fg]
        new = next(r['schedulable'] for r in reports if r['id'] == bid)
        quota_changes.append({'branch': bid, 'background_parent_group': selected_bg,
            'foreground_parent_group': selected_fg, 'original_pair_quota': old,
            'scheduled_pair_quota': new, 'increase': new - old,
            'original_pair_quotas': branch['background_foreground_shares'],
            'scheduled_pair_quotas': {bg_id: {fg_id: (new if (bg_id, fg_id) == (selected_bg, selected_fg) else 0)
                                                 for fg_id in groups}
                                      for bg_id, groups in branch['background_foreground_shares'].items()},
            'reason': '现有角色关系及入口参数仅支持该候选来源组合；按足额要求重分配，保留来源集中事实。'})
    result = {k: source[k] for k in ('schema_version', 'model_key', 'target_split',
                                     'uncertainty_weight', 'constraints')}
    result.update({'budget': len(strata), 'sampling_seed': seed,
        'purpose': '固定种子候选合成排程，不生成图像',
        'independence_basis': '同一背景帧和同一牌具视频前景反复组合，不是独立采集',
        'background_inputs': [entry], 'strata': strata,
        'batch_report': {'branches': reports, 'quota_reallocations': quota_changes,
            'role_relationship_source': SOURCE.relative_to(ROOT).as_posix(),
            'distribution_source': DISTRIBUTION.relative_to(ROOT).as_posix(),
            'status': 'planned_not_produced',
            'limitations': ['前景及背景为 candidate；不提升原始素材或输出的准入状态。',
                '完整牌具分支 700、截断 120 均来自同一背景帧和同一牌具视面；不代表其他父源已覆盖。',
                '地面多边形仅作入口参数边界，并不证明所有落点真实接地。',
                '直接与 Blender 分支的缺口限于本次选用关系文件和当前可读输入，不代表全工作空间没有素材。',
                '未执行生产守卫；生产守卫及实际像素与标签结果未知。']}})
    measured._prepare(result, {}, [entry], json.loads(
        (measured.base.ANNOTATED / 'manifest.json').read_text(encoding='utf8')), ROOT)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'lab/features/sign/training/config').resolve()) or output.exists():
        raise ValueError('Output must be a new sign training config')
    source = json.loads(SOURCE.read_text(encoding='utf8'))
    distribution = json.loads(DISTRIBUTION.read_text(encoding='utf8'))
    result = build(source, distribution, args.seed)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        stream.write('{\n')
        items = list(result.items())
        for index, (key, value) in enumerate(items):
            stream.write('  ' + json.dumps(key) + ': ')
            if key == 'strata':
                stream.write('[\n')
                for state_index, state in enumerate(value):
                    stream.write('    ' + json.dumps(state, ensure_ascii=False, separators=(',', ':')))
                    stream.write(',\n' if state_index + 1 < len(value) else '\n')
                stream.write('  ]')
            else:
                stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
            stream.write(',\n' if index + 1 < len(items) else '\n')
        stream.write('}\n')
    print(json.dumps({'plan': output.relative_to(ROOT).as_posix(), 'budget': result['budget'],
                      'branches': result['batch_report']['branches']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
