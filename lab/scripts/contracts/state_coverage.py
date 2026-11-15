#!/usr/bin/env python3
"""Report training-target coverage, combination-axis coverage and quota gaps.

Training targets are the classes, keypoints, OCR fields and counts the model
should learn (config/task_targets.yaml). Combination axes are nuisance or
domain conditions used to exclude features the model should not rely on
(config/combination_axes.yaml). The two are reported separately.

Reads selected plan JSON, runs the existing Bayesian allocator, and writes a
generated block in docs/DATASET_DESIGN.md. It never writes plan files or data.
Values are extracted only from fields already present in a stratum; an axis
without evidence is reported as unknown, not guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TARGETS = ROOT / 'config/task_targets.yaml'
AXES = ROOT / 'config/combination_axes.yaml'
DOC = ROOT / 'docs/DATASET_DESIGN.md'
START = '<!-- STATE_COVERAGE:START -->'
END = '<!-- STATE_COVERAGE:END -->'
TASK_ORDER = ['person', 'e_bike', 'fire', 'building_name_paper', 'sign',
              'waste_bin_state', 'waste_sticker', 'meter']


def load_targets(path=TARGETS):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('task targets 版本必须为 1')
    tasks = data.get('tasks')
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError('task targets 缺少 tasks')
    for task, spec in tasks.items():
        if not isinstance(spec, dict) or not isinstance(spec.get('plans'), list) or not spec['plans']:
            raise ValueError('{} 缺少 plans'.format(task))
        if not isinstance(spec.get('classes'), list) or not spec['classes']:
            raise ValueError('{} 缺少 classes'.format(task))
    return data


def load_axes(path=AXES):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('combination axes 版本必须为 1')
    tasks = data.get('tasks')
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError('combination axes 缺少 tasks')
    for task, spec in tasks.items():
        if not isinstance(spec, dict) or not isinstance(spec.get('plans'), list) or not spec['plans']:
            raise ValueError('{} 缺少 plans'.format(task))
        if not isinstance(spec.get('axes'), dict) or not spec['axes']:
            raise ValueError('{} 缺少 axes'.format(task))
        for name, values in spec['axes'].items():
            if values != 'dynamic' and not isinstance(values, list):
                raise ValueError('{}.{} 的 values 必须是列表或 dynamic'.format(task, name))
    return data


def _task_target_images(root, task):
    path = root / 'datasets/scripts' / task / 'config/dataset.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    target = data.get('train_target_images')
    return target if isinstance(target, int) else None


def _allocate(plan):
    from lab.training_common.bayesian_allocation import allocate
    return allocate(plan)


def _count_bucket(number):
    if number == 0:
        return ['empty']
    if number == 1:
        return ['single']
    return ['multi']


def _gain_bucket(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return ['normal'] if abs(value - 1.0) < 1e-9 else ['adjusted']


def _exposure_bucket(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return ['normal'] if abs(value) < 1e-9 else ['adjusted']


def _tone_bucket(tone):
    if tone in ('original', 'normal'):
        return ['normal']
    if tone:
        return ['adjusted']
    return None


def _person_targets(stratum):
    actors = (stratum.get('target') or {}).get('actors') or []
    people = [item for item in actors if isinstance(item, dict) and item.get('kind') == 'person']
    classes = sorted({item.get('category') for item in people
                      if item.get('category') in ('community', 'noncommunity')})
    match = re.search(r'_(community|noncommunity)_(none|upright|fallen)_tone\d+$',
                      str(stratum.get('id', '')))
    if match:
        classes = [match.group(1)]
    return {'classes': classes or None, 'count': _count_bucket(len(people))}


def _person_axes(stratum):
    out = {}
    actors = (stratum.get('target') or {}).get('actors') or []
    people = [item for item in actors if isinstance(item, dict) and item.get('kind') == 'person']
    vehicles = [item for item in actors if isinstance(item, dict) and item.get('kind') != 'person']
    views = sorted({item.get('view') for item in people if item.get('view')})
    out['view'] = views or None
    augmentation = stratum.get('augmentation') or {}
    out['lighting'] = _gain_bucket(augmentation.get('value_gain'))
    out['saturation'] = _gain_bucket(augmentation.get('saturation_gain'))
    overlaps = (stratum.get('parameters') or {}).get('required_overlaps') or []
    out['mutual_occlusion'] = ['present'] if overlaps else ['none']
    match = re.search(r'_(community|noncommunity)_(none|upright|fallen)_tone\d+$',
                      str(stratum.get('id', '')))
    if match:
        out['vehicle_occlusion'] = [match.group(2)]
    else:
        occlusions = sorted({item.get('category') for item in vehicles
                             if item.get('category') in ('upright', 'fallen')})
        out['vehicle_occlusion'] = occlusions or None
    return out


def _e_bike_targets(stratum):
    target = stratum.get('target') or {}
    if 'vehicles' in target:
        vehicles = target.get('vehicles') or []
        classes = sorted({item.get('class') for item in vehicles
                          if isinstance(item, dict) and item.get('class') in ('upright', 'fallen')})
        return {'classes': classes or ['empty'], 'count': _count_bucket(len(vehicles))}
    classes = sorted({value for value in (target.get('classes') or [])
                      if value in ('upright', 'fallen')})
    return {'classes': classes or None, 'count': None}


def _e_bike_axes(stratum):
    tone = (stratum.get('augmentation') or {}).get('tone')
    return {'lighting': _tone_bucket(tone), 'saturation': _tone_bucket(tone)}


def _fire_targets(stratum):
    target = stratum.get('target') or {}
    classes = ['fire_sticker'] if target.get('kind') == 'fire_sticker' else None
    count = target.get('count')
    if isinstance(count, int):
        bucket = ['zero'] if count == 0 else ['single'] if count == 1 else ['multi']
    else:
        bucket = None
    return {'classes': classes, 'count': bucket}


def _fire_axes(stratum):
    conditions = [value for value in (stratum.get('conditions') or []) if value.startswith('facade_')]
    facade = conditions[0][7:] if conditions else (stratum.get('parameters') or {}).get('source_id')
    return {'facade': [facade] if facade else None}


def _building_name_paper_targets(stratum):
    target = stratum.get('target') or {}
    papers = target.get('papers') or []
    texts = target.get('texts') if isinstance(target.get('texts'), dict) else {}
    if 'papers' in target or 'texts' in target:
        paper = 'paper' if (papers or texts) else 'no_paper'
    elif target.get('quad_xy') and target.get('text_policy'):
        paper = 'paper'
    else:
        paper = None
    names = {item.get('text') for item in papers if isinstance(item, dict) and item.get('text')}
    names |= set(texts.values())
    count = _count_bucket(len(names)) if names else (['zero'] if paper == 'no_paper' else None)
    return {'classes': [paper] if paper else None, 'count': count,
            'ocr': ['text'] if names else None}


def _building_name_paper_axes(stratum):
    out = {}
    buildings = (stratum.get('background') or {}).get('buildings') or []
    out['building_count'] = _count_bucket(len(buildings)) if buildings else None
    yaw = (stratum.get('parameters') or {}).get('yaw_degrees')
    out['yaw'] = ['frontal' if abs(yaw) <= 2 else 'oblique'] if isinstance(yaw, (int, float)) else None
    augmentation = stratum.get('augmentation') or {}
    out['lighting'] = _exposure_bucket(augmentation.get('exposure_ev'))
    return out


def _sign_targets(stratum):
    state = (stratum.get('target') or {}).get('state')
    if state not in ('placed', 'absent'):
        return {'classes': None, 'presence': None}
    return {'classes': ['sign_fixture' if state == 'placed' else 'no_sign'],
            'presence': [state]}


def _sign_axes(stratum):
    out = {}
    conditions = set(stratum.get('conditions') or [])
    views = [value for value in ('front', 'oblique', 'side', 'back') if value in conditions]
    out['view'] = views or None
    parameters = stratum.get('parameters') or {}
    kind = parameters.get('visible_kind')
    retain = parameters.get('retain_lower_half')
    if kind == 'identifiable_pole':
        out['truncation'] = ['pole_only']
    elif retain:
        out['truncation'] = ['partial']
    elif kind == 'fixture':
        out['truncation'] = ['none']
    else:
        out['truncation'] = None
    tone = (stratum.get('augmentation') or {}).get('tone')
    if tone == 'normal':
        out['lighting'] = ['normal']
        out['saturation'] = ['normal']
    elif tone == 'dark_low_saturation':
        out['lighting'] = ['dark']
        out['saturation'] = ['low']
    else:
        out['lighting'] = None
        out['saturation'] = None
    return out


def _waste_bin_state_targets(stratum):
    bins = (stratum.get('target') or {}).get('bins') or {}
    classes = sorted(set(bins.values())) if bins else ['empty']
    return {'classes': classes, 'count': _count_bucket(len(bins))}


def _waste_bin_state_axes(stratum):
    bins = (stratum.get('target') or {}).get('bins') or {}
    return {'color': sorted(bins) if bins else ['not_applicable']}


def _waste_sticker_targets(stratum):
    target = stratum.get('target') or {}
    if 'placements' not in target:
        return {'classes': None, 'count': None}
    placements = target.get('placements') or []
    classes = sorted({str(item.get('class_id')) for item in placements
                      if isinstance(item, dict) and item.get('class_id') is not None})
    return {'classes': classes or None, 'count': _count_bucket(len(placements))}


def _waste_sticker_axes(stratum):
    out = {}
    target = stratum.get('target') or {}
    placements = target.get('placements') or []
    out['bin_color'] = sorted({item.get('bin_id') for item in placements
                               if isinstance(item, dict) and item.get('bin_id')}) or None
    augmentation = stratum.get('augmentation') or {}
    if 'exposure_ev' in augmentation:
        out['lighting'] = _exposure_bucket(augmentation.get('exposure_ev'))
    else:
        out['lighting'] = _gain_bucket(augmentation.get('gain', augmentation.get('value_gain')))
    out['saturation'] = _gain_bucket(augmentation.get('saturation', augmentation.get('saturation_gain')))
    return out


def _meter_targets(stratum):
    target = stratum.get('target') or {}
    kind = target.get('kind')
    out = {'classes': ['meter_dial'] if kind else None,
           'presence': ['single'] if kind in ('dual_needle_meter', 'rendered_station') else None}
    parameters = stratum.get('parameters') or {}
    if 'temperature_fraction' in parameters and 'humidity_fraction' in parameters:
        out['needle_position'] = ['{}:{}'.format(
            parameters['temperature_fraction'], parameters['humidity_fraction'])]
    else:
        out['needle_position'] = None
    return out


def _meter_axes(stratum):
    parameters = stratum.get('parameters') or {}
    if 'temperature_fraction' not in parameters:
        return {'distance': None, 'lighting': None, 'saturation': None}
    return {
        'distance': ['ground_angle={}'.format(parameters.get('ground_angle_deg'))],
        'lighting': _exposure_bucket(parameters.get('exposure_ev')),
        'saturation': _gain_bucket(parameters.get('saturation')),
    }


TARGET_EXTRACTORS = {
    'person': _person_targets,
    'e_bike': _e_bike_targets,
    'fire': _fire_targets,
    'building_name_paper': _building_name_paper_targets,
    'sign': _sign_targets,
    'waste_bin_state': _waste_bin_state_targets,
    'waste_sticker': _waste_sticker_targets,
    'meter': _meter_targets,
}
AXIS_EXTRACTORS = {
    'person': _person_axes,
    'e_bike': _e_bike_axes,
    'fire': _fire_axes,
    'building_name_paper': _building_name_paper_axes,
    'sign': _sign_axes,
    'waste_bin_state': _waste_bin_state_axes,
    'waste_sticker': _waste_sticker_axes,
    'meter': _meter_axes,
}


def _target_keys(spec):
    keys = ['classes']
    if spec.get('counts'):
        keys.append('count')
    if spec.get('presence'):
        keys.append('presence')
    if spec.get('ocr'):
        keys.append('ocr')
    if spec.get('needle_axes'):
        keys.append('needle_position')
    return keys


def _summarize_plan(root, task, relative, target_spec, axis_spec):
    path = root / 'datasets/scripts' / task / relative
    plan = json.loads(path.read_text(encoding='utf-8'))
    strata = plan.get('strata')
    if not isinstance(strata, list) or not strata:
        raise ValueError('{} 缺少 strata'.format(path))
    try:
        allocated = _allocate(plan)['strata']
    except Exception as exc:
        return {
            'path': relative, 'split': plan.get('target_split'), 'budget': plan.get('budget'),
            'planned_total': None, 'capacity': None, 'strata_count': len(strata),
            'targets': {}, 'target_unknown': {}, 'axes': {}, 'unknown': {},
            'error': '{}: {}'.format(type(exc).__name__, exc),
        }
    if len(allocated) != len(strata):
        raise ValueError('{} 分配结果与 strata 数量不一致'.format(path))
    target_keys = _target_keys(target_spec)
    target_counts = {key: {} for key in target_keys}
    target_unknown = {key: 0 for key in target_keys}
    axis_counts = {axis: {} for axis in axis_spec}
    unknown = {axis: 0 for axis in axis_spec}
    target_fn = TARGET_EXTRACTORS.get(task)
    axis_fn = AXIS_EXTRACTORS.get(task)
    for source, planned in zip(strata, allocated):
        count = planned.get('planned_count', 0)
        target_values = target_fn(source) if target_fn else {}
        for key in target_keys:
            values = target_values.get(key)
            if values is None:
                target_unknown[key] += count
            else:
                for value in values:
                    target_counts[key][str(value)] = target_counts[key].get(str(value), 0) + count
        axis_values = axis_fn(source) if axis_fn else {}
        for axis in axis_spec:
            values = axis_values.get(axis)
            if values is None:
                unknown[axis] += count
            else:
                for value in values:
                    axis_counts[axis][str(value)] = axis_counts[axis].get(str(value), 0) + count
    return {
        'path': relative, 'split': plan.get('target_split'), 'budget': plan.get('budget'),
        'planned_total': sum(item.get('planned_count', 0) for item in allocated),
        'capacity': sum(item.get('max_count', 0) for item in strata if isinstance(item, dict)),
        'strata_count': len(strata), 'targets': target_counts,
        'target_unknown': target_unknown, 'axes': axis_counts, 'unknown': unknown,
        'error': None,
    }


def summarize(root=ROOT, targets_path=TARGETS, axes_path=AXES):
    targets = load_targets(targets_path)
    axes = load_axes(axes_path)
    if set(targets['tasks']) != set(axes['tasks']):
        raise ValueError('task_targets.yaml 与 combination_axes.yaml 的任务集合不一致')
    tasks = {}
    for task in TASK_ORDER:
        if task not in targets['tasks']:
            continue
        target_spec = targets['tasks'][task]
        axis_spec = axes['tasks'][task]
        if list(target_spec['plans']) != list(axis_spec['plans']):
            raise ValueError('{} 的目标与组合轴计划列表不一致'.format(task))
        target_images = _task_target_images(root, task)
        plans = [_summarize_plan(root, task, relative, target_spec, axis_spec['axes'])
                 for relative in target_spec['plans']]
        train = [plan for plan in plans if plan['split'] == 'train']
        planned = sum(plan['planned_total'] or 0 for plan in train)
        capacity = sum(plan['capacity'] or 0 for plan in train)
        target_totals = {key: {} for key in _target_keys(target_spec)}
        target_unknown = {key: 0 for key in _target_keys(target_spec)}
        axis_totals = {axis: {} for axis in axis_spec['axes']}
        unknown_totals = {axis: 0 for axis in axis_spec['axes']}
        for plan in plans:
            for key, counts in plan['targets'].items():
                for value, count in counts.items():
                    target_totals[key][value] = target_totals[key].get(value, 0) + count
            for key, count in plan['target_unknown'].items():
                target_unknown[key] += count
            for axis, counts in plan['axes'].items():
                for value, count in counts.items():
                    axis_totals[axis][value] = axis_totals[axis].get(value, 0) + count
            for axis, count in plan['unknown'].items():
                unknown_totals[axis] += count
        tasks[task] = {
            'target_images': target_images,
            'target_spec': target_spec,
            'axes_spec': axis_spec['axes'],
            'plans': plans,
            'targets': target_totals,
            'target_unknown': target_unknown,
            'axes': axis_totals,
            'unknown': unknown_totals,
            'train_planned': planned,
            'train_capacity': capacity,
            'budget_gap': max(0, target_images - planned) if target_images is not None else None,
            'capacity_gap': max(0, target_images - capacity) if target_images is not None else None,
        }
    return {'version': 1, 'tasks': tasks}


def _sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _format_counts(counts):
    return '、'.join('{}={}'.format(value, counts[value]) for value in sorted(counts, key=_sort_key))


def render_markdown(summary):
    lines = [
        '> 自动生成：修改 `config/task_targets.yaml`、`config/combination_axes.yaml` 或任务计划后运行 `python lab/scripts/contracts/state_coverage.py --write`。训练目标与组合轴分开统计；`unknown` 表示当前计划字段无法证明该值。计划配额是待生产目标，不是实际产量。',
        '',
        '| 任务 | 计划 | split | budget | planned_total | strata | 容量上限 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for task in TASK_ORDER:
        item = summary['tasks'].get(task)
        if not item:
            continue
        for plan in item['plans']:
            lines.append('| {} | `{}` | {} | {} | {} | {} | {} |'.format(
                task, plan['path'], plan['split'] or '-',
                plan['budget'] if plan['budget'] is not None else '-',
                plan['planned_total'] if plan['planned_total'] is not None else 'ERROR',
                plan['strata_count'], plan['capacity'] if plan['capacity'] is not None else '-'))
    lines.extend(['', '**训练目标覆盖（计划张数，可重叠）**', ''])
    for task in TASK_ORDER:
        item = summary['tasks'].get(task)
        if not item:
            continue
        pieces = []
        for key, counts in item['targets'].items():
            if counts:
                pieces.append('{} {}'.format(key, _format_counts(counts)))
        if not pieces:
            pieces.append('无可提取目标值')
        unknown_parts = ['{}={}'.format(key, count)
                         for key, count in item['target_unknown'].items() if count]
        lines.append('- {}：{}；未识别目标字段：{}。'.format(
            task, '；'.join(pieces), '、'.join(unknown_parts) or '无'))
    lines.extend(['', '**组合轴覆盖（计划张数，可重叠；这些不是训练目标）**', ''])
    for task in TASK_ORDER:
        item = summary['tasks'].get(task)
        if not item:
            continue
        pieces = []
        unmapped = []
        for axis, counts in item['axes'].items():
            if counts:
                pieces.append('{} {}'.format(axis, _format_counts(counts)))
        for axis, count in item['unknown'].items():
            if count:
                unmapped.append('{}(unknown={})'.format(axis, count))
            elif not item['axes'].get(axis):
                unmapped.append(axis)
        if not pieces:
            pieces.append('无可提取轴值')
        lines.append('- {}：{}；未映射：{}。'.format(
            task, '；'.join(pieces), '、'.join(dict.fromkeys(unmapped)) or '无'))
    lines.extend([
        '',
        '**全量缺口（最终 train 目标与现有 train 计划比较；计划用途可能重叠）**',
        '',
        '| 任务 | 最终 train | 现有 train 计划预算 | 计划容量上限 | 按预算缺口 | 容量缺口 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ])
    for task in TASK_ORDER:
        item = summary['tasks'].get(task)
        if not item or item['target_images'] is None:
            continue
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            task, item['target_images'], item['train_planned'], item['train_capacity'],
            item['budget_gap'], item['capacity_gap']))
    lines.extend([
        '',
        '容量上限是计划 strata 的 `max_count` 之和。容量缺口大于 0 表示不能只放大 budget，必须新增可行组合或状态；具体全量计划由后续设计补齐，本报告不生成配额。',
    ])
    return '\n'.join(lines)


def replace_section(path, body):
    text = path.read_text(encoding='utf-8')
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError('文档需要一对有序的状态覆盖生成标记: {}'.format(path))
    return text[:text.index(START)] + START + '\n' + body.rstrip('\n') + '\n' + text[text.index(END):]


def write_doc(summary, doc=DOC):
    body = render_markdown(summary)
    doc.write_text(replace_section(doc, body), encoding='utf-8', newline='\n')
    return body


def check_staged_inputs(root, targets, axes, summary):
    """Never stage a report derived from an uncommitted input."""
    paths = [targets, axes]
    for task, item in summary['tasks'].items():
        for plan in item['plans']:
            paths.append(root / 'datasets/scripts' / task / plan['path'])
    for path in paths:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        indexed = subprocess.run(['git', '-C', str(root), 'show', ':' + relative],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if indexed.returncode:
            raise ValueError('状态覆盖输入未暂存或不可读取: ' + relative)
        working = path.read_text(encoding='utf-8')
        if indexed.stdout.decode('utf-8').replace('\r\n', '\n') != working:
            raise ValueError('状态覆盖输入含未暂存修改: ' + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='工作空间根目录，默认本仓库')
    parser.add_argument('--targets', help='训练目标配置，默认 <root>/config/task_targets.yaml')
    parser.add_argument('--axes', help='组合轴配置，默认 <root>/config/combination_axes.yaml')
    parser.add_argument('--doc', help='目标文档，默认 <root>/docs/DATASET_DESIGN.md')
    parser.add_argument('--json', action='store_true', help='输出 JSON 摘要')
    parser.add_argument('--write', action='store_true', help='写入文档生成区块')
    parser.add_argument('--check', action='store_true', help='检查文档生成区块是否同步')
    parser.add_argument('--staged', action='store_true', help='要求输入文件已暂存')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT
    targets = Path(args.targets) if args.targets else root / 'config/task_targets.yaml'
    axes = Path(args.axes) if args.axes else root / 'config/combination_axes.yaml'
    doc = Path(args.doc) if args.doc else root / 'docs/DATASET_DESIGN.md'
    try:
        summary = summarize(root, targets, axes)
        if args.staged:
            check_staged_inputs(root, targets, axes, summary)
        has_error = any(plan['error'] for item in summary['tasks'].values() for plan in item['plans'])
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1 if has_error else 0
        if args.write:
            body = write_doc(summary, doc)
            print('已更新: {}'.format(doc.relative_to(root).as_posix()))
            print(body)
            return 1 if has_error else 0
        body = render_markdown(summary)
        if args.check:
            text = doc.read_text(encoding='utf-8')
            if text.count(START) != 1 or text.count(END) != 1:
                raise ValueError('文档需要一对有序的状态覆盖生成标记: {}'.format(doc))
            current = text[text.index(START) + len(START):text.index(END)].strip('\n')
            changed = current != body
            print(('待更新: ' if changed else '已同步: ') + doc.relative_to(root).as_posix())
            return 1 if changed or has_error else 0
        print(body)
        return 1 if has_error else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print('状态覆盖检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
