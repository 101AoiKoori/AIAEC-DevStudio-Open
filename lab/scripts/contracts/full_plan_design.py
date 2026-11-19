#!/usr/bin/env python3
"""Validate and render the draft full-train plan allocation.

The allocation in config/full_train_allocation.yaml is a design draft: group
quotas and source mix, not active production plans. It does not write
production_plan.json. Training targets and combination axes remain owned by
config/task_targets.yaml and config/combination_axes.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ROOT / 'config/task_targets.yaml'
ALLOCATION = ROOT / 'config/full_train_allocation.yaml'
DOC = ROOT / 'docs/FULL_TRAIN_PLAN_DESIGN.md'
TASK_ORDER = ['person', 'e_bike', 'fire', 'building_name_paper', 'sign',
              'waste_bin_state', 'waste_sticker', 'meter']
ALLOWED_NEGATIVES = {'empty', 'no_paper', 'no_sign', 'ordinary_pole', 'absent'}


def source_share(value):
    return f'{value * 100:.2f}'.rstrip('0').rstrip('.') + '%'


def load_targets(path=TARGETS):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('task targets 版本必须为 1')
    return data['tasks']


def load_allocation(path=ALLOCATION):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('全量计划设计版本必须为 1')
    if data.get('status') != 'design_draft':
        raise ValueError('全量计划设计必须是 design_draft 状态')
    tasks = data.get('tasks')
    if not isinstance(tasks, dict) or set(tasks) != set(TASK_ORDER):
        raise ValueError('全量计划设计必须覆盖八个任务')
    return data


def _target_images(root, task):
    path = root / 'datasets/scripts' / task / 'config/dataset.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    value = data.get('train_target_images')
    if not isinstance(value, int) or value <= 0:
        raise ValueError('{} 缺少有效 train_target_images'.format(task))
    return value


def _allowed_targets(spec):
    allowed = {str(value) for value in spec.get('classes', [])}
    allowed |= {str(value) for value in spec.get('presence', [])}
    allowed |= {str(value) for value in spec.get('negatives', [])}
    allowed |= ALLOWED_NEGATIVES
    return allowed


def validate(allocation, targets, root=ROOT):
    result = {}
    for task in TASK_ORDER:
        spec = allocation['tasks'][task]
        total = spec.get('total')
        expected = _target_images(root, task)
        if total != expected:
            raise ValueError('{} total={} 与 dataset.yaml {} 不一致'.format(task, total, expected))
        mix = spec.get('source_mix')
        if not isinstance(mix, dict) or set(mix) != {'direct_real', 'real_composite', 'blender'}:
            raise ValueError('{} source_mix 必须包含 direct_real/real_composite/blender'.format(task))
        if any(not isinstance(value, (int, float)) or value < 0 for value in mix.values()):
            raise ValueError('{} source_mix 必须是非负数'.format(task))
        if abs(sum(mix.values()) - 1.0) > 1e-9:
            raise ValueError('{} source_mix 之和必须为 1'.format(task))
        mix_status = spec.get('source_mix_status', 'design_reference')
        if mix_status not in {'design_reference', 'historical_reference'}:
            raise ValueError('{} source_mix_status 无效'.format(task))
        cap = spec.get('blender_cap')
        if not isinstance(cap, (int, float)) or cap < 0 or cap > 1:
            raise ValueError('{} blender_cap 必须在 [0,1]'.format(task))
        allowed = _allowed_targets(targets[task])
        groups = spec.get('groups')
        if groups is not None:
            if not isinstance(groups, list) or not groups:
                raise ValueError('{} groups 必须是非空列表'.format(task))
            ids = set()
            group_total = 0
            for index, item in enumerate(groups):
                if not isinstance(item, dict):
                    raise ValueError('{}.groups[{}] 必须是映射'.format(task, index))
                group_id = item.get('id')
                if not isinstance(group_id, str) or not group_id or group_id in ids:
                    raise ValueError('{} group id 无效或重复: {}'.format(task, group_id))
                ids.add(group_id)
                quota = item.get('quota')
                if not isinstance(quota, int) or quota <= 0:
                    raise ValueError('{} {} quota 必须是正整数'.format(task, group_id))
                target = item.get('target')
                if target not in allowed:
                    raise ValueError('{} {} target 不在 task_targets: {}'.format(task, group_id, target))
                if not isinstance(item.get('axes', {}), dict):
                    raise ValueError('{} {} axes 必须是映射'.format(task, group_id))
                group_total += quota
            if group_total != total:
                raise ValueError('{} 组配额和 {} 不等于 total {}'.format(task, group_total, total))
            result[task] = {'total': total, 'groups': groups, 'mode': 'groups'}
        else:
            grid = spec.get('grid')
            if not isinstance(grid, dict):
                raise ValueError('{} 缺少 groups 或 grid'.format(task))
            count = grid.get('class_count')
            per = grid.get('per_class')
            if type(count) is not int or count <= 0 or count != len(targets[task].get('classes', [])):
                raise ValueError('{} grid.class_count 与 task_targets 中的类别数不一致'.format(task))
            if not isinstance(per, int) or per <= 0:
                raise ValueError('{} grid.per_class 无效'.format(task))
            grid_total = count * per
            if grid_total != total:
                raise ValueError('{} grid 合计 {} 不等于 total {}'.format(task, grid_total, total))
            result[task] = {'total': total, 'grid': grid, 'mode': 'grid'}
        result[task]['source_mix'] = mix
        result[task]['source_mix_status'] = mix_status
        result[task]['blender_cap'] = cap
        result[task]['notes'] = spec.get('notes', '')
        result[task]['startup_batch'] = spec.get('startup_batch', '')
        result[task]['target_spec'] = targets[task]
        split_design = spec.get('split_design')
        if split_design is not None:
            splits = split_design['splits']
            branches = ('direct_real','real_background_composite','blender_background_composite','empty_negative')
            if set(splits) != {'train','val','test'} or splits['train']['total'] != total:
                raise ValueError(f'{task} split_design 与 train 总额不一致')
            if sum(row['total'] for row in splits.values()) != split_design['total']:
                raise ValueError(f'{task} split_design 三集合之和不一致')
            for split,row in splits.items():
                if any(type(row.get(key)) is not int or row[key]<0 for key in branches):
                    raise ValueError(f'{task}/{split} 分支配额必须为非负整数')
                if sum(row[key] for key in branches) != row['total']:
                    raise ValueError(f'{task}/{split} 分支配额之和不一致')
            for axis,values in split_design['synthetic_positive_distributions'].items():
                if any(v<0 for v in values.values()) or abs(sum(values.values())-1)>1e-9:
                    raise ValueError(f'{task}/{axis} 分布必须归一化')
            groups = [set(row['indices']) for row in split_design['direct_record_groups'].values()]
            if any(a & b for i,a in enumerate(groups) for b in groups[i+1:]):
                raise ValueError(f'{task} 原生实拍不可跨集合重复')
            result[task]['split_design'] = split_design
    return result


def _plan_stats(root, task, targets):
    budget = 0
    capacity = 0
    plans = 0
    spec = targets[task]
    source_only = spec.get('plan_usage', {})
    for relative in spec['plans']:
        if relative in source_only:
            continue
        path = root / 'datasets/scripts' / task / relative
        plan = json.loads(path.read_text(encoding='utf-8'))
        if plan.get('target_split') != 'train':
            continue
        plans += 1
        budget += plan.get('budget', 0) or 0
        capacity += sum(item.get('max_count', 0) for item in plan.get('strata', [])
                        if isinstance(item, dict))
    return {'budget': budget, 'capacity': capacity, 'plans': plans}


def _target_summary(spec):
    parts = []
    if spec.get('classes'):
        parts.append('classes: ' + ', '.join(str(value) for value in spec['classes']))
    if spec.get('counts'):
        parts.append('counts: ' + ', '.join(spec['counts']))
    if spec.get('presence'):
        parts.append('presence: ' + ', '.join(spec['presence']))
    if spec.get('ocr'):
        parts.append('ocr: ' + spec['ocr'])
    if spec.get('keypoints'):
        parts.append('keypoints: {}'.format(spec['keypoints']))
    if spec.get('needle_axes'):
        parts.append('needle_axes: ' + ', '.join(spec['needle_axes']))
    return '；'.join(parts) or '-'


def render_design(allocation, validated, targets, root=ROOT):
    parent_reference = allocation.get('allocation_policy', {}).get('parent_share_reference', 0.8)
    lines = [
        '# 八任务全量 train 计划设计（草案）',
        '',
        '> 状态：`design_draft`。本文由 `config/full_train_allocation.yaml` 生成，不是已执行计划，也不覆盖各任务当前的 `production_plan.json`。组配额是设计分配，不是实际产量、实测分布或模型效果。',
        '',
        '## 设计原则',
        '',
        '- 训练目标（类别、关键点、OCR、计数）与组合轴（光照、视角、遮挡、背景等噪声）分开；目标见 `config/task_targets.yaml`，轴见 `config/combination_axes.yaml`。',
        '- 对象分类：本文件是 GeneratedDoc，由 PlanDraft `config/full_train_allocation.yaml` 生成；PlanDraft 不是 ProductionPlan，不被生产守卫消费。编译后的 `datasets/scripts/<任务>/config/production_plan.json` 才是 ProductionPlan。',
        '- 组配额只安排已声明的可行组合；不要求每个组合都有独立实拍对应证据。',
        '- 来源组成和 Blender 参考份额读取 `source_mix` 和 `blender_cap`；标为历史参考的比例不规定现行编批。按可用直接实拍和可行合成编批，实际份额按批次清单记录。',
        '- {:.0%} 参考份额用于提示父组集中度。编批时分配已合格的库存。同源 val/test 记录诊断范围，独立实拍评估另核物理父组与场次。底图容量和成图配额分别计数。'.format(parent_reference),
        '- 由各任务现用计划入口编制具体状态。生成、训练和评估各自使用对应入口。',
        '',
        '## 总览',
        '',
        '| 任务 | train 目标（张） | 现有 train 计划预算（张） | 现有计划容量（strata max_count 合计，张） | 需要新增容量（张） | 直接实拍 | 实拍合成 | Blender |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for task in TASK_ORDER:
        item = validated[task]
        stats = _plan_stats(root, task, targets)
        mix = item['source_mix']
        new_capacity = max(0, item['total'] - stats['capacity'])
        shares = (['按实际编批'] * 3 if item['source_mix_status'] == 'historical_reference' else
                  [source_share(mix[key]) for key in ('direct_real', 'real_composite', 'blender')])
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            task, item['total'], stats['budget'], stats['capacity'], new_capacity, *shares))
    lines.extend([
        '',
        '预算、计划容量和需要新增容量均以最终训练图片张数计。计划容量为数据集用途的 train 计划 strata 的 `max_count` 之和，源渲染和小样另见组合轴覆盖表。父组数见各任务计划与 `docs/PARENT_SOURCE_CAPACITY.md`。',
        '',
        '## 分任务设计',
        '',
    ])
    for task in TASK_ORDER:
        item = validated[task]
        spec = item['target_spec']
        stats = _plan_stats(root, task, targets)
        lines.extend([
            '### {}'.format(task),
            '',
            '- 训练目标：{}'.format(_target_summary(spec)),
            '- 全量目标：{} 张；现有 train 计划预算 {} 张；现有计划容量 {} 张；需要新增容量 {} 张。'.format(
                item['total'], stats['budget'], stats['capacity'],
                max(0, item['total'] - stats['capacity'])),
        ])
        if item['source_mix_status'] == 'historical_reference':
            lines.append('- 来源组成按当批可用的直接实拍目标和合成确定；旧的直接实拍 {:.0%}、实拍合成 {:.0%}、Blender {:.0%} 及 Blender {:.0%} 提示线仅供历史参考，不约束现行编批。'.format(
                item['source_mix']['direct_real'], item['source_mix']['real_composite'],
                item['source_mix']['blender'], item['blender_cap']))
        else:
            lines.append('- 来源组成参考为直接实拍 {}、实拍背景合成 {}、Blender {}，Blender 份额提示线为 {:.0%}。'.format(
                source_share(item['source_mix']['direct_real']), source_share(item['source_mix']['real_composite']),
                source_share(item['source_mix']['blender']), item['blender_cap']))
        if item['startup_batch']:
            lines.append('- ' + item['startup_batch'])
        if item['notes']:
            lines.append('- 设计说明：{}'.format(item['notes']))
        lines.append('')
        if item.get('split_design'):
            design=item['split_design']
            caption=('**三集合配额（当前工作集已生成）**' if design.get('status')=='executed_current_working_dataset'
                     else '**三集合配额（设计，尚未生成）**')
            lines.extend([caption,'',
                '| 集合 | 总数 | 原生实拍 | 实拍底板合成 | Blender 底板合成 | 无表负例 |',
                '| --- | ---: | ---: | ---: | ---: | ---: |'])
            for split,row in design['splits'].items():
                lines.append('| {} | {} | {} | {} | {} | {} |'.format(split,row['total'],row['direct_real'],row['real_background_composite'],row['blender_background_composite'],row['empty_negative']))
            lines.extend(['',design['scale_basis'],'',design['distribution_scope'],''])
            for axis,values in design['synthetic_positive_distributions'].items():
                lines.append('- {}：{}'.format(axis,'、'.join(f'{key} {value:.0%}' for key,value in values.items())))
            lines.extend(['','**来源和评估范围**',''])
            for key,value in design['background_policy'].items():
                lines.append(f'- {key}：{value}')
            lines.extend(['','**后续编批项**',''])
            lines.extend('- '+gap for gap in design['resource_gaps'])
            lines.extend(['',design['execution_boundary'],''])
        if item['mode'] == 'groups':
            lines.extend([
                '| 组 | 目标 | 组合轴 | 配额 |',
                '| --- | --- | --- | ---: |',
            ])
            for group in item['groups']:
                axes = ', '.join('{}={}'.format(key, value)
                                 for key, value in sorted(group.get('axes', {}).items()))
                lines.append('| {} | {} | {} | {} |'.format(
                    group['id'], group.get('target', '-'), axes or '-', group['quota']))
        else:
            grid = item['grid']
            lines.append('| 组 | 配额 |')
            lines.append('| --- | ---: |')
            lines.append('| {} 类 × 每类 {} 张 | {} |'.format(
                grid['class_count'], grid['per_class'], item['total']))
        lines.append('')
    lines.extend([
        '## 编制可执行批次',
        '',
        '1. 查询已有资格的共享底图、Blender 或实拍来源，纳入当批可行状态并分配预算。定向补判仍缺证据的用途。',
        '2. 在计划中分配 split 并记录父组用途和已知接触。同一母场景或拍摄原件按实际谱系计数。合格底图可承载多个有效布局，按最终图签计产量。',
        '3. 把组配额展开为 `production_plan.json` 的 strata，写清背景、目标、标签、增强、参数、来源和上下限。',
        '4. 按出图清单登记来源、覆盖和未补容量，按实际状态调整当批份额。设计或参考计划变更后同步组合轴覆盖文档。',
        '5. 当前可运行的批次见 [生产进度](DATASET_PRODUCTION.md#当前生产进度)。',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='工作空间根目录，默认本仓库')
    parser.add_argument('--allocation', help='全量分配配置，默认 <root>/config/full_train_allocation.yaml')
    parser.add_argument('--targets', help='训练目标配置，默认 <root>/config/task_targets.yaml')
    parser.add_argument('--doc', help='目标文档，默认 <root>/docs/FULL_TRAIN_PLAN_DESIGN.md')
    parser.add_argument('--json', action='store_true', help='输出 JSON 摘要')
    parser.add_argument('--write', action='store_true', help='写入设计文档')
    parser.add_argument('--check', action='store_true', help='检查设计文档是否同步')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT
    allocation_path = Path(args.allocation) if args.allocation else root / 'config/full_train_allocation.yaml'
    targets_path = Path(args.targets) if args.targets else root / 'config/task_targets.yaml'
    doc = Path(args.doc) if args.doc else root / 'docs/FULL_TRAIN_PLAN_DESIGN.md'
    try:
        allocation = load_allocation(allocation_path)
        targets = load_targets(targets_path)
        validated = validate(allocation, targets, root)
        summary = {
            'version': 1,
            'status': allocation['status'],
            'tasks': {
                task: {
                    'total': validated[task]['total'],
                    'source_mix': validated[task]['source_mix'],
                    'source_mix_status': validated[task]['source_mix_status'],
                    'blender_cap': validated[task]['blender_cap'],
                    'current': _plan_stats(root, task, targets),
                    'new_capacity': max(0, validated[task]['total'] - _plan_stats(root, task, targets)['capacity']),
                    'mode': validated[task]['mode'],
                }
                for task in TASK_ORDER
            },
        }
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        body = render_design(allocation, validated, targets, root)
        if args.write:
            doc.write_text(body, encoding='utf-8', newline='\n')
            print('已更新: {}'.format(doc.relative_to(root).as_posix()))
            return 0
        if args.check:
            changed = not doc.is_file() or doc.read_text(encoding='utf-8') != body
            print(('待更新: ' if changed else '已同步: ') + doc.relative_to(root).as_posix())
            return 1 if changed else 0
        print(body)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print('全量计划设计检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
