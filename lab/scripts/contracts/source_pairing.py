#!/usr/bin/env python3
"""Validate task-level source pairing declarations.

Checks the pairing block in datasets/scripts/<task>/config/source.yaml:
unique IDs, split/action enums, capture placeholders and capture_id links,
path/evidence existence, and consistency with config/full_train_allocation.yaml.
It does not write plans, quotas, data or reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
TASKS = ['person', 'e_bike', 'fire', 'building_name_paper', 'sign',
         'waste_bin_state', 'waste_sticker', 'meter']
ALLOCATION = Path('config/full_train_allocation.yaml')
SPLITS = {'train', 'val', 'test', 'shared', 'unassigned', 'inherit'}
STATUSES = {'proposed', 'reserved', 'confirmed'}
ACTIONS = {'selected', 'reserved', 'to_review', 'to_label', 'to_extract', 'to_capture', 'blocked'}
CAPTURE_SESSIONS = {'planned', 'proven', 'unknown', 'same_session'}
CAPTURE_STATUSES = {'proposed', 'conditional', 'confirmed'}
ROLES = {'direct_target', 'composite_target', 'occluder', 'composite_background',
         'empty_negative', 'geometry_reference', 'material_reference', 'mask_source',
         'target_foreground', 'ocr_target', 'ordinary_pole', 'reference', 'surface'}
ALIGNMENT_STATUSES = {'aligned', 'stale', 'unknown'}


def _relative(value):
    return str(value).replace('\\', '/').rstrip('/')


def _exists(root, relative):
    return (root / relative).exists()


def _normalize(value):
    return str(value).replace('\\', '/').rstrip('/')


def validate_pairing(root, task, data, allocation=None, data_root=None, check_paths=True):
    data_root = Path(data_root) if data_root else root
    errors = []
    pairing = data.get('pairing')
    if not isinstance(pairing, dict):
        return ['{}: 缺少 pairing'.format(task)]
    if pairing.get('schema_version') != 1:
        errors.append('{}: pairing.schema_version 必须为 1'.format(task))
    if pairing.get('status') not in {'draft', 'reserved', 'executable'}:
        errors.append('{}: pairing.status 无效'.format(task))
    policy = pairing.get('policy')
    if not isinstance(policy, dict):
        errors.append('{}: 缺少 pairing.policy'.format(task))
    composition = pairing.get('composition_policy')
    if not isinstance(composition, dict):
        errors.append('{}: 缺少 pairing.composition_policy'.format(task))
    else:
        for key in ('target_source', 'blender_priority', 'target_components',
                    'background_components'):
            if key not in composition:
                errors.append('{}: composition_policy 缺少 {}'.format(task, key))
        for key in ('target_components', 'background_components'):
            values = composition.get(key)
            if not isinstance(values, list) or not values or not all(
                    isinstance(value, str) and value for value in values):
                errors.append('{}: composition_policy.{} 必须是非空字符串列表'.format(task, key))
        if composition.get('blender_priority') not in {'background_only', 'increased'}:
            errors.append('{}: composition_policy.blender_priority 无效'.format(task))
        if (allocation and isinstance(allocation.get('tasks', {}).get(task), dict)
                and allocation['tasks'][task].get('source_mix_status') != 'historical_reference'):
            expected = allocation['tasks'][task].get('source_mix', {}).get('blender')
            actual = composition.get('blender_share_target', 0.30)
            if expected is not None and abs(float(actual) - float(expected)) > 1e-9:
                errors.append('{}: composition_policy.blender_share_target={} 与全量来源组成 {} 不一致'.format(
                    task, actual, expected))
    ids = []
    captures = {}
    for item in pairing.get('captures', []) or []:
        cid = item.get('id') if isinstance(item, dict) else None
        if not cid or cid in captures:
            errors.append('{}: captures ID 无效或重复: {}'.format(task, cid))
            continue
        captures[cid] = item
        if item.get('status') not in CAPTURE_STATUSES:
            errors.append('{}: {} captures.status 无效'.format(task, cid))
    placeholders = []
    for key in ('task_real', 'backgrounds', 'data_sources', 'blender'):
        for item in pairing.get(key, []) or []:
            if not isinstance(item, dict):
                errors.append('{}: {} 下存在非映射条目'.format(task, key))
                continue
            item_id = item.get('id')
            if not item_id or item_id in ids:
                errors.append('{}: {} 下 ID 无效或重复: {}'.format(task, key, item_id))
            ids.append(item_id)
            split = item.get('split') or {}
            primary = split.get('primary')
            if primary is not None and primary not in SPLITS:
                errors.append('{}: {} split.primary 无效: {}'.format(task, item_id, primary))
            if split.get('status') is not None and split['status'] not in STATUSES:
                errors.append('{}: {} split.status 无效: {}'.format(task, item_id, split.get('status')))
            if split.get('action') is not None and split['action'] not in ACTIONS:
                errors.append('{}: {} split.action 无效: {}'.format(task, item_id, split.get('action')))
            if item.get('capture_session') is not None and item['capture_session'] not in CAPTURE_SESSIONS:
                errors.append('{}: {} capture_session 无效: {}'.format(task, item_id, item.get('capture_session')))
            roles = item.get('roles') or []
            if not isinstance(roles, list) or any(not isinstance(role, str) or role not in ROLES for role in roles):
                errors.append('{}: {} 角色未登记: {}'.format(task, item_id, roles))
            if 'empty_negative' in roles and (item.get('review') or {}).get('state') not in (
                    'confirmed_empty', 'candidate_empty', 'candidate_complete_negative'):
                errors.append('{}: {} empty_negative 需记录空场判定'.format(task, item_id))
            is_planned = split.get('action') == 'to_capture'
            if is_planned:
                placeholders.append((item_id, item))
                cid = item.get('capture_id')
                if cid not in captures:
                    errors.append('{}: {} 缺少有效 capture_id'.format(task, item_id))
                elif key == 'task_real':
                    if captures[cid].get('target_split') != primary:
                        errors.append('{}: {} capture_id 的 split 与占位 split 不一致'.format(task, item_id))
                    else:
                        save_to = _normalize(captures[cid].get('save_to', ''))
                        paths = item.get('paths') or ([item.get('path')] if item.get('path') else [])
                        if not any(_normalize(path) == save_to for path in paths):
                            errors.append('{}: {} 的路径与 capture save_to 不一致'.format(task, item_id))
                    if item.get('capture_session') != 'planned':
                        errors.append('{}: {} 计划补采必须使用 capture_session: planned'.format(task, item_id))
            for path in (item.get('paths') or ([item.get('path')] if item.get('path') else [])):
                if path and not is_planned and check_paths and not _exists(data_root, path):
                    errors.append('{}: {} 路径不存在: {}'.format(task, item_id, path))
            for evidence in item.get('evidence', []) or []:
                if isinstance(evidence, dict) and evidence.get('manifest'):
                    if check_paths and not _exists(data_root, evidence['manifest']):
                        errors.append('{}: {} 证据不存在: {}'.format(task, item_id, evidence['manifest']))
            parent = item.get('parent_group')
            if key == 'data_sources' and parent:
                if parent not in {p.get('id') for k in ('task_real', 'backgrounds')
                                  for p in (pairing.get(k, []) or []) if isinstance(p, dict)}:
                    errors.append('{}: {} parent_group 不存在: {}'.format(task, item_id, parent))
                if is_planned is False:
                    parent_item = next((p for k in ('task_real', 'backgrounds')
                                        for p in (pairing.get(k, []) or [])
                                        if isinstance(p, dict) and p.get('id') == parent), None)
                    if parent_item and (parent_item.get('split') or {}).get('action') == 'to_capture':
                        errors.append('{}: {} 的父组是计划补采，派生源也必须标记 to_capture'.format(task, item_id))
    for cid, item in captures.items():
        linked = [p for p in placeholders if p[1].get('capture_id') == cid]
        if not linked:
            errors.append('{}: captures {} 没有 task_real 占位'.format(task, cid))
        if item.get('status') == 'confirmed':
            save_to = item.get('save_to')
            if save_to and check_paths and not _exists(data_root, save_to):
                errors.append('{}: confirmed capture 的 save_to 不存在: {}'.format(task, save_to))
    bindings = pairing.get('plan_bindings') or {}
    for split, paths in bindings.items():
        if not isinstance(paths, list):
            errors.append('{}: plan_bindings.{} 必须是列表'.format(task, split))
            continue
        for path in paths:
            if check_paths and not _exists(data_root, path):
                errors.append('{}: plan_bindings.{} 路径不存在: {}'.format(task, split, path))
    alignment = pairing.get('plan_alignment')
    if alignment is not None:
        if not isinstance(alignment, dict):
            errors.append('{}: plan_alignment 必须是映射'.format(task))
        else:
            if alignment.get('status') not in ALIGNMENT_STATUSES:
                errors.append('{}: plan_alignment.status 无效'.format(task))
            if not isinstance(alignment.get('note'), str) or not alignment['note'].strip():
                errors.append('{}: plan_alignment 缺少说明'.format(task))
            conflicts = alignment.get('conflicts', [])
            if not isinstance(conflicts, list):
                errors.append('{}: plan_alignment.conflicts 必须是列表'.format(task))
            else:
                for index, conflict in enumerate(conflicts):
                    prefix = '{}: plan_alignment.conflicts[{}]'.format(task, index)
                    if not isinstance(conflict, dict):
                        errors.append(prefix + ' 必须是映射')
                        continue
                    parent = conflict.get('parent')
                    if parent not in ids:
                        errors.append(prefix + ' parent 不存在: {}'.format(parent))
                    if not isinstance(conflict.get('plan'), str) or not conflict['plan']:
                        errors.append(prefix + ' 缺少 plan')
                    elif check_paths and not _exists(data_root, conflict['plan']):
                        errors.append(prefix + ' plan 路径不存在: {}'.format(conflict['plan']))
                    if conflict.get('pairing_split') not in SPLITS:
                        errors.append(prefix + ' pairing_split 无效')
                    if not isinstance(conflict.get('plan_use'), str) or not conflict['plan_use']:
                        errors.append(prefix + ' 缺少 plan_use')
                if alignment.get('status') == 'stale' and not conflicts:
                    errors.append('{}: stale plan_alignment 必须列出 conflicts'.format(task))
    return errors


def validate_all(root=ROOT, data_root=None, check_paths=None, task=None):
    if task is not None and task not in TASKS:
        raise ValueError('未知数据任务: {}'.format(task))
    data_root = Path(data_root) if data_root else root
    if check_paths is None:
        check_paths = ((data_root / 'datasets/raw').is_dir()
                       or (data_root / 'datasets/annotated').is_dir())
    allocation_path = root / ALLOCATION
    allocation = yaml.safe_load(allocation_path.read_text(encoding='utf-8')) if allocation_path.is_file() else None
    errors = []
    for selected_task in ([task] if task is not None else TASKS):
        path = root / 'datasets/scripts' / selected_task / 'config/source.yaml'
        if not path.is_file():
            errors.append('{}: source.yaml 不存在'.format(selected_task))
            continue
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        errors.extend(validate_pairing(root, selected_task, data, allocation, data_root, check_paths))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='工作空间根目录（读取配置），默认本仓库')
    parser.add_argument('--data-root', help='数据/证据存在性检查根目录，默认同 --root；worktree 中可指向主检出')
    parser.add_argument('--json', action='store_true', help='输出 JSON')
    parser.add_argument('--check', action='store_true', help='检查（默认）')
    parser.add_argument('--task', choices=TASKS, help='只检查指定数据任务；不指定则检查全部任务')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT
    data_root = Path(args.data_root).resolve() if args.data_root else root
    try:
        errors = validate_all(root, data_root, task=args.task)
        if not ((data_root / 'datasets/raw').is_dir() or (data_root / 'datasets/annotated').is_dir()):
            print('数据树未展开，跳过路径存在性检查（结构、ID 和 capture 关联仍检查）。')
        if args.json:
            print(json.dumps({'errors': errors}, ensure_ascii=False, indent=2))
        else:
            for error in errors:
                print('ERROR: ' + error)
            print('父子源配对检查：{} 个错误。'.format(len(errors)))
        return 1 if errors else 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print('父子源配对检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
