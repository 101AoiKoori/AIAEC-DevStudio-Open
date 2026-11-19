#!/usr/bin/env python3
"""Summarize selected plans and describe the paired checks for actual exports.

Plan counts are allocated image quotas from the listed reference plans.
--actual adds read-only counts from current fire/sign manifests and labels.
Learning targets, scene coverage and observed conditions have separate fields.
Generation controls alone do not establish observed appearance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TARGETS = ROOT / 'config/task_targets.yaml'
AXES = ROOT / 'config/combination_axes.yaml'
OBSERVATIONS = ROOT / 'config/combination_axis_observations.yaml'
DOC = ROOT / 'docs/DATASET_DESIGN.md'
START = '<!-- STATE_COVERAGE:START -->'
END = '<!-- STATE_COVERAGE:END -->'
TASK_ORDER = ['person', 'e_bike', 'fire', 'building_name_paper', 'sign',
              'waste_bin_state', 'waste_sticker', 'meter']
TASK_START = '<!-- TASK_DESIGN:START -->'
TASK_END = '<!-- TASK_DESIGN:END -->'


def render_task_design(task, target, axis):
    """Render one read-only task view from the shared design sources."""
    def display(value):
        if isinstance(value, list):
            return '、'.join(str(item) for item in value)
        return str(value)

    lines = ['## 学习目标与组合设计', '',
             '本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) '
             '和 [combination_axes.yaml](../../../config/combination_axes.yaml)，'
             '运行 `python lab/scripts/contracts/state_coverage.py --write`。', '',
             '**学习及标注**', '']
    for key, title in (('classes', '类别'), ('keypoints', '关键点'), ('ocr', '文字'),
                       ('needle_axes', '指针'), ('counts', '数量'), ('presence', '有无'),
                       ('negatives', '负例')):
        if key in target:
            lines.append('- {}：{}'.format(title, display(target[key])))
    lines.extend(['', target.get('annotation', '标注范围未定义。')])
    if target.get('notes'):
        lines.extend(['', target['notes']])
    if target.get('source_inputs'):
        lines.extend(['', '**素材用途**', ''])
        for role, requirement in target['source_inputs'].items():
            lines.append('- `{}`：{}'.format(role, requirement))
    lines.extend(['', '**组合轴与对照**', '',
                  '以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。', ''])
    for name, values in axis['axes'].items():
        lines.append('- `{}`：{}'.format(name, display(values)))
    if axis.get('placement'):
        lines.extend(['', '**布局**：' + axis['placement']])
    for cross in axis.get('crosses', []):
        lines.extend(['', '**{}**：{}'.format(cross['id'], cross['purpose']), '',
                      '- 安排：' + cross['pairs'], '- 核对：' + cross['check']])
    lines.extend(['', '对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。'])
    return '\n'.join(lines)


def update_task_designs(root, targets, axes, *, write):
    """Update/check generated blocks without changing task-specific navigation."""
    changed = []
    for task in TASK_ORDER:
        path = root / 'datasets/scripts' / task / 'README.md'
        text = path.read_text(encoding='utf-8')
        if text.count(TASK_START) != 1 or text.count(TASK_END) != 1 or text.index(TASK_START) >= text.index(TASK_END):
            raise ValueError('任务说明需要一对有序生成标记: {}'.format(path))
        body = render_task_design(task, targets['tasks'][task], axes['tasks'][task])
        start = text.index(TASK_START) + len(TASK_START)
        end = text.index(TASK_END)
        result = text[:start] + '\n\n' + body + '\n\n' + text[end:]
        if result != text:
            changed.append(path)
            if write:
                path.write_text(result, encoding='utf-8')
    return changed


def check_staged_task_designs(root, targets, axes):
    """Ensure indexed task views correspond to the staged design inputs."""
    for task in TASK_ORDER:
        relative = 'datasets/scripts/{}/README.md'.format(task)
        indexed = subprocess.run(['git', '-C', str(root), 'show', ':' + relative],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if indexed.returncode:
            raise ValueError('任务说明未暂存或不可读取: ' + relative)
        text = indexed.stdout.decode('utf-8').replace('\r\n', '\n')
        if (text.count(TASK_START) != 1 or text.count(TASK_END) != 1 or
                text.index(TASK_START) >= text.index(TASK_END)):
            raise ValueError('暂存任务说明缺少生成标记: ' + relative)
        body = text[text.index(TASK_START) + len(TASK_START):text.index(TASK_END)].strip('\n')
        if body != render_task_design(task, targets['tasks'][task], axes['tasks'][task]):
            raise ValueError('暂存任务说明尚未同步: ' + relative)


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
        usage = spec.get('plan_usage', {})
        if (not isinstance(usage, dict) or set(usage) - set(spec['plans'])
                or any(value not in ('source_render', 'sample', 'fine_tune') for value in usage.values())):
            raise ValueError('{} 的 plan_usage 必须指向已列出的源渲染、小样或微调计划'.format(task))
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
        if not isinstance(spec, dict) or not isinstance(spec.get('axes'), dict) or not spec['axes']:
            raise ValueError('{} 缺少 axes'.format(task))
        for name, values in spec['axes'].items():
            if values != 'dynamic' and not isinstance(values, list):
                raise ValueError('{}.{} 的 values 必须是列表或 dynamic'.format(task, name))
        crosses = spec.get('crosses', [])
        if not isinstance(crosses, list) or any(not isinstance(cross, dict) for cross in crosses):
            raise ValueError('{} 的 crosses 必须是对象列表'.format(task))
        ids = [cross.get('id') for cross in crosses]
        if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError('{} 的交叉 ID 必须非空且在任务内唯一'.format(task))
        structure = spec.get('structure', {})
        dimensions = structure.get('dimensions', {})
        objects = structure.get('objects', {})
        for name, dimension in dimensions.items():
            if (dimension.get('object') not in objects or dimension.get('role') not in ('target', 'nuisance')
                    or not isinstance(dimension.get('value_source'), str)):
                raise ValueError('{}.{} 的维度声明无效'.format(task, name))
        for cross in crosses:
            structured = cross.get('structured')
            if not structured:
                continue
            for contrast in structured.get('contrasts', []):
                hold, vary = contrast.get('hold', []), contrast.get('vary', [])
                if not vary or set(hold) & set(vary) or (set(hold) | set(vary)) - dimensions.keys():
                    raise ValueError('{}.{} 的对照维度无效'.format(task, cross['id']))
                variation = 'single' if len(vary) == 1 else 'unresolved'
                if contrast.get('variation') != variation:
                    raise ValueError('{}.{} 的变化关系须明确标识'.format(task, cross['id']))
    return data


def load_observations(path):
    """Read a historical snapshot; an absent snapshot means unknown, never current coverage."""
    if not path.is_file():
        return {'tasks': {}}
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('tasks'), dict):
        raise ValueError('组合轴观察记录格式无效')
    return data


def _task_target_images(root, task):
    path = root / 'datasets/scripts' / task / 'config/dataset.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    target = data.get('train_target_images')
    return target if isinstance(target, int) else None


def _allocate(plan):
    from lab.training_common.bayesian_allocation import allocate
    return allocate(plan)


def _count_bucket(number, zero='empty'):
    if number == 0:
        return [zero]
    if number == 1:
        return ['single']
    return ['multi']


def _person_targets(stratum):
    actors = (stratum.get('target') or {}).get('actors')
    if actors is None:
        return {}
    people = [actor for actor in actors if actor.get('kind') == 'person' and actor.get('view') == 'front']
    classes = sorted({actor.get('category') for actor in people
                      if actor.get('category') in ('community', 'noncommunity')})
    return {'classes': classes or (None if people else []), 'count': _count_bucket(len(people))}


def _person_axes(stratum):
    actors = (stratum.get('target') or {}).get('actors')
    if actors is None:
        return {}
    people = {actor['id']: actor for actor in actors
              if actor.get('kind') == 'person' and actor.get('view') == 'front'}
    vehicles = {actor['id']: actor.get('category') for actor in actors if actor.get('kind') == 'vehicle'}
    views = sorted({actor['view'] for actor in people.values()
                    if actor.get('view') in ('front', 'oblique')})
    out = {'view': views or None}
    if not people:
        return dict.fromkeys(('view', 'mutual_occlusion', 'vehicle_occlusion'), ['not_applicable'])
    overlaps = (stratum.get('parameters') or {}).get('required_overlaps') or []
    mutual = any(a in people and b in people for a, b in overlaps)
    if len(people) == 1:
        out['mutual_occlusion'] = ['not_applicable']
    elif mutual:
        out['mutual_occlusion'] = ['present']
    categories = {vehicles[b] for a, b in overlaps if a in people and b in vehicles}
    categories |= {vehicles[a] for a, b in overlaps if a in vehicles and b in people}
    out['vehicle_occlusion'] = sorted(categories & {'upright', 'fallen'}) or (
        None if vehicles else ['none'])
    return out


def _e_bike_targets(stratum):
    target = stratum.get('target') or {}
    if isinstance(target.get('vehicles'), list):
        labels = [vehicle.get('class') for vehicle in target['vehicles']]
    elif isinstance(target.get('classes'), list):
        # In the direct-photo plan this is an ordered class per box.
        labels = target['classes']
    else:
        return {}
    classes = sorted(set(labels) & {'upright', 'fallen'})
    return {'classes': classes or (None if labels else []),
            'count': None if target.get('complete_camera_label') is False else _count_bucket(len(labels))}


def _fire_targets(stratum):
    target = stratum.get('target') or {}
    counts = target.get('fire_counts')
    if isinstance(counts, dict):
        if any(type(value) is not int or value < 0 for value in counts.values()):
            return {}
        count = sum(counts.values())
    elif 'count' in target:
        count = target.get('count')
    else:
        return {}
    if type(count) is not int or count < 0:
        return {}
    return {'classes': ['fire_sticker'] if count else [], 'count': _count_bucket(count, 'zero')}


def _fire_axes(stratum):
    target = stratum.get('target') or {}
    facades = list(target.get('fire_counts') or stratum.get('building_counts') or {})
    if not facades and target.get('building'):
        facades = [target['building']]
    if not facades:
        facade = (stratum.get('parameters') or {}).get('source_id')
        facade = facade or next((value[7:] for value in stratum.get('conditions', [])
                                if value.startswith('facade_')), None)
        facades = [facade] if facade else []
    out = {'facade': facades} if facades else {}
    if _fire_targets(stratum).get('count') == ['zero']:
        out['position'] = ['not_applicable']
    distractor = target.get('distractor') or (stratum.get('context') or {}).get('distractor')
    if not distractor:
        distractor = (stratum.get('distractor') or {}).get('hard_negative_type')
    if isinstance(distractor, str):
        out['distractor'] = [distractor]
    return out


def _paper_stats(values, complete=True):
    count = len(values)
    text_count = sum(isinstance(value, str) and bool(value.strip()) for value in values)
    out = {'classes': ['paper'] if count else None, 'presence': ['present'] if count else None,
           'ocr': ['text'] if text_count else None}
    if complete:
        out.update(instance_count=[count], text_count=[text_count])
        if not count:
            out.update(classes=[], presence=['absent'], ocr=[])
    return out


def _building_name_paper_targets(stratum):
    target = stratum.get('target') or {}
    if type(target.get('paper_count')) is int and target['paper_count'] >= 0:
        return _paper_stats([None] * target['paper_count'])
    if isinstance(target.get('papers'), str):
        return _paper_stats([None] * len(target['papers']))
    if isinstance(target.get('texts'), dict):
        # Renderer text keys denote paper slots, even when strings match.
        return _paper_stats(list(target['texts'].values()))
    if target.get('quad_xy') and target.get('text_policy'):
        # One explicit placement on a background of unknown full-frame coverage.
        return _paper_stats([None], complete=False)
    if not isinstance(target.get('papers'), list):
        return {}
    papers = target['papers']
    generated = [paper.get('text') for paper in papers]
    candidates = (stratum.get('background') or {}).get('candidates') or []
    if not candidates:
        return _paper_stats(generated, complete=False)
    slots = {paper['slot_id'] for paper in papers}
    stats = []
    for candidate in candidates:
        replacements = {region['replacement_of'] for region in candidate.get('regions', [])
                        if region.get('slot_id') in slots and 'replacement_of' in region}
        existing = [paper.get('text') for paper in candidate.get('existing_papers', [])
                    if paper.get('building_id') not in replacements]
        stats.append(_paper_stats(existing + generated,
                                  complete=candidate.get('coverage') == 'complete_candidate'))
    # One stratum can draw distinct candidate backgrounds. Only count invariant
    # values against its full quota.
    return {key: value for key, value in stats[0].items()
            if all(other.get(key) == value for other in stats[1:])}


def _building_name_paper_axes(stratum):
    buildings = ((stratum.get('background') or {}).get('buildings') or
                 (stratum.get('target') or {}).get('buildings'))
    yaw = (stratum.get('parameters') or {}).get('yaw_degrees')
    return {'building_count': _count_bucket(len(buildings)) if buildings else None,
            'yaw': ['frontal' if abs(yaw) <= 15 else 'oblique'] if isinstance(yaw,(int,float)) else None}


def _sign_targets(stratum):
    if (stratum.get('labels') or {}).get('kind') == 'retained_background_boxes':
        return {'classes': ['sign_fixture'], 'presence': ['placed']}
    state = ((stratum.get('target') or {}).get('state') or
             (stratum.get('actual') or {}).get('state'))
    if state == 'placed':
        return {'classes': ['sign_fixture'], 'presence': ['placed']}
    if state in ('absent', 'negative'):
        return {'classes': [], 'presence': ['absent']}
    return {}


def _sign_axes(stratum):
    if _sign_targets(stratum).get('presence') == ['absent']:
        return {'view': ['not_applicable'], 'truncation': ['not_applicable']}
    conditions = set(stratum.get('conditions') or [])
    actual = stratum.get('actual') or {}
    views = [actual['view']] if actual.get('view') in ('front', 'oblique', 'side', 'back') else [
        value for value in ('front', 'oblique', 'side', 'back') if value in conditions]
    parameters = actual.get('geometry') or stratum.get('parameters') or {}
    layout = actual.get('layout') or stratum.get('layout') or parameters.get('layout')
    truncation = None
    if parameters.get('retain_lower_half') or parameters.get('visible_kind') == 'identifiable_pole':
        truncation = ['pole_only']
    elif actual and stratum.get('image_size') and all(
            key in parameters for key in ('left', 'top', 'sprite_width_px', 'sprite_height_px')):
        width, height = stratum['image_size']
        clipped = (parameters['left'] < 0 or parameters['top'] < 0 or
                   parameters['left'] + parameters['sprite_width_px'] > width or
                   parameters['top'] + parameters['sprite_height_px'] > height)
        truncation = ['partial' if clipped else 'none']
    elif layout in ('top_clipped', 'partial'):
        truncation = ['partial']
    elif layout in ('full', 'far_full', 'near_full', 'small', 'medium', 'occluded'):
        truncation = ['none']
    return {'view': views or None,
            'truncation': truncation}


def _waste_bin_state_targets(stratum):
    bins = (stratum.get('target') or {}).get('bins')
    if bins is None:
        return {}
    classes = sorted(set(bins.values()) & {'open', 'closed'})
    return {'classes': classes or (None if bins else []), 'count': _count_bucket(len(bins))}


def _waste_bin_state_axes(stratum):
    bins = (stratum.get('target') or {}).get('bins')
    if bins is None:
        return {}
    return {'color': sorted(bins) if bins else ['not_applicable']}


def _waste_sticker_targets(stratum):
    placements = (stratum.get('target') or {}).get('placements')
    if placements is None:
        return {}
    classes = sorted({str(item['class_id']) for item in placements if item.get('class_id') is not None})
    return {'classes': classes or (None if placements else []),
            'count': _count_bucket(len(placements)),
            'presence': ['present' if placements else 'absent']}


def _waste_sticker_axes(stratum):
    target = stratum.get('target') or {}
    if target.get('kind') == 'explicit_bin_stickers' and target.get('placements'):
        return {'carrier': ['bin_face']}
    return {}


def _meter_targets(stratum):
    kind = (stratum.get('target') or {}).get('kind')
    if kind == 'empty_meter_scene':
        return {'classes': [], 'presence': ['absent']}
    if kind not in ('dual_needle_meter', 'rendered_station'):
        return {}
    out = {'classes': ['meter_dial'], 'presence': ['single']}
    parameters = stratum.get('parameters') or {}
    if 'temperature_fraction' in parameters and 'humidity_fraction' in parameters:
        out['needle_position'] = ['{}:{}'.format(
            parameters['temperature_fraction'], parameters['humidity_fraction'])]
    return out


def _meter_axes(stratum):
    """Read declared camera/material controls, without claiming observed glare."""
    p = stratum.get('parameters') or {}
    out = {}
    for axis in ('saturation','exposure','light_relation'):
        if axis in p:
            out[axis] = [p[axis]]
    if 'reading_sampling' in p:out['reading_sampling']=[p['reading_sampling']]
    if 'placement_controls' in p:out['placement']=['randomized']
    elif p.get('branch') in ('real_background_composite','blender_background_composite'):out['placement']=['fixed_contact']
    if 'degradation' in p:out['degradation']=[p['degradation']['kind']]
    if p.get('face_material') in ('white', 'colored'):
        out['face_material'] = [p['face_material']]
    if 'distance_m' in p:
        out['distance'] = [str(p['distance_m'])]
    if 'yaw_deg' in p and 'elevation_deg' in p:
        out['view'] = ['frontal' if p['yaw_deg'] == 0 and p['elevation_deg'] == 0 else 'oblique']
    if 'glare' in p:
        out['glare'] = ['none' if p['glare'] == 'none' else 'present']
    if p.get('branch'):
        out['background'] = [p['source_image'] if p['branch'].startswith('direct_task_')
                             else p.get('background_key',p.get('background_record_id', 'open_neutral_preview'))]
    if p.get('branch') == 'empty_negative':
        out.update({key:['not_applicable'] for key in ('distance','view','glare','face_material','light_relation')})
        out.update({key:['not_applicable'] for key in ('reading_sampling','placement')})
    return out


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
    'meter': _meter_axes,
    'person': _person_axes,
    'fire': _fire_axes,
    'building_name_paper': _building_name_paper_axes,
    'sign': _sign_axes,
    'waste_bin_state': _waste_bin_state_axes,
    'waste_sticker': _waste_sticker_axes,
}


def _target_keys(spec):
    keys = ['classes']
    if spec.get('ocr'):
        keys.append('ocr')
    if spec.get('needle_axes'):
        keys.append('needle_position')
    return keys


def _coverage_keys(task, spec):
    if task == 'building_name_paper':
        return ['presence', 'instance_count', 'text_count']
    keys = []
    if spec.get('counts'):
        keys.append('count')
    if spec.get('presence'):
        keys.append('presence')
    return keys


def _batch_request(plan):
    # These declarations are written by the sign and e-bike plan compilers.
    declared = plan.get('quota_gap') or plan.get('allocation_evidence') or {}
    requested = declared.get('requested_budget')
    if requested is None:
        return {}
    executable = declared.get('executable_budget')
    shortfall = declared.get('remaining_shortfall')
    if (type(requested) is not int or type(executable) is not int or type(shortfall) is not int
            or executable != plan['budget'] or not 0 <= executable <= requested
            or shortfall != requested - executable):
        raise ValueError('批次需求、可执行预算和本批缺额不一致')
    return {'requested_budget': requested, 'remaining_shortfall': shortfall}


def _summarize_plan(root, task, relative, target_spec, axis_spec):
    path = root / 'datasets/scripts' / task / relative
    plan = json.loads(path.read_text(encoding='utf-8'))
    usage = target_spec.get('plan_usage', {}).get(relative, 'dataset')
    if task == 'waste_bin_state' and plan.get('purpose') == 'four_color_real_appearance_finetune':
        train = plan['train']
        total, photos, composites, replay = (train[key] for key in
                                             ('total', 'source_photos', 'real_cutout_composites', 'replay_images'))
        colors, states = ('blue', 'gray', 'green', 'red'), ('open', 'closed')
        negative_replay = train.get('replay_empty_images', 0)
        if (any(type(value) is not int or value < 0 for value in
                (total, photos, composites, replay, negative_replay)) or
                total != photos + composites + replay or negative_replay > replay):
            raise ValueError('{} 的微调图像配额不一致'.format(path))
        # These counters retain the global image-presence semantics: a mixed
        # four-bin frame contributes one image to each state present, not four.
        composite_classes, composite_colors = Counter(), Counter()
        class_instances, color_instances, color_state_instances = Counter(), Counter(), Counter()
        composite_counts = Counter()
        if train.get('composition_groups') is not None:
            from itertools import combinations, product

            if train.get('composition_recipe') != 'all_color_combinations_all_binary_states_equal_repetitions':
                raise ValueError('{} 缺少可解析的多桶状态组合配方'.format(path))
            groups = train['composition_groups']
            if not isinstance(groups, list):
                raise ValueError('{} 的多桶图像分组必须为列表'.format(path))
            for group in groups:
                count, images = group.get('count'), group.get('images')
                if (type(count) is not int or not 1 <= count <= len(colors) or
                        type(images) is not int or images < 0):
                    raise ValueError('{} 的多桶图像分组无效'.format(path))
                recipes = [(subset, pattern) for subset in combinations(colors, count)
                           for pattern in product(states, repeat=count)]
                if images % len(recipes):
                    raise ValueError('{} 的图像配额不能均匀覆盖颜色和开闭组合'.format(path))
                repetitions = images // len(recipes)
                composite_counts['single' if count == 1 else 'multi'] += images
                for subset, pattern in recipes:
                    composite_classes.update({state: repetitions for state in set(pattern)})
                    composite_colors.update({color: repetitions for color in subset})
                    for color, state in zip(subset, pattern):
                        class_instances[state] += repetitions
                        color_instances[color] += repetitions
                        color_state_instances[f'{color}:{state}'] += repetitions
            if sum(composite_counts.values()) != composites:
                raise ValueError('{} 的多桶图像分组与合成总量不一致'.format(path))
        else:
            # Historical single-bin recipes remain readable without inventing
            # a new composition recipe in their immutable run snapshots.
            per_color_state = train['composites_per_color_state']
            if (type(per_color_state) is not int or per_color_state < 0 or
                    composites != len(colors) * len(states) * per_color_state):
                raise ValueError('{} 的微调图像配额不一致'.format(path))
            composite_classes.update(dict.fromkeys(states, len(colors) * per_color_state))
            composite_colors.update(dict.fromkeys(colors, len(states) * per_color_state))
            class_instances.update(composite_classes)
            color_instances.update(composite_colors)
            color_state_instances.update({f'{color}:{state}': per_color_state
                                          for color in colors for state in states})
            composite_counts['single'] = composites
        # The plan fixes the composite allocation. Photo colors/states must come
        # from its real source annotations; replay counts do not imply labels.
        photo_rows = []
        annotation = root / plan.get('annotation_manifest', '__missing_reshoot_annotations__')
        if annotation.is_file():
            candidates = json.loads(annotation.read_text(encoding='utf-8'))['records']
            if (len(candidates) == photos and all(row.get('color') in colors and row.get('state') in states
                                                and len(row.get('bins', [])) == 1 for row in candidates)):
                photo_rows = candidates
        photo_states = Counter(row['state'] for row in photo_rows)
        photo_colors = Counter(row['color'] for row in photo_rows)
        class_instances.update(photo_states)
        color_instances.update(photo_colors)
        color_state_instances.update(f"{row['color']}:{row['state']}" for row in photo_rows)
        missing_photos = photos - len(photo_rows)
        target_keys, coverage_keys = _target_keys(target_spec), _coverage_keys(task, target_spec)
        targets = {key: {} for key in target_keys}
        target_unknown = {key: total for key in target_keys}
        targets['classes'] = {state: composite_classes[state] + photo_states[state] for state in states}
        target_unknown['classes'] = replay - negative_replay + missing_photos
        coverage = {key: {} for key in coverage_keys}
        coverage_unknown = {key: total for key in coverage_keys}
        if 'count' in coverage:
            coverage['count'] = {key: value for key, value in
                                 {'single': photos + composite_counts['single'],
                                  'multi': composite_counts['multi'], 'empty': negative_replay}.items() if value}
            coverage_unknown['count'] = replay - negative_replay
        axes = {axis: {} for axis in axis_spec}
        unknown = {axis: total for axis in axis_spec}
        if 'color' in axes:
            axes['color'] = {color: composite_colors[color] + photo_colors[color] for color in colors}
            unknown['color'] = replay - negative_replay + missing_photos
        not_applicable = {axis: negative_replay if axis == 'color' else 0 for axis in axis_spec}
        return {'usage': 'fine_tune', 'path': relative, 'split': 'train', 'budget': total,
                'planned_total': total, 'capacity': None, 'strata_count': 0, 'direct_capture_ids': [],
                'targets': targets, 'target_unknown': target_unknown,
                'coverage': coverage, 'coverage_unknown': coverage_unknown,
                'axes': axes, 'unknown': unknown, 'not_applicable': not_applicable,
                'instance_quotas': {'classes': dict(class_instances), 'color': dict(color_instances),
                                    'color_state': dict(color_state_instances),
                                    'known_instances': sum(class_instances.values()),
                                    'unresolved_images': replay - negative_replay + missing_photos},
                'generation_methods': {'direct_real': photos, 'real_cutout_composite': composites, 'replay': replay},
                'scope': 'Targets and axes count images containing each value; instance_quotas count planned boxes. '
                         'Replay labels and axes remain unknown except explicitly allocated empty images; '
                         'photo color/state counts require the referenced annotation manifest.',
                'error': None}
    strata = plan.get('strata')
    if not isinstance(strata, list) or not strata:
        raise ValueError('{} 缺少 strata'.format(path))
    try:
        request = _batch_request(plan)
        allocated = _allocate(plan)['strata']
    except Exception as exc:
        return {
            'usage': usage,
            'path': relative, 'split': plan.get('target_split'), 'budget': plan.get('budget'),
            'planned_total': None, 'capacity': None, 'strata_count': len(strata),
            'targets': {}, 'target_unknown': {}, 'coverage': {}, 'coverage_unknown': {},
            'axes': {}, 'unknown': {}, 'not_applicable': {},
            'error': '{}: {}'.format(type(exc).__name__, exc),
        }
    if len(allocated) != len(strata):
        raise ValueError('{} 分配结果与 strata 数量不一致'.format(path))
    target_keys = _target_keys(target_spec)
    target_counts = {key: {} for key in target_keys}
    target_unknown = {key: 0 for key in target_keys}
    coverage_keys = _coverage_keys(task, target_spec)
    coverage_counts = {key: {} for key in coverage_keys}
    coverage_unknown = {key: 0 for key in coverage_keys}
    axis_counts = {axis: {} for axis in axis_spec}
    unknown = {axis: 0 for axis in axis_spec}
    not_applicable = {axis: 0 for axis in axis_spec}
    target_fn = TARGET_EXTRACTORS.get(task)
    axis_fn = AXIS_EXTRACTORS.get(task)
    for source, planned in zip(strata, allocated):
        count = planned.get('planned_count', 0)
        target_values = target_fn(source) if target_fn else {}
        for totals, missing in ((target_counts, target_unknown), (coverage_counts, coverage_unknown)):
            for key, counts in totals.items():
                values = target_values.get(key)
                if values is None:
                    missing[key] += count
                else:
                    for value in values:
                        counts[str(value)] = counts.get(str(value), 0) + count
        axis_values = axis_fn(source) if axis_fn else {}
        if 'background' in axis_spec:
            background = source.get('background') or {}
            image = background.get('image') or (source.get('parameters') or {}).get('background_image')
            identifier = background.get('id') or (Path(image).stem if image else None)
            if identifier:
                axis_values['background'] = [identifier]
        for axis in axis_spec:
            values = axis_values.get(axis)
            if values is None or values == ['unknown']:
                unknown[axis] += count
            elif values == ['not_applicable']:
                not_applicable[axis] += count
            else:
                for value in values:
                    axis_counts[axis][str(value)] = axis_counts[axis].get(str(value), 0) + count
    return {
        **request,
        'usage': usage,
        'path': relative, 'split': plan.get('target_split'), 'budget': plan.get('budget'),
        'planned_total': sum(item.get('planned_count', 0) for item in allocated),
        'direct_capture_ids': plan.get('direct_capture_ids', []),
        'capacity': sum(item.get('max_count', 0) for item in strata if isinstance(item, dict)),
        'strata_count': len(strata), 'targets': target_counts,
        'target_unknown': target_unknown, 'coverage': coverage_counts,
        'coverage_unknown': coverage_unknown, 'axes': axis_counts, 'unknown': unknown,
        'not_applicable': not_applicable,
        'error': None,
    }


def _summarize_actual_batch(root, task, target_spec=None, targets_path=None):
    """Read current fire/sign pairs without treating a plan as produced data."""
    from PIL import Image

    relative = 'datasets/samples/{}/manifest.json'.format(task)
    result = {'path': relative, 'recorded_images': None, 'images': None, 'instances': None,
              'error': None, 'missing_files': []}
    try:
        targets_path = targets_path or root / 'config/task_targets.yaml'
        if target_spec is None:
            target_spec = load_targets(targets_path)['tasks'][task]
        classes = target_spec['classes']
        config = root / 'datasets/scripts' / task / 'config/dataset.yaml'
        design = yaml.safe_load(config.read_text(encoding='utf-8'))
        manifest = ((root / design['actual_manifest']) if design.get('actual_manifest') else
                    (root / design['sample_output'] / 'manifest.json')).resolve()
        relative = manifest.relative_to(root.resolve()).as_posix()
        result['path'] = relative
        before = manifest.stat()
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        records = data.get('records', data.get('samples'))
        if not isinstance(records, list):
            raise ValueError('manifest 缺少 records/samples 列表')
        counts, branches, class_counts, axis_counts, unknown = Counter(), Counter(), Counter(), {}, Counter()
        inapplicable, buildings, joint, splits = Counter(), {}, Counter(), {}
        manifest_plan = data.get('plan')
        planned_split = manifest_plan.get('target_split') if isinstance(manifest_plan, dict) else None
        images = instances = 0
        for row in records:
            files = []
            for key in ('image', 'label'):
                path = root / row[key]
                if not path.is_file():
                    path = manifest.parent / row[key]
                path = path.resolve()
                path.relative_to(root.resolve())
                files.append(path)
            missing = [p.relative_to(root.resolve()).as_posix() for p in files if not p.is_file()]
            result['missing_files'].extend(missing)
            if missing:
                continue
            with Image.open(files[0]) as image:
                image_size = list(image.size)
            labels = [line.split() for line in files[1].read_text(encoding='utf-8').splitlines()
                      if line.strip()]
            if any(len(fields) != 5 or not 0 <= int(fields[0]) < len(classes) or
                   not all(math.isfinite(float(v)) for v in fields[1:]) for fields in labels):
                raise ValueError('实际标签格式或类别无效: {}；类别见 {} 的 tasks.{}.classes'.format(
                    files[1], targets_path, task))
            class_counts.update(str(classes[int(fields[0])]) for fields in labels)
            count = len(labels)
            counts[str(count)] += 1
            images += 1
            instances += count
            split = row.get('split') or data.get('target_split') or data.get('split') or planned_split or 'unknown'
            split_counts = splits.setdefault(split, {'images': 0, 'instances': 0, 'negative_images': 0})
            split_counts['images'] += 1
            split_counts['instances'] += count
            split_counts['negative_images'] += count == 0
            branches[row.get('branch', 'unknown')] += 1
            if task == 'fire':
                actual = {**row, 'target': {'kind': 'fire_sticker', 'count': count}}
                values = _fire_axes(actual)
                for building, observation in (row.get('building_counts') or {}).items():
                    visible = observation.get('visible_count')
                    if type(visible) is int and visible >= 0:
                        buildings.setdefault(building, Counter())[str(visible)] += 1
            else:
                values = _sign_axes({**row, 'image_size': image_size})
                if values['view'] and values['truncation'] and values['view'] != ['not_applicable']:
                    joint['{}/{}'.format(values['view'][0], values['truncation'][0])] += 1
            for name in ('facade', 'position', 'distractor') if task == 'fire' else ('view', 'truncation'):
                items = values.get(name)
                if items is None:
                    unknown[name] += 1
                elif items == ['not_applicable']:
                    inapplicable[name] += 1
                else:
                    axis_counts.setdefault(name, Counter()).update(items)
        after = manifest.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('读取期间清单发生变化，本次实际数量未知')
        result.update(recorded_images=len(records), images=images, instances=instances,
                      counts_by_instances=dict(counts), branches=dict(branches),
                      class_instances=dict(class_counts),
                      axes={name: dict(values) for name, values in axis_counts.items()},
                      unknown=dict(unknown), not_applicable=dict(inapplicable),
                      building_counts={name: dict(values) for name, values in buildings.items()},
                      view_truncation=dict(joint), splits=splits)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        result['error'] = str(exc)
    return result


def summarize(root=ROOT, targets_path=TARGETS, axes_path=AXES, observations_path=None,
              *, include_actual=False):
    targets = load_targets(targets_path)
    axes = load_axes(axes_path)
    observations = load_observations(observations_path or root / 'config/combination_axis_observations.yaml')
    if set(targets['tasks']) != set(axes['tasks']):
        raise ValueError('task_targets.yaml 与 combination_axes.yaml 的任务集合不一致')
    tasks = {}
    for task in TASK_ORDER:
        if task not in targets['tasks']:
            continue
        target_spec = targets['tasks'][task]
        axis_spec = axes['tasks'][task]
        observation = observations['tasks'].get(task, {})
        for name, dimension in axis_spec.get('structure', {}).get('dimensions', {}).items():
            source = dimension['value_source']
            if source == 'unknown':
                continue
            prefix, separator, field = source.partition('.')
            origin = {'axes': axis_spec['axes'], 'targets': target_spec}.get(prefix)
            if not separator or origin is None or field not in origin:
                raise ValueError('{}.{} 的值域引用无效: {}'.format(task, name, source))
        target_images = _task_target_images(root, task)
        plans = [_summarize_plan(root, task, relative, target_spec, axis_spec['axes'])
                 for relative in target_spec['plans']]
        train = [plan for plan in plans if plan['split'] == 'train' and plan['usage'] == 'dataset']
        planned = sum(plan['planned_total'] or 0 for plan in train)
        capacity = sum(plan['capacity'] or 0 for plan in train)
        direct = len({capture for plan in train for capture in plan.get('direct_capture_ids', [])})
        tasks[task] = {
            'target_images': target_images,
            'target_spec': target_spec,
            'axes_spec': axis_spec['axes'],
            'structure': axis_spec.get('structure', {}),
            'crosses': [{**cross, 'evidence': observation.get('crosses', {}).get(cross['id'], '未知')}
                        for cross in axis_spec.get('crosses', [])],
            'source_exposure': observation.get('source_exposure'),
            'plans': plans,
            'targets': _merge_counts(plans, 'targets', _target_keys(target_spec)),
            'target_unknown': _merge_missing(plans, 'target_unknown', _target_keys(target_spec)),
            'coverage': _merge_counts(plans, 'coverage', _coverage_keys(task, target_spec)),
            'coverage_unknown': _merge_missing(plans, 'coverage_unknown', _coverage_keys(task, target_spec)),
            'axes': _merge_counts(plans, 'axes', axis_spec['axes']),
            'unknown': _merge_missing(plans, 'unknown', axis_spec['axes']),
            'not_applicable': _merge_missing(plans, 'not_applicable', axis_spec['axes']),
            'train_planned': planned,
            'train_capacity': capacity,
            'train_planned_direct': direct,
            'budget_gap': max(0, target_images - planned - direct) if target_images is not None else None,
            'capacity_gap': max(0, target_images - capacity - direct) if target_images is not None else None,
        }
        if include_actual and task in ('fire', 'sign'):
            tasks[task]['actual'] = _summarize_actual_batch(root, task, target_spec, targets_path)
    return {'version': 1, 'structured_semantics': axes.get('structured_semantics', {}), 'observation_provenance': {
        'status': observations.get('status', 'unknown'),
        'source': observations.get('source'), 'observed_at': observations.get('observed_at'),
    }, 'tasks': tasks}


def _merge_counts(plans, field, keys):
    totals = {key: {} for key in keys}
    for plan in plans:
        for key, counts in plan[field].items():
            for value, count in counts.items():
                totals[key][value] = totals[key].get(value, 0) + count
    return totals


def _merge_missing(plans, field, keys):
    return {key: sum(plan[field].get(key, 0) for plan in plans) for key in keys}


def _sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _format_counts(counts):
    return '、'.join('{}={}'.format(value, counts[value]) for value in sorted(counts, key=_sort_key))


def _render_actual_batches(summary):
    rows = [(task, item['actual']) for task, item in summary['tasks'].items() if 'actual' in item]
    if not rows:
        return []
    lines = ['', '**当前实际小样（清单与配对标签，只读）**', '',
             '| 任务 | 清单 | 清单记录 | 图签齐备 | 实例框 | 缺失文件 | 读取错误 |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for task, actual in rows:
        error = (actual['error'] or '-').replace('|', '\\|').replace('\n', ' ')
        values = [actual[key] if actual[key] is not None else '未知'
                  for key in ('recorded_images', 'images', 'instances')]
        lines.append('| {} | `{}` | {} | {} | {} | {} | {} |'.format(
            task, actual['path'], *values, len(actual['missing_files']), error))
    lines.extend(['', '下列数量只统计本次读到的完整图签。计划目标和实际实例分别计数；缺失或变化中的清单保持未知。', ''])
    for task, actual in rows:
        if actual['error']:
            continue
        axes = ['{} {}'.format(name, _format_counts(values)) for name, values in actual['axes'].items()]
        lines.append('- {}：类别框 {}；每图实例数 {}；实际轴 {}；未记录轴 {}；不适用 {}。'.format(
            task, _format_counts(actual['class_instances']), _format_counts(actual['counts_by_instances']), '；'.join(axes) or '无',
            _format_counts(actual['unknown']) or '无', _format_counts(actual['not_applicable']) or '无'))
        if actual['view_truncation']:
            lines.append('  视面与截断联合数量 {}。'.format(_format_counts(actual['view_truncation'])))
        for split, counts in actual['splits'].items():
            lines.append('  {} 图签 {} 张、{} 框、空标签 {} 张。'.format(
                split, counts['images'], counts['instances'], counts['negative_images']))
        for building, counts in actual['building_counts'].items():
            lines.append('  楼面 {} 的可见实例数 {}。'.format(building, _format_counts(counts)))
    return lines


def _plan_rows(summary):
    return [(task, plan) for task, item in summary['tasks'].items() for plan in item['plans']]


def _render_plan_budgets(rows):
    lines = [
        '| 任务 | 计划 | split | 用途 | 本批请求 | 已编预算 | 本批缺额 | 已分配 | strata | 状态容量 |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    usages = {'dataset': '数据集', 'source_render': '源渲染', 'sample': '小样', 'fine_tune': '微调'}
    for task, plan in rows:
        capacity = plan['capacity']
        lines.append('| {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            task, plan['path'], plan['split'] or '-', usages[plan['usage']], plan.get('requested_budget', '-'),
            plan['budget'] if plan['budget'] is not None else '-', plan.get('remaining_shortfall', '-'),
            plan['planned_total'] if plan['planned_total'] is not None else 'ERROR',
            plan['strata_count'], capacity if capacity is not None else '-'))
    return lines


def _render_train_gaps(summary):
    lines = [
        '**参考计划与全量目标的配额差**',
        '',
        '| 任务 | 最终 train | 状态预算 | 状态容量上限 | 单列实拍 | 按预算差额 | 按容量差额 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for task, item in summary['tasks'].items():
        if item['target_images'] is not None:
            lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                task, item['target_images'], item['train_planned'], item['train_capacity'],
                item['train_planned_direct'], item['budget_gap'], item['capacity_gap']))
    lines.extend([
        '',
        '全量差额使用数据集用途的 train 计划，源渲染、小样和微调单列。状态容量为适用 strata 的 `max_count` 之和，单列实拍来自计划的 `direct_capture_ids`。实际进度查各 split 的图片、标签和清单。',
    ])
    return lines


def _render_source_exposure(summary):
    exposure = [(task, item['source_exposure']) for task, item in summary['tasks'].items()
                if item['source_exposure']]
    if not exposure:
        return []
    lines = ['', '以下是历史来源接触及当时用途的迁移快照，时间及逐条依据未知；当前情况以回执和实际清单为准。', '',
             '| 任务 | 历史接触叙述 | 当时用途叙述 |', '| --- | --- | --- |']
    for task, entry in exposure:
        lines.append('| {} | {} | {} |'.format(task, entry['history'], entry['scope']))
    return lines


def render_markdown(summary):
    lines = [
        '> 修改目标、组合轴或所列计划后，由文档负责人运行 `python lab/scripts/contracts/state_coverage.py --write`。下方计数来自所列参考计划的分配，包含小批与可能重叠的用途。各 split 实际批次以任务当前计划和导出清单为准。',
        '',
        '入库容量另用 `python lab/scripts/contracts/state_coverage.py --source-status` 查看，按任务、角色和 split 核对可用现货。可合成张数取决于这些来源能编成多少可行状态。',
        '',
        '火焰与禁行牌当前清单及标签的实际对账使用 `python -B lab/scripts/contracts/state_coverage.py --actual --json`，同时返回计划分配和实际图签数量。该查询不生成数据、不更新清单。',
        '',
        '实际交叉用同一实例或可比场景的标签、观察和来源核对，记录各联合格的图片、实例与父组数。照度、饱和度和视角需要相应观测，增益、曝光、平面旋转是生成控制。直接实拍是来源方式。',
        '',
        '组合轴配置的结构化维度与保持／变化关系是对原设计的部分编码，值域未知仍记为未知；它们尚未自动接入计划生成，也不要求维度全排列。',
        '',
        '下表观察列来自历史迁移快照，时间及逐条依据未知，不代表实时产出；未记录的交叉显示“未知”。',
        '',
        '| 任务 | 交叉 ID | 防止误学 | 配对 | 历史观察与缺口 | 实际产物核对 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for task, item in summary['tasks'].items():
        for cross in item['crosses']:
            cells = [task] + [cross[key] for key in ('id', 'purpose', 'pairs', 'evidence', 'check')]
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend([
        '',
        '本批缺额按该计划明确登记的请求计算。全量差额按数据集用途的 train 计划另计。',
        '',
        *_render_plan_budgets(_plan_rows(summary)),
    ])
    lines.extend(_render_source_exposure(summary))
    lines.extend(['', '**训练目标覆盖（所列分支计划张数，可重叠）**', ''])
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
    lines.extend(['', '**画面覆盖（所列分支计划张数）**', ''])
    for task, item in summary['tasks'].items():
        pieces = ['{} {}'.format(key, _format_counts(counts))
                  for key, counts in item['coverage'].items() if counts]
        unknown = _format_counts({key: value for key, value in item['coverage_unknown'].items() if value})
        negatives = '、'.join(item['target_spec'].get('negatives', [])) or '无声明'
        lines.append('- {}：{}；待补证 {}；设计负例类型 {}。'.format(
            task, '；'.join(pieces) or '无可提取场景值', unknown or '无', negatives))
    lines.extend(['', '**组合轴覆盖（所列分支计划张数，可重叠）**', '',
                  '`unknown` 表示适用且缺少映射或证据；`not_applicable` 表示已知该条件不适用。', ''])
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
        inapplicable = _format_counts({key: value for key, value in item['not_applicable'].items() if value})
        if not pieces:
            pieces.append('无可提取轴值')
        lines.append('- {}：{}；待补证 {}；不适用 {}。'.format(
            task, '；'.join(pieces), '、'.join(unmapped) or '无', inapplicable or '无'))
    lines.extend(['', *_render_train_gaps(summary)])
    lines.extend(_render_actual_batches(summary))
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


def check_staged_inputs(root, targets, axes, summary, observations=None):
    """Never stage a report derived from an uncommitted input."""
    paths = [targets, axes]
    observations = observations or root / 'config/combination_axis_observations.yaml'
    if observations.is_file():
        paths.append(observations)
    else:
        relative = observations.resolve().relative_to(root.resolve()).as_posix()
        indexed = subprocess.run(['git', '-C', str(root), 'show', ':' + relative],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if indexed.returncode == 0:
            raise ValueError('状态覆盖观察记录删除尚未暂存: ' + relative)
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


def _requested_batches(summary):
    return [(task, plan) for task, plan in _plan_rows(summary)
            if 'requested_budget' in plan or plan['error']]


def _plan_progress(summary):
    fields = ('target_images', 'train_planned', 'train_capacity', 'train_planned_direct',
              'budget_gap', 'capacity_gap')
    batch_fields = ('path', 'split', 'usage', 'requested_budget', 'budget', 'remaining_shortfall',
                    'planned_total', 'capacity', 'error')
    return {
        'observation_provenance': summary['observation_provenance'],
        'batches': [{'task': task, **{key: plan.get(key) for key in batch_fields}}
                    for task, plan in _requested_batches(summary)],
        'source_exposure': {task: item['source_exposure'] for task, item in summary['tasks'].items()
                            if item['source_exposure']},
        'train': {task: {key: item[key] for key in fields} for task, item in summary['tasks'].items()},
    }


def render_source_status(summary, plan_summary=None):
    lines = [
        '逐帧资格库存，已登记 {} 帧版本、{} 个路径。'.format(
            summary['registered_frames'], summary['registered_paths']),
        '',
        '每格依次列合格帧版本数、去重捕获数、父组数。',
        '',
        '| 任务 | 角色 | 全部合格 | 未分配 | train | val | test |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    counts = '{frames} / {captures} / {parents}'
    buckets = {(row['task'], row['role'], row['split']): counts.format(**row)
               for row in summary['eligible_by_split']}
    for row in summary['eligible_stock']:
        task, role = row['task'], row['role']
        cells = [buckets.get((task, role, split), '0 / 0 / 0')
                 for split in ('unassigned', 'train', 'val', 'test')]
        lines.append('| ' + ' | '.join([task, role, counts.format(**row), *cells]) + ' |')
    lines.extend([
        '',
        '`direct_target` 对应标签完整的直接实拍正例；'
        '`empty_negative` 对应全帧空标签监督，放置空间未知或不可用时仍可计入；'
        '`composite_background` 另需适用的可放或可贴空间。',
        '同一帧可有多个任务角色。直接实拍配额按捕获身份去重；逐角色库存分别列出。'
        '查看待补证、不可用状态、父组及媒体来源类型明细时加 `--json`。',
    ])
    if plan_summary is not None:
        batches = _requested_batches(plan_summary)
        if batches:
            lines.extend(['', '**登记的本批请求与已编计划**', '', *_render_plan_budgets(batches)])
        lines.extend(_render_source_exposure(plan_summary))
        lines.extend(['', *_render_train_gaps(plan_summary)])
        lines.extend(_render_actual_batches(plan_summary))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='工作空间根目录，默认本仓库')
    parser.add_argument('--targets', help='训练目标配置，默认 <root>/config/task_targets.yaml')
    parser.add_argument('--axes', help='组合轴配置，默认 <root>/config/combination_axes.yaml')
    parser.add_argument('--observations', help='历史观察快照，默认 <root>/config/combination_axis_observations.yaml')
    parser.add_argument('--doc', help='目标文档，默认 <root>/docs/DATASET_DESIGN.md')
    parser.add_argument('--json', action='store_true', help='输出 JSON 摘要')
    parser.add_argument('--source-status', action='store_true', help='只读查询逐帧库存和已登记计划的配额差')
    parser.add_argument('--actual', action='store_true', help='只读核对火焰和禁行牌当前数据清单及配对标签')
    parser.add_argument('--write', action='store_true', help='写入文档生成区块')
    parser.add_argument('--check', action='store_true', help='检查文档生成区块是否同步')
    parser.add_argument('--staged', action='store_true', help='要求输入文件已暂存')
    args = parser.parse_args()
    if args.source_status and (args.write or args.check or args.staged):
        parser.error('--source-status 与文档生成及检查选项互斥')
    if args.actual and (args.write or args.check or args.staged):
        parser.error('--actual 是只读清单查询，与文档生成及暂存检查选项互斥')
    root = Path(args.root).resolve() if args.root else ROOT
    targets = Path(args.targets) if args.targets else root / 'config/task_targets.yaml'
    axes = Path(args.axes) if args.axes else root / 'config/combination_axes.yaml'
    observations = Path(args.observations) if args.observations else root / 'config/combination_axis_observations.yaml'
    doc = Path(args.doc) if args.doc else root / 'docs/DATASET_DESIGN.md'
    try:
        if args.source_status:
            from datasets.scripts.common.frame_environment import DEFAULT_DATABASE, quota_summary

            database = root / DEFAULT_DATABASE.relative_to(ROOT)
            if not database.is_file():
                raise FileNotFoundError('逐帧环境库尚未建立: {}'.format(database))
            inventory = quota_summary(database=database)
            plans = summarize(root, targets, axes, observations, include_actual=args.actual) if targets.is_file() and axes.is_file() else None
            if args.json and plans is not None:
                inventory['plan_progress'] = _plan_progress(plans)
                if args.actual:
                    inventory['actual_batches'] = {task: item['actual'] for task, item in plans['tasks'].items()
                                                 if 'actual' in item}
            print(json.dumps(inventory, ensure_ascii=False, indent=2) if args.json
                  else render_source_status(inventory, plans))
            return 0
        summary = summarize(root, targets, axes, observations, include_actual=args.actual)
        if args.staged:
            check_staged_inputs(root, targets, axes, summary, observations)
            if not args.doc and args.check:
                check_staged_task_designs(root, load_targets(targets), load_axes(axes))
        has_error = any(plan['error'] for item in summary['tasks'].values() for plan in item['plans'])
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1 if has_error else 0
        if args.write:
            body = write_doc(summary, doc)
            if not args.doc:
                update_task_designs(root, load_targets(targets), load_axes(axes), write=True)
            print('已更新: {}'.format(doc.relative_to(root).as_posix()))
            if not args.doc:
                print('已更新: 八任务说明生成区块')
            print(body)
            return 1 if has_error else 0
        body = render_markdown(summary)
        if args.check:
            text = doc.read_text(encoding='utf-8')
            if text.count(START) != 1 or text.count(END) != 1:
                raise ValueError('文档需要一对有序的状态覆盖生成标记: {}'.format(doc))
            current = text[text.index(START) + len(START):text.index(END)].strip('\n')
            changed = current != body
            task_changes = update_task_designs(root, load_targets(targets), load_axes(axes), write=False) if not args.doc else []
            print(('待更新: ' if changed else '已同步: ') + doc.relative_to(root).as_posix())
            for path in task_changes:
                print('待更新: ' + path.relative_to(root).as_posix())
            return 1 if changed or task_changes or has_error else 0
        print(body)
        return 1 if has_error else 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, yaml.YAMLError) as exc:
        print('状态覆盖检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
