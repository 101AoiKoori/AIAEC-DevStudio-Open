#!/usr/bin/env python3
"""Validate the ID/count policy and render docs/ID_AND_COUNT_POLICY.md.

config/id_policy.yaml owns identifier namespaces, the three task-scope
vocabularies, counting units and independent annotation-status layers. The
checker classifies every tagged requirement ID as the canonical TASK-RNN family
or a frozen alias, compares the declared task scopes with their actual configs,
and keeps the generated policy document in sync. It does not rewrite
requirements, plans or data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sync_docs

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / 'config/id_policy.yaml'
DOC = ROOT / 'docs/ID_AND_COUNT_POLICY.md'
REQUIRED_UNITS = {
    'image', 'instance', 'parent_group', 'state', 'plan_budget',
    'plan_capacity', 'quota',
}
REQUIRED_LAYERS = (
    'source_available', 'annotated', 'candidate', 'auto_validated',
    'human_confirmed', 'training_admitted',
)
REQUIRED_CONFIRMATION_DIMENSIONS = ('semantic', 'box', 'mask')


def _string(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('{} 必须是非空字符串'.format(where))
    return value


def _strings(value, where, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty) or not all(
            isinstance(item, str) and item.strip() for item in value):
        raise ValueError('{} 必须是{}字符串列表'.format(where, '可空' if allow_empty else '非空'))
    return value


def _relative(value, where):
    _string(value, where)
    path = Path(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('{} 必须是工作空间相对路径'.format(where))
    return path


def _display(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def validate_policy(data):
    """Validate the policy structure and return it."""
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('编号与计数口径版本必须为 1')
    if data.get('status') not in {'active', 'draft'}:
        raise ValueError('status 必须为 active 或 draft')
    _string(data.get('note'), 'note')

    requirements = data.get('requirement_ids')
    if not isinstance(requirements, dict):
        raise ValueError('缺少 requirement_ids')
    canonical = requirements.get('canonical')
    if not isinstance(canonical, dict):
        raise ValueError('缺少 requirement_ids.canonical')
    if canonical.get('family') != 'TASK-RNN':
        raise ValueError('requirement_ids.canonical.family 必须为 TASK-RNN')
    for key in ('owner', 'validator', 'description'):
        _string(canonical.get(key), 'requirement_ids.canonical.' + key)
    pattern = _string(canonical.get('pattern'), 'requirement_ids.canonical.pattern')
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError('requirement_ids.canonical.pattern 不是有效正则: {}'.format(exc))
    if not compiled.fullmatch('TASK-R01'):
        raise ValueError('规范 ID 正则必须匹配 TASK-R01')
    if compiled.fullmatch('TASK-1'):
        raise ValueError('规范 ID 正则不得匹配裸数字 ID TASK-1')

    aliases = requirements.get('frozen_aliases')
    if not isinstance(aliases, dict):
        raise ValueError('缺少 requirement_ids.frozen_aliases')
    semantic = _strings(aliases.get('semantic'), 'frozen_aliases.semantic', allow_empty=True)
    numeric = _strings(aliases.get('legacy_numeric'), 'frozen_aliases.legacy_numeric', allow_empty=True)
    seen = set()
    for value in semantic + numeric:
        if value in seen:
            raise ValueError('冻结别名重复: ' + value)
        seen.add(value)
        if compiled.fullmatch(value):
            raise ValueError('冻结别名不能匹配规范族: ' + value)
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*-[A-Z0-9][A-Z0-9_-]*', value):
            raise ValueError('冻结别名格式无效: ' + value)
    for value in semantic:
        if re.fullmatch(r'[A-Z][A-Z0-9_]*-[0-9]+', value):
            raise ValueError('语义别名不能是裸数字 ID: ' + value)
    for value in numeric:
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*-[0-9]+', value):
            raise ValueError('裸数字别名格式无效: ' + value)

    pairing = data.get('pairing_ids')
    if not isinstance(pairing, dict):
        raise ValueError('缺少 pairing_ids')
    if _string(pairing.get('canonical_requirements'), 'pairing_ids.canonical_requirements') != 'PAIR-RNN':
        raise ValueError('pairing_ids.canonical_requirements 必须为 PAIR-RNN')
    if _string(pairing.get('canonical_rules'), 'pairing_ids.canonical_rules') != 'PAIR-VNN':
        raise ValueError('pairing_ids.canonical_rules 必须为 PAIR-VNN')
    frozen = pairing.get('frozen')
    if not isinstance(frozen, dict):
        raise ValueError('缺少 pairing_ids.frozen')
    for key in ('requirements', 'rules', 'diagnostics'):
        _string(frozen.get(key), 'pairing_ids.frozen.' + key)
    reserved = _strings(pairing.get('reserved_diagnostics'), 'pairing_ids.reserved_diagnostics',
                        allow_empty=True)
    for code in reserved:
        if not re.fullmatch(r'BSL[0-9]{3}', code):
            raise ValueError('保留诊断码格式无效: ' + code)
    _strings(pairing.get('notes'), 'pairing_ids.notes')

    scopes = data.get('task_scopes')
    if not isinstance(scopes, dict):
        raise ValueError('缺少 task_scopes')
    sources = scopes.get('sources')
    if not isinstance(sources, dict):
        raise ValueError('缺少 task_scopes.sources')
    for key in ('perception', 'dataset', 'model_packages'):
        _relative(sources.get(key), 'task_scopes.sources.' + key)
    _strings(sources.get('model_packages_extra'), 'task_scopes.sources.model_packages_extra',
             allow_empty=True)
    mapping = scopes.get('mapping')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('task_scopes.mapping 必须是非空映射')
    mapped = []
    for task_id, values in mapping.items():
        if not isinstance(task_id, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', task_id):
            raise ValueError('task_scopes.mapping 的任务 ID 无效: {}'.format(task_id))
        _strings(values, 'task_scopes.mapping.' + task_id, allow_empty=True)
        mapped.extend(values)
    if len(mapped) != len(set(mapped)):
        raise ValueError('task_scopes.mapping 中数据任务重复')
    _strings(scopes.get('notes'), 'task_scopes.notes')

    units = data.get('count_units')
    if not isinstance(units, dict):
        raise ValueError('缺少 count_units')
    missing = REQUIRED_UNITS - set(units)
    if missing:
        raise ValueError('count_units 缺少: {}'.format(', '.join(sorted(missing))))
    for name, spec in units.items():
        if not isinstance(spec, dict):
            raise ValueError('count_units.{} 必须是映射'.format(name))
        _string(spec.get('definition'), 'count_units.{}.definition'.format(name))
        _string(spec.get('authority'), 'count_units.{}.authority'.format(name))

    layers = data.get('annotation_layers')
    if not isinstance(layers, dict):
        raise ValueError('缺少 annotation_layers')
    independent = layers.get('independent_layers')
    if not isinstance(independent, list) or not independent:
        raise ValueError('annotation_layers.independent_layers 必须是非空列表')
    layer_ids = []
    for index, item in enumerate(independent):
        where = 'annotation_layers.independent_layers[{}]'.format(index)
        if not isinstance(item, dict):
            raise ValueError(where + ' 必须是映射')
        layer_id = _string(item.get('id'), where + '.id')
        if layer_id in layer_ids:
            raise ValueError('annotation_layers.independent_layers 的 id 重复: ' + layer_id)
        layer_ids.append(layer_id)
        _string(item.get('definition'), where + '.definition')
        _string(item.get('authority'), where + '.authority')
    if tuple(layer_ids) != REQUIRED_LAYERS:
        raise ValueError('annotation_layers.independent_layers 必须依次定义: {}'.format(
            ', '.join(REQUIRED_LAYERS)))
    _string(layers.get('count_rule'), 'annotation_layers.count_rule')
    _string(layers.get('unit_rule'), 'annotation_layers.unit_rule')

    pending = layers.get('pending')
    if not isinstance(pending, dict):
        raise ValueError('缺少 annotation_layers.pending')
    _string(pending.get('definition'), 'annotation_layers.pending.definition')
    _string(pending.get('rule'), 'annotation_layers.pending.rule')

    confirmation = layers.get('confirmation_dimensions')
    if not isinstance(confirmation, dict):
        raise ValueError('缺少 annotation_layers.confirmation_dimensions')
    for dimension in REQUIRED_CONFIRMATION_DIMENSIONS:
        spec = confirmation.get(dimension)
        if not isinstance(spec, dict):
            raise ValueError('缺少 annotation_layers.confirmation_dimensions.' + dimension)
        _string(spec.get('definition'),
                'annotation_layers.confirmation_dimensions.{}.definition'.format(dimension))
        _string(spec.get('rule'),
                'annotation_layers.confirmation_dimensions.{}.rule'.format(dimension))
    _string(confirmation.get('separation_rule'),
            'annotation_layers.confirmation_dimensions.separation_rule')

    negative = layers.get('negative_states')
    if not isinstance(negative, dict):
        raise ValueError('缺少 annotation_layers.negative_states')
    values = _strings(negative.get('values'), 'annotation_layers.negative_states.values')
    if set(values) != {'rejected', 'excluded'} or len(values) != 2:
        raise ValueError('annotation_layers.negative_states.values 必须只含 rejected 和 excluded')
    _string(negative.get('evidence_required'),
            'annotation_layers.negative_states.evidence_required')
    _string(negative.get('rule'), 'annotation_layers.negative_states.rule')

    summary = layers.get('summary_contract')
    if not isinstance(summary, dict):
        raise ValueError('缺少 annotation_layers.summary_contract')
    summary_layers = _strings(summary.get('required_layers'),
                              'annotation_layers.summary_contract.required_layers')
    if tuple(summary_layers) != REQUIRED_LAYERS:
        raise ValueError('annotation_layers.summary_contract.required_layers 必须依次为六层')
    if _string(summary.get('pending_counter'),
               'annotation_layers.summary_contract.pending_counter') != 'pending':
        raise ValueError('annotation_layers.summary_contract.pending_counter 必须为 pending')
    summary_dimensions = _strings(
        summary.get('required_confirmation_dimensions'),
        'annotation_layers.summary_contract.required_confirmation_dimensions')
    if tuple(summary_dimensions) != REQUIRED_CONFIRMATION_DIMENSIONS:
        raise ValueError(
            'annotation_layers.summary_contract.required_confirmation_dimensions 必须依次为 '
            'semantic、box、mask')
    _string(summary.get('rule'), 'annotation_layers.summary_contract.rule')

    authorities = data.get('authorities')
    if not isinstance(authorities, list) or not authorities:
        raise ValueError('authorities 必须是非空列表')
    for index, item in enumerate(authorities):
        if not isinstance(item, dict):
            raise ValueError('authorities[{}] 必须是映射'.format(index))
        _string(item.get('fact'), 'authorities[{}].fact'.format(index))
        _string(item.get('source'), 'authorities[{}].source'.format(index))

    _strings(data.get('rules'), 'rules')
    numbering = data.get('section_numbering')
    if not isinstance(numbering, dict):
        raise ValueError('缺少 section_numbering')
    _string(numbering.get('rule'), 'section_numbering.rule')
    return data


def load_policy(path=POLICY):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    return validate_policy(data)


def load_actual(root, policy):
    """Read the three task-scope sources and return their actual members."""
    sources = policy['task_scopes']['sources']
    perception_path = root / _relative(sources['perception'], 'task_scopes.sources.perception')
    dataset_path = root / _relative(sources['dataset'], 'task_scopes.sources.dataset')
    model_path = root / _relative(sources['model_packages'], 'task_scopes.sources.model_packages')

    perception_data = json.loads(perception_path.read_text(encoding='utf-8'))
    perception = perception_data.get('tasks')
    if not isinstance(perception, list) or not perception:
        raise ValueError('识别任务配置必须包含非空 tasks 列表')
    perception_ids = []
    for task in perception:
        task_id = task.get('id') if isinstance(task, dict) else None
        if not isinstance(task_id, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', task_id):
            raise ValueError('识别任务 ID 无效: {!r}'.format(task_id))
        perception_ids.append(task_id)

    dataset_data = yaml.safe_load(dataset_path.read_text(encoding='utf-8'))
    dataset_tasks = dataset_data.get('datasets', {}).get('tasks')
    if not isinstance(dataset_tasks, dict) or not dataset_tasks:
        raise ValueError('目录契约必须包含非空 datasets.tasks')
    dataset_ids = list(dataset_tasks.keys())

    model_data = yaml.safe_load(model_path.read_text(encoding='utf-8'))
    model_tasks = model_data.get('tasks')
    if not isinstance(model_tasks, dict) or not model_tasks:
        raise ValueError('训练目标配置必须包含非空 tasks')
    model_ids = list(model_tasks.keys())
    for name in sources.get('model_packages_extra', []):
        if name not in model_ids:
            model_ids.append(name)

    return {
        'perception': perception,
        'perception_ids': perception_ids,
        'dataset_ids': dataset_ids,
        'model_package_ids': model_ids,
    }


def validate_scopes(policy, actual):
    errors = []
    mapping = policy['task_scopes']['mapping']
    perception = set(actual['perception_ids'])
    dataset = set(actual['dataset_ids'])
    model = set(actual['model_package_ids'])
    extra = set(policy['task_scopes']['sources'].get('model_packages_extra', []))

    missing = sorted(perception - set(mapping))
    if missing:
        errors.append('task_scopes.mapping 缺少识别任务: ' + ', '.join(missing))
    unknown = sorted(set(mapping) - perception)
    if unknown:
        errors.append('task_scopes.mapping 含未知识别任务: ' + ', '.join(unknown))

    mapped = []
    for task_id, values in mapping.items():
        for value in values:
            if value not in dataset:
                errors.append('{}: 映射到未知数据任务 {}'.format(task_id, value))
            mapped.append(value)
    if len(mapped) != len(set(mapped)):
        errors.append('task_scopes.mapping 中数据任务重复')
    missing_dataset = sorted(dataset - set(mapped))
    if missing_dataset:
        errors.append('task_scopes.mapping 未覆盖数据任务: ' + ', '.join(missing_dataset))
    if extra & dataset:
        errors.append('model_packages_extra 不能与数据任务重叠: ' + ', '.join(sorted(extra & dataset)))
    expected_model = dataset | extra
    if model != expected_model:
        missing_model = sorted(expected_model - model)
        extra_model = sorted(model - expected_model)
        if missing_model:
            errors.append('模型包缺少: ' + ', '.join(missing_model))
        if extra_model:
            errors.append('模型包含未登记任务: ' + ', '.join(extra_model))
    return errors


def validate_requirements(policy, actual):
    """Classify every tagged requirement ID and report unknown or stale entries."""
    canonical = re.compile(policy['requirement_ids']['canonical']['pattern'])
    aliases = policy['requirement_ids']['frozen_aliases']
    semantic = set(aliases['semantic'])
    numeric = set(aliases['legacy_numeric'])
    all_aliases = semantic | numeric
    prefixes = {task_id.upper() + '-' for task_id in actual['perception_ids']}
    errors = []
    for alias in sorted(all_aliases):
        if not any(alias.startswith(prefix) for prefix in prefixes):
            errors.append('冻结别名不属于已知任务前缀: ' + alias)

    counts = {'canonical': 0, 'semantic': 0, 'legacy_numeric': 0}
    rows = {}
    seen = set()
    used_aliases = set()
    for task in actual['perception']:
        task_id = task['id']
        prefix = task_id.upper() + '-'
        row = {'canonical': [], 'semantic': [], 'legacy_numeric': []}
        for state, item_id, line in sync_docs.iter_requirement_items(task):
            if state not in sync_docs.REQUIREMENT_STATES:
                errors.append('{}: 需求状态无效: {}'.format(task_id, line.strip()))
            if not item_id:
                errors.append('{}: 无法解析需求 ID: {}'.format(task_id, line.strip()))
                continue
            if item_id in seen:
                errors.append('需求事项 ID 重复: ' + item_id)
            seen.add(item_id)
            if not item_id.startswith(prefix):
                errors.append('{}: ID 不属于任务前缀: {}'.format(task_id, item_id))
            if canonical.fullmatch(item_id):
                row['canonical'].append(item_id)
                counts['canonical'] += 1
            elif item_id in semantic:
                row['semantic'].append(item_id)
                counts['semantic'] += 1
                used_aliases.add(item_id)
            elif item_id in numeric:
                row['legacy_numeric'].append(item_id)
                counts['legacy_numeric'] += 1
                used_aliases.add(item_id)
            else:
                errors.append('{}: 未登记的需求 ID: {}'.format(task_id, item_id))
        rows[task_id] = row
    for alias in sorted(all_aliases - used_aliases):
        errors.append('冻结别名未在需求配置中出现: ' + alias)
    return counts, rows, errors


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _ids(values):
    return '、'.join('`{}`'.format(value) for value in values) if values else '—'


def render_doc(policy, actual, counts, rows):
    total = sum(counts.values())
    canonical = policy['requirement_ids']['canonical']
    lines = [
        '# 编号与计数口径',
        '',
        '> 状态：`{}`。本文由 [`config/id_policy.yaml`](../config/id_policy.yaml) 生成，修改请改 YAML 后运行 `python lab/scripts/contracts/id_policy.py --write`；检查用 `python lab/scripts/contracts/id_policy.py --check`。'.format(
            policy['status']),
        '>',
        '> 本文定义编号命名空间、三套任务口径、计数单位、标注状态层和权威源，不修改需求、计划、数据或产量。',
        '',
        '## 需求 ID',
        '',
        '新需求项只使用规范族 `{}`，模式 `{}`。语义 ID 和裸数字 ID 是冻结别名，保留原含义，不再新增；修改需求文字时保留原 ID。'.format(
            canonical['family'], canonical['pattern']),
        '',
        '当前需求配置共 **{}** 条：规范 `TASK-RNN` **{}** 条，冻结语义别名 **{}** 条，冻结裸数字别名 **{}** 条。'.format(
            total, counts['canonical'], counts['semantic'], counts['legacy_numeric']),
        '',
        '| 任务 | 规范 `TASK-RNN` | 冻结语义别名 | 冻结裸数字别名 |',
        '| --- | --- | --- | --- |',
    ]
    for task_id in actual['perception_ids']:
        row = rows[task_id]
        lines.append('| `{}` | {} | {} | {} |'.format(
            task_id, _ids(row['canonical']), _ids(row['semantic']), _ids(row['legacy_numeric'])))
    lines.extend([
        '',
        '## 配对编号',
        '',
        '| 命名空间 | 状态 | 用途 |',
        '| --- | --- | --- |',
        '| `PAIR-RNN` | 规范 | 新增配对设计需求 |',
        '| `PAIR-VNN` | 规范 | 新增配对校核规则；`source_pairing.py` 的新错误码使用此族 |',
        '| `{}` | 冻结 | 现有配对设计需求 |'.format(policy['pairing_ids']['frozen']['requirements']),
        '| `{}` | 冻结 | 现有配对校核规则；V11 尚无自动 checker，编译前专项核对 |'.format(
            policy['pairing_ids']['frozen']['rules']),
        '| `{}` | 冻结 | `source_ledger` 派生报告诊断；不是准入 gate |'.format(
            policy['pairing_ids']['frozen']['diagnostics']),
    ])
    if policy['pairing_ids']['reserved_diagnostics']:
        lines.append('| {} | 保留空号 | 当前实现未使用；不得复用为其他含义，待 ledger 维护者登记 |'.format(
            '、'.join('`{}`'.format(code) for code in policy['pairing_ids']['reserved_diagnostics'])))
    lines.extend(['', '配对口径说明：', ''])
    lines.extend('- {}'.format(note) for note in policy['pairing_ids']['notes'])
    lines.extend([
        '',
        '## 三套任务口径',
        '',
        '| 口径 | 成员 | 来源 |',
        '| --- | --- | --- |',
        '| 识别任务 | {} | `{}` |'.format(
            _ids(actual['perception_ids']), policy['task_scopes']['sources']['perception']),
        '| 数据任务 | {} | `{}` |'.format(
            _ids(actual['dataset_ids']), policy['task_scopes']['sources']['dataset']),
        '| 模型包 | {} | `{}` + `model_packages_extra` |'.format(
            _ids(actual['model_package_ids']), policy['task_scopes']['sources']['model_packages']),
        '',
        '识别任务到数据任务的映射：',
        '',
        '| 识别任务 | 数据任务 |',
        '| --- | --- |',
    ])
    for task_id in actual['perception_ids']:
        lines.append('| `{}` | {} |'.format(
            task_id, _ids(policy['task_scopes']['mapping'][task_id])))
    lines.extend(['', '任务口径说明：', ''])
    lines.extend('- {}'.format(note) for note in policy['task_scopes']['notes'])
    lines.extend([
        '',
        '## 计数单位',
        '',
        '| 单位 | 定义 | 权威源 |',
        '| --- | --- | --- |',
    ])
    for name in ('image', 'instance', 'parent_group', 'state', 'plan_budget', 'plan_capacity', 'quota'):
        spec = policy['count_units'][name]
        lines.append('| `{}` | {} | {} |'.format(
            name, _cell(spec['definition']), _cell(spec['authority'])))
    lines.extend([
        '',
        '## 标注状态层',
        '',
        policy['annotation_layers']['count_rule'],
        '',
        policy['annotation_layers']['unit_rule'],
        '',
        '| 层 | 定义 | 权威源 |',
        '| --- | --- | --- |',
    ])
    for item in policy['annotation_layers']['independent_layers']:
        lines.append('| `{}` | {} | {} |'.format(
            item['id'], _cell(item['definition']), _cell(item['authority'])))
    lines.extend([
        '',
        '全流程盘点中的 `pending` 计量尚未完成或尚未确认的记录，可与已有标签或候选计数重叠。',
        '',
        policy['annotation_layers']['pending']['definition'],
        '',
        policy['annotation_layers']['pending']['rule'],
        '',
        '### 人工确认维度',
        '',
        '| 维度 | 定义 | 规则 |',
        '| --- | --- | --- |',
    ])
    confirmation = policy['annotation_layers']['confirmation_dimensions']
    for name in REQUIRED_CONFIRMATION_DIMENSIONS:
        spec = confirmation[name]
        lines.append('| `{}` | {} | {} |'.format(
            name, _cell(spec['definition']), _cell(spec['rule'])))
    lines.extend([
        '',
        confirmation['separation_rule'],
        '',
        '### 负状态证据门槛',
        '',
        '允许的负状态：{}。{}'.format(
            _ids(policy['annotation_layers']['negative_states']['values']),
            policy['annotation_layers']['negative_states']['evidence_required']),
        '',
        policy['annotation_layers']['negative_states']['rule'],
        '',
        '### 摘要契约',
        '',
        '盘点全流程标注状态时，列出计数单位、六层计数（{}）、`{}` 数，以及适用的人工语义、框和掩膜确认计数。{}'.format(
            '、'.join('`{}`'.format(name) for name in REQUIRED_LAYERS),
            policy['annotation_layers']['summary_contract']['pending_counter'],
            policy['annotation_layers']['summary_contract']['rule']),
        '',
        '## 权威源',
        '',
        '| 事实 | 权威源 |',
        '| --- | --- |',
    ])
    for item in policy['authorities']:
        lines.append('| {} | {} |'.format(_cell(item['fact']), _cell(item['source'])))
    lines.extend([
        '',
        '## 规则',
        '',
    ])
    lines.extend('- {}'.format(rule) for rule in policy['rules'])
    lines.extend([
        '',
        '## 章节序号',
        '',
        policy['section_numbering']['rule'],
        '',
        '## 维护',
        '',
        '```bash',
        'python lab/scripts/contracts/id_policy.py --write',
        'python lab/scripts/contracts/id_policy.py --check',
        '```',
        '',
        '本文只校验编号和口径声明，不判断需求证据是否充分、计划是否能执行、数据是否已生产或模型效果。',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='工作空间根目录，默认本仓库')
    parser.add_argument('--policy', help='编号与计数口径配置，默认 <root>/config/id_policy.yaml')
    parser.add_argument('--doc', help='生成文档，默认 <root>/docs/ID_AND_COUNT_POLICY.md')
    parser.add_argument('--write', action='store_true', help='写入生成文档')
    parser.add_argument('--check', action='store_true', help='检查生成文档是否同步')
    parser.add_argument('--json', action='store_true', help='输出 JSON 摘要')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT
    policy_path = Path(args.policy).resolve() if args.policy else root / 'config/id_policy.yaml'
    doc = Path(args.doc).resolve() if args.doc else root / 'docs/ID_AND_COUNT_POLICY.md'
    try:
        data = load_policy(policy_path)
        actual = load_actual(root, data)
        errors = validate_scopes(data, actual)
        counts, rows, requirement_errors = validate_requirements(data, actual)
        errors.extend(requirement_errors)
        if errors:
            if args.json:
                print(json.dumps({'errors': errors}, ensure_ascii=False, indent=2))
            else:
                for error in errors:
                    print('ERROR: ' + error)
                print('编号与计数口径检查：{} 个错误。'.format(len(errors)))
            return 1
        body = render_doc(data, actual, counts, rows)
        summary = {
            'status': 'ok',
            'policy': _display(policy_path, root),
            'doc': _display(doc, root),
            'counts': counts,
            'requirements_total': sum(counts.values()),
            'perception_tasks': actual['perception_ids'],
            'dataset_tasks': actual['dataset_ids'],
            'model_packages': actual['model_package_ids'],
        }
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.write:
            doc.parent.mkdir(parents=True, exist_ok=True)
            with doc.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write(body)
            print('已更新: ' + _display(doc, root))
            return 0
        if args.check:
            changed = not doc.is_file() or doc.read_text(encoding='utf-8') != body
            print(('待更新: ' if changed else '已同步: ') + _display(doc, root))
            return 1 if changed else 0
        print(body)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print('编号与计数口径检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
