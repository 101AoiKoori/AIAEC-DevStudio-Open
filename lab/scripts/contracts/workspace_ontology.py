#!/usr/bin/env python3
"""Validate and render the workspace object ontology.

The ontology in config/workspace_ontology.yaml defines object classes, parents,
owners, authorities, versioning, mutability, lifecycle and relations. The
checker maps key paths to classes, verifies required artifacts and the plan
draft status, and generates docs/WORKSPACE_ONTOLOGY.md. It never scans data
contents or writes plan/data files.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY = ROOT / 'config/workspace_ontology.yaml'
DOC = ROOT / 'docs/WORKSPACE_ONTOLOGY.md'
AUTHORITIES = {'git', 'manifest', 'media_index', 'production_receipt', 'blender_registry',
               'run_record', 'model_record', 'archive_index', 'external_repo', 'none', 'mixed'}
MUTABILITIES = {'abstract', 'immutable', 'append_only', 'rebuildable', 'current_only',
                'versioned', 'generated', 'disposable', 'mixed'}
TRACKED = {True, False, 'mixed'}


def load_ontology(path=ONTOLOGY):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    return validate_ontology(data)


def _relative(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('{} 必须是非空路径'.format(where))
    text = value.replace('\\', '/')
    if text.startswith('/') or '..' in Path(text).parts:
        raise ValueError('{} 必须是工作区相对路径'.format(where))
    return text


def _relations(classes, data):
    relations = data.get('relations')
    if not isinstance(relations, dict) or not relations:
        raise ValueError('缺少 relations')
    for name, spec in relations.items():
        if not isinstance(spec, dict):
            raise ValueError('relations.{} 必须是映射'.format(name))
        for side in ('from', 'to'):
            values = spec.get(side)
            if not isinstance(values, list) or not values:
                raise ValueError('relations.{} 缺少 {}'.format(name, side))
            for value in values:
                if value not in classes:
                    raise ValueError('relations.{} 引用未知类: {}'.format(name, value))
        if not isinstance(spec.get('description'), str) or not spec['description'].strip():
            raise ValueError('relations.{} 缺少说明'.format(name))
    return relations


def _ancestors(classes, name):
    result = set()
    current = classes[name].get('parent')
    while current:
        if current in result:
            raise ValueError('类层次存在循环: ' + name)
        result.add(current)
        current = classes[current].get('parent')
    return result


def validate_ontology(data):
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('对象分类学版本必须为 1')
    classes = data.get('classes')
    if not isinstance(classes, dict) or not classes:
        raise ValueError('缺少 classes')
    for name, spec in classes.items():
        if not isinstance(spec, dict):
            raise ValueError('类 {} 必须是映射'.format(name))
        parent = spec.get('parent')
        if parent is not None and parent not in classes:
            raise ValueError('类 {} 的父类不存在: {}'.format(name, parent))
        if not isinstance(spec.get('abstract'), bool):
            raise ValueError('类 {} 缺少 abstract'.format(name))
        if not isinstance(spec.get('description'), str) or not spec['description'].strip():
            raise ValueError('类 {} 缺少说明'.format(name))
        prefixes = spec.get('owner_prefixes')
        if not isinstance(prefixes, list):
            raise ValueError('类 {} 的 owner_prefixes 必须是列表'.format(name))
        for index, prefix in enumerate(prefixes):
            _relative(prefix, 'classes.{}.owner_prefixes[{}]'.format(name, index))
        if spec.get('authority') not in AUTHORITIES:
            raise ValueError('类 {} 的 authority 无效'.format(name))
        if spec.get('mutability') not in MUTABILITIES:
            raise ValueError('类 {} 的 mutability 无效'.format(name))
        if spec.get('tracked') not in TRACKED:
            raise ValueError('类 {} 的 tracked 无效'.format(name))
        if not isinstance(spec.get('lifecycle'), list) or not all(
                isinstance(value, str) and value for value in spec['lifecycle']):
            raise ValueError('类 {} 的 lifecycle 必须是字符串列表'.format(name))
        _relations(classes, data)
        for relation in spec.get('relations', []):
            if relation not in data['relations']:
                raise ValueError('类 {} 引用未知关系: {}'.format(name, relation))
        _ancestors(classes, name)
    policies = data.get('versioning_policies')
    if not isinstance(policies, dict) or not policies:
        raise ValueError('缺少 versioning_policies')
    for name, spec in classes.items():
        if spec.get('versioning') not in policies:
            raise ValueError('类 {} 的 versioning 未在策略中定义'.format(name))
    if set(data.get('mutability_values', [])) != MUTABILITIES:
        raise ValueError('mutability_values 必须与检查器支持值一致')
    prefixes = {}
    for name, spec in classes.items():
        for prefix in spec.get('owner_prefixes', []):
            prefixes.setdefault(prefix, set()).add(name)
    for prefix, names in prefixes.items():
        if len(names) > 1:
            names = list(names)
            related = all(names[0] in _ancestors(classes, other)
                          or other in _ancestors(classes, names[0])
                          for other in names[1:])
            if not related:
                raise ValueError('路径前缀被多个无关类占用: {} -> {}'.format(prefix, names))
    artifacts = data.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('缺少 artifacts')
    exact = set()
    used = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ValueError('artifacts[{}] 必须是映射'.format(index))
        class_name = item.get('class')
        if class_name not in classes or classes[class_name].get('abstract'):
            raise ValueError('artifacts[{}] 的 class 无效: {}'.format(index, class_name))
        if ('path' in item) == ('pattern' in item):
            raise ValueError('artifacts[{}] 必须且只能有 path 或 pattern'.format(index))
        value = item.get('path', item.get('pattern'))
        _relative(value, 'artifacts[{}]'.format(index))
        key = ('path' if 'path' in item else 'pattern', value)
        if key in exact:
            raise ValueError('artifacts 重复: {}'.format(value))
        exact.add(key)
        used.add(class_name)
    for name, spec in classes.items():
        if spec.get('abstract'):
            continue
        if name not in used and not spec.get('owner_prefixes'):
            raise ValueError('具体类 {} 没有路径归属'.format(name))
    checks = data.get('checks')
    if not isinstance(checks, dict):
        raise ValueError('缺少 checks')
    status = checks.get('plan_draft_status')
    if not isinstance(status, dict) or not all(
            isinstance(status.get(key), str) for key in ('path', 'field', 'value')):
        raise ValueError('checks.plan_draft_status 无效')
    _relative(status['path'], 'checks.plan_draft_status.path')
    for key in ('required_artifacts', 'generated_docs'):
        values = checks.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError('checks.{} 必须是字符串列表'.format(key))
        for value in values:
            _relative(value, 'checks.' + key)
    return data


def _class_depth(classes, name):
    return len(_ancestors(classes, name))


def classify(path, ontology):
    """Return (class_name, reason) for a workspace-relative path, or (None, reason)."""
    path = path.replace('\\', '/')
    if path.startswith('./'):
        path = path[2:]
    for item in ontology['artifacts']:
        if 'path' in item:
            if path == item['path']:
                return item['class'], 'artifact path'
        elif fnmatch.fnmatch(path, item['pattern']):
            return item['class'], 'artifact pattern'
    matches = []
    for name, spec in ontology['classes'].items():
        for prefix in spec.get('owner_prefixes', []):
            if path == prefix.rstrip('/') or path.startswith(prefix):
                matches.append((len(prefix), name))
    if not matches:
        return None, 'unclassified'
    longest = max(length for length, _ in matches)
    names = sorted({name for length, name in matches if length == longest})
    if len(names) == 1:
        return names[0], 'owner prefix'
    classes = ontology['classes']
    depths = {name: _class_depth(classes, name) for name in names}
    max_depth = max(depths.values())
    deepest = [name for name, depth in depths.items() if depth == max_depth]
    if len(deepest) == 1:
        return deepest[0], 'owner prefix (deepest)'
    raise ValueError('路径分类不唯一: {} -> {}'.format(path, names))


def check_workspace(root, ontology):
    errors = []
    warnings = []
    checks = ontology['checks']
    for relative in checks['required_artifacts']:
        if not (root / relative).exists():
            errors.append('缺少必需对象: ' + relative)
    for relative in checks['generated_docs']:
        if not (root / relative).exists():
            errors.append('缺少生成文档: ' + relative)
    status = checks['plan_draft_status']
    path = root / status['path']
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        if data.get(status['field']) != status['value']:
            errors.append('计划草案状态必须为 {}: {}'.format(status['value'], status['path']))
    config_dir = root / 'config'
    if config_dir.is_dir():
        for item in sorted(config_dir.iterdir()):
            if item.is_file():
                relative = item.relative_to(root).as_posix()
                class_name, _ = classify(relative, ontology)
                if class_name is None:
                    errors.append('config 文件未分类: ' + relative)
    for item in sorted((root / 'datasets/scripts').glob('*/config/*')):
        if item.is_file():
            relative = item.relative_to(root).as_posix()
            class_name, _ = classify(relative, ontology)
            if class_name is None:
                warnings.append('任务配置未分类: ' + relative)
    return errors, warnings


def render_doc(ontology):
    lines = [
        '# 工作空间对象分类学',
        '',
        '> 自动生成：修改 `config/workspace_ontology.yaml` 后运行 `python lab/scripts/contracts/workspace_ontology.py --write`。本文件不是目录契约；目录结构见 `config/workspace_layout.yaml`。',
        '',
        ontology.get('description', ''),
        '',
        '## 类层次',
        '',
        '| 类 | 父类 | 抽象 | 所有者前缀 | 权威源 | 版本策略 | 可变性 | Git | 生命周期 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, spec in ontology['classes'].items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name, spec.get('parent') or '-', '是' if spec.get('abstract') else '否',
            ', '.join(spec.get('owner_prefixes', [])) or '-',
            spec.get('authority'), spec.get('versioning'), spec.get('mutability'),
            spec.get('tracked'), ', '.join(spec.get('lifecycle', [])) or '-'))
    lines.extend([
        '',
        '## 关系',
        '',
        '| 关系 | 从 | 到 | 说明 |',
        '| --- | --- | --- | --- |',
    ])
    for name, spec in ontology['relations'].items():
        lines.append('| {} | {} | {} | {} |'.format(
            name, ', '.join(spec['from']), ', '.join(spec['to']), spec['description']))
    lines.extend([
        '',
        '## 版本策略',
        '',
        '| 策略 | 说明 |',
        '| --- | --- |',
    ])
    for name, text in ontology['versioning_policies'].items():
        lines.append('| {} | {} |'.format(name, text))
    lines.extend([
        '',
        '## 关键对象映射',
        '',
        '| 路径或模式 | 类 |',
        '| --- | --- |',
    ])
    for item in ontology['artifacts']:
        value = item.get('path', item.get('pattern'))
        lines.append('| `{}` | {} |'.format(value, item['class']))
    lines.extend([
        '',
        '## 检查规则',
        '',
        '- 必需对象：' + '、'.join('`{}`'.format(value) for value in ontology['checks']['required_artifacts']),
        '- 生成文档：' + '、'.join('`{}`'.format(value) for value in ontology['checks']['generated_docs']),
        '- 计划草案状态：`{}` 的 `{}` 必须为 `{}`。'.format(
            ontology['checks']['plan_draft_status']['path'],
            ontology['checks']['plan_draft_status']['field'],
            ontology['checks']['plan_draft_status']['value']),
        '',
        '## 新增对象',
        '',
        '1. 在 `classes` 中选择父类，定义 `owner_prefixes`、`authority`、`versioning`、`mutability`、`tracked`、`lifecycle` 和 `relations`。',
        '2. 在 `artifacts` 中添加精确路径或模式；同一路径只能属于一个类。',
        '3. 运行 `python lab/scripts/contracts/workspace_ontology.py --write` 生成本文件，再运行 `--check`。',
        '4. 不另建与现有类同义的目录、清单或版本命名目录；历史由 Git 或归档承载。',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', help='工作空间根目录，默认本仓库')
    parser.add_argument('--ontology', help='对象分类学，默认 <root>/config/workspace_ontology.yaml')
    parser.add_argument('--doc', help='目标文档，默认 <root>/docs/WORKSPACE_ONTOLOGY.md')
    parser.add_argument('--json', action='store_true', help='输出 JSON 结果')
    parser.add_argument('--write', action='store_true', help='写入生成文档')
    parser.add_argument('--check', action='store_true', help='检查文档与关键对象')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT
    ontology_path = Path(args.ontology) if args.ontology else root / 'config/workspace_ontology.yaml'
    doc = Path(args.doc) if args.doc else root / 'docs/WORKSPACE_ONTOLOGY.md'
    try:
        ontology = load_ontology(ontology_path)
        errors, warnings = check_workspace(root, ontology)
        body = render_doc(ontology)
        summary = {'version': 1, 'classes': len(ontology['classes']),
                   'errors': errors, 'warnings': warnings}
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 1 if errors else 0
        if args.write:
            doc.write_text(body, encoding='utf-8', newline='\n')
            print('已更新: {}'.format(doc.relative_to(root).as_posix()))
            return 1 if errors else 0
        if args.check:
            changed = not doc.is_file() or doc.read_text(encoding='utf-8') != body
            print(('待更新: ' if changed else '已同步: ') + doc.relative_to(root).as_posix())
            for error in errors:
                print('ERROR: ' + error)
            for warning in warnings:
                print('WARNING: ' + warning)
            return 1 if changed or errors else 0
        for error in errors:
            print('ERROR: ' + error)
        for warning in warnings:
            print('WARNING: ' + warning)
        print('类 {} 个；错误 {}；警告 {}。'.format(len(ontology['classes']), len(errors), len(warnings)))
        return 1 if errors else 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print('对象分类学检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
