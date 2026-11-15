#!/usr/bin/env python3
"""Check, suggest and scaffold workspace paths from config/workspace_layout.yaml.

Requires PyYAML. The tool reads directory names and declared files only. It
does not read data, labels, model weights or secrets, and it never deletes or
overwrites anything.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / 'config/workspace_layout.yaml'
DATASET_ROLES = {
    'dataset-script', 'dataset-config', 'dataset-annotation', 'dataset-sample',
    'dataset-export', 'dataset-shared-script', 'dataset-shared-annotation',
}
OTHER_ROLES = {'feature-root', 'ros-package', 'repo-tool', 'temporary'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _string(value, message):
    _require(isinstance(value, str) and bool(value.strip()), message + ' 必须是非空字符串')
    return value


def _string_list(value, message):
    _require(isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value),
        message + ' 必须是字符串列表')
    return value


def _relative(value, message):
    _string(value, message)
    path = PurePosixPath(value.replace('\\', '/'))
    _require(not path.is_absolute() and '..' not in path.parts and bool(path.parts),
             message + ' 必须是工作区相对路径')
    return value


def validate_contract(data):
    """Validate contract structure and path strings; return the contract."""
    _require(isinstance(data, dict), '契约顶层必须是映射')
    _require(data.get('version') == 1, '契约版本必须为 1')
    for key in ('generated_docs', 'root_entries', 'datasets', 'features', 'packages'):
        _require(key in data, '契约缺少字段: {}'.format(key))

    docs = data['generated_docs']
    _require(isinstance(docs, dict), 'generated_docs 必须是映射')
    for key in ('readme', 'dataset_design'):
        _relative(docs.get(key), 'generated_docs.' + key)

    roots = data['root_entries']
    _require(isinstance(roots, list) and roots, 'root_entries 必须是非空列表')
    seen = set()
    for index, entry in enumerate(roots):
        prefix = 'root_entries[{}]'.format(index)
        _require(isinstance(entry, dict), prefix + ' 必须是映射')
        path = _relative(entry.get('path'), prefix + '.path')
        _require(path not in seen, 'root_entries 路径重复: ' + path)
        seen.add(path)
        _require(entry.get('kind', 'dir') in {'dir', 'file'}, prefix + '.kind 必须是 dir 或 file')
        _string(entry.get('purpose'), prefix + '.purpose')
        if 'required' in entry:
            _require(isinstance(entry['required'], bool), prefix + '.required 必须是布尔值')
        if 'local' in entry:
            _require(isinstance(entry['local'], bool), prefix + '.local 必须是布尔值')

    datasets = data['datasets']
    _require(isinstance(datasets, dict), 'datasets 必须是映射')
    markers = datasets.get('materialized_if_any', [])
    _string_list(markers, 'datasets.materialized_if_any')
    for marker in markers:
        _relative(marker, 'datasets.materialized_if_any')
    shared = datasets.get('shared')
    _require(isinstance(shared, list), 'datasets.shared 必须是列表')
    shared_paths = set()
    for index, entry in enumerate(shared):
        prefix = 'datasets.shared[{}]'.format(index)
        _require(isinstance(entry, dict), prefix + ' 必须是映射')
        path = _relative(entry.get('path'), prefix + '.path')
        _require(path not in shared_paths, 'datasets.shared 路径重复: ' + path)
        shared_paths.add(path)
        _string(entry.get('purpose'), prefix + '.purpose')
        if 'required' in entry:
            _require(isinstance(entry['required'], bool), prefix + '.required 必须是布尔值')
        if 'create' in entry:
            _require(isinstance(entry['create'], bool), prefix + '.create 必须是布尔值')
        for name in _string_list(entry.get('required_files', []), prefix + '.required_files'):
            _require('/' not in name and '\\' not in name and name not in {'.', '..'},
                     prefix + '.required_files 必须是文件名: ' + name)

    tasks = datasets.get('tasks')
    _require(isinstance(tasks, dict) and tasks, 'datasets.tasks 必须是非空映射')
    for task_id, spec in tasks.items():
        _require(re.fullmatch(r'[a-z][a-z0-9_]*', task_id) is not None,
                 '数据任务 ID 必须是小写字母、数字和下划线: {}'.format(task_id))
        prefix = 'datasets.tasks.' + task_id
        _require(isinstance(spec, dict), prefix + ' 必须是映射')
        _relative(spec.get('scripts'), prefix + '.scripts')
        config = spec.get('config')
        _require(isinstance(config, dict), prefix + '.config 必须是映射')
        _string_list(config.get('required'), prefix + '.config.required')
        conditional = config.get('conditional', [])
        _require(isinstance(conditional, list), prefix + '.config.conditional 必须是列表')
        for index, item in enumerate(conditional):
            item_prefix = '{}.config.conditional[{}]'.format(prefix, index)
            _require(isinstance(item, dict), item_prefix + ' 必须是映射')
            _string(item.get('path'), item_prefix + '.path')
            _string(item.get('reason'), item_prefix + '.reason')
        annotations = _string_list(spec.get('annotation_roots'), prefix + '.annotation_roots')
        for annotation in annotations:
            _relative(annotation, prefix + '.annotation_roots')
        if 'annotation_roots_owned' in spec:
            _require(isinstance(spec['annotation_roots_owned'], bool),
                     prefix + '.annotation_roots_owned 必须是布尔值')
        _relative(spec.get('exports'), prefix + '.exports')
        splits = _string_list(spec.get('exports_splits'), prefix + '.exports_splits')
        _require(bool(splits), prefix + '.exports_splits 不能为空')
        _relative(spec.get('samples'), prefix + '.samples')
        _string_list(spec.get('optional_config', []), prefix + '.optional_config')
        if 'annotation_note' in spec:
            _require(isinstance(spec['annotation_note'], str),
                     prefix + '.annotation_note 必须是字符串')

    features = data['features']
    _require(isinstance(features, dict), 'features 必须是映射')
    _relative(features.get('perception_tasks'), 'features.perception_tasks')
    _relative(features.get('root'), 'features.root')
    _string_list(features.get('required_files'), 'features.required_files')
    branches = _string_list(features.get('branches'), 'features.branches')
    _string_list(features.get('branch_entries'), 'features.branch_entries')
    local_entries = features.get('branch_local_entries', {})
    _require(isinstance(local_entries, dict), 'features.branch_local_entries 必须是映射')
    for branch, entries in local_entries.items():
        _require(branch in branches, 'features.branch_local_entries 含未声明分支: ' + branch)
        _string_list(entries, 'features.branch_local_entries.' + branch)
    if 'require_branches' in features:
        _require(isinstance(features['require_branches'], bool),
                 'features.require_branches 必须是布尔值')
    extra = features.get('extra_dirs', {})
    _require(isinstance(extra, dict), 'features.extra_dirs 必须是映射')
    for name, note in extra.items():
        _require(re.fullmatch(r'[a-z][a-z0-9_]*', name) is not None,
                 'features.extra_dirs 名称无效: ' + str(name))
        _string(note, 'features.extra_dirs.' + name)
    aliases = features.get('feature_aliases', {})
    _require(isinstance(aliases, dict), 'features.feature_aliases 必须是映射')
    for name, target in aliases.items():
        _require(re.fullmatch(r'[a-z][a-z0-9_]*', name) is not None,
                 'features.feature_aliases 名称无效: ' + str(name))
        _require(re.fullmatch(r'[a-z][a-z0-9_]*', target) is not None,
                 'features.feature_aliases 目标无效: ' + str(target))
    fixed = _string_list(features.get('fixed_roots'), 'features.fixed_roots')
    for root in fixed:
        _relative(root, 'features.fixed_roots')

    packages = data['packages']
    _require(isinstance(packages, dict), 'packages 必须是映射')
    _relative(packages.get('root'), 'packages.root')
    pattern = _string(packages.get('name_pattern'), 'packages.name_pattern')
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError('packages.name_pattern 无效: {}'.format(exc)) from exc
    _string_list(packages.get('required_files'), 'packages.required_files')
    template = _string(packages.get('python_package'), 'packages.python_package')
    _require('{name}' in template, 'packages.python_package 必须包含 {name}')
    _string_list(packages.get('python_package_exceptions', []),
                 'packages.python_package_exceptions')
    return data


def load_contract(path):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError('读取 YAML 契约需要 PyYAML；请在使用该工具的 Python 中安装 PyYAML') from exc
    with open(path, encoding='utf-8') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError('契约 YAML 解析失败: {}'.format(exc)) from exc
    return validate_contract(data)


def resolve_path(root, relative):
    path = PurePosixPath(str(relative).replace('\\', '/').strip('/'))
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('路径必须位于工作区内: {}'.format(relative))
    resolved = (root / Path(*path.parts)).resolve()
    try:
        inside = os.path.commonpath([str(root.resolve()), str(resolved)]) == str(root.resolve())
    except ValueError:
        inside = False
    if not inside:
        raise ValueError('路径指向工作区外: {}'.format(relative))
    return root / Path(*path.parts)


def _item(severity, path, message):
    return {'severity': severity, 'path': path, 'message': message}


def _exists(root, relative, kind):
    path = resolve_path(root, relative)
    return path.is_file() if kind == 'file' else path.is_dir()


def _perception_tasks(root, contract):
    source = resolve_path(root, contract['features']['perception_tasks'])
    return json.loads(source.read_text(encoding='utf-8'))['tasks']


def _data_present(root, contract):
    markers = contract['datasets'].get('materialized_if_any', [])
    return not markers or any(_exists(root, marker, 'dir') for marker in markers)


def _check_datasets(root, contract, task):
    items = []
    tasks = contract['datasets']['tasks']
    if task is not None and task not in tasks:
        raise ValueError('数据任务未在契约中定义: {}'.format(task))
    if not _data_present(root, contract):
        return [_item('info', 'datasets', '本地数据树未展开，跳过数据集结构检查')]
    for entry in contract['datasets']['shared']:
        if entry.get('required') and not _exists(root, entry['path'], 'dir'):
            items.append(_item('error', entry['path'], '必需共享目录缺失'))
        for name in entry.get('required_files', []):
            relative = entry['path'].rstrip('/') + '/' + name
            if not _exists(root, relative, 'file'):
                items.append(_item('error', relative, '共享目录缺少必需文件'))
    for task_id, spec in tasks.items():
        if task is not None and task_id != task:
            continue
        prefix = spec['scripts']
        if not _exists(root, prefix, 'dir'):
            items.append(_item('error', prefix, '任务脚本目录缺失'))
        config_dir = prefix + '/config'
        if not _exists(root, config_dir, 'dir'):
            items.append(_item('error', config_dir, '任务配置目录缺失'))
        for name in spec['config']['required']:
            relative = config_dir + '/' + name
            if not _exists(root, relative, 'file'):
                items.append(_item('error', relative, '必需任务配置缺失'))
        for conditional in spec['config'].get('conditional', []):
            relative = config_dir + '/' + conditional['path']
            if not _exists(root, relative, 'file'):
                items.append(_item('info', relative, '条件配置缺失：{}'.format(conditional['reason'])))
        for annotation in spec['annotation_roots']:
            if not _exists(root, annotation, 'dir'):
                items.append(_item('error', annotation, '任务声明的标注根缺失'))
        for split in spec['exports_splits']:
            relative = spec['exports'] + '/' + split
            if not _exists(root, relative, 'dir'):
                items.append(_item('info', relative, '数据集集合目录尚未生成'))
        if not _exists(root, spec['samples'], 'dir'):
            items.append(_item('info', spec['samples'], '当前小样目录尚未生成'))
    return items


def check(root, contract, task=None):
    """Return findings; severity is error, warning or info."""
    items = []

    data_present = _data_present(root, contract)
    for entry in contract['root_entries']:
        if not entry.get('required'):
            continue
        if entry.get('local') and not data_present:
            continue
        if not _exists(root, entry['path'], entry.get('kind', 'dir')):
            items.append(_item('error', entry['path'], '必需顶层条目缺失'))

    items.extend(_check_datasets(root, contract, task))

    feature_root = contract['features']['root']
    feature_path = resolve_path(root, feature_root)
    feature_ids = {entry['id'] for entry in _perception_tasks(root, contract)}
    extra_dirs = contract['features'].get('extra_dirs', {})
    branch_entries = set(contract['features']['branch_entries'])
    branch_local = contract['features'].get('branch_local_entries', {})
    for entry in _perception_tasks(root, contract):
        task_id = entry['id']
        for name in contract['features']['required_files']:
            relative = '{}/{}/{}'.format(feature_root, task_id, name)
            if not _exists(root, relative, 'file'):
                items.append(_item('error', relative, '识别任务缺少说明文件'))
        for branch in contract['features']['branches']:
            branch_path = resolve_path(root, '{}/{}/{}'.format(feature_root, task_id, branch))
            if not branch_path.is_dir():
                if contract['features'].get('require_branches'):
                    items.append(_item('error', '{}/{}/{}'.format(feature_root, task_id, branch),
                                       '识别任务缺少实验分支'))
                continue
            allowed = branch_entries | set(branch_local.get(branch, [])) | set(
                contract['features']['required_files']) | {'.gitkeep'}
            for child in sorted(branch_path.iterdir()):
                if child.name not in allowed:
                    items.append(_item('warning', '{}/{}/{}/{}'.format(
                        feature_root, task_id, branch, child.name), '实验分支下出现未登记条目'))
    if feature_path.is_dir():
        for child in sorted(feature_path.iterdir()):
            if child.is_dir() and child.name not in feature_ids and child.name not in extra_dirs:
                items.append(_item('warning', '{}/{}'.format(feature_root, child.name),
                                   'feature 目录未在契约登记'))
    for name in extra_dirs:
        if not _exists(root, '{}/{}'.format(feature_root, name), 'dir'):
            items.append(_item('info', '{}/{}'.format(feature_root, name), '契约登记的数据任务实验目录不存在'))
    for relative in contract['features']['fixed_roots']:
        if not _exists(root, relative, 'dir'):
            items.append(_item('error', relative, '固定实验目录缺失'))

    package_root = contract['packages']['root']
    package_path = resolve_path(root, package_root)
    if not package_path.is_dir():
        items.append(_item('error', package_root, 'ROS 包根目录缺失'))
    else:
        pattern = re.compile(contract['packages']['name_pattern'])
        for child in sorted(package_path.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith('aec_'):
                items.append(_item('info', '{}/{}'.format(package_root, name), '外部或厂商包沿用原名，不按项目命名规则检查'))
                continue
            if not pattern.fullmatch(name):
                items.append(_item('error', '{}/{}'.format(package_root, name), '项目包名不符合 aec_<职责>'))
            for required in contract['packages']['required_files']:
                relative = '{}/{}/{}'.format(package_root, name, required)
                if not _exists(root, relative, 'file'):
                    items.append(_item('error', relative, 'ROS 包必需文件缺失'))
            python_package = contract['packages']['python_package'].format(name=name)
            if name not in contract['packages'].get('python_package_exceptions', []):
                relative = '{}/{}/{}'.format(package_root, name, python_package)
                if not _exists(root, relative, 'file'):
                    items.append(_item('error', relative, 'Python 包入口缺失'))

    return items


def scaffold(root, contract, tasks, include_shared=False, dry_run=False):
    """Create declared directories only; return created paths and missing files."""
    if not tasks:
        raise ValueError('scaffold 需要 --task 或 --all')
    selected = contract['datasets']['tasks']
    for task in tasks:
        if task not in selected:
            raise ValueError('数据任务未在契约中定义: {}'.format(task))
    targets = []
    if include_shared:
        targets.extend(entry['path'] for entry in contract['datasets']['shared']
                       if entry.get('required') and entry.get('create', True))
        targets.extend(entry['path'] for entry in contract['root_entries']
                       if entry.get('local') and entry.get('kind', 'dir') == 'dir'
                       and entry['path'].startswith('datasets/'))
    missing_files = []
    for task in tasks:
        spec = selected[task]
        targets.append(spec['scripts'])
        targets.append(spec['scripts'] + '/config')
        if spec.get('annotation_roots_owned', True):
            targets.extend(spec['annotation_roots'])
        targets.append(spec['samples'])
        targets.extend(spec['exports'] + '/' + split for split in spec['exports_splits'])
        config_dir = resolve_path(root, spec['scripts'] + '/config')
        for name in spec['config']['required']:
            if not (config_dir / name).is_file():
                missing_files.append('{}/config/{}'.format(spec['scripts'], name))
    created = []
    for relative in targets:
        path = resolve_path(root, relative)
        if path.is_dir():
            continue
        created.append(relative)
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)
    return created, missing_files


def suggest(contract, role, task=None, name=None, root=None):
    """Return candidate paths for a role without creating them."""
    if role in DATASET_ROLES:
        if role == 'dataset-shared-script':
            return ['datasets/scripts/common']
        if role == 'dataset-shared-annotation':
            return ['datasets/annotated/common/review', 'datasets/annotated/common/perspective']
        if task is None:
            raise ValueError('{} 需要 --task'.format(role))
        tasks = contract['datasets']['tasks']
        if task not in tasks:
            raise ValueError('数据任务未在契约中定义: {}'.format(task))
        spec = tasks[task]
        if role == 'dataset-script':
            return [spec['scripts']]
        if role == 'dataset-config':
            return [spec['scripts'] + '/config']
        if role == 'dataset-annotation':
            return list(spec['annotation_roots'])
        if role == 'dataset-sample':
            return [spec['samples']]
        if role == 'dataset-export':
            return [spec['exports'] + '/' + split for split in spec['exports_splits']]
    if role == 'feature-root':
        if task is None:
            raise ValueError('feature-root 需要 --task')
        resolved = contract['features'].get('feature_aliases', {}).get(task, task)
        if root is not None:
            known = {entry['id'] for entry in _perception_tasks(root, contract)}
            known |= set(contract['features'].get('extra_dirs', {}))
            if resolved not in known:
                raise ValueError('识别或数据任务未在契约中定义: {}'.format(task))
        return [contract['features']['root'] + '/' + resolved]
    if role == 'ros-package':
        if name is None:
            raise ValueError('ros-package 需要 --name')
        if not re.fullmatch(contract['packages']['name_pattern'], name):
            raise ValueError('项目包名不符合 aec_<职责>: {}'.format(name))
        return [contract['packages']['root'] + '/' + name]
    if role == 'repo-tool':
        return ['lab/scripts']
    if role == 'temporary':
        return ['tmp']
    raise ValueError('未知角色: {}'.format(role))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'scaffold', 'suggest'])
    parser.add_argument('--root', help='工作空间根目录，默认本仓库')
    parser.add_argument('--contract', help='契约路径，默认 <root>/config/workspace_layout.yaml')
    parser.add_argument('--task', help='数据任务 ID；feature-root 使用识别任务 ID')
    parser.add_argument('--all', action='store_true', help='scaffold 全部数据任务')
    parser.add_argument('--shared', action='store_true', help='scaffold 同时创建共享目录')
    parser.add_argument('--dry-run', action='store_true', help='scaffold 只报告，不创建')
    parser.add_argument('--role', choices=sorted(DATASET_ROLES | OTHER_ROLES), help='suggest 的角色')
    parser.add_argument('--name', help='ros-package 的包名')
    parser.add_argument('--json', action='store_true', help='以 JSON 输出 check 结果')
    parser.add_argument('--quiet', action='store_true', help='check 只输出错误和汇总')
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else ROOT
    contract_path = Path(args.contract) if args.contract else root / 'config/workspace_layout.yaml'
    if not contract_path.is_absolute():
        contract_path = root / contract_path
    try:
        contract = load_contract(contract_path)
        if args.command == 'check':
            items = check(root, contract, task=args.task)
            errors = sum(item['severity'] == 'error' for item in items)
            warnings = sum(item['severity'] == 'warning' for item in items)
            infos = sum(item['severity'] == 'info' for item in items)
            if args.json:
                print(json.dumps({'errors': errors, 'warnings': warnings, 'infos': infos,
                                  'items': items}, ensure_ascii=False, indent=2))
            else:
                for item in items:
                    if args.quiet and item['severity'] != 'error':
                        continue
                    print('{}: {}: {}'.format(item['severity'].upper(), item['path'], item['message']))
                print('发现 {} 项：{} 项错误，{} 项警告，{} 项提示。'.format(
                    len(items), errors, warnings, infos))
            return 1 if errors else 0
        if args.command == 'scaffold':
            tasks = list(contract['datasets']['tasks']) if args.all else ([args.task] if args.task else [])
            created, missing = scaffold(root, contract, tasks, include_shared=args.shared,
                                        dry_run=args.dry_run)
            for relative in created:
                print('{}: {}'.format('WOULD CREATE' if args.dry_run else 'CREATED', relative))
            for relative in missing:
                print('MISSING FILE: {}（未自动创建，需按模板填写）'.format(relative))
            print('目录 {}：{} 个；缺失配置文件：{} 个。'.format(
                '待创建' if args.dry_run else '已创建', len(created), len(missing)))
            return 1 if missing else 0
        if not args.role:
            parser.error('suggest 需要 --role')
        for relative in suggest(contract, args.role, task=args.task, name=args.name, root=root):
            print(relative)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print('目录契约检查失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
