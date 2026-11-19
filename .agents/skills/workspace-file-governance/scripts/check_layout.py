#!/usr/bin/env python3
"""Check workspace paths without reading file contents or modifying files."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
CONTRACTS = ROOT / 'lab/scripts/contracts'
if str(CONTRACTS) not in sys.path:
    sys.path.insert(0, str(CONTRACTS))
ROOT_DIRS = {'.agents', '.claude', '.codex', '.githooks', '.git', '.venv', '.vscode', 'AIAEC-DevStudio-Open',
             'Archive', 'config', 'datasets', 'docs', 'lab', 'models', 'refs', 'runs', 'tmp',
             'training_packages', 'ws'}
ROOT_FILES = {'.gitignore', 'AGENTS.md', 'README.md', '国赛规则说明.pdf'}
WEIGHTS = {'.pt', '.pth', '.onnx', '.engine', '.pdparams', '.pdiparams'}
DATASET_KINDS = {'dataset-script', 'dataset-config', 'dataset-annotation', 'dataset-sample',
                 'dataset-export', 'dataset-shared-script', 'dataset-shared-annotation'}
PACKAGE_KIND = 'ros-package'
SUGGEST_KINDS = ['training-script', 'training-config', 'training-test',
                 'inference-script', 'inference-config', 'inference-test', 'shared-core',
                 'package-test', 'navigation', 'integration', 'repo-tool', 'temporary',
                 PACKAGE_KIND] + sorted(DATASET_KINDS)


def git_paths(root, *args):
    result = subprocess.run(['git', '-C', str(root)] + list(args) + ['-z'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip())
    return {part.decode('utf-8', errors='surrogateescape') for part in result.stdout.split(b'\0') if part}


def indexed_links(root):
    result = subprocess.run(['git', '-C', str(root), 'ls-files', '--stage', '-z'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip())
    paths = set()
    for record in result.stdout.split(b'\0'):
        if record:
            metadata, path = record.split(b'\t', 1)
            if metadata.split(b' ', 1)[0] in (b'120000', b'160000'):
                paths.add(path.decode('utf-8', errors='surrogateescape'))
    return paths


def classify(path, indexed):
    """Return (severity, reason); severity is error, review or ok."""
    parts = PurePosixPath(path).parts
    if not parts or '..' in parts or PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive:
        return 'error', '路径必须位于工作区内'
    top, name = parts[0], parts[-1]
    if re.fullmatch(r'datasets/(?:samples/[^/]+|exports/[^/]+/(?:train|val|test))/\.gitkeep', path):
        return 'ok', ''
    if (top in {'.venv', 'tmp'} or '__pycache__' in parts or name.endswith('.pyc')
            or path.startswith(('datasets/raw/', 'datasets/annotated/', 'datasets/exports/', 'datasets/samples/',
                                'models/weights/', 'Archive/Assets/', 'ws/build/', 'ws/devel/',
                                'ws/install/', 'ws/log/'))
            or path.startswith('ws/src/aec_vision/models/')
            or name == '.env' or (name.startswith('.env.') and name not in {'.env.example', '.env.sample'}) or name == '相关登录信息.txt'
            or PurePosixPath(name).suffix.lower() in WEIGHTS | {'.bag', '.db3'}
            or (len(parts) > 4 and parts[:2] == ('lab', 'features')
                and ((parts[3] == 'training' and parts[4] in {'runs', 'checkpoints'})
                     or (parts[3] == 'inference' and parts[4] == 'outputs')))):
        return 'error', '数据、权重、凭据、临时文件或生成产物不能提交'
    if name in {'.env.example', '.env.sample'}:
        return 'review', '环境配置示例需确认不含凭据'
    if len(parts) == 1:
        return ('ok', '') if name in ROOT_FILES else ('review', '根目录文件未登记职责')
    if top not in ROOT_DIRS:
        return 'review', '顶层目录未登记职责'
    if top == 'ws' and len(parts) >= 3 and parts[1] == 'src' and parts[2] != '.gitkeep':
        package = parts[2]
        if not re.fullmatch(r'aec_[a-z]+(?:_[a-z]+)*', package):
            return 'review', 'ws/src 下的包名称或外部包需核实'
        if 'ws/src/{}/package.xml'.format(package) not in indexed:
            return 'error', 'ws/src 包缺少已跟踪或已暂存的 package.xml'
    if top == 'lab' and len(parts) > 1 and parts[1] not in {'README.md', 'features', 'navigation', 'integration', 'training_common', 'scripts'}:
        return 'review', '实验区子目录职责未登记'
    if top == 'lab' and len(parts) > 2 and parts[1] == 'scripts' and parts[2] not in {
            'README.md', '__init__.py', 'contracts', 'media', 'blender', 'data', 'audit', 'common'}:
        return 'review', '全局脚本分类未登记'
    if top == 'lab' and len(parts) > 3 and parts[1] == 'features' and parts[3] not in {'.gitkeep', 'README.md', 'training', 'inference'}:
        return 'review', '任务实验子目录职责需核实'
    if (top == 'lab' and len(parts) > 4 and parts[1] == 'features'
            and parts[3] in {'training', 'inference'}
            and parts[4] not in {'scripts', 'config', 'tests', 'README.md', '.gitkeep'}):
        return 'review', '训练或推理实验内容应确认 scripts、config 或 tests 归属'
    return 'ok', ''


def suggest(root, kind, task, name=None):
    if kind in DATASET_KINDS or kind == PACKAGE_KIND:
        import workspace_layout
        contract = workspace_layout.load_contract(root / 'config/workspace_layout.yaml')
        return workspace_layout.suggest(contract, kind, task=task, name=name, root=root)
    if kind.startswith(('training-', 'inference-')):
        if not task:
            raise ValueError('识别实验需要 --task 指定任务 ID')
        data = json.loads((root / 'config/perception_tasks.json').read_text(encoding='utf-8'))
        if task not in {item['id'] for item in data['tasks']}:
            raise ValueError('任务 ID 未在 config/perception_tasks.json 中定义: {}'.format(task))
        branch, role = kind.split('-', 1)
        return 'lab/features/{}/{}/{}'.format(task, branch, {'script': 'scripts', 'config': 'config', 'test': 'tests'}[role])
    if task:
        raise ValueError('--task 仅用于识别训练或推理实验')
    return {'shared-core': 'ws/src/aec_vision/src/aec_vision',
            'package-test': 'ws/src/aec_vision/tests', 'navigation': 'lab/navigation',
            'integration': 'lab/integration', 'repo-tool': 'lab/scripts',
            'temporary': 'tmp'}[kind]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--staged', action='store_true', help='检查暂存变化及受影响包')
    mode.add_argument('--path', help='评估拟使用的工作区相对路径；不读取或创建该路径')
    mode.add_argument('--suggest', choices=SUGGEST_KINDS,
                      help='按任务类型建议目录；不会创建文件')
    parser.add_argument('--task', help='识别或数据任务 ID，仅与 --suggest 配合')
    parser.add_argument('--name', help='ROS 包名，仅与 --suggest ros-package 配合')
    args = parser.parse_args()
    try:
        if args.suggest:
            suggestions = suggest(ROOT, args.suggest, args.task, args.name)
            for item in ([suggestions] if isinstance(suggestions, str) else suggestions):
                print('建议目录: {}'.format(item))
            print('请结合用途、权威目录说明和具体文件再次判断；该建议不会创建目录。')
            return 0
        if args.task:
            parser.error('--task 只能与 --suggest 配合')
        indexed = git_paths(ROOT, 'ls-files', '--cached')
        links = indexed_links(ROOT)
        if args.path:
            paths = {args.path.replace('\\', '/')}
        elif args.staged:
            changed = git_paths(ROOT, 'diff', '--cached', '--name-only', '--no-renames')
            paths = changed & indexed
            # A staged manifest deletion can invalidate files that did not themselves change.
            affected = {p.split('/')[2] for p in changed if p.startswith('ws/src/') and len(p.split('/')) > 3}
            paths |= {p for p in indexed if p.startswith('ws/src/') and len(p.split('/')) > 3 and p.split('/')[2] in affected}
        else:
            paths = indexed | git_paths(ROOT, 'ls-files', '--others', '--exclude-standard')
            indexed = paths  # A new, untracked package may already contain its manifest.
        errors = reviews = 0
        for path in sorted(paths):
            severity, reason = classify(path, indexed)
            if args.path and severity != 'error':
                resolved = (ROOT / path).resolve()
                if os.path.commonpath([str(ROOT.resolve()), str(resolved)]) != str(ROOT.resolve()):
                    severity, reason = 'error', '路径经符号链接或挂载点指向工作区外'
            if path in links and severity == 'ok':
                severity, reason = 'review', '符号链接或子模块需人工确认目标与用途'
            if args.path and severity == 'error' and reason.startswith('数据、'):
                severity, reason = 'review', '该路径可用于本机产物，但符合禁止提交规则；确认用途与忽略规则'
            if args.path and severity == 'error' and 'package.xml' in reason:
                severity, reason = 'review', '新包使用前需确认职责并在提交时包含 package.xml'
            if severity != 'ok':
                print('{}: {}: {}'.format(severity.upper(), path, reason))
                errors += severity == 'error'
                reviews += severity == 'review'
            elif args.path:
                print('OK: {}: 路径未触发机械规则；仍需按用途判断'.format(path))
        if not args.staged and not args.path:
            for item in sorted(ROOT.iterdir()):
                if item.is_dir() and item.name not in ROOT_DIRS:
                    print('REVIEW: {}/: 顶层目录未登记职责（未检查其内容）'.format(item.name))
                    reviews += 1
        print('检查 {} 个路径：{} 项违规，{} 项待人工核实。'.format(len(paths), errors, reviews))
        print('范围：{}；忽略目录的内容、文件语义、动态依赖和凭据内容未检查。'.format(
            '拟定路径' if args.path else ('暂存路径' if args.staged else '已跟踪及未忽略的未跟踪文件')))
        return 1 if errors else 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print('检查无法完成：{}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
