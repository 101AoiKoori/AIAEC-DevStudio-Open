#!/usr/bin/env python3
"""Check workspace paths without reading file contents or modifying files."""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
ROOT_DIRS = {'.agents', '.githooks', '.git', '.venv', '.vscode', 'Archive', 'config',
             'datasets', 'docs', 'lab', 'models', 'refs', 'tmp', 'tools', 'ws'}
ROOT_FILES = {'.gitignore', 'AGENTS.md', 'README.md', '国赛规则说明.pdf'}
WEIGHTS = {'.pt', '.pth', '.onnx', '.engine', '.pdparams', '.pdiparams'}


def git_paths(root, *args):
    result = subprocess.run(['git', '-C', str(root)] + list(args) + ['-z'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip())
    return {part.decode('utf-8', errors='surrogateescape') for part in result.stdout.split(b'\0') if part}


def classify(path, indexed):
    """Return (severity, reason); severity is error, review or ok."""
    parts = PurePosixPath(path).parts
    if not parts or '..' in parts or PurePosixPath(path).is_absolute():
        return 'error', '路径必须位于工作区内'
    top, name = parts[0], parts[-1]
    if (top in {'.venv', 'tmp'} or '__pycache__' in parts or name.endswith('.pyc')
            or path.startswith(('datasets/raw/', 'datasets/annotated/', 'datasets/exports/',
                                'models/weights/', 'Archive/Assets/', 'ws/build/', 'ws/devel/',
                                'ws/install/', 'ws/log/'))
            or (top == 'ws' and 'models' in parts)
            or name == '.env' or name.startswith('.env.') or name == '相关登录信息.txt'
            or PurePosixPath(name).suffix.lower() in WEIGHTS | {'.bag', '.db3'}
            or (top == 'lab' and any(p in {'runs', 'checkpoints', 'outputs'} for p in parts))):
        return 'error', '数据、权重、凭据、临时文件或生成产物不能提交'
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
    if top == 'lab' and len(parts) > 1 and parts[1] not in {'README.md', 'features', 'navigation', 'integration'}:
        return 'review', '实验区子目录职责未登记'
    if top == 'lab' and len(parts) > 3 and parts[1] == 'features' and parts[3] not in {'.gitkeep', 'README.md', 'training', 'inference'}:
        return 'review', '任务实验子目录职责需核实'
    return 'ok', ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='仅检查暂存的新增、修改或重命名路径')
    parser.add_argument('--path', help='评估拟使用的工作区相对路径；不读取或创建该路径')
    args = parser.parse_args()
    try:
        indexed = git_paths(ROOT, 'ls-files', '--cached')
        if args.path:
            paths = {args.path.replace('\\', '/')}
        elif args.staged:
            paths = git_paths(ROOT, 'diff', '--cached', '--name-only', '--diff-filter=ACMR')
        else:
            paths = indexed | git_paths(ROOT, 'ls-files', '--others', '--exclude-standard')
        errors = reviews = 0
        for path in sorted(paths):
            severity, reason = classify(path, indexed)
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
    except (OSError, RuntimeError) as exc:
        print('检查无法完成：{}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
