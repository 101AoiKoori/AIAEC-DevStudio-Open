#!/usr/bin/env python3
"""Render requirement documents from the single task configuration."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'config/perception_tasks.json'
START = '<!-- TASKS:START -->'
END = '<!-- TASKS:END -->'
NOTICE = '> 自动生成：请修改 `config/perception_tasks.json`，然后运行 `python tools/sync_docs.py`。\n\n'


def replace_section(path, lines):
    text = path.read_text(encoding='utf-8')
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError('文档需要一对有序的生成标记: {}'.format(path))
    return text[:text.index(START)] + START + '\n' + '\n'.join(lines) + '\n' + text[text.index(END):]


def render():
    data = json.loads(CONFIG.read_text(encoding='utf-8'))
    tasks = data['tasks']
    ids = [task['id'] for task in tasks]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r'[a-z][a-z0-9_]*', task_id) for task_id in ids):
        raise ValueError('任务 ID 必须唯一且仅包含小写字母、数字和下划线')
    outputs = {}
    index = ['| 任务 | 识别结果 | 分值 | 处理策略 | 共享推理模块 |', '| --- | --- | --- | --- | --- |']
    matrix = ['| 模块 | 输出与判定 | 验证重点 |', '| --- | --- | --- |']
    for task in tasks:
        task_id = task['id']
        module = ROOT / 'ws/src/competition_perception/src/competition_perception' / (task_id + '.py')
        package = module.with_suffix('') / '__init__.py'
        status = '源码已建立' if module.is_file() or package.is_file() else '待开发'
        index.append('| [{0}]({0}/README.md) | {1} | {2} | {3} | {4} |'.format(task_id, task['summary'], task['score'], task['scheduling']['mode'], status))
        matrix.append('| {matrix_name} | {summary} | {matrix_check} |'.format(**task))
        brief = '**识别摘要：** {}\n\n**分值：** {}\n\n**调度方案：** {}。{}\n\n'.format(task['summary'], task['score'], task['scheduling']['mode'], task['scheduling']['approach'])
        outputs[ROOT / 'lab/features' / task_id / 'README.md'] = NOTICE + brief + task['requirements_markdown'].strip() + '\n'
    index_path = ROOT / 'lab/features/README.md'
    matrix_path = ROOT / 'docs/TASK_MATRIX.md'
    outputs[index_path] = replace_section(index_path, index)
    outputs[matrix_path] = replace_section(matrix_path, matrix)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='检查生成内容是否已同步')
    parser.add_argument('--list', action='store_true', help='列出生成文件')
    parser.add_argument('--stage', action='store_true', help='生成后暂存生成文档')
    args = parser.parse_args()
    try:
        outputs = render()
        if args.list:
            for path in outputs:
                print(path.relative_to(ROOT).as_posix())
            return 0
        changed = []
        for path, content in outputs.items():
            if not path.exists() or path.read_text(encoding='utf-8') != content:
                changed.append(path.relative_to(ROOT).as_posix())
                if not args.check:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('w', encoding='utf-8', newline='\n') as stream:
                        stream.write(content)
        print(('待更新: ' if args.check else '已更新: ') + (', '.join(changed) if changed else '无'))
        if args.stage and not args.check:
            subprocess.check_call(['git', 'add', '--'] + [str(p.relative_to(ROOT)) for p in outputs], cwd=str(ROOT))
        return 1 if args.check and changed else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print('文档生成失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
