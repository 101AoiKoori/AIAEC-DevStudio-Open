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
REQUIREMENT_STATES = {'已明确', '待解决', '待全局规划', '未定', '已解决'}
STATUS_LINE = re.compile(r'^\s*(?:[-*]\s+)?`?\[([^\]]+)\]`?(?=\s|$)\s*(.*)$')


def validate_requirement_items(tasks):
    """Validate tagged item identity, not the truth of its state or evidence."""
    seen = set()
    for task in tasks:
        prefix = task['id'].upper() + '-'
        fence = None
        for line in task['requirements_markdown'].splitlines():
            delimiter = re.match(r'^\s*(`{3,}|~{3,})(.*)$', line)
            if delimiter:
                mark, suffix = delimiter.groups()
                if fence is None:
                    fence = mark
                elif mark[0] == fence[0] and len(mark) >= len(fence) and not suffix.strip():
                    fence = None
                continue
            if fence is not None:
                continue
            match = STATUS_LINE.match(line)
            if not match:
                continue
            state, body = match.groups()
            item = re.fullmatch(r'([A-Z][A-Z0-9_]*-[A-Z0-9]+(?:-[A-Z0-9]+)*)[：:]\s*(\S.*)', body)
            if state not in REQUIREMENT_STATES or not item or not item[1].startswith(prefix):
                raise ValueError('需求状态或事项 ID 无效（{}）: {}'.format(task['id'], line))
            if item[1] in seen:
                raise ValueError('需求事项 ID 重复: {}'.format(item[1]))
            seen.add(item[1])


def replace_section(path, lines):
    text = path.read_text(encoding='utf-8')
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError('文档需要一对有序的生成标记: {}'.format(path))
    return text[:text.index(START)] + START + '\n' + '\n'.join(lines) + '\n' + text[text.index(END):]


def check_staged_config():
    """Never stage documents derived from an uncommitted portion of the source."""
    indexed = subprocess.check_output(
        ['git', 'show', ':' + CONFIG.relative_to(ROOT).as_posix()],
        cwd=str(ROOT), encoding='utf-8')
    if indexed != CONFIG.read_text(encoding='utf-8'):
        raise ValueError('需求配置含未暂存修改；请先确定并暂存本阶段配置，再生成暂存文档')


def render():
    data = json.loads(CONFIG.read_text(encoding='utf-8'))
    tasks = data['tasks']
    ids = [task['id'] for task in tasks]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r'[a-z][a-z0-9_]*', task_id) for task_id in ids):
        raise ValueError('任务 ID 必须唯一且仅包含小写字母、数字和下划线')
    validate_requirement_items(tasks)
    outputs = {}
    index = ['| 任务 | 识别结果 | 分值 | 处理策略 | 共享推理模块 |', '| --- | --- | --- | --- | --- |']
    matrix = ['| 模块 | 输出与判定 | 验证重点 |', '| --- | --- | --- |']
    for task in tasks:
        task_id = task['id']
        if task_id == 'waste_bin':
            items = task.get('waste_items')
            allowed = {'recyclable', 'food', 'hazardous', 'residual'}
            if not isinstance(items, dict) or len(items) != 24 or any(
                    not isinstance(k, str) or not k or not isinstance(v, dict)
                    or set(v) != {'name', 'category'} or not isinstance(v['name'], str)
                    or not v['name'] or v['category'] not in allowed
                    for k, v in items.items()):
                raise ValueError('垃圾图例映射必须包含 24 个有效名称及四分类')
            outputs[ROOT / 'ws/src/aec_vision/config/waste_bin/items.json'] = (
                json.dumps({'items': items}, ensure_ascii=False, indent=2) + '\n')
        module = ROOT / 'ws/src/aec_vision/src/aec_vision' / (task_id + '.py')
        package = module.with_suffix('') / '__init__.py'
        status = '源码已建立' if module.is_file() or package.is_file() else '待开发'
        index.append('| [{0}]({0}/README.md) | {1} | {2} | {3} | {4} |'.format(task_id, task['summary'], task['score'], task['scheduling']['mode'], status))
        matrix.append('| {matrix_name} | {summary} | {matrix_check} |'.format(**task))
        brief = '**识别摘要：** {}\n\n**分值：** {}\n\n**调度方案：** {}。{}\n\n'.format(task['summary'], task['score'], task['scheduling']['mode'], task['scheduling']['approach'])
        research = '[开源项目分析与方法研究]({})\n\n'.format(task['research_doc'])
        outputs[ROOT / 'lab/features' / task_id / 'README.md'] = NOTICE + brief + research + task['requirements_markdown'].strip() + '\n'
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
        if args.stage and not args.check and not args.list:
            check_staged_config()
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
