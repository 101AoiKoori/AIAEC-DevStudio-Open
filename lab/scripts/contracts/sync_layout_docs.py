#!/usr/bin/env python3
"""Render workspace directory documentation from config/workspace_layout.yaml.

The generated blocks are the only human-readable inventories of top-level
directories and dataset task layout. Edit the contract, not the blocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workspace_layout import load_contract, resolve_path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / 'config/workspace_layout.yaml'
README_START = '<!-- WORKSPACE_LAYOUT:START -->'
README_END = '<!-- WORKSPACE_LAYOUT:END -->'
DESIGN_START = '<!-- DATASET_LAYOUT:START -->'
DESIGN_END = '<!-- DATASET_LAYOUT:END -->'


def _cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def _path_cell(path, kind='dir'):
    value = str(path).rstrip('/')
    return '`{}`'.format(value + ('/' if kind == 'dir' else ''))


def render_readme_table(contract):
    lines = ['| 路径 | 用途 |', '| --- | --- |']
    for entry in contract['root_entries']:
        lines.append('| {} | {} |'.format(
            _path_cell(entry['path'], entry.get('kind', 'dir')),
            _cell(entry['purpose'])))
    return lines


def render_dataset_block(contract):
    target_subdirs = contract['datasets'].get('task_target_subdirs', [])
    target_rows = ['    <任务>/targets/{{{}}}/             任务目标源工作副本'.format(','.join(target_subdirs)),
                   '    shared/backgrounds/{camera,rendered}/  跨任务共享背景工作图'] if target_subdirs else []
    lines = [
        '```text',
        'datasets/',
        '  raw/                               按任务和共享背景整理的来源工作副本',
        *target_rows,
        '  annotated/',
        '    common/review/                   跨任务审查观测与批次清单',
        '    common/perspective/              跨任务透视模板',
        '    common/sources/                  任务目标源、共享背景池与来源总账',
        '    media_index/                     查询缓存与生产回执',
        '    <任务声明的标注根>/               按任务声明，可能是共享批次目录',
        '  scripts/',
        '    common/                          跨任务共享脚本代码',
        '    <任务>/config/                   任务配置',
        '  exports/<任务>/{train,test,val}/   当前数据集',
        '  samples/<任务>/                    当前小样',
        '  assets/blender/                    Blender 编辑工程',
        '  assets/person/                     只读人物候选来源',
        '```',
        '',
        '目录树为结构示意；共享根与任务路径以下表为准。',
        '',
        '| 共享根 | 用途 |',
        '| --- | --- |',
    ]
    for entry in contract['datasets']['shared']:
        lines.append('| {} | {} |'.format(_path_cell(entry['path']), _cell(entry['purpose'])))
    lines.extend([
        '',
        '| 任务 | 脚本入口 | 必需配置 | 标注根 | 集合与小样 |',
        '| --- | --- | --- | --- | --- |',
    ])
    notes = []
    for task_id, spec in contract['datasets']['tasks'].items():
        required = '、'.join('`{}`'.format(name) for name in spec['config']['required'])
        conditional = spec['config'].get('conditional', [])
        if conditional:
            required += '；' + '；'.join(
                '`{}`（条件：{}）'.format(item['path'], item['reason']) for item in conditional)
        annotations = '、'.join('`{}/`'.format(path.rstrip('/')) for path in spec['annotation_roots'])
        exports = '`{}/{{{}}}/`'.format(spec['exports'].rstrip('/'), ','.join(spec['exports_splits']))
        lines.append('| {} | `{}/` | {} | {} | {}、`{}/` |'.format(
            task_id, spec['scripts'].rstrip('/'), required, annotations,
            exports, spec['samples'].rstrip('/')))
        if spec.get('annotation_note'):
            notes.append('- {}：{}'.format(task_id, spec['annotation_note']))
    optional = []
    for task_id, spec in contract['datasets']['tasks'].items():
        names = spec.get('optional_config', [])
        if names:
            optional.append('`{}`：{}'.format(task_id, '、'.join('`{}`'.format(name) for name in names)))
    if optional:
        lines.extend(['', '可选配置（存在时使用）：' + '；'.join(optional)])
    lines.extend([
        '',
        '标注根按任务声明；`datasets/annotated/<任务>/` 不是通用规则。生成方式：修改 `config/workspace_layout.yaml`，运行 `python lab/scripts/contracts/sync_layout_docs.py`。',
    ])
    if notes:
        lines.extend(['', '标注根说明：', *notes])
    return lines


def replace_section(path, start, end, body):
    text = path.read_text(encoding='utf-8')
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError('文档需要一对有序的生成标记: {}'.format(path))
    return text[:text.index(start)] + start + '\n' + body.rstrip('\n') + '\n' + text[text.index(end):]


def render(root=ROOT, contract_path=CONTRACT):
    contract = load_contract(contract_path)
    docs = contract['generated_docs']
    readme = resolve_path(root, docs['readme'])
    design = resolve_path(root, docs['dataset_design'])
    readme_body = '\n'.join(render_readme_table(contract))
    design_body = '\n'.join(render_dataset_block(contract))
    return {
        readme: replace_section(readme, README_START, README_END, readme_body),
        design: replace_section(design, DESIGN_START, DESIGN_END, design_body),
    }


def check_staged_contract(contract_path=CONTRACT):
    """Never stage documents derived from an uncommitted portion of the contract."""
    relative = contract_path.relative_to(ROOT).as_posix()
    indexed = subprocess.check_output(
        ['git', 'show', ':' + relative], cwd=str(ROOT), encoding='utf-8')
    if indexed.replace('\r\n', '\n') != contract_path.read_text(encoding='utf-8'):
        raise ValueError('目录契约含未暂存修改；请先确定并暂存契约，再生成暂存文档')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='检查生成内容是否已同步')
    parser.add_argument('--list', action='store_true', help='列出生成文件')
    parser.add_argument('--stage', action='store_true', help='生成后暂存生成文档')
    args = parser.parse_args()
    try:
        if args.stage and not args.check and not args.list:
            check_staged_contract()
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
            pending = []
            for path, content in outputs.items():
                name = path.relative_to(ROOT).as_posix()
                indexed = subprocess.run(['git', 'show', ':' + name], cwd=str(ROOT),
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if indexed.returncode or indexed.stdout.decode('utf-8').replace('\r\n', '\n') != content:
                    pending.append(name)
            if pending:
                subprocess.check_call(['git', 'add', '--'] + pending, cwd=str(ROOT))
        return 1 if args.check and changed else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print('目录文档生成失败: {}'.format(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
