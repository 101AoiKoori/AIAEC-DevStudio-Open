"""Copy eligible synthetic fire candidates into an isolated YOLO directory.

The output is an unreviewed train candidate set, not a name/OCR dataset or an
independent evaluation set. Source files and earlier exports are never changed.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/. Production remains paused.")

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from fire_blender_labels import fire_label_suppression, yolo_row


ROOT = Path.cwd().resolve()
FIRE_EXPORT_ROOT = ROOT / 'datasets/exports/fire'
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


def child_file(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('清单路径必须为相对路径')
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'清单文件缺失或位于来源目录外: {value}')
    return path


def decide(item, resolution):
    """Check label eligibility and derive expected rows without file I/O."""
    if item.get('split') != 'train' or item.get('status') != 'candidate_requires_visual_review':
        raise ValueError('仅支持待人工复核的 train 合成候选')
    requested = item.get('requested_fire_counts')
    instances = item.get('instances')
    active = item.get('active_buildings')
    if (not isinstance(requested, dict) or not isinstance(instances, list)
            or not isinstance(active, list) or not active or set(requested) != set(active)):
        raise ValueError('缺少火焰数量或实例清单')
    fires = [o for o in instances if o.get('kind') == 'fire_sticker']
    if any(type(n) is not int or n < 0 for n in requested.values()):
        raise ValueError('火焰数量必须为非负整数')
    counts = Counter(o.get('building') for o in fires)
    if set(counts) - set(requested) or any(counts[k] != n for k, n in requested.items()):
        raise ValueError('火焰实例与请求数量不一致')
    if type(item.get('occluder')) is not bool:
        raise ValueError('缺少明确的遮挡状态')
    reason = fire_label_suppression([o.get('state') for o in fires], item['occluder'])
    if reason != item.get('fire_label_suppression_reason'):
        raise ValueError('火焰抑制原因与实例状态不一致')
    label = item.get('fire_label')
    if (reason is None) != (isinstance(label, str) and bool(label)):
        raise ValueError('火焰标签路径与抑制状态不一致')
    rows = [yolo_row(o['pixel_bbox_xyxy'], resolution, resolution)
            for o in fires if o.get('state') == 'visible']
    return reason, rows


def plan_source(source):
    source = source.resolve()
    run = json.loads((source / 'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'complete_candidates' or run.get('completed') != run.get('requested'):
        raise ValueError('来源运行未完整结束')
    resolution = run.get('resolution')
    if type(resolution) is not int or resolution <= 0:
        raise ValueError('来源分辨率无效')
    items = [json.loads(line) for line in (source / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(items) != run['completed'] or not items:
        raise ValueError('清单数量与运行记录不一致或清单为空')
    included, excluded, seen = [], [], set()
    for item in items:
        reason, rows = decide(item, resolution)
        image = child_file(source, item.get('image'))
        if image.suffix.lower() != '.png' or image.stem in seen:
            raise ValueError('图片必须为名称不重复的 PNG')
        seen.add(image.stem)
        record = {'source_image': item['image'], 'scenario': item.get('scenario'),
                  'source_group': item.get('source_group'), 'split': 'train'}
        if reason is not None:
            record['reason'] = reason
            excluded.append(record)
            continue
        label = child_file(source, item['fire_label'])
        if label.suffix.lower() != '.txt' or label.stem != image.stem:
            raise ValueError('图片与火焰标签名称不匹配')
        if label.read_text(encoding='utf-8').splitlines() != rows:
            raise ValueError(f'火焰标签与实例记录不一致: {item["fire_label"]}')
        record.update({'source_fire_label': item['fire_label'], 'fire_count': len(rows),
                       'image': f'images/train/{image.name}', 'label': f'labels/train/{label.name}'})
        included.append((image, label, record))
    return included, excluded


def _export_impl(source, output):
    source, output = source.resolve(), output.resolve()
    if not source.is_relative_to(FIRE_EXPORT_ROOT) or not output.is_relative_to(FIRE_EXPORT_ROOT):
        raise ValueError('来源和输出必须位于 datasets/exports/fire 中')
    if output.exists() or source == output or output.is_relative_to(source):
        raise ValueError('输出已存在或位于来源目录内')
    included, excluded = plan_source(source)
    if not included:
        raise ValueError('没有可导出的火焰候选')
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'source_run': source.relative_to(ROOT).as_posix(),
                'source_manifest': (source / 'manifest.jsonl').relative_to(ROOT).as_posix(),
                'included': len(included), 'excluded': len(excluded),
                'usage': 'unreviewed synthetic fire train candidates only; no name/OCR labels or independent evaluation',
                'class_id': 0, 'empty_label': 'source manifest reports no visible in-frame fire instance'}
    try:
        (output / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        for folder in ('images/train', 'labels/train'):
            (output / folder).mkdir(parents=True)
        for image, label, record in included:
            shutil.copyfile(image, output / record['image'])
            shutil.copyfile(label, output / record['label'])
        for filename, rows in [('manifest.jsonl', [r for _, _, r in included]), ('excluded.jsonl', excluded)]:
            (output / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        metadata['status'] = 'complete_candidates'
    except Exception:
        metadata['status'] = 'failed'
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(included), len(excluded)


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    included, excluded = plan_source(source)
    media = [image for image, _, _ in included]
    # Excluded inputs influence the output manifest too.
    media += [child_file(source, row['source_image']) for row in excluded]
    procedural = [source / 'run.json', source / 'manifest.jsonl'] + [label for _, label, _ in included]
    with generation_guard('fire.export_blender_candidates/' + output.relative_to(ROOT).as_posix(), media + procedural, [output],
                          {}, code_paths=[Path(__file__), Path(__file__).with_name('fire_blender_labels.py'),
                                          ROOT / 'datasets/scripts/common/annotations.py'],
                          procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
            return metadata['included'], metadata['excluded']
        result = _export_impl(source, output)
        metadata_path = output / 'run.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata['source_refs'] = run.source_refs
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


def main():
    parser = argparse.ArgumentParser(description='隔离导出 Blender 火焰 YOLO 待复核候选')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        included, excluded = export(args.source, args.output)
        print(f'完成：候选 {included} 张，排除 {excluded} 张；均未经过人工验收')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
