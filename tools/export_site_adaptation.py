"""Add visually reviewed field samples to existing YOLO data splits."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.config import load_config, resolve
from engine.data import EXTENSIONS, dataset_header, inspect_dataset, resolve_sources


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def label_lines(boxes, width, height, classes, keypoints):
    lines = []
    for box in boxes:
        class_id = box['class_id']
        x1, y1, x2, y2 = box['xyxy']
        if type(class_id) is not int or not 0 <= class_id < classes:
            raise ValueError(f'无效类别：{class_id}')
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'边界框越界：{box}')
        values = [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
        points = box.get('keypoints', [])
        if len(points) != keypoints:
            raise ValueError(f'关键点数量须为 {keypoints}：{box}')
        for x, y, visible in points:
            if (visible not in (0, 1, 2) or not (0 <= x <= width and 0 <= y <= height)
                    or (visible == 0 and (x or y))):
                raise ValueError(f'无效关键点：{(x, y, visible)}')
            values.extend((x/width, y/height, visible))
        if not all(math.isfinite(value) for value in values):
            raise ValueError('标注须为有限数值')
        lines.append(str(class_id) + ' ' + ' '.join(f'{value:.8f}' for value in values))
    return ''.join(line + '\n' for line in lines)


def prepare(plan, heldout):
    config = load_config(ROOT, plan['config'])
    header = dataset_header(ROOT, config['data'])
    out = resolve(ROOT, plan['output'])
    exports = (ROOT / 'datasets/exports').resolve()
    if not out.is_relative_to(exports) or out == exports or out.exists():
        raise ValueError('输出必须是 datasets/exports 下的新目录')
    source_records = []
    seen = set()
    for row in plan['samples']:
        image = resolve(ROOT, row['image'])
        if image in heldout or image in seen:
            raise ValueError(f'评估留用或重复图片：{row["image"]}')
        if not image.is_relative_to(ROOT / 'datasets') or image.suffix.lower() not in EXTENSIONS:
            raise ValueError(f'需要工作空间 datasets 中的有效图像：{image}')
        boxes = row['boxes']
        status = 'visual_candidate' if boxes else 'visual_negative'
        if row['review_status'] != status or not row['source_group']:
            raise ValueError('样本须有来源及完整的目视候选状态')
        with Image.open(image) as pixels:
            pixels.load()
            width, height = pixels.size
        labels = label_lines(boxes, width, height, len(header['names']), header.get('kpt_shape', [0])[0])
        seen.add(image)
        source_records.append((row, image, labels))
    if not source_records:
        raise ValueError('补充集为空')
    split_sources = {
        split: [{**source, 'images': relative(source['images']), 'labels': relative(source['labels'])}
                for source in resolve_sources(ROOT, config, split)]
        for split in ('train', 'val', 'test')
    }
    return config, header, out, source_records, split_sources


def export(plan, heldout):
    base, header, out, samples, split_sources = prepare(plan, heldout)
    images, labels = out / 'images/train', out / 'labels/train'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rows = []
    try:
        for index, (row, source, line) in enumerate(samples, start=1):
            stem = f'field_{index:04d}_{source.stem}'
            target, label = images / (stem + source.suffix.lower()), labels / (stem + '.txt')
            shutil.copy2(source, target)
            label.write_text(line, encoding='utf-8')
            rows.append({**row, 'image': relative(target), 'label': relative(label), 'split': 'train',
                         'workspace_image': relative(source), 'annotation_method': 'visual_review_candidate'})
        data = out / 'data.yaml'
        data.write_text(yaml.safe_dump({'train': 'images/train', **header}, sort_keys=False), encoding='utf-8')
        baseline_manifest = resolve(ROOT, base['data']).parent / 'workspace_sources.json'
        baseline_rows = read_json(baseline_manifest)['records'] if baseline_manifest.is_file() else []
        write_json(out / 'workspace_sources.json', {'records': baseline_rows + rows})
        check = deepcopy(base)
        check.update(data=relative(data), version='site_adaptation_data_v1', data_overrides=split_sources)
        split_sources['train'].append({'images': relative(images), 'labels': relative(labels),
                                       'source_group': relative(out)})
        check['source_notes'] = [*base.get('source_notes', []),
                                 'Same-site visual candidates added to train. Existing val/test splits retained.',
                                 'The field diagnostic images are reserved, but physical objects may recur.']
        (out / 'dataset_check.yaml').write_text(yaml.safe_dump(check, sort_keys=False, allow_unicode=True), encoding='utf-8')
        _, splits, report = inspect_dataset(ROOT, check)
        if any(report['exact_path_overlap'].values()):
            raise ValueError(f'数据集分割路径重叠：{report}')
        observed = {key: sorted({cls for row in values for cls in row['classes']})
                    for key, values in splits.items()}
        write_json(out / 'summary.json', {'status': 'format_checked_visual_candidates',
                                         'base_config': plan['config'], 'samples_added': len(rows),
                                         'added_review_status': dict(Counter(row['review_status'] for row in rows)),
                                         'added_classes': dict(Counter(box['class_id'] for row in rows for box in row['boxes'])),
                                         'available_classes_by_split': observed, **report})
    except (Exception, KeyboardInterrupt) as exc:
        write_json(out / 'summary.json', {'status': 'failed_or_interrupted', 'error': str(exc),
                                         'samples_written': len(rows)})
        raise
    print(f'{base["model_key"]}: {len(rows)} 张实拍补充，{report["available_images"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--heldout', required=True)
    args = parser.parse_args()
    try:
        heldout = {resolve(ROOT, row['image']) for row in read_json(resolve(ROOT, args.heldout))}
        for plan in read_json(resolve(ROOT, args.plan)):
            export(plan, heldout)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
