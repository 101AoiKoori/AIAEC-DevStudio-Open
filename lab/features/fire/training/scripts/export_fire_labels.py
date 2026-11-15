"""Export reviewed frame annotations as OCR crops and YOLO fire labels."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/. Production remains paused.")

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard, plan_production

from generate_fire_dataset import read_image, write_image
from prepare_fire_videos import workspace_path


def box(value, width, height):
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError('框必须为 [x1,y1,x2,y2] 整数像素坐标')
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('标注框超出图像或面积为零')
    return x1, y1, x2, y2


def _export_impl(source, output, source_refs):
    records = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('标注清单为空')
    if output.exists():
        raise ValueError('输出目录已存在')
    prepared, seen = [], set()
    for item in records:
        if item.get('reviewed') is not True or item.get('split') not in ('train', 'val', 'test'):
            raise ValueError('每张图必须明确 split 并完成人工复核')
        path = workspace_path(item['image'])
        if path in seen:
            raise ValueError('图像重复出现')
        seen.add(path)
        image = read_image(path)
        h, w = image.shape[:2]
        if not isinstance(item.get('fires'), list) or not isinstance(item.get('names'), list):
            raise ValueError('fires 与 names 必须是列表；空列表表示已确认无目标')
        fires = [box(b, w, h) for b in item['fires']]
        names = []
        for name in item['names']:
            text = name.get('text')
            if not isinstance(text, str) or not text.strip() or '\t' in text or '\n' in text:
                raise ValueError('楼名文字不能为空或包含制表符和换行')
            names.append((box(name['box'], w, h), text))
        prepared.append((item, image, fires, names))
    output.mkdir(parents=True)
    ocr = {s: [] for s in ('train', 'val', 'test')}
    for index, (item, image, fires, names) in enumerate(prepared, 1):
        split = item['split']; name = f'frame_{index:06d}'; h, w = image.shape[:2]
        imgs = output / 'fire' / 'images' / split
        labels = output / 'fire' / 'labels' / split
        name_imgs = output / 'name_sign' / 'images' / split
        name_labels = output / 'name_sign' / 'labels' / split
        crops = output / 'ocr' / 'images' / split
        for folder in (imgs, labels, name_imgs, name_labels, crops):
            folder.mkdir(parents=True, exist_ok=True)
        write_image(imgs / (name + '.png'), image)
        write_image(name_imgs / (name + '.png'), image)
        lines = [f'0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}'
                 for x1, y1, x2, y2 in fires]
        (labels / (name + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        name_lines = [f'0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}'
                      for (x1, y1, x2, y2), _ in names]
        (name_labels / (name + '.txt')).write_text('\n'.join(name_lines) + ('\n' if name_lines else ''), encoding='utf-8')
        for j, ((x1, y1, x2, y2), text) in enumerate(names, 1):
            cropname = f'{name}_{j:02d}.png'
            write_image(crops / cropname, image[y1:y2, x1:x2])
            ocr[split].append(f'images/{split}/{cropname}\t{text}')
    for split, lines in ocr.items():
        (output / 'ocr' / (split + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    indexed = []
    for item in records:
        media_key = workspace_path(item['image']).resolve().relative_to(ROOT).as_posix()
        if media_key not in source_refs:
            raise ValueError(f'数据库来源引用缺失: {media_key}')
        indexed.append({**item, 'source_refs': source_refs[media_key]})
    (output / 'sources.json').write_text(json.dumps(indexed, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(prepared)


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    records = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('标注清单为空')
    media = [workspace_path(row['image']) for row in records]
    with generation_guard('fire.export_reviewed_labels/' + output.relative_to(ROOT).as_posix(), media + [source], [output], {},
                          code_paths=[Path(__file__), Path(__file__).with_name('generate_fire_dataset.py'),
                                      Path(__file__).with_name('prepare_fire_videos.py')],
                          procedural_inputs=[source], root=ROOT) as run:
        if run.reused:
            return len(records)
        for path in media:
            media_key = path.resolve().relative_to(ROOT).as_posix()
            if media_key not in run.source_refs:
                raise ValueError(f'数据库来源引用缺失: {media_key}')
        return _export_impl(source, output, run.source_refs)


def export_existing_candidates(source, output):
    """Copy existing candidate labels and images without upgrading their review status."""
    source, output = Path(source).resolve(), Path(output).resolve()
    allowed = (ROOT / 'datasets/exports/authoritative/fire_sticker').resolve()
    if not source.is_relative_to(allowed) or not output.is_relative_to(allowed):
        raise ValueError('candidate selection and output must be within the fire task export')
    selection = json.loads(source.read_text(encoding='utf-8'))
    records = selection.get('selected')
    if not isinstance(records, list) or not records:
        raise ValueError('candidate selection is empty')
    media, metadata, pairs = [], {source}, []
    seen = set()
    for item in records:
        if item.get('label_state') not in ('complete_positive', 'complete_negative',
                                           'candidate_complete_positive', 'candidate_complete_negative'):
            raise ValueError('partial or unknown target state cannot enter supervised candidates')
        image = workspace_path(item['image']).resolve()
        label = workspace_path(item['label']).resolve()
        manifest = workspace_path(item['manifest']).resolve()
        if image in seen or not image.is_file() or not label.is_file() or not manifest.is_file():
            raise ValueError('candidate image is duplicated or its source files are missing')
        seen.add(image)
        media.append(image)
        metadata.update((label, manifest))
        pairs.append((item, image, label))
    with generation_guard('fire.export_existing_candidates:' + output.relative_to(ROOT).as_posix(),
                          [*media, *sorted(metadata)], [output], {'selection_count': len(pairs)},
                          code_paths=[Path(__file__), Path(__file__).with_name('generate_fire_dataset.py'),
                                      Path(__file__).with_name('prepare_fire_videos.py')],
                          procedural_inputs=sorted(metadata), root=ROOT) as run:
        if run.reused:
            return len(pairs)
        if output.exists():
            raise ValueError('output already exists')
        (output / 'images/train').mkdir(parents=True)
        (output / 'labels/train').mkdir(parents=True)
        exported = []
        for index, (item, image, label) in enumerate(pairs, 1):
            stem = f'real_{index:06d}'
            image_out = output / 'images/train' / (stem + image.suffix.lower())
            label_out = output / 'labels/train' / (stem + '.txt')
            shutil.copyfile(image, image_out)
            shutil.copyfile(label, label_out)
            image_key = image.relative_to(ROOT).as_posix()
            exported.append({**item, 'output_image': image_out.relative_to(ROOT).as_posix(),
                             'output_label': label_out.relative_to(ROOT).as_posix(),
                             'source_refs': run.source_refs[image_key],
                             'current_split': 'train_candidate',
                             'status': 'source_annotation_candidate_not_accepted'})
        result = {'status': 'complete_candidates_not_accepted',
                  'scope': 'Existing real-image candidate annotations copied unchanged; '
                           'prior split and source review state retained; no human acceptance claimed.',
                  'records': exported}
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return len(exported)


def export_signal_negatives(source, output):
    """Create new candidate empty labels for four directly inspected signal frames."""
    source, output = Path(source).resolve(), Path(output).resolve()
    allowed_root = (ROOT / 'datasets/exports/authoritative/fire_sticker').resolve()
    if not source.is_relative_to(allowed_root) or not output.is_relative_to(allowed_root):
        raise ValueError('plan and output must be within the fire task export')
    plan = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(plan, dict) or set(plan) != {'schema_version', 'model_key', 'target_split', 'purpose',
            'budget', 'uncertainty_weight', 'independence_basis', 'constraints', 'strata'}:
        raise ValueError('unsupported signal-negative plan fields')
    if (plan.get('model_key') != 'fire_sticker' or plan.get('target_split') != 'train' or
            plan.get('constraints') != {'route_context': 'signal_without_building',
                'label_policy': 'no_recognizable_fire_in_visible_frame_candidate',
                'human_acceptance': 'not_confirmed'}):
        raise ValueError('unsupported signal-negative model, split or constraints')
    allowed_images = {
        'datasets/raw/car_real_capture_20261112/P01_TL1.png': 'car_real_capture_20261112',
        'datasets/raw/car_real_capture_20261112/P12_TL2.png': 'car_real_capture_20261112',
        'datasets/raw/site_capture_20261112/onboard_camera_images/vehicle_0043.png': 'site_capture_20261112_vehicle_route',
        'datasets/raw/site_capture_20261112/onboard_camera_images/vehicle_0088.png': 'site_capture_20261112_vehicle_route',
    }
    states = plan.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('signal-negative states are required')
    images, seen = [], set()
    for state in states:
        if not isinstance(state, dict) or set(state) != {'id', 'failure_event', 'min_count', 'max_count',
                'prior', 'exposure_weight', 'failure_cost', 'conditions', 'background', 'target',
                'labels', 'augmentation', 'sources', 'parameters'}:
            raise ValueError('unsupported signal-negative state fields')
        params = state.get('parameters')
        image = params.get('image') if isinstance(params, dict) else None
        group = allowed_images.get(image)
        if not group or image in seen:
            raise ValueError('source was not inspected for this signal-only candidate or is duplicated')
        seen.add(image)
        expected = {'id': 'signal_' + Path(image).stem, 'failure_event': 'unrecognized_fire_or_context_uncertainty',
                    'conditions': ['traffic_signal', 'no_visible_building', 'no_recognizable_fire'],
                    'background': {'image': image, 'route': 'signal_without_building'},
                    'target': {'class': 'fire_sticker', 'visible_count': 0},
                    'labels': {'format': 'yolo_v8', 'class_id': 0, 'status': 'ai_visual_candidate_not_human_accepted'},
                    'augmentation': {'method': 'none_original_capture'}, 'sources': [group],
                    'parameters': {'image': image}}
        if any(state.get(k) != v for k, v in expected.items()) or state.get('min_count') != 1 or state.get('max_count') != 1:
            raise ValueError('signal-negative state does not match the inspected source and label policy')
        path = workspace_path(image).resolve()
        if not path.is_file():
            raise ValueError('signal source missing: ' + image)
        images.append(path)
    planned = plan_production(plan, ROOT)
    with generation_guard('fire.export_signal_negatives:' + output.relative_to(ROOT).as_posix(),
                          [*images, source], [output], {'annotation': 'new_empty_yolo_candidate'},
                          code_paths=[Path(__file__), Path(__file__).with_name('generate_fire_dataset.py'),
                                      Path(__file__).with_name('prepare_fire_videos.py'),
                                      ROOT / 'lab/training_common/bayesian_allocation.py'],
                          procedural_inputs=[source], root=ROOT, production_plan=plan,
                          sample_output_paths=[output / 'images/train']) as run:
        if run.plan != planned:
            raise ValueError('guard plan differs from pre-generation plan')
        if run.reused:
            return planned['planned_total']
        if output.exists():
            raise ValueError('output already exists')
        (output / 'images/train').mkdir(parents=True)
        (output / 'labels/train').mkdir(parents=True)
        records = []
        for state in run.plan['strata']:
            if state['planned_count'] != 1:
                raise ValueError('each inspected signal source is required exactly once')
            image = workspace_path(state['parameters']['image']).resolve()
            stem = state['id']
            image_out = output / 'images/train' / (stem + image.suffix.lower())
            label_out = output / 'labels/train' / (stem + '.txt')
            shutil.copyfile(image, image_out)
            label_out.write_text('', encoding='utf-8')
            record = {'image': image_out.relative_to(ROOT).as_posix(),
                      'label': label_out.relative_to(ROOT).as_posix(),
                      'source_image': image.relative_to(ROOT).as_posix(),
                      'source_group': state['sources'][0], 'generation_method': 'camera_capture',
                      'annotation_method': 'Codex visual inspection of original image, 2026-11-14',
                      'annotation_state': 'candidate_complete_negative_not_human_accepted',
                      'scope': 'no recognizable fire or building in visible frame; independent capture not established',
                      'source_refs': run.source_refs[image.relative_to(ROOT).as_posix()]}
            records.append(record)
            run.record_sample(state['id'], output=image_out, sources=state['sources'],
                              parameters=state['parameters'], actual={'source_image': record['source_image'],
                                  'label': record['label'], 'visible_fire_count': 0,
                                  'annotation_state': record['annotation_state']})
        (output / 'manifest.json').write_text(json.dumps({'status': 'complete_candidates_not_accepted',
            'records': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return len(records)


def main():
    parser = argparse.ArgumentParser(description='导出人工复核的火焰框和楼名裁剪')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--annotations')
    inputs.add_argument('--candidate-selection')
    inputs.add_argument('--signal-negative-plan')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.signal_negative_plan:
            n = export_signal_negatives(workspace_path(args.signal_negative_plan), workspace_path(args.output))
            print(f'新建 {n} 帧信号场景空标签候选；不代表人工验收')
        elif args.candidate_selection:
            n = export_existing_candidates(workspace_path(args.candidate_selection), workspace_path(args.output))
            print(f'导出 {n} 帧既有候选；不代表人工验收')
        else:
            n = export(workspace_path(args.annotations), workspace_path(args.output))
            print(f'导出 {n} 帧人工复核数据')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
