"""Create a bounded, unsplit set of auditable occlusion candidates."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import preview_person_occlusions as core


def write_manifest(output, config, samples, status, error=None):
    record = {'status': status, 'purpose': 'review candidates only; not validated ground truth',
              'source_group': config['group'], 'split': None,
              'source_plan': config['source_plan'], 'samples': samples}
    if error:
        record['error'] = error
    (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def build(selection, output):
    if output.exists():
        raise ValueError('输出目录已存在，请选择新目录')
    config = json.loads(selection.read_text(encoding='utf-8'))
    if config.get('split') is not None or not config.get('group'):
        raise ValueError('计划必须指定来源组且不得预先切分')
    people, vehicles = config['people'], config['vehicles']
    if not people or not vehicles or len(people) > 10 or len(vehicles) > 8:
        raise ValueError('人物或车辆素材数量超出本次有界范围')
    allowed = config.get('approved_source_videos')
    if not isinstance(allowed, list) or not allowed:
        raise ValueError('来源视频白名单缺失')
    review_manifest = core.workspace_path(config['review_manifest'])
    review = json.loads(review_manifest.read_text(encoding='utf-8'))
    review_root = review_manifest.parent
    source_frames = {}
    for source in review['sources']:
        source_video = core.workspace_path(source['source'])
        for frame in source['frames']:
            frame_path = (review_root / frame).resolve()
            try:
                frame_path.relative_to((review_root / 'images').resolve())
            except ValueError as exc:
                raise ValueError('源帧越出复核图像目录') from exc
            source_frames[frame_path] = source_video
    allowed_paths = {core.workspace_path(p) for p in allowed}
    prepared_people, prepared_vehicles = [], []
    for item in people:
        if item['class'] not in ('community', 'noncommunity'):
            raise ValueError('人物类别无效')
        path = core.workspace_path(item['path'])
        prepared_people.append((item, path, core.load_rgba(path)))
    for item in vehicles:
        if item['class'] not in ('upright', 'fallen'):
            raise ValueError('车辆类别无效')
        path = core.workspace_path(item['path'])
        video = core.workspace_path(item['source_video'])
        frame = core.workspace_path(item['source_frame'])
        if not video.is_file() or not frame.is_file():
            raise ValueError('车辆视频或源帧不存在')
        if video not in allowed_paths or source_frames.get(frame) != video:
            raise ValueError('车辆源帧与已批准视频不一致')
        prepared_vehicles.append((item, path, core.load_rgba(path)))
    config['source_plan'] = selection.relative_to(core.ROOT).as_posix()
    output.mkdir(parents=True)
    samples, sheets = [], []
    status, error = 'interrupted', '生成中断'
    write_manifest(output, config, samples, 'running')
    try:
        for person_item, person_path, person_rgba in prepared_people:
            for bike_item, bike_path, bike_rgba in prepared_vehicles:
                for variant in range(2):
                    number = len(samples) + 1
                    stem = f'candidate_{number:03d}'
                    canvas = np.full((core.HEIGHT, core.WIDTH, 3),
                                     (205 - variant * 12, 208 - variant * 8, 210 - variant * 4), np.uint8)
                    canvas[580:] = (150, 151, 151) if variant == 0 else (165, 160, 153)
                    floor = 665
                    person = core.resized(person_rgba, height=410 if variant == 0 else 440)
                    bike = core.resized(bike_rgba, width=330 if variant == 0 else 360)
                    if bike.shape[0] > 380:
                        bike = core.resized(bike_rgba, height=380)
                    person_x = 420 if variant == 0 else 475
                    bike_x = person_x + person.shape[1] // 2 - bike.shape[1] // 2 + (-45 if variant == 0 else 45)
                    person_full = core.mask_and_place(canvas, person, person_x, floor - person.shape[0])
                    vehicle = core.mask_and_place(canvas, bike, bike_x, floor - bike.shape[0])
                    visible = np.logical_and(person_full, np.logical_not(vehicle)).astype(np.uint8)
                    if not visible.sum() or visible.sum() == person_full.sum():
                        raise ValueError(f'{stem} 没有有效人物遮挡')
                    core.write_image(output / f'{stem}.png', canvas)
                    for label, mask in [('person_full', person_full), ('person_visible', visible), ('vehicle', vehicle)]:
                        core.write_image(output / f'{stem}_{label}.png', mask * 255)
                    pbox, vbox = core.box(visible), core.box(vehicle)
                    display = canvas.copy()
                    cv2.rectangle(display, tuple(pbox[:2]), (pbox[2] - 1, pbox[3] - 1), (0, 255, 0), 2)
                    cv2.rectangle(display, tuple(vbox[:2]), (vbox[2] - 1, vbox[3] - 1), (255, 80, 20), 2)
                    thumb = cv2.resize(display, (256, 192), interpolation=cv2.INTER_AREA)
                    cv2.putText(thumb, stem, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
                    sheets.append(thumb)
                    samples.append({
                        'image': f'{stem}.png', 'status': 'AI_candidate_not_human_verified',
                        'source_group': config['group'], 'split': None,
                        'person_class': person_item['class'], 'vehicle_class': bike_item['class'],
                        'person_source': person_path.relative_to(core.ROOT).as_posix(),
                        'vehicle_mask_source': bike_path.relative_to(core.ROOT).as_posix(),
                        'vehicle_frame': bike_item['source_frame'], 'vehicle_video': bike_item['source_video'],
                        'vehicle_visual_notes': bike_item.get('visual_notes', ''),
                        'person_full_mask': f'{stem}_person_full.png',
                        'person_visible_mask': f'{stem}_person_visible.png',
                        'vehicle_mask': f'{stem}_vehicle.png',
                        'person_full_bbox_xyxy': core.box(person_full),
                        'person_visible_bbox_xyxy': pbox, 'vehicle_bbox_xyxy': vbox,
                        'person_visible_fraction': int(visible.sum()) / int(person_full.sum()),
                        'bbox_convention': 'xyxy pixels; right and bottom exclusive',
                    })
        for page, first in enumerate(range(0, len(sheets), 12), 1):
            thumbnails = sheets[first:first + 12]
            while len(thumbnails) < 12:
                thumbnails.append(np.full((192, 256, 3), 230, np.uint8))
            sheet = np.vstack([np.hstack(thumbnails[i:i + 4]) for i in range(0, 12, 4)])
            core.write_image(output / f'review_{page:02d}.png', sheet)
        status, error = 'complete_candidates', None
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_manifest(output, config, samples, status, error)
    return samples


def main():
    parser = argparse.ArgumentParser(description='生成人车遮挡候选及可见掩码；不做训练/验证切分')
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        samples = build(core.workspace_path(args.selection), core.workspace_path(args.output))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'失败：{exc}\n')
    print(f'生成 {len(samples)} 张人车遮挡候选，尚未人工复核且未切分')


if __name__ == '__main__':
    main()
