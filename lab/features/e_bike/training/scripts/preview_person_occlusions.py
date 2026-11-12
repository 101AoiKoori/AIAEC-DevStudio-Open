"""Create inspectable person/e-bike occlusion previews, not a training dataset."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HEIGHT, WIDTH = 768, 1024


def workspace_path(value):
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('路径必须位于工作空间内') from exc
    return path


def load_rgba(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f'图片必须是带透明通道的 PNG：{path}')
    pixels = np.argwhere(image[:, :, 3] > 127)
    if not len(pixels):
        raise ValueError(f'透明素材没有前景：{path}')
    y1, x1 = pixels.min(axis=0)
    y2, x2 = pixels.max(axis=0) + 1
    return image[y1:y2, x1:x2]


def resized(image, width=None, height=None):
    h, w = image.shape[:2]
    scale = width / w if width is not None else height / h
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def mask_and_place(canvas, rgba, x, y):
    h, w = rgba.shape[:2]
    if x < 0 or y < 0 or x + w > WIDTH or y + h > HEIGHT:
        raise ValueError('前景放置超出画布')
    opacity = rgba[:, :, 3:4].astype(np.float32) / 255
    target = canvas[y:y + h, x:x + w]
    target[:] = (rgba[:, :, :3] * opacity + target * (1 - opacity)).astype(np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[y:y + h, x:x + w] = (rgba[:, :, 3] > 127).astype(np.uint8)
    return mask


def box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('可见掩码为空')
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def write_image(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'无法编码图片：{path}')
    data.tofile(str(path))


def preview(community, noncommunity, upright, fallen, output):
    if output.exists():
        raise ValueError('输出目录已存在，请选择新目录')
    people = [('community', community, load_rgba(community)),
              ('noncommunity', noncommunity, load_rgba(noncommunity))]
    bikes = [('upright', upright, load_rgba(upright)), ('fallen', fallen, load_rgba(fallen))]
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text(json.dumps({'status': 'running', 'samples': []}), encoding='utf-8')
    records, thumbs = [], []
    status, error = 'interrupted', '生成中断'
    try:
        for person_name, person_path, person_rgba in people:
            for bike_name, bike_path, bike_rgba in bikes:
                idx = len(records) + 1
                canvas = np.full((HEIGHT, WIDTH, 3), (208, 211, 211), np.uint8)
                canvas[580:] = (150, 151, 151)
                person = resized(person_rgba, height=420)
                bike = resized(bike_rgba, width=350)
                floor = 660
                person_xy = (440, floor - person.shape[0])
                bike_xy = (330 if bike_name == 'upright' else 355, floor - bike.shape[0])
                full = mask_and_place(canvas, person, *person_xy)
                vehicle = mask_and_place(canvas, bike, *bike_xy)
                visible = np.logical_and(full, np.logical_not(vehicle)).astype(np.uint8)
                visible_count, full_count = int(visible.sum()), int(full.sum())
                if not visible_count or visible_count == full_count:
                    raise ValueError('样例未形成可见的人物遮挡')
                stem = f'preview_{idx:02d}'
                write_image(output / f'{stem}.png', canvas)
                for suffix, mask in [('person_full', full), ('person_visible', visible), ('vehicle', vehicle)]:
                    write_image(output / f'{stem}_{suffix}.png', mask * 255)
                pbox, vbox = box(visible), box(vehicle)
                annotated = canvas.copy()
                cv2.rectangle(annotated, tuple(pbox[:2]), (pbox[2] - 1, pbox[3] - 1), (0, 255, 0), 2)
                cv2.rectangle(annotated, tuple(vbox[:2]), (vbox[2] - 1, vbox[3] - 1), (255, 80, 20), 2)
                thumbs.append(cv2.resize(annotated, (512, 384), interpolation=cv2.INTER_AREA))
                records.append({'image': f'{stem}.png', 'person_class': person_name, 'vehicle_class': bike_name,
                                'person_source': person_path.relative_to(ROOT).as_posix(),
                                'vehicle_source': bike_path.relative_to(ROOT).as_posix(),
                                'person_full_mask': f'{stem}_person_full.png',
                                'person_visible_mask': f'{stem}_person_visible.png',
                                'vehicle_mask': f'{stem}_vehicle.png',
                                'person_full_bbox_xyxy': box(full), 'person_visible_bbox_xyxy': pbox,
                                'vehicle_bbox_xyxy': vbox,
                                'person_visible_fraction': visible_count / full_count,
                                'bbox_convention': 'xyxy pixel coordinates, right and bottom exclusive',
                                'status': 'candidate_not_ground_truth'})
        write_image(output / 'comparison.png', np.vstack([np.hstack(thumbs[:2]), np.hstack(thumbs[2:])]))
        status = 'complete_preview'
        error = None
    except Exception as exc:
        status = 'failed'
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest = {'status': status, 'purpose': 'visual and mask audit only; not training or evaluation data',
                    'background': 'programmatically generated, not a real scene',
                    'limitation': 'vehicle masks are unreviewed SAM2 candidates with known defects',
                    'split': None, 'samples': records}
        if error:
            manifest['error'] = error
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def main():
    parser = argparse.ArgumentParser(description='人物和电动车遮挡的少量候选预览')
    parser.add_argument('--community', required=True)
    parser.add_argument('--noncommunity', required=True)
    parser.add_argument('--upright', required=True)
    parser.add_argument('--fallen', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        sources = [workspace_path(p) for p in (args.community, args.noncommunity, args.upright, args.fallen)]
        result = preview(*sources, workspace_path(args.output))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'失败：{exc}\n')
    print(f'生成 {len(result)} 张遮挡预览及对应掩码；不是训练集')


if __name__ == '__main__':
    main()
