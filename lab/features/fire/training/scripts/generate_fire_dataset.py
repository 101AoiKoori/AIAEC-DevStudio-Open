"""Generate fire-sticker training images from verified, manually marked floor bands."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_fire_videos import ROOT, workspace_path


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片：{path}')
    return image


def write_image(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'无法编码图片：{path}')
    data.tofile(str(path))


def generate(source, sticker, output, count, seed):
    if output.exists():
        raise ValueError('输出目录已存在，请选择新目录')
    if not 1 <= count <= 10000:
        raise ValueError('生成数量必须在 1～10000 之间')
    entries = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(entries, list) or not entries:
        raise ValueError('需要至少一个经确认的无火焰楼层区域')
    candidates = {}
    for item in entries:
        if item.get('split') != 'train' or item.get('verified_empty') is not True:
            raise ValueError('每个背景必须为训练集且人工确认无火焰')
        image_path = workspace_path(item['image'])
        im = read_image(image_path)
        quad = np.asarray(item['quad'], dtype=np.float32)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError('quad 必须是四个有限的像素坐标')
        h, w = im.shape[:2]
        if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= w) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= h):
            raise ValueError('区域坐标超出图片')
        if not cv2.isContourConvex(quad.astype(np.float32)) or abs(cv2.contourArea(quad)) < 100:
            raise ValueError('区域须为顺序排列的有效凸四边形')
        key = str(image_path)
        if key not in candidates:
            candidates[key] = {'image': im, 'source': item['image'], 'quads': []}
        candidates[key]['quads'].append(quad)
    candidates = list(candidates.values())
    icon = read_image(sticker)
    # The reference image contains the printed icon and white background.
    # Paper is rendered at the confirmed 2.5:3.5 aspect ratio.
    paper = np.full((140, 100, 3), 255, np.uint8)
    icon = cv2.resize(icon, (88, 88), interpolation=cv2.INTER_AREA)
    paper[26:114, 6:94] = icon
    rng = np.random.default_rng(seed)
    (output / 'images' / 'train').mkdir(parents=True)
    (output / 'labels' / 'train').mkdir(parents=True)
    records = []
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    for i in range(count):
        candidate = candidates[int(rng.integers(len(candidates)))]
        background = candidate['image']
        h, w = background.shape[:2]
        result = background.copy()
        boxes, projected = [], []
        quads = candidate['quads']
        number = int(rng.integers(0, min(3, len(quads)) + 1))
        for chosen in rng.choice(len(quads), size=number, replace=False):
            quad = quads[int(chosen)]
            # Each annotated quadrilateral describes one allowed sticker footprint.
            x, y = rng.uniform(0.0, 0.12, 2)
            s = float(rng.uniform(0.78, 0.88))
            corners = np.array([[x, y], [x+s, y], [x+s, y+s], [x, y+s]], np.float32)
            projection = cv2.getPerspectiveTransform(unit, quad)
            target = cv2.perspectiveTransform(corners.reshape(1, -1, 2), projection)[0]
            ph, pw = paper.shape[:2]
            src = np.float32([[0, 0], [pw-1, 0], [pw-1, ph-1], [0, ph-1]])
            matrix = cv2.getPerspectiveTransform(src, target)
            warped = cv2.warpPerspective(paper, matrix, (w, h))
            mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), matrix, (w, h))
            ys, xs = np.where(mask > 0)
            if not len(xs):
                raise ValueError('投影为空')
            if any(np.any(mask[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes):
                raise ValueError('标注区域相互重叠，请调整后重新生成')
            result[mask > 0] = warped[mask > 0]
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
            projected.append(target.tolist())
        name = f'fire_{i+1:06d}'
        write_image(output / 'images' / 'train' / (name + '.png'), result)
        lines = [f'0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}'
                 for x1, y1, x2, y2 in boxes]
        (output / 'labels' / 'train' / (name + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        records.append({'image': name + '.png', 'source': candidate['source'],
                        'projected': projected, 'seed': seed, 'review': 'pending'})
    (output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(records)


def main():
    parser = argparse.ArgumentParser(description='基于人工确认的无火楼层区域生成火焰训练候选')
    parser.add_argument('--regions', required=True, help='人工确认区域 JSON')
    parser.add_argument('--sticker', default='datasets/raw/fire/team_assets/image1.png')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        n = generate(workspace_path(args.regions), workspace_path(args.sticker), workspace_path(args.output), args.count, args.seed)
        print(f'生成 {n} 张训练候选；须人工复核，真实验证集需单独标注')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
