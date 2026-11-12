"""Export reviewed frame annotations as OCR crops and YOLO fire labels."""
import argparse
import json

from generate_fire_dataset import read_image, write_image
from prepare_fire_videos import workspace_path


def box(value, width, height):
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError('框必须为 [x1,y1,x2,y2] 整数像素坐标')
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('标注框超出图像或面积为零')
    return x1, y1, x2, y2


def export(source, output):
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
    (output / 'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(prepared)


def main():
    parser = argparse.ArgumentParser(description='导出人工复核的火焰框和楼名裁剪')
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        n = export(workspace_path(args.annotations), workspace_path(args.output))
        print(f'导出 {n} 帧人工复核数据')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
