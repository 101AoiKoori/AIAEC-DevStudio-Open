"""Mark verified empty sticker footprints for synthetic fire images."""
import argparse
import json

import cv2

from generate_fire_dataset import read_image
from prepare_fire_videos import ROOT, workspace_path


def pick_quad(image):
    title = 'Click upper left, upper right, lower right, lower left; ENTER confirms'
    points = []
    display = image.copy()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow(title, display)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click)
    cv2.imshow(title, display)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10) and len(points) == 4:
            break
        if key == 27:
            points = []
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            points = []
            break
    cv2.destroyWindow(title)
    return points


def main():
    parser = argparse.ArgumentParser(description='标定无火焰训练帧上的贴纸落点')
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        prepared, output = workspace_path(args.prepared), workspace_path(args.output)
        if output.exists():
            raise ValueError('输出文件已存在，请选择新文件名')
        manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
        entries = []
        action = ''
        for video in manifest:
            video_id = video['video_id']
            if video['split'] != 'train':
                continue
            if not video_id.startswith('video_') or not video_id[6:].isdigit():
                raise ValueError('视频清单格式错误')
            for frame in sorted((prepared / 'images' / 'train' / video_id).glob('frame_*.png')):
                print(f'\n{video_id} / {frame.name}')
                action = input('无火焰且贴面清晰，标定此帧？[y/回车/q]：').strip().lower()
                if action == 'q':
                    break
                if action != 'y':
                    continue
                image = read_image(frame)
                while input('添加一个贴纸落点？[y/回车]：').strip().lower() == 'y':
                    quad = pick_quad(image)
                    if quad:
                        entries.append({'image': frame.relative_to(ROOT).as_posix(),
                                        'split': 'train', 'verified_empty': True, 'quad': quad})
            if action == 'q':
                break
        if not entries:
            raise ValueError('没有标定区域')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(entries, stream, ensure_ascii=False, indent=2)
        print(f'已保存 {len(entries)} 个区域；请核对后生成')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
