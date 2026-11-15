"""Mark verified empty sticker footprints for synthetic fire images."""
import argparse
import json
from pathlib import Path

import cv2

from generate_fire_dataset import read_image
from prepare_fire_videos import ROOT, workspace_path
from lab.training_common.production_guard import generation_guard


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
        manifest_path = prepared / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        frames = []
        videos = set()
        for video in manifest:
            if video['split'] != 'train':
                continue
            video_id = video['video_id']
            if not video_id.startswith('video_') or not video_id[6:].isdigit():
                raise ValueError('视频清单格式错误')
            if video.get('source'):
                videos.add(workspace_path(video['source']))
            frames.extend(sorted((prepared / 'images' / 'train' / video_id).glob('frame_*.png')))
        with generation_guard('fire.annotate_fire_regions:' + output.relative_to(ROOT).as_posix(),
                              [manifest_path, *frames, *sorted(videos)], [output], {'prepared': prepared.relative_to(ROOT).as_posix()},
                              code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/generate_fire_dataset.py',
                                          ROOT / 'lab/features/fire/training/scripts/prepare_fire_videos.py'],
                              procedural_inputs=[manifest_path], root=ROOT) as run:
            if run.reused:
                print('已复用已有标定')
                return
            entries = []
            for frame in frames:
                print(f'\n{frame.parent.name} / {frame.name}')
                action = input('无火焰且贴面清晰，标定此帧？[y/回车/q]：').strip().lower()
                if action == 'q':
                    break
                if action != 'y':
                    continue
                image = read_image(frame)
                while input('添加一个贴纸落点？[y/回车]：').strip().lower() == 'y':
                    quad = pick_quad(image)
                    if quad:
                        image_path = frame.relative_to(ROOT).as_posix()
                        entries.append({'image': image_path, 'split': 'train', 'verified_empty': True,
                                        'quad': quad, 'source_references': run.source_refs[image_path]})
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
