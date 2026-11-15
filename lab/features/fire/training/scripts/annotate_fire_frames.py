"""Interactively annotate selected extracted frames with OpenCV selection boxes."""
import argparse
import json
from pathlib import Path

import cv2

from generate_fire_dataset import read_image
from prepare_fire_videos import ROOT, workspace_path
from lab.training_common.production_guard import generation_guard


def select_box(image, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    return [int(x), int(y), int(x + w), int(y + h)] if w and h else None


def annotate(prepared, output):
    if output.exists():
        raise ValueError('标注文件已存在，请选择新文件名')
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    entries = []
    for video in manifest:
        split, video_id = video['split'], video['video_id']
        if split not in ('train', 'val', 'test') or not video_id.startswith('video_') or not video_id[6:].isdigit():
            raise ValueError('视频清单格式错误')
        for frame in sorted((prepared / 'images' / split / video_id).glob('frame_*.png')):
            print(f'\n{split} / {video_id} / {frame.name}')
            action = input('标注此帧 [y]，跳过 [回车]，结束 [q]：').strip().lower()
            if action == 'q':
                return entries
            if action != 'y':
                continue
            image = read_image(frame)
            fires, names = [], []
            while input('添加火焰框？[y/回车]：').strip().lower() == 'y':
                rect = select_box(image, 'Select fire, ENTER confirms')
                if rect:
                    fires.append(rect)
            while input('添加楼名框？[y/回车]：').strip().lower() == 'y':
                rect = select_box(image, 'Select name, ENTER confirms')
                if rect:
                    text = input('输入楼名纸条完整文字：').strip()
                    if not text:
                        raise ValueError('楼名不能为空')
                    names.append({'box': rect, 'text': text})
            print(f'火焰 {len(fires)} 个，楼名 {len(names)} 个。请确认目标全部可判读并已标完。')
            if input('确认完整并保存？[y/回车]：').strip().lower() == 'y':
                entries.append({'image': frame.relative_to(ROOT).as_posix(), 'split': split,
                                'reviewed': True, 'fires': fires, 'names': names})
    return entries


def main():
    parser = argparse.ArgumentParser(description='人工复核框选火焰及楼名纸条')
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        prepared, output = workspace_path(args.prepared), workspace_path(args.output)
        manifest_path = prepared / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for video in manifest:
            if (video['split'] not in ('train', 'val', 'test') or
                    not video['video_id'].startswith('video_') or not video['video_id'][6:].isdigit()):
                raise ValueError('视频清单格式错误')
        frames = [frame for video in manifest
                  for frame in sorted((prepared / 'images' / video['split'] / video['video_id']).glob('frame_*.png'))]
        videos = sorted({workspace_path(video['source']) for video in manifest if video.get('source')})
        with generation_guard('fire.annotate_fire_frames:' + output.relative_to(ROOT).as_posix(),
                              [manifest_path, *frames, *videos], [output], {'prepared': prepared.relative_to(ROOT).as_posix()},
                              code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/generate_fire_dataset.py',
                                          ROOT / 'lab/features/fire/training/scripts/prepare_fire_videos.py'],
                              procedural_inputs=[manifest_path], root=ROOT) as run:
            if run.reused:
                print('已复用已有标注')
                return
            entries = annotate(prepared, output)
            if not entries:
                raise ValueError('没有确认的标注')
            for entry in entries:
                entry['source_references'] = run.source_refs[entry['image']]
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as stream:
                json.dump(entries, stream, ensure_ascii=False, indent=2)
        print(f'已保存 {len(entries)} 帧的标注')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'失败：{exc}\n')


if __name__ == '__main__':
    main()
