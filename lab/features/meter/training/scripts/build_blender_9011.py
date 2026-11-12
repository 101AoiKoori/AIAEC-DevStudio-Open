#!/usr/bin/env python3
"""Use the selected face texture and invoke Blender for a small experiment.

Output must be a new directory. PDF photo, inpainted texture, model and renders
are workspace data, not files to commit. Requires Python OpenCV and Blender.
"""
import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', required=True, help='Blender executable')
    parser.add_argument('--face-texture', required=True, help='User-selected square dial texture')
    parser.add_argument('--geometry', required=True, help='Approximate geometry for this texture')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=9011)
    args = parser.parse_args()
    if args.count < 1 or args.count > 100:
        parser.error('count must be between 1 and 100')
    source = cv2.imread(args.face_texture)
    if source is None or source.shape[:2] != (1254, 1254):
        parser.error('face texture must be the selected 1254 x 1254 image')
    if not Path(args.blender).is_file() or not Path(args.geometry).is_file():
        parser.error('Blender executable or geometry file does not exist')
    out = Path(args.output).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('output directory already exists; refusing to overwrite')
    shutil.copyfile(args.face_texture, out / 'face_texture.png')
    script = Path(__file__).with_name('render_blender_9011.py')
    command = [str(Path(args.blender).resolve()), '--background', '--factory-startup',
               '--python-exit-code', '1', '--python', str(script), '--',
               '--output', str(out), '--ticks', str(Path(args.geometry).resolve()),
               '--count', str(args.count), '--seed', str(args.seed)]
    print('Running Blender in a separate background process...', flush=True)
    subprocess.run(command, check=True)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    preview_dir = out / 'previews'
    preview_dir.mkdir()
    for item in manifest['samples'][:8]:
        name = item['id']
        image = cv2.imread(str(out / 'images' / 'train' / (name + '.png')))
        if image is None:
            raise RuntimeError('cannot read rendered sample ' + name)
        for i, ((x, y, _), state) in enumerate(zip(item['keypoints'], item['visibility'])):
            if state:
                p = (int(round(x * image.shape[1])), int(round(y * image.shape[0])))
                cv2.circle(image, p, 4, (0, 255, 0) if state == 2 else (0, 165, 255), -1)
                cv2.putText(image, str(i), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 0, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(preview_dir / (name + '.png')), image):
            raise RuntimeError('cannot write preview ' + name)
    print('Experimental model and data:', out)


if __name__ == '__main__':
    main()
