#!/usr/bin/env python3
"""Prepare PDF-derived face texture and invoke Blender for a small experiment.

Output must be a new directory. PDF photo, inpainted texture, model and renders
are workspace data, not files to commit. Requires Python OpenCV and Blender.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', required=True, help='Blender executable')
    parser.add_argument('--source', required=True, help='Extracted PDF image 005')
    parser.add_argument('--ticks', required=True, help='PDF estimated tick configuration')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=9011)
    args = parser.parse_args()
    if args.count < 1 or args.count > 100:
        parser.error('count must be between 1 and 100')
    source = cv2.imread(args.source)
    if source is None or source.shape[:2] != (672, 641):
        parser.error('source must be the 641 x 672 PDF photograph')
    if not Path(args.blender).is_file() or not Path(args.ticks).is_file():
        parser.error('Blender executable or tick file does not exist')
    out = Path(args.output).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('output directory already exists; refusing to overwrite')
    # Original hands and pivots are baked into the PDF photo. Inpainting removes
    # them approximately; tiny overlapped scale marks can be damaged.
    mask = np.zeros(source.shape[:2], np.uint8)
    cv2.line(mask, (296, 395), (435, 135), 255, 25)
    cv2.circle(mask, (320, 355), 27, 255, -1)
    cv2.line(mask, (364, 478), (230, 550), 255, 21)
    cv2.circle(mask, (325, 497), 25, 255, -1)
    repaired = cv2.inpaint(source, mask, 9, cv2.INPAINT_TELEA)
    texture = repaired[35:635, 20:620]
    if not cv2.imwrite(str(out / 'face_inpainted.png'), texture):
        raise RuntimeError('failed to write texture')
    script = Path(__file__).with_name('render_blender_9011.py')
    command = [str(Path(args.blender).resolve()), '--background', '--factory-startup',
               '--python-exit-code', '1', '--python', str(script), '--',
               '--output', str(out), '--ticks', str(Path(args.ticks).resolve()),
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
