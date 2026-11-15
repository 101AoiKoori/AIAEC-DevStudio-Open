"""Create a source-artwork contact sheet at realistic paper-side pixel sizes."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import composite_bin_real_scene as base

sys.path.insert(0, str(base.ROOT))
from lab.training_common.production_guard import generation_guard


def make_paper(icon, size):
    rgb, alpha = icon
    paper = np.full((size, size, 3), 237, dtype=np.uint8)
    margin = max(1, round(size * .12))
    inner = size - 2*margin
    scale = min(inner / rgb.shape[1], inner / rgb.shape[0])
    width, height = max(1, round(rgb.shape[1]*scale)), max(1, round(rgb.shape[0]*scale))
    ink = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    ink = np.clip(237.0 + (ink.astype(np.float32)-237.0)*1.35, 0, 255)
    mask = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)[:, :, None]
    x, y = (size-width)//2, (size-height)//2
    paper[y:y+height, x:x+width] = np.clip(ink*mask + paper[y:y+height, x:x+width]*(1-mask), 0, 255)
    return paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = (base.ROOT / args.out).resolve()
    if not path.is_relative_to((base.ROOT/'runs').resolve()):
        parser.error('out must be under runs')
    config = yaml.safe_load(base.CLASSES.read_text(encoding='utf-8'))
    names = [item['name'] for item in config['items']]
    inputs = [base.CACHE / (name + '.npz') for name in names]
    procedural = [base.CLASSES, base.ROOT / 'config/perception_tasks.json']
    with generation_guard('review_sticker_sources:' + path.relative_to(base.ROOT).as_posix(),
                          inputs, [path], {}, code_paths=[__file__, base.__file__],
                          procedural_inputs=procedural) as run:
        if run.reused:
            return
        _render(path)


def _render(path):
    names, icons, _, _ = base.load_icons()
    ids = (0, 2, 11, 13, 14, 19, 20, 23)
    sizes = (28, 40, 56, 80)
    sheet = np.full((len(ids)*115, len(sizes)*125, 3), 170, dtype=np.uint8)
    for row, class_id in enumerate(ids):
        for col, size in enumerate(sizes):
            x, y = col*125+20, row*115+24
            sheet[y:y+size, x:x+size] = make_paper(icons[class_id], size)
            cv2.putText(sheet, f'{class_id}: {size}px', (col*125+5,row*115+15),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1)
        print(class_id, names[class_id])
    ok, encoded = cv2.imencode('.png', sheet)
    if not ok:
        raise OSError('failed to encode contact sheet')
    encoded.tofile(str(path))


if __name__ == '__main__':
    main()
