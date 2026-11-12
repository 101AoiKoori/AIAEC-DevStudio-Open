#!/usr/bin/env python3
"""Read a calibrated image and save experimental needle overlay and JSON."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws' / 'src' / 'aec_vision' / 'src'))
from aec_vision.meter import annotate, detect_needle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True, help='Output image path')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    image = cv2.imread(args.image)
    if image is None:
        p.error('image could not be read')
    if list(image.shape[1::-1]) != config['image_size']:
        p.error('image size differs from calibration')
    results = {name: detect_needle(image, config[name]) for name in ('temperature', 'humidity')}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), annotate(image, results)):
        raise RuntimeError('could not write overlay')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
