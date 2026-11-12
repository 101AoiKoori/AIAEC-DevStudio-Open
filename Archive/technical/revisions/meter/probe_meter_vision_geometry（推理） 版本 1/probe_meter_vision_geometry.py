#!/usr/bin/env python3
"""Probe an independently obtained meter-vision checkout with synthetic points."""
import argparse
import importlib.util
import math
from pathlib import Path


def load_reader(checkout):
    path = Path(checkout) / 'backend' / 'gauge_reader.py'
    spec = importlib.util.spec_from_file_location('meter_vision_gauge_reader', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load {}'.format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_gauge_reading


def probe(reader, horizontal_scale):
    keypoints = [{'kp_id': 0, 'x': 0, 'y': 0, 'conf': 1.0}]

    def add(keypoint_id, degrees, radius):
        angle = math.radians(degrees)
        keypoints.append({'kp_id': keypoint_id,
                          'x': horizontal_scale * radius * math.cos(angle),
                          'y': radius * math.sin(angle), 'conf': 1.0})

    add(1, 55, 0.5)
    add(2, 55, 1.0)
    for index in range(7):
        add(index + 3, 10 + index * 20, 1.0)
    return reader({'people': [{'keypoints': keypoints}]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', help='Path to an independent meter-vision checkout')
    args = parser.parse_args()
    reader = load_reader(args.checkout)
    for scale in (1.0, 0.4):
        result = probe(reader, scale)
        print('horizontal_scale={}, expected=2.25, result={}'.format(scale, result))


if __name__ == '__main__':
    main()
