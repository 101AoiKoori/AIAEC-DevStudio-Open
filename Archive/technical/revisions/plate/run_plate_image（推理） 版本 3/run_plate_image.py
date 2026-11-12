#!/usr/bin/env python3
"""Run the shared plate code on one image or test the CNN on a plate crop."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PACKAGE = ROOT / 'ws/src/aec_vision'
sys.path.insert(0, str(PACKAGE / 'src'))

import cv2
import numpy as np
from aec_vision.plate import CNNReader, load_recognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--cnn-crop', action='store_true', help='Input is an existing crop; exercise only the CNN.')
    parser.add_argument('--device', default=None, help='Override the configured inference device.')
    args = parser.parse_args()
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('cannot read image: {}'.format(args.image))
    if args.cnn_crop:
        reader = CNNReader(PACKAGE / 'models/plate_ocr_best.pt', PACKAGE / 'config/plate/charset.json', args.device or 'cpu')
        text, score = reader.read(image)
        result = {'mode': 'cnn_only_test', 'cnn_text': text, 'cnn_confidence': score}
    else:
        recognizer = load_recognizer(PACKAGE, device=args.device)
        result = {'mode': 'dual_ocr', 'detections': recognizer.infer(image)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
