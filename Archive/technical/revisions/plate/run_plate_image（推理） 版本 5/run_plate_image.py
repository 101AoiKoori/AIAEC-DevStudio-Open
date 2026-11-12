#!/usr/bin/env python3
"""Run shared plate inference on images, optionally saving JSON, boxes and crops."""
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
    parser.add_argument('image', nargs='*')
    parser.add_argument('--province-test', action='store_true', help='Read provincial test.txt and run on corresponding full detection images.')
    parser.add_argument('--limit', type=int, help='Use the first N provincial test entries, without selecting by prediction.')
    parser.add_argument('--cnn-crop', action='store_true', help='Input is an existing crop; exercise only the CNN.')
    parser.add_argument('--device', default=None, help='Override the configured inference device.')
    parser.add_argument('--output-dir', type=Path, help='New output directory for JSON, padded boxes and OCR crops.')
    args = parser.parse_args()
    expected = {}
    if args.province_test:
        if args.image or args.cnn_crop:
            parser.error('--province-test uses full test images; do not combine with image paths or --cnn-crop')
        source = ROOT / 'datasets/exports/plate/provincial_ccpd'
        rows = [line.split('\t') for line in (source / 'ocr-recognition/test.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if args.limit is not None:
            if args.limit <= 0:
                parser.error('--limit must be positive')
            rows = rows[:args.limit]
        for relative, text in rows:
            path = source / 'detection/images/test' / Path(relative).name
            if not path.is_file():
                raise FileNotFoundError(path)
            expected[str(path)] = text
        args.image = list(expected)
    elif not args.image or args.limit is not None:
        parser.error('provide image paths, or use --province-test [--limit N]')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.cnn_crop:
        reader = CNNReader(PACKAGE / 'models/plate_ocr_best.pt', PACKAGE / 'config/plate/charset.json', args.device or 'cpu')
    else:
        recognizer = load_recognizer(PACKAGE, device=args.device)
    for index, filename in enumerate(args.image):
        image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('cannot read image: {}'.format(filename))
        if args.cnn_crop:
            text, score = reader.read(image)
            result = {'mode': 'cnn_only_test', 'cnn_text': text, 'cnn_confidence': score}
        else:
            result = {'mode': 'dual_ocr', 'detections': recognizer.infer(image)}
        result.update(image=str(Path(filename).resolve()), image_size=[image.shape[1], image.shape[0]])
        if args.province_test:
            result.update(dataset='provincial_ccpd_detection_test', expected_text=expected[filename],
                          selection='test.txt order', evaluated_count=len(args.image))
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        print(payload, flush=True)
        if args.output_dir:
            stem = '{}_{}'.format(index, Path(filename).stem)
            (args.output_dir / (stem + '.json')).write_text(payload + '\n', encoding='utf-8')
            marked = image.copy()
            for number, detection in enumerate(result.get('detections', [])):
                x1, y1, x2, y2 = detection['bbox']
                save_png(args.output_dir / '{}_crop_{}.png'.format(stem, number), image[y1:y2, x1:x2])
                cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 180, 0), 3)
                label = 'crop {} det={:.3f} {}'.format(number, detection['detector_confidence'], detection['ocr']['status'])
                cv2.putText(marked, label, (x1, max(24, y1-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2)
            save_png(args.output_dir / (stem + '_annotated.png'), marked)


def save_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('cannot encode output: {}'.format(path))
    encoded.tofile(str(path))


if __name__ == '__main__':
    main()
