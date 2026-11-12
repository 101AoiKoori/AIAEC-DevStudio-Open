"""Extract unverified SAM2 masks from the two existing flat-background orbits."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parents[5]
SOURCES = ROOT / 'datasets/annotated/e_bike/video_review_20261110/images'
WEIGHTS = ROOT / 'models/weights/e_bike/sam2_t.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', '0'), default='cpu')
    args = parser.parse_args()
    dest = (ROOT / args.output).resolve()
    if not dest.is_relative_to(ROOT / 'datasets/annotated/e_bike') or dest.exists():
        parser.error('output must be a new directory in datasets/annotated/e_bike')
    if not WEIGHTS.is_file():
        parser.error('local SAM2 weights not found')
    dest.mkdir(parents=True)
    items = []
    def save(status, error=None):
        payload = {'status': status, 'device': args.device, 'records': items}
        if error:
            payload['error'] = error
        (dest/'manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save('running')
    try:
        model = SAM(str(WEIGHTS))
        for folder, kind in (('source_001', 'fallen'), ('source_002', 'upright')):
            paths = sorted((SOURCES/folder).glob('*.jpg'))
            if not paths:
                raise ValueError(f'no source frames in {folder}')
            for path in paths:
                img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f'unreadable frame: {path}')
                h, w = img.shape[:2]
                scale = 1024/max(h, w)
                small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                sh, sw = small.shape[:2]
                region = [.025, .39 if kind == 'fallen' else .23, .975, .70]
                box = [region[0]*sw, region[1]*sh, region[2]*sw, region[3]*sh]
                result = model.predict(small, bboxes=[box], device=args.device, imgsz=1024, verbose=False)[0]
                if result.masks is None or len(result.masks.data) != 1:
                    raise ValueError(f'no unique mask: {path}')
                mask = result.masks.data[0].cpu().numpy().astype(np.uint8)*255
                mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
                points = cv2.findNonZero(mask)
                if points is None:
                    raise ValueError(f'empty mask: {path}')
                x, y, bw, bh = cv2.boundingRect(points)
                rgba = np.dstack((small, mask))[y:y+bh, x:x+bw]
                name = f'{folder}_{path.stem}.png'
                ok, encoded = cv2.imencode('.png', rgba)
                if not ok:
                    raise OSError(f'cannot encode {name}')
                encoded.tofile(str(dest/name))
                items.append({'source_frame': path.relative_to(ROOT).as_posix(), 'class': kind,
                              'path': (dest/name).relative_to(ROOT).as_posix(),
                              'status': 'auto_candidate_unreviewed', 'box_prompt_xyxy_normalized': region})
                save('running')
        save('complete_unreviewed')
        print(f'extracted {len(items)} unreviewed masks')
    except BaseException as exc:
        save('failed_or_interrupted', f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
