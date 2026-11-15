"""Create a traceable thicker-tick derivative of the calibrated station face."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

SOURCE = ROOT / 'datasets/assets/blender/meter/v4_parametric_station_v2_20261113/face.png'
OUT = ROOT / 'runs/meter/v4_tick_face_r4_20261114'
PARAMS = {'threshold': 64, 'kernel': 'ellipse', 'radius_pixels': 4,
          'method': 'dilate dark tick mask; retain original pixels outside mask',
          'coordinate_transform': 'identity'}

def main():
    with generation_guard('meter.derive_tick_face_v4:' + OUT.relative_to(ROOT).as_posix(),
                          [SOURCE], [OUT], PARAMS, code_paths=[Path(__file__)]) as run:
        if run.reused:
            print('REUSED', OUT)
            return
        if OUT.exists():
            raise ValueError('output exists')
        src = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
        if src is None or src.shape != (1254, 1254, 3):
            raise ValueError('unexpected source face')
        mask = np.all(src < PARAMS['threshold'], axis=2).astype(np.uint8)
        if not mask.any():
            raise ValueError('no dark ticks found')
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        result = src.copy()
        result[mask != 0] = 9
        OUT.mkdir(parents=True)
        if not cv2.imwrite(str(OUT / 'face.png'), result):
            raise RuntimeError('face write failed')
        (OUT / 'manifest.json').write_text(json.dumps({'source': SOURCE.relative_to(ROOT).as_posix(),
            'output': 'face.png', 'parameters': PARAMS, 'semantics': 'synthetic tick positions unchanged; no numeric readings added'},
            ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        run.set_result({'output': (OUT / 'face.png').relative_to(ROOT).as_posix(), **PARAMS})

if __name__ == '__main__':
    main()
