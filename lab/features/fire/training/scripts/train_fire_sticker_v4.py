"""Fire-sticker training and target-size audit for the reviewed V4 export."""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
PACKAGED = (PACKAGE_ROOT / 'scripts/common/engine').is_dir()
ROOT = PACKAGE_ROOT if PACKAGED else HERE.parents[5]
COMMON = ROOT / 'scripts/common' if PACKAGED else ROOT / 'lab/training_common'
sys.path.insert(0, str(COMMON))
from engine.cli import main as engine_main
from engine.config import load_config
from engine.data import inspect_dataset

DEFAULT_CONFIG = 'lab/features/fire/training/config/fire_sticker_v4.yaml'
if PACKAGED:
    DEFAULT_CONFIG = 'config/train.yaml'


def audit_target_sizes(root, config):
    """Count negatives and exported-image box sizes; label truth requires review."""
    from PIL import Image

    root = Path(root).resolve()
    header, splits, report = inspect_dataset(root, config)
    if header['names'] != ['fire_sticker'] or config['model_key'] != 'fire_sticker':
        raise ValueError('Fire audit requires the single fire_sticker class')
    sizes = {}
    for split, records in splits.items():
        item = {'images': len(records), 'positive_images': 0, 'negative_images': 0,
                'instances': 0, 'exported_short_side_pixels': {'0_8': 0, '8_16': 0, '16_32': 0, 'over_32': 0},
                'estimated_input_short_side_pixels': {'0_8': 0, '8_16': 0, '16_32': 0, 'over_32': 0}}
        for record in records:
            lines = (root / record['label']).read_text(encoding='utf-8').splitlines()
            labels = [line.split() for line in lines if line.strip()]
            if labels:
                item['positive_images'] += 1
            else:
                item['negative_images'] += 1
                continue
            with Image.open(root / record['image']) as image:
                width, height = image.size
            ratio = config['imgsz'] / max(width, height)
            resized_width = min(math.ceil(width * ratio), config['imgsz'])
            resized_height = min(math.ceil(height * ratio), config['imgsz'])
            for values in labels:
                if len(values) != 5 or int(float(values[0])) != 0:
                    raise ValueError('Fire labels must contain one class and four box values')
                for field, w, h in (('exported_short_side_pixels', width, height),
                                     ('estimated_input_short_side_pixels', resized_width, resized_height)):
                    size = min(float(values[3]) * w, float(values[4]) * h)
                    bucket = '0_8' if size <= 8 else '8_16' if size <= 16 else '16_32' if size <= 32 else 'over_32'
                    item[field][bucket] += 1
                item['instances'] += 1
        sizes[split] = item
    return {'status': 'descriptive_only', 'splits': sizes, 'dataset_report': report,
            'scope': ('Exported pixels and estimated unaugmented aspect-preserving model-input pixels; '
                      'not verified source resolution or recovered detail. Padding and augmentation can change actual inputs; '
                      'empty labels alone do not establish verified negative or distractor type.')}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'audit':
        parser = argparse.ArgumentParser(description=audit_target_sizes.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--config', default=DEFAULT_CONFIG)
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            print(json.dumps(audit_target_sizes(root, load_config(root, args.config)), ensure_ascii=False, indent=2))
            return 0
        except (OSError, ValueError, KeyError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
