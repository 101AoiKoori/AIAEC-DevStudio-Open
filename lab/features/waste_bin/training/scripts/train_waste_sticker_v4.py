"""Waste-sticker training and per-class plus multi-sticker diagnostics."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
PACKAGED = (PACKAGE_ROOT / 'scripts/common/engine').is_dir()
ROOT = PACKAGE_ROOT if PACKAGED else HERE.parents[5]
sys.path.insert(0, str(ROOT / 'scripts/common' if PACKAGED else ROOT / 'lab/training_common'))
from engine.cli import main as engine_main
from engine.config import resolve
from engine.sticker_report import sticker_class_report, write_sticker_report

DEFAULT_CONFIG = 'config/train.yaml' if PACKAGED else 'lab/features/waste_bin/training/config/waste_sticker_v4.yaml'




def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'class-report':
        parser = argparse.ArgumentParser(description=sticker_class_report.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--evaluation', required=True)
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve(); evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts: raise ValueError('Evaluation must be inside runs')
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'waste_sticker': raise ValueError('Evaluation is not waste sticker')
            path = write_sticker_report(evaluation, summary['evaluation_config']['class_names'])
            print(path.name + ' ready'); return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
