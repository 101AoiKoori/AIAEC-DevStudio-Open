"""Internal bounded-experiment subprocess entry; no data or environment installation."""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.config import load_config
from engine.runner import train_run, threshold_scan_in_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'thresholds'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-name')
    parser.add_argument('--resume')
    parser.add_argument('--attempt')
    parser.add_argument('--controller-wall-seconds', type=float)
    parser.add_argument('--run-dir')
    parser.add_argument('--weights')
    args = parser.parse_args()
    root = args.root.resolve(); config = load_config(root, args.config)
    if args.mode == 'train':
        train_run(root, config, args.run_name, args.resume, attempt=args.attempt,
                  controller_wall_seconds=args.controller_wall_seconds)
    else:
        if not args.weights or not args.run_dir:
            parser.error('thresholds requires --weights and --run-dir')
        threshold_scan_in_process(root, config, args.weights, run_dir=root / args.run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
