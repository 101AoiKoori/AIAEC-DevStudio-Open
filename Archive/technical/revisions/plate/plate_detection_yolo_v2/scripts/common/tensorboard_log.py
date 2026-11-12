"""Launch TensorBoard for YOLOv8 event files under the configured project.

Ultralytics writes training metrics to event files during training. This
separate program presents the records; starting it never launches training.
"""
import argparse
from pathlib import Path
import sys

import yaml
from train import load_yaml, within

ROOT = Path(__file__).resolve().parent


def log_directory(root=ROOT):
    """Use the same project path resolution as the training entry point."""
    config = load_yaml(within(root, 'config/train.yaml'))
    return within(root, config.get('project', 'runs'))


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args(argv)
    try:
        logdir = log_directory(root)
        logdir.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print('ERROR:', error, file=sys.stderr)
        return 1
    from tensorboard import program
    board = program.TensorBoard()
    board.configure(argv=[None, '--logdir', str(logdir), '--host', args.host, '--port', str(args.port)])
    print('TensorBoard:', board.launch(), flush=True)
    try:
        import threading
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\nTensorBoard stopped.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
