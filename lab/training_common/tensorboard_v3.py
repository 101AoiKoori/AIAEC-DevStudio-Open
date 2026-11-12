"""Open TensorBoard on this copied package's runs directory."""
import argparse
import os
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root/'runs/_viewer'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(cache), TMP=str(cache), MPLCONFIGDIR=str(cache))
    from tensorboard import main as tensorboard_main
    sys.argv = ['tensorboard', '--logdir', str(root/'runs'), '--host', args.host, '--port', str(args.port)]
    tensorboard_main.run_main()

if __name__ == '__main__':
    main()
