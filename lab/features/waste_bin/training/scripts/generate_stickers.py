"""Import compatibility for datasets/scripts/waste_sticker/generate_stickers.py; generation CLI retired."""
import importlib
from pathlib import Path
import sys

if __name__ == '__main__':
    raise SystemExit('Legacy helper CLI is retired; use datasets/scripts/waste_sticker/generate_stickers.py. Production remains paused.')
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
_implementation = importlib.import_module('datasets.scripts.waste_sticker.generate_stickers')

sys.modules[__name__] = _implementation

def __getattr__(name):
    return getattr(_implementation, name)
