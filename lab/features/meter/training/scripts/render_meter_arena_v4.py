"""Compatibility import; implementation is task-owned."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.meter import render_meter_arena_v4 as _impl
if __name__ == '__main__':
    _impl.main()
else:
    sys.modules[__name__] = _impl
    globals().update({k: v for k, v in vars(_impl).items() if not k.startswith('__')})
