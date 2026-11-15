"""Compatibility import; implementation is task-owned."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.building_name_paper import package_real_candidates as _impl
if __name__ == '__main__':
    _impl.main()
else:
    sys.modules[__name__] = _impl
    globals().update({k: v for k, v in vars(_impl).items() if not k.startswith('__')})
