"""Compatibility entry for the e-bike compositor in datasets/scripts/e_bike."""
import importlib.util
from pathlib import Path
import runpy
import sys


_target = (Path(__file__).resolve().parents[5] /
           'datasets/scripts/e_bike/compose_v4_grounded_train.py')
if __name__ == '__main__':
    runpy.run_path(str(_target), run_name='__main__')
else:
    _spec = importlib.util.spec_from_file_location(__name__, _target)
    if _spec is None or _spec.loader is None:
        raise ImportError(f'Cannot load e-bike compositor: {_target}')
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
