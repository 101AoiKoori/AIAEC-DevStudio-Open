"""Compatibility entry for the building-name compositor in datasets/scripts."""
import importlib.util
from pathlib import Path
import runpy
import sys


_target = (Path(__file__).resolve().parents[5] /
           'datasets/scripts/building_name_paper/compose_rooftop_papers.py')
if __name__ == '__main__':
    runpy.run_path(str(_target), run_name='__main__')
else:
    _spec = importlib.util.spec_from_file_location(__name__, _target)
    if _spec is None or _spec.loader is None:
        raise ImportError(f'Cannot load building-name compositor: {_target}')
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
