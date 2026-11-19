"""Identify the task design consulted by a plan without asserting its coverage."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .config import read_yaml
from .paths import WORKSPACE


def design_context(task: str, root: Path = WORKSPACE) -> dict:
    """Describe the task's design snapshot; unavailable inputs remain a local gap.

    The result references source files and explicitly marks every contrast as
    unverified. It is not a compiler, coverage check or production permit.
    """
    root = Path(root)
    result = {'task': task, 'status': 'unavailable', 'executable': False,
              'coverage': 'unverified', 'sources': {}, 'crosses': []}
    try:
        documents = {}
        for key, relative in (('targets', 'config/task_targets.yaml'),
                              ('axes', 'config/combination_axes.yaml')):
            path = root / relative
            before = path.read_bytes()
            parsed = read_yaml(path)
            if path.read_bytes() != before:
                raise ValueError(f'{relative} changed while reading')
            if not isinstance(parsed, dict) or not isinstance(parsed.get('tasks'), dict):
                raise ValueError(f'{relative} has no task map')
            documents[key] = parsed
            result['sources'][key] = {'path': relative, 'sha256': hashlib.sha256(before).hexdigest()}
        target = documents['targets']['tasks'].get(task)
        axis = documents['axes']['tasks'].get(task)
        if not isinstance(target, dict) or not isinstance(axis, dict):
            raise ValueError(f'task {task} is missing from design configuration')
        crosses = axis.get('crosses', [])
        if not isinstance(crosses, list):
            raise ValueError('crosses must be a list')
        seen = set()
        for cross in crosses:
            if not isinstance(cross, dict) or not isinstance(cross.get('id'), str) or not cross['id']:
                raise ValueError('cross needs a stable id')
            if cross['id'] in seen:
                raise ValueError('duplicate cross id: ' + cross['id'])
            seen.add(cross['id'])
            result['crosses'].append({'id': cross['id'],
                                      'completeness': (cross.get('structured') or {}).get('completeness', 'unknown'),
                                      'coverage': 'unverified'})
        result['status'] = ('referenced' if crosses and isinstance(target.get('classes'), list)
                            else 'incomplete')
    except Exception as exc:
        # A malformed or unavailable design must not stop unrelated feasible
        # production; it is reported rather than turned into coverage proof.
        result['crosses'] = []
        result['reason'] = str(exc)
    return result
