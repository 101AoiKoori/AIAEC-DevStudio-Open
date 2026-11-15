"""Load a task's design and source configuration without creating output."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .paths import TASKS, WORKSPACE


def read_yaml(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    try:
        import yaml
    except ModuleNotFoundError as exc:
        if exc.name != 'yaml':
            raise
        interpreter = WORKSPACE / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        result = subprocess.run([str(interpreter), '-I', '-B', '-c',
            'import sys,json,yaml; print(json.dumps(yaml.safe_load(sys.stdin.buffer.read().decode("utf-8")),ensure_ascii=True))'],
            input=text.encode('utf-8'), capture_output=True, check=True, timeout=30, env=env)
        return json.loads(result.stdout)
    return yaml.safe_load(text)


def source_path(task, key):
    _, source = load_task_config(task)
    value = source.get(key)
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f'{task}.{key} must be a workspace-relative source path')
    path = (WORKSPACE / value).resolve()
    if not path.is_relative_to(WORKSPACE) or not path.exists():
        raise ValueError(f'{task}.{key} is missing or outside the workspace: {value}')
    return path


def load_task_config(task: str) -> tuple[dict, dict]:
    """Return the task design and its own source configuration.

    Loading a configuration does not establish that a generator consumes it or
    that every source is labelled or ready for production.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown V4 dataset task: {task!r}")
    root = WORKSPACE / "datasets" / "scripts" / task
    design = read_yaml(root / 'config' / 'dataset.yaml')
    if not isinstance(design, dict) or design.get("task") != task:
        raise ValueError("Dataset configuration task does not match its directory")
    if design.get("source_config") != "config/source.yaml":
        raise ValueError("Dataset configuration must reference its own source.yaml")
    source = read_yaml(root / 'config' / 'source.yaml')
    if not isinstance(source, dict) or source.get("task") != task:
        raise ValueError("Source configuration task does not match its directory")
    return design, source
