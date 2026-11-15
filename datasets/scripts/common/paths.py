"""Stable locations shared by the eight V4 dataset tasks.

These functions describe locations only; callers own production, replacement,
and the applicable production guard. They never create or remove files.
"""
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
TASKS = frozenset({
    "person", "e_bike", "fire", "building_name_paper", "sign",
    "waste_bin_state", "waste_sticker", "meter",
})
SPLITS = frozenset({"train", "test", "val"})


def _task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"Unknown V4 dataset task: {task!r}")
    return task


def sample_dir(task: str) -> Path:
    """Return the one current sample directory for a task."""
    return WORKSPACE / "datasets" / "samples" / _task(task)


def dataset_dir(task: str) -> Path:
    """Return the one current dataset directory for a task."""
    return WORKSPACE / "datasets" / "exports" / _task(task)


def split_dir(task: str, split: str) -> Path:
    """Return one of the three formal dataset split directories."""
    if split not in SPLITS:
        raise ValueError(f"Unknown dataset split: {split!r}")
    return dataset_dir(task) / split


def validate_guarded_outputs(code_paths, output_paths, procedural_paths, *, allow_outputless=False):
    """Enforce V4 locations for relocated task entries on workspace-relative paths.

    Return the task when a relocated generator is declared, or ``None`` for
    unrelated legacy operations. This verifies declared configuration paths,
    not whether a generator has correctly consumed every source or state.
    """
    tasks = set()
    for value in code_paths:
        parts = Path(value).as_posix().split('/')
        if len(parts) >= 4 and parts[:2] == ['datasets', 'scripts'] and parts[2] in TASKS:
            tasks.add(parts[2])
    if not tasks:
        return None
    if len(tasks) != 1:
        raise ValueError('A dataset production entry cannot include another task generator')
    task = tasks.pop()
    required = {f'datasets/scripts/{task}/config/dataset.yaml',
                f'datasets/scripts/{task}/config/source.yaml'}
    if not required.issubset(set(procedural_paths)):
        raise ValueError('Task production must declare its dataset.yaml and source.yaml')
    permitted = {f'datasets/samples/{task}'} | {
        f'datasets/exports/{task}/{split}' for split in SPLITS
    }
    if not output_paths and allow_outputless:
        return task
    if not output_paths or any(value not in permitted for value in output_paths):
        raise ValueError('Task outputs must be its current sample or train/test/val directories')
    return task
