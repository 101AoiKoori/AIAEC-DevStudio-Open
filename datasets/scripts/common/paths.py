"""Stable locations shared by the eight V4 dataset tasks.

These functions describe locations only; callers own production, replacement,
and the applicable production guard. They never create or remove files.
"""
from pathlib import Path
import re


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


def sticker_orbit_sample_dir(frame: int) -> Path:
    return sample_dir("waste_sticker") / f"orbit_frame_{frame:04d}"


def sticker_surface_sample_dir() -> Path:
    return sample_dir("waste_sticker") / "surface_composite"


def bin_state_foreground_dir(split: str = "train") -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown V4 dataset split: {split!r}")
    task = WORKSPACE / "datasets" / "annotated" / "waste_bin_state"
    return task / "foreground" if split == "train" else task / "full_evaluation" / split / "foreground"


def bin_state_arena_render_dir() -> Path:
    return bin_state_foreground_dir().with_name("arena_render")


def bin_state_sample_composites_dir() -> Path:
    return bin_state_foreground_dir().with_name("sample_composites")


def building_name_sample_work_dir() -> Path:
    return WORKSPACE / "datasets/annotated/fire/building_name_paper_sample"


def building_name_full_work_dir() -> Path:
    return WORKSPACE / "datasets/annotated/fire/building_name_paper_full"


def bin_state_full_composites_dir(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown V4 dataset split: {split!r}")
    return bin_state_foreground_dir().with_name("full_composites") / split


def bin_photo_orbit_diagnostic_dir(*, compare_lids: bool = False) -> Path:
    name = "photo_orbit_lid_comparison" if compare_lids else "photo_orbit"
    return bin_state_foreground_dir().with_name(name)


BIN_STATE_RENDER_FOLDERS = (
    'images', 'labels_candidate', 'mask_visible', 'mask_isolated',
    'mask_state_visible', 'mask_state_isolated',
)


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
    if 'datasets/scripts/e_bike/prepare_baseline_data.py' in code_paths:
        permitted.add('datasets/exports/e_bike')
    if 'datasets/scripts/fire/render_hard_negative_preview.py' in code_paths:
        permitted.update({'datasets/samples/fire/hard_negative_preview',
                          'datasets/annotated/fire/production_inputs/hard_negative_preview.blend'})
    if 'datasets/scripts/fire/render_hard_negatives.py' in code_paths:
        permitted.add('datasets/annotated/fire/hard_negative_replacement/rendered')
    if 'datasets/scripts/fire/render_contrast_positives.py' in code_paths:
        permitted.add('datasets/annotated/fire/training_preparation/rendered')
    if 'datasets/scripts/e_bike/extract_video_views.py' in code_paths:
        permitted.update(value for value in output_paths
                         if re.fullmatch(r'datasets/annotated/e_bike/intermediate_views_[0-9]{8}(?:_refined)?', value))
    if 'datasets/scripts/sign/compose_sign_v4_measured.py' in code_paths:
        permitted.update(f'datasets/samples/sign/{split}' for split in SPLITS)
    orbit_entry = 'datasets/scripts/waste_sticker/render_sticker_arena_v4.py' in code_paths
    if orbit_entry:
        permitted.update(value for value in output_paths
                         if re.fullmatch(r'datasets/samples/waste_sticker/orbit_frame_[0-9]{4}', value))
    if 'datasets/scripts/waste_sticker/generate_sticker_real_photo_v4.py' in code_paths:
        permitted.add(sticker_surface_sample_dir().relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/waste_bin_state/render_bins_photo_orbit.py' in code_paths:
        permitted.update(bin_photo_orbit_diagnostic_dir(compare_lids=compare).relative_to(WORKSPACE).as_posix()
                         for compare in (False, True))
    if 'datasets/scripts/waste_bin_state/render_blender_waste_bin.py' in code_paths:
        permitted.update((bin_state_foreground_dir(split) / name).relative_to(WORKSPACE).as_posix()
                         for split in SPLITS for name in (*BIN_STATE_RENDER_FOLDERS, 'manifest.json'))
    if 'datasets/scripts/waste_bin_state/render_arena_waste_v4.py' in code_paths:
        permitted.add(bin_state_arena_render_dir().relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/waste_bin_state/generate_waste_v4_grouped.py' in code_paths:
        permitted.discard(sample_dir('waste_bin_state').relative_to(WORKSPACE).as_posix())
        permitted.add(bin_state_sample_composites_dir().relative_to(WORKSPACE).as_posix())
        permitted.update(bin_state_full_composites_dir(split).relative_to(WORKSPACE).as_posix() for split in SPLITS)
    if 'datasets/scripts/waste_bin_state/build_full_dataset.py' in code_paths:
        output = dataset_dir('waste_bin_state')
        permitted.update((output / name).relative_to(WORKSPACE).as_posix()
                         for name in ('data.yaml', 'manifest.json', 'workspace_sources.json'))
        permitted.add(bin_state_foreground_dir().with_name('full_export_history').relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/building_name_paper/render_blender_names.py' in code_paths:
        permitted.update((folder / 'blender').relative_to(WORKSPACE).as_posix()
                         for folder in (sample_dir('building_name_paper'), split_dir('building_name_paper', 'train')))
        train = split_dir('building_name_paper', 'train')
        permitted.update((train / folder / 'blender').relative_to(WORKSPACE).as_posix()
                         for folder in ('images', 'labels'))
        permitted.add((building_name_sample_work_dir() / 'blender').relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/building_name_paper/build_current_sample.py' in code_paths:
        permitted.update((building_name_sample_work_dir() / name).relative_to(WORKSPACE).as_posix()
                         for name in ('blender', 'blender_plan.json'))
    if 'datasets/scripts/building_name_paper/render_hybrid_sample.py' in code_paths:
        permitted.add((building_name_sample_work_dir() / 'hybrid/layers').relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/building_name_paper/hybrid_sample.py' in code_paths:
        permitted.add((building_name_sample_work_dir() / 'hybrid/composites').relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/building_name_paper/render_full_layers.py' in code_paths:
        permitted.add((building_name_full_work_dir() / 'layers').relative_to(WORKSPACE).as_posix())
        permitted.add((building_name_full_work_dir() / 'supplement_layers').relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/building_name_paper/build_full_dataset.py' in code_paths:
        permitted.add(dataset_dir('building_name_paper').relative_to(WORKSPACE).as_posix())
        permitted.add((dataset_dir('building_name_paper') / 'workspace_sources.json').relative_to(WORKSPACE).as_posix())
        permitted.add((building_name_full_work_dir() / 'visible_masks').relative_to(WORKSPACE).as_posix())
    if 'datasets/scripts/building_name_paper/build_video_finetune.py' in code_paths:
        permitted.add('training_packages/building_name_paper/finetune_dataset')
    if 'datasets/scripts/building_name_paper/package_detection_dataset.py' in code_paths:
        base = dataset_dir('building_name_paper')
        permitted.update((base / name).relative_to(WORKSPACE).as_posix()
                         for name in ('data.yaml', 'workspace_sources.json'))
    if 'datasets/scripts/meter/build_full_dataset.py' in code_paths:
        permitted.add(dataset_dir('meter').relative_to(WORKSPACE).as_posix())
        permitted.update((dataset_dir('meter')/name).relative_to(WORKSPACE).as_posix()
                         for name in ('data.yaml','manifest.json','workspace_sources.json'))
        permitted.update(value for value in output_paths if
                         re.fullmatch(r'datasets/annotated/meter/full_production/(?:production_inputs(?:/pilot)?|(?:train|val|test)/(?:foreground|backgrounds|composites)/chunk_[0-9]{4})',value))
    if not output_paths or any(value not in permitted for value in output_paths):
        raise ValueError('Task outputs must use the declared sample, annotated intermediate, or train/test/val paths')
    return task
