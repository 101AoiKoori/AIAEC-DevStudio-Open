"""Create traceable dark, low-saturation camera-tone review pairs.

These profiles are simulations, not measured camera calibration. Outputs are
unassigned or development-regression candidates without labels; visibility
must be reviewed per task. No train/val/test split is assigned here.
"""
import argparse
import filecmp
import json
from pathlib import Path
import platform
import re
import shutil
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILES = {
    'mild': {'value_gain': 0.90, 'saturation_gain': 0.85},
    'moderate': {'value_gain': 0.70, 'saturation_gain': 0.60},
}


def apply_tone(image, *, value_gain, saturation_gain):
    """Apply bounded HSV V/S gains to uint8 BGR without changing geometry."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an 8-bit three-channel BGR image')
    for name, value in (('value_gain', value_gain), ('saturation_gain', saturation_gain)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(name + ' must be finite and in (0, 1]')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.rint(hsv[:, :, 1].astype(np.float32) * saturation_gain).clip(0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.rint(hsv[:, :, 2].astype(np.float32) * value_gain).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def statistics(image):
    """Summarize brightness and saturation; dark-pixel hue is unreliable."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visible = hsv[:, :, 2] >= 16
    return {'mean_gray_0_255': float(gray.mean()), 'mean_saturation_0_255': float(hsv[:, :, 1].mean()),
            'mean_saturation_value_ge_16_0_255': float(hsv[:, :, 1][visible].mean()) if visible.any() else None,
            'value_ge_16_pixels': int(visible.sum()), 'pixels': int(gray.size)}


def workspace_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Path must remain within workspace')
    return path


def read_image(path):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return result


def write_image(path, value):
    ok, encoded = cv2.imencode('.png', value)
    if not ok:
        raise OSError('Cannot encode image: ' + str(path))
    encoded.tofile(str(path))


def source_role(root, source, known, force_regression=False):
    """Return regression, unassigned, or unknown historical exposure."""
    if force_regression:
        return 'regression', 'explicit_development_regression'
    if (not isinstance(known, dict) or not isinstance(known.get('exposed_images'), list) or
            not isinstance(known.get('exposed_video_frames'), list)):
        raise ValueError('Known development exposure inventory is required')
    relative = source.relative_to(root).as_posix()
    unavailable = {item['path'] for item in known.get('unavailable_exposure_paths', [])}
    for item in known['exposed_images']:
        prior = workspace_path(root, item)
        if relative == item:
            return 'regression', 'known_exposed_image_or_identical_bytes'
        if (item not in unavailable and prior.is_file() and prior.stat().st_size == source.stat().st_size and
                                filecmp.cmp(prior, source, shallow=False)):
            return 'regression', 'known_exposed_image_or_identical_bytes'
    # These exact sequences have an auditable frame-to-time contract.
    from lab.scripts.data.v4_source_gate import (PERSON_SEQUENCE, PERSON_VIDEO, SITE_SEQUENCE, SITE_VIDEO,
                                      person_sequence_times, site_sequence_times)
    for sequence, video, loader in ((SITE_SEQUENCE, SITE_VIDEO, site_sequence_times),
                                     (PERSON_SEQUENCE, PERSON_VIDEO, person_sequence_times)):
        if relative.startswith(sequence + '/'):
            match = re.fullmatch(r'frame_(\d+)\.png', relative[len(sequence) + 1:])
            if not match:
                raise ValueError('Unrecognized sequence frame name')
            seconds = loader(root)[int(match.group(1))]
            if any(frame['video'] == video and abs(seconds - frame['seconds']) <= .25 + 1e-6
                   for frame in known['exposed_video_frames']):
                return 'regression', 'known_diagnostic_time_buffer'
    if known.get('exposure_status') == 'incomplete' or known.get('missing_exposure_paths') or unavailable:
        return 'unknown', 'historical_exposure_incomplete'
    return 'unassigned', 'source_usage_requires_review'


def _validate_declaration(declaration, source_names):
    """Reject states this tone-only, unlabeled candidate generator cannot execute."""
    if not isinstance(declaration, dict):
        raise ValueError('A production declaration is required')
    allowed = {'schema_version', 'model_key', 'target_split', 'purpose', 'budget',
               'uncertainty_weight', 'independence_basis', 'constraints', 'strata'}
    if set(declaration) - allowed or declaration.get('target_split') != 'unassigned':
        raise ValueError('Tone candidates require unassigned target_split and supported declaration fields')
    if declaration.get('constraints') != {'max_long_side': 1280, 'max_short_side': 720,
                                          'geometry': 'unchanged'}:
        raise ValueError('Unsupported camera-tone constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('Camera-tone states are required')
    allowed_state = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight',
                     'failure_cost', 'conditions', 'observations', 'background', 'target', 'labels',
                     'augmentation', 'sources', 'parameters'}
    seen = set()
    for state in states:
        if not isinstance(state, dict) or set(state) - allowed_state:
            raise ValueError('Unsupported camera-tone state fields')
        background = state.get('background')
        source = background.get('source') if isinstance(background, dict) else None
        augmentation = state.get('augmentation')
        profile = augmentation.get('profile') if isinstance(augmentation, dict) else None
        if (not isinstance(source, str) or source not in source_names or
                not isinstance(profile, str) or profile not in PROFILES or
                background != {'source': source} or augmentation != {'profile': profile} or
                state.get('target') != {'kind': 'existing_image_content'} or
                state.get('labels') != {'status': 'unknown'} or
                state.get('sources') != [source] or state.get('parameters') != PROFILES[profile] or
                state.get('conditions') != ['camera_tone:' + profile] or state.get('observations', []) or
                not isinstance(state.get('prior'), dict) or set(state['prior']) != {'alpha', 'beta'} or
                type(state.get('min_count')) is not int or type(state.get('max_count')) is not int or
                not 0 <= state['min_count'] <= state['max_count'] <= 1):
            raise ValueError('Unsupported camera-tone source, target, label, profile, parameters or quota')
        if (source, profile) in seen:
            raise ValueError('Duplicate camera-tone source/profile state')
        seen.add((source, profile))


def make_preview(root, image_paths, output, *, known, production_plan, force_regression=False):
    """Reuse a complete registered preview before decoding or writing images."""
    from lab.training_common.production_guard import generation_guard

    root = Path(root).resolve()
    destination = workspace_path(root, output)
    owner = root / 'datasets/exports/mixed_scene'
    if destination == owner or not destination.is_relative_to(owner):
        raise ValueError('Output must be within datasets/exports/mixed_scene')
    sources = [workspace_path(root, item) for item in image_paths]
    if not sources:
        raise ValueError('At least one source image is required')
    names = [source.relative_to(root).as_posix() for source in sources]
    if len(names) != len(set(names)):
        raise ValueError('Source images must be unique')
    _validate_declaration(production_plan, set(names))
    unavailable = {item['path'] for item in known.get('unavailable_exposure_paths', [])}
    dependencies = {workspace_path(root, item) for item in known.get('exposed_images', [])
                    if item not in unavailable and workspace_path(root, item).is_file()}
    from lab.scripts.data.v4_source_gate import SITE_SEQUENCE, PERSON_SEQUENCE
    procedural = []
    for prefix, filename in ((SITE_SEQUENCE, 'ffmpeg.log'), (PERSON_SEQUENCE, 'frames_probe.json')):
        if any(p.relative_to(root).as_posix().startswith(prefix + '/') for p in sources):
            procedural.append(workspace_path(root, prefix + '/' + filename))
    with generation_guard('camera_tone:' + destination.relative_to(root).as_posix(),
                          sorted(set(sources) | dependencies | set(procedural)),
                          [destination], {'known': known, 'force_regression': force_regression,
                                          'profiles': PROFILES, 'opencv': cv2.__version__, 'numpy': np.__version__},
                          code_paths=[Path(__file__), root / 'lab/scripts/data/v4_source_gate.py'],
                          procedural_inputs=procedural, root=root, production_plan=production_plan,
                          sample_output_paths=[destination]) as run:
        if run.reused:
            return json.loads((destination / 'manifest.json').read_text(encoding='utf8'))
        if destination.exists():
            # The guard permits replacement only for an unchanged owned output
            # with a prior complete receipt and changed declared inputs.
            shutil.rmtree(destination)
        return _make_preview(root, names, output, known=known, run=run, force_regression=force_regression)


def _make_preview(root, image_paths, output, *, known, run, force_regression=False):
    """Write candidates with source paths, exact gains and no copied labels."""
    from lab.training_common.production_guard import ProductionRun
    if not isinstance(run, ProductionRun) or not run.active or run.plan is None:
        raise ValueError('Camera-tone generation requires an active planned production run')
    root = Path(root).resolve()
    owner = workspace_path(root, 'datasets/exports/mixed_scene')
    output = workspace_path(root, output)
    if not output.is_relative_to(owner) or output == owner or output.exists():
        raise ValueError('Output must be a new directory in datasets/exports/mixed_scene')
    if not image_paths:
        raise ValueError('At least one source image is required')
    sources = []
    for item in image_paths:
        path = workspace_path(root, item)
        if not path.is_file():
            raise FileNotFoundError('Missing source image: ' + str(path))
        image = read_image(path)
        height, width = image.shape[:2]
        if min(width, height) > 720 or max(width, height) > 1280:
            raise ValueError('Source image exceeds 720p: ' + str(path))
        role, role_reason = source_role(root, path, known, force_regression)
        sources.append((path, image, role, role_reason))
    # Identical input files cannot get different usage roles within one batch.
    for index, (path, image, role, reason) in enumerate(sources):
        if role not in ('unassigned', 'unknown'):
            continue
        for other, _, other_role, _ in sources:
            if (other_role == 'regression' and other.stat().st_size == path.stat().st_size and
                    filecmp.cmp(other, path, shallow=False)):
                sources[index] = (path, image, 'regression', 'identical_to_regression_input_in_batch')
                break
    output.mkdir(parents=True)
    report = {'status': 'running', 'role': 'review_candidates',
              'scope': ('Fitted HSV gains only; no camera calibration or model-quality claim. '
                        'All task labels and visibility require review; no train/val/test assigned.'),
              'software': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__},
              'profiles': PROFILES, 'samples': []}
    try:
        states_by_source = {source.relative_to(root).as_posix(): [] for source, _, _, _ in sources}
        for state in run.plan['strata']:
            if state['planned_count']:
                states_by_source[state['background']['source']].append(state)
        for index, (source, original, role, role_reason) in enumerate(sources):
            source_name = source.relative_to(root).as_posix()
            for state in states_by_source[source_name]:
                profile = state['augmentation']['profile']
                values = state['parameters']
                result = apply_tone(original, **values)
                filename = f'{index:02d}_{source.stem}_{profile}.png'
                write_image(output / filename, result)
                report['samples'].append({'image': filename, 'parent_image': source.relative_to(root).as_posix(),
                    'parent_ids': [source.relative_to(root).as_posix()], 'generation_method': 'augmentation',
                    'processing': {'method': 'HSV V and S multiplication', **values, 'geometric_transform': 'none',
                                   'resize': 'none', 'compression': 'PNG lossless', 'blur': 'none'},
                    'image_size': [original.shape[1], original.shape[0]], 'profile': profile,
                    'source_statistics': statistics(original), 'output_statistics': statistics(result),
                    'split': 'unassigned' if role == 'unknown' else role,
                    'historical_exposure_status': role, 'role_reason': role_reason,
                    'source_review_status': 'not_verified_by_preview',
                    'task_state': 'unreviewed', 'labels': None})
                run.record_sample(state['id'], output=output / filename, sources=state['sources'],
                                  parameters=values, actual={'parent_image': source_name,
                                  'value_gain': values['value_gain'], 'saturation_gain': values['saturation_gain'],
                                  'manifest': (output / 'manifest.json').relative_to(root).as_posix(),
                                  'record_index': len(report['samples']) - 1})
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', action='append', required=True, help='Workspace-relative source; repeat for pairs')
    parser.add_argument('--output', required=True, help='New directory under datasets/exports/mixed_scene')
    parser.add_argument('--plan', required=True, help='Workspace-relative JSON production declaration')
    parser.add_argument('--regression', action='store_true', help='Mark every output as development regression')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from lab.scripts.data.v4_source_gate import build_queue
    declaration = json.loads(workspace_path(ROOT, args.plan).read_text(encoding='utf-8-sig'))
    result = make_preview(ROOT, args.image, args.output, known=build_queue(ROOT),
                          production_plan=declaration, force_regression=args.regression)
    print(json.dumps({'status': result['status'], 'count': len(result['samples'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
