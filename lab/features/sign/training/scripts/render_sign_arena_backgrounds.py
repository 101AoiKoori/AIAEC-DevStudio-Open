"""Render planned Blender arena background candidates, with no target label."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/sign/. Production remains paused.")

import argparse
import json
import sys
from pathlib import Path

import bpy

ROOT = Path.cwd().resolve()
OUTPUT_ROOT = ROOT / 'runs/sign'
SCENE = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend'
VIEWS = ((-0.6, -0.75, -0.35), (-0.25, -1.1, -0.12),
         (0.2, -0.9, 0.18), (0.5, -1.2, 0.32))


def validate_declaration(declaration, sources):
    """Reject states this fixed-view background renderer cannot implement."""
    constraints = {'image_size': [640, 480], 'cycles_samples': 16, 'device': 'OPTIX'}
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'sign':
        raise ValueError('Explicit sign production plan is required')
    if declaration.get('constraints') != constraints:
        raise ValueError('Unsupported arena render constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('At least one arena render state is required')
    seen = set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('Arena state must be an object')
        if (state.get('background') != {'scene': SCENE} or
                state.get('target') != {'kind': 'background', 'sign_absence': 'unknown'} or
                state.get('labels') != {'kind': 'none'} or
                state.get('augmentation') != {'kind': 'none'}):
            raise ValueError('Unsupported arena background, target, labels or augmentation')
        params = state.get('parameters')
        if not isinstance(params, dict) or set(params) != {'view'}:
            raise ValueError('Arena parameters must specify exactly a view')
        view = params['view']
        if not isinstance(view, list) or len(view) != 3 or any(type(n) not in (int, float) for n in view):
            raise ValueError('Arena view must contain three numeric values')
        view = tuple(view)
        if view not in VIEWS or view in seen:
            raise ValueError('Arena view is unsupported or repeated')
        seen.add(view)
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('Each fixed camera view permits at most one sample')
        if state.get('sources') != sources:
            raise ValueError('Declared sources differ from actual arena inputs')
        if state.get('conditions') != ['background_only']:
            raise ValueError('Unsupported arena background conditions')


def _main(output, plan, run):
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve() or output.exists():
        raise ValueError('Output must be a new sign export directory')
    scene = bpy.context.scene
    camera = bpy.data.objects['Vehicle camera']
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    devices = [device for device in prefs.devices if device.type == 'OPTIX']
    if not devices:
        raise RuntimeError('No Cycles OPTIX device available')
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    (output / 'images').mkdir(parents=True)
    rows = []
    try:
        for state in plan['strata']:
            if state['planned_count'] == 0:
                continue
            x, y, yaw = state['parameters']['view']
            camera.location.x, camera.location.y = x, y
            camera.rotation_euler.z = yaw
            name = f'arena_{len(rows):02}.png'
            target = output / 'images' / name
            scene.render.filepath = str(target)
            bpy.ops.render.render(write_still=True)
            actual = {'scene': SCENE, 'camera_location': list(camera.location),
                      'camera_rotation_euler': list(camera.rotation_euler),
                      'lens_mm': camera.data.lens, 'device': 'OPTIX',
                      'view': state['parameters']['view']}
            run.record_sample(state['id'], output=target, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            rows.append({'image': 'images/' + name, 'state_id': state['id'],
                         **actual, 'review_status': 'pending'})
        status = 'background_candidates_pending_review'
    except BaseException:
        status = 'failed_or_interrupted'
        raise
    finally:
        result = {'status': status, 'source_scene': SCENE, 'method': 'Blender scene render',
                  'size': [640, 480], 'samples': rows, 'split': plan['target_split'],
                  'note': 'Same scene and shared geometry; not independent real captures.'}
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts task entry and its source.yaml")


if __name__ == '__main__':
    main()
