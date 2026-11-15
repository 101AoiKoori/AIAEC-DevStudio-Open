"""Render bounded four-bin arena candidates with per-instance/state raster evidence.

Run with Blender -b ARENA --python this_file -- --out NEW_DIR --count N.
Only synthetic geometry is used; calibrated diagnostic photos are not composited.
"""

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from datasets.scripts.waste_bin_state.bin_render_contract import (BIN_NAMES, MASK_COLORS, OPEN_POSES, arena_planned_jobs,
                                 attach_state_evidence, decode_mask, mouth_pixel_contrast, STATE_CONTRACT)
from datasets.scripts.waste_bin_state.render_blender_waste_bin import (GEOMETRY_TYPES, correct_hanging_hinge_clearance,
                                     configure_device, create_mouth_proxies, read_rgba, targets_inside_frame)

ROOT = Path(__file__).resolve().parents[3]
from datasets.scripts.common.config import source_path, load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir
SOURCE = source_path('waste_bin_state', 'arena_bin_asset')
ARENA = source_path('waste_bin_state', 'arena_scene')
EXPORT = sample_dir('waste_bin_state')
QUANTITY_SCHEDULE = (4, 3, 2, 1, 0, 4, 3, 2, 4, 3, 2, 4, 3, 4, 2, 4, 3, 4, 1, 4)


def sampling_plan(index):
    """Cycle each quantity's color/state combinations before repeating them."""
    count = QUANTITY_SCHEDULE[index % len(QUANTITY_SCHEDULE)]
    cycle, position = divmod(index, len(QUANTITY_SCHEDULE))
    occurrence = cycle * QUANTITY_SCHEDULE.count(count) + QUANTITY_SCHEDULE[:position].count(count)
    subsets = list(itertools.combinations(BIN_NAMES, count))
    selected = subsets[occurrence % len(subsets)]
    state_index = (occurrence // len(subsets)) % (1 << count)
    states = {name: ('open' if name in selected and state_index & (1 << selected.index(name)) else 'closed')
              for name in BIN_NAMES}
    total = len(subsets) * (1 << count)
    dark = (bool(occurrence % 2) if count == 0 else
            bool((occurrence % len(subsets) + occurrence // len(subsets) + occurrence // total) % 2))
    return count, occurrence, selected, states, dark


def mask_visibility(visible, isolated, inside):
    """Use per-pixel intersection; Workbench edge coverage differs by pass."""
    visible_count, isolated_count = int(visible.sum()), int(isolated.sum())
    fraction = int(np.logical_and(visible, isolated).sum()) / isolated_count if isolated_count else None
    reasons = []
    if not inside:
        reasons.append('clipped_or_behind_camera')
    if not isolated_count:
        reasons.append('no_isolated_pixels')
    if visible_count < 24:
        reasons.append('too_few_visible_pixels')
    if fraction is not None and fraction < .70:
        reasons.append('occluded_below_threshold')
    box = None
    if visible_count:
        y, x = np.nonzero(visible)
        h, w = visible.shape
        box = [int(x.min()) / w, int(y.min()) / h, (int(x.max()) + 1) / w, (int(y.max()) + 1) / h]
    return {'candidate_xyxy_normalized': box, 'visible_pixels': visible_count,
            'isolated_pixels': isolated_count, 'visible_fraction': fraction,
            'in_frame': bool(inside), 'label_eligible': not reasons, 'rejection_reasons': reasons,
            'visibility_verified': 'workbench_intersection_estimate'}


def masks(scene, camera, parts, output, stem, tag, force_visible=()):
    from datasets.scripts.common.blender_masks import masks as shared_masks
    return shared_masks(scene, camera, parts, output, stem, tag, force_visible, palette=MASK_COLORS)


def _generate(asset_objects, run, media):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scene', default=ARENA.relative_to(ROOT).as_posix())
    ap.add_argument('--count', type=int, default=5)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261212)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--max-seconds', type=int, default=360)
    ap.add_argument('--plan', required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = (ROOT / args.out).resolve()
    if out != EXPORT.resolve() or (out.exists() and any(out.iterdir())):
        ap.error('arena render output must be the empty current waste_bin_state sample')
    arena = (ROOT / args.scene).resolve()
    if (not arena.is_relative_to(ROOT) or not arena.is_file() or
            Path(bpy.data.filepath).resolve() != arena or
            not 1 <= args.count <= 20 or not 1 <= args.samples <= 8):
        ap.error('load the workspace scene specified by --scene; count 1..20 and samples 1..8')
    if args.offset < 0 or not 20 <= args.max_seconds <= 360:
        ap.error('offset must be nonnegative; max-seconds must be 20..360')
    jobs = arena_planned_jobs(run.plan, arena.relative_to(ROOT).as_posix(), SOURCE.relative_to(ROOT).as_posix())
    if len(jobs) != args.count:
        raise ValueError('count must equal the planned budget')
    scene = bpy.context.scene
    for obj in asset_objects:
        scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    pivots = {name: bpy.data.objects[name + ' lid pivot - rotate X'] for name in BIN_NAMES}
    clearance = correct_hanging_hinge_clearance(list(pivots.values()))
    root = bpy.data.objects['Measured height scale']
    root.scale = (.1, .1, .1)
    root.location = (0, 1.0, 0)
    bpy.context.view_layer.update()
    parts = {name: [o for o in scene.objects if o.type in GEOMETRY_TYPES and o.name.startswith(name + ' ')]
             for name in BIN_NAMES}
    lids = {name: [o for o in parts[name] if o.parent == pivots[name]] for name in BIN_NAMES}
    proxies = create_mouth_proxies(scene)
    camera = scene.objects['Vehicle camera']
    scene.camera = camera
    bpy.context.view_layer.update()
    original_camera = {'location': list(camera.location), 'rotation_euler': list(camera.rotation_euler),
                       'matrix_world': [list(row) for row in camera.matrix_world],
                       'lens_mm': camera.data.lens, 'sensor_width_mm': camera.data.sensor_width}
    # Fitted approach view: estimated 140 px bin height at the existing lens;
    # the fixed ground placement and framing require verification in the pilot.
    camera.location = (0.0, 0.64, 0.18)
    fitted_aim = Vector((0.0, 1.0, 0.075))
    camera.rotation_euler = (fitted_aim - camera.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 640, 480, 100
    scene.render.threads_mode, scene.render.threads = 'FIXED', 2
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine, scene.cycles.samples = 'CYCLES', args.samples
    devices = configure_device(scene, 'OPTIX')
    tone_tree = bpy.data.node_groups.new('Waste arena low saturation', 'CompositorNodeTree')
    tone_tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layer_node = tone_tree.nodes.new('CompositorNodeRLayers')
    tone = tone_tree.nodes.new('CompositorNodeHueSat')
    tone_output = tone_tree.nodes.new('NodeGroupOutput')
    tone_tree.links.new(layer_node.outputs['Image'], tone.inputs['Image'])
    tone_tree.links.new(tone.outputs['Image'], tone_output.inputs['Image'])
    scene.compositing_node_group = tone_tree
    scene.render.use_compositing = True
    out.mkdir(parents=True, exist_ok=True)
    for sub in ('images', 'masks', 'labels_candidate'):
        (out / sub).mkdir()
    report = {'status': 'running', 'seed': args.seed, 'offset': args.offset, 'requested': args.count, 'requested_images': args.count, 'label_contract_id': STATE_CONTRACT,
              'min_visibility': .70, 'min_visible_pixels': 24,
              'resolution': [640, 480], 'render_device': 'OPTIX', 'devices': devices, 'cycles_samples': args.samples,
              'wall_seconds_limit': args.max_seconds, 'scene_file': arena.relative_to(ROOT).as_posix(),
              'bin_source': SOURCE.relative_to(ROOT).as_posix(),
              'source_model_units': '1 unit approximately 10 cm per asset construction; arena units interpreted as metres',
              'scale_fitted': .1, 'layout_fitted_m': [0, 1.0, 0], 'camera_fitted_not_measured': True,
              'source_camera': original_camera,
              'camera_changes': 'plan specifies position and aim; lens and sensor width retained',
              'camera_fitted_aim': list(fitted_aim),
              'reference_only': ['datasets/raw/car_real_capture_20261112/P12_TL2.png', 'frame_000181',
                                 'datasets/raw/car_real_capture_20261112/P02_PS1.png', 'datasets/raw/car_real_capture_20261112/P04_PS2.png',
                                 'datasets/raw/car_real_capture_20261112/P05_NOENTRY.png'],
              'lineage': 'one synthetic arena and one four-bin asset; real diagnostics were referenced only, not composited; not independent capture',
              'split': run.plan['target_split'], 'hinge_clearance': clearance, 'samples': []}
    def save():
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    save()
    start = time.monotonic()
    try:
        for i, job in enumerate(jobs):
            if time.monotonic() - start >= args.max_seconds:
                report['status'] = 'time_limit_partial'
                break
            index = args.offset + i
            selected = set(job['target']['bins'])
            count = len(selected)
            states = {name: job['target']['bins'].get(name, 'closed') for name in BIN_NAMES}
            root.location = job['parameters']['bin_location']
            camera.location = job['parameters']['camera_location']
            fitted_aim = Vector(job['parameters']['camera_aim'])
            camera.rotation_euler = (fitted_aim - camera.location).to_track_quat('-Z', 'Y').to_euler()
            for name in BIN_NAMES:
                for obj in parts[name]:
                    obj.hide_render = name not in selected
                pivots[name].rotation_euler.x = math.radians(OPEN_POSES['upright'] if states[name] == 'open' else 0)
            scene.view_settings.exposure = job['augmentation']['exposure_ev']
            tone.inputs['Saturation'].default_value = job['augmentation']['saturation']
            scene.cycles.seed = args.seed + index
            bpy.context.view_layer.update()
            stem = f'arena_{index:04d}'
            scene.render.filepath = str(out / 'images' / (stem + '.png'))
            bpy.ops.render.render(write_still=True)
            if selected:
                selected_parts = {name: parts[name] for name in BIN_NAMES if name in selected}
                target_mask, target = masks(scene, camera, selected_parts, out, stem, 'target')
                state_parts = {name: ([proxies[name]] if states[name] == 'open' else lids[name]) for name in BIN_NAMES if name in selected}
                state_mask, evidence = masks(scene, camera, state_parts, out, stem, 'state',
                                             [proxies[name] for name in selected if states[name] == 'open'])
                labels = [attach_state_evidence(t, e, states[t['bin']]) for t, e in zip(target, evidence)]
                image_pixels = read_rgba(out / 'images' / (stem + '.png'))
                visible_pixels = read_rgba(out / target_mask)
                mouth_pixels = read_rgba(out / state_mask)
                for item in labels:
                    if item['state'] == 'open':
                        contrast = mouth_pixel_contrast(image_pixels,
                            decode_mask(mouth_pixels, item['bin']), decode_mask(visible_pixels, item['bin']))
                        item['state_evidence']['image_pixel_contrast'] = contrast
                        if not contrast['passed']:
                            item['label_eligible'] = False
                            item['rejection_reasons'].append('state_evidence:mouth_pixel_contrast_unconfirmed')
            else:
                target_mask, state_mask, labels = None, None, []
            label_path = None
            if all(x['label_eligible'] for x in labels):
                label_path = 'labels_candidate/' + stem + '.txt'
                rows = []
                for item in labels:
                    a, b, c, d = item['candidate_xyxy_normalized']
                    rows.append(f"{0 if item['state'] == 'open' else 1} {(a+c)/2:.6f} {(b+d)/2:.6f} {c-a:.6f} {d-b:.6f}")
                (out / label_path).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf8')
            report['samples'].append({'id': stem, 'image': 'images/' + stem + '.png', 'bin_count': count,
                'visible_bins': [n for n in BIN_NAMES if n in selected], 'states': states,
                'plan_state_id': job['id'], 'bin_group_location_fitted': list(root.location),
                'camera_aim_fitted': list(fitted_aim),
                'candidate_label': label_path, 'labels': labels, 'mask_visible': target_mask, 'mask_state_visible': state_mask,
                'camera_location_fitted': list(camera.location),
                'camera_rotation_euler': list(camera.rotation_euler),
                'camera_matrix_world': [list(row) for row in camera.matrix_world],
                'lens_mm_fitted': camera.data.lens, 'sensor_width_mm': camera.data.sensor_width,
                'exposure_ev': scene.view_settings.exposure,
                'saturation': tone.inputs['Saturation'].default_value})
            run.record_sample(job['id'], output=out / 'images' / (stem + '.png'),
                              sources=job['sources'], parameters=job['parameters'],
                              actual={'state_id': job['id'], 'visible_bins': sorted(selected),
                                      'bin_states': {name: states[name] for name in sorted(selected)},
                                      'bin_location': list(root.location), 'camera_location': list(camera.location),
                                      'camera_aim': list(fitted_aim), 'exposure_ev': scene.view_settings.exposure,
                                      'saturation': tone.inputs['Saturation'].default_value,
                                      'render_seed': args.seed + index, 'candidate_label': label_path,
                                      'scene': arena.relative_to(ROOT).as_posix(),
                                      'bin_source': SOURCE.relative_to(ROOT).as_posix(),
                                      'media_inputs': [p.relative_to(ROOT).as_posix() for p in media]})
            report['elapsed_seconds'] = time.monotonic() - start
            save()
            print('ARENA_WASTE_PROGRESS', i + 1, '/', args.count, flush=True)
        else:
            report['status'] = 'complete_candidates'
    except BaseException as exc:
        report['status'] = 'failed_or_interrupted'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        save()


def _guard_inputs():
    """Declare the scene, its file images, and the bin asset's photo source."""
    if not bpy.app.background:
        raise ValueError('asset dependency discovery requires background Blender')
    if bpy.data.is_dirty:
        raise ValueError('loaded Blender scene has unsaved changes; use a saved workspace copy')
    scene = Path(bpy.data.filepath).resolve()
    asset_manifest = SOURCE.parent / 'manifest.json'
    asset = json.loads(asset_manifest.read_text(encoding='utf8'))
    _, config = load_task_config('waste_bin_state')
    if asset['source_scene'] != config['asset_parent_manifest_path']:
        raise ValueError('bin asset manifest parent differs from the configured registry alias')
    procedural = {scene, SOURCE, asset_manifest, source_path('waste_bin_state', 'asset_parent_scene')}
    media = {ROOT / asset['photo']}
    existing_images = {image.as_pointer() for image in bpy.data.images}
    with bpy.data.libraries.load(str(SOURCE), link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n == 'Measured height scale' or
                       any(n.startswith(name + ' ') for name in BIN_NAMES)]
    asset_objects = list(dst.objects)
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            if not image.filepath:
                raise ValueError('scene image has no source path: ' + image.name)
            if image.as_pointer() in existing_images:
                path = Path(bpy.path.abspath(image.filepath, library=image.library)).resolve()
            else:
                path = Path(bpy.path.abspath(image.filepath, start=str(SOURCE.parent),
                                             library=image.library)).resolve()
            if path.is_file():
                media.add(path)
            else:
                raise ValueError('scene image source missing: ' + str(path))
    return sorted(media), sorted(procedural), asset_objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scene', default=ARENA.relative_to(ROOT).as_posix())
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261212)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--max-seconds', type=int, default=360)
    parser.add_argument('--plan', required=True)
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    out = (ROOT / args.out).resolve()
    if out != EXPORT.resolve():
        parser.error('arena render output must be datasets/samples/waste_bin_state')
    plan_path = (ROOT / args.plan).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('plan must be an existing workspace file')
    declaration = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    arena = (ROOT / args.scene).resolve()
    if arena != ARENA or not arena.is_file() or Path(bpy.data.filepath).resolve() != arena:
        raise ValueError('load the workspace arena declared by --scene')
    media, procedural, asset_objects = _guard_inputs()
    code = [Path(__file__).resolve(), ROOT / 'datasets/scripts/common/blender_masks.py', ROOT / 'datasets/scripts/waste_bin_state/render_blender_waste_bin.py',
            ROOT / 'datasets/scripts/waste_bin_state/bin_render_contract.py']
    with generation_guard('waste_bin.render_arena:' + str(out.relative_to(ROOT)),
                          media, [out], {'argv': sys.argv[sys.argv.index('--') + 1:],
                                         'blender_version': bpy.app.version_string},
                          code_paths=code, procedural_inputs=[*procedural, plan_path, ROOT/'datasets/scripts/waste_bin_state/config/source.yaml', ROOT/'datasets/scripts/waste_bin_state/config/dataset.yaml'], root=ROOT,
                          production_plan=declaration, sample_output_paths=[out / 'images'],
                          auxiliary_output_paths=[out / 'masks']) as run:
        jobs = arena_planned_jobs(run.plan, arena.relative_to(ROOT).as_posix(), SOURCE.relative_to(ROOT).as_posix())
        if len(jobs) != args.count:
            raise ValueError('count must equal the planned budget')
        if run.reused:
            return
        _generate(asset_objects, run, media)
        if json.loads((out / 'manifest.json').read_text(encoding='utf8'))['status'] != 'complete_candidates':
            raise RuntimeError('render did not finish; output retained without a completion receipt')


if __name__ == '__main__':
    main()
