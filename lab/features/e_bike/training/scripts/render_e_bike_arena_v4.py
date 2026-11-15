"""Render a bounded e-bike pilot using real cutouts in the fitted arena.

Run in an independent Blender background process with the user-confirmed arena
as the input blend. Sprite geometry is planar; every output remains a candidate.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/. Production remains paused.")

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_standees import render_instance_masks, standee  # noqa: E402

ARENA = ROOT / 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
VEHICLES = ROOT / 'datasets/annotated/person/v4_vehicle_occluders_20261113/manifest.json'
PLANS = [
    ('upright', 'sam2_06_upright', 0.025, -0.30),
    ('fallen', 'sam2_01_fallen', 0.025, -0.30),
]


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = (ROOT / args.output).resolve()
    sources = json.loads(VEHICLES.read_text(encoding='utf-8'))['records']
    names = {name for _, name, _, _ in PLANS}
    selected = [row for row in sources if Path(row['path']).stem in names]
    if len(selected) != len(PLANS):
        raise ValueError('required reviewed vehicle cutouts are unavailable or duplicated')
    media = [ROOT / row[key] for row in selected for key in
             ('path', 'cropped_image', 'source_frame', 'source_video')]
    procedural = [ARENA, VEHICLES]
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    operation = 'e_bike.render_arena_v4:' + out.relative_to(ROOT).as_posix()
    with generation_guard(operation, list(dict.fromkeys(procedural + media)), [out],
                          {'engine': 'CYCLES', 'device': 'OPTIX', 'samples': 12},
                          code_paths=[Path(__file__), ROOT / 'datasets/scripts/common/blender_standees.py',
                                      ROOT / 'lab/training_common/production_guard.py'],
                          procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _render(out)


def _render(out):
    owner = (ROOT / 'runs/e_bike').resolve()
    if not out.is_relative_to(owner) or out == owner or out.exists():
        raise ValueError('output must be a new e-bike export subdirectory')
    if Path(bpy.data.filepath).resolve() != ARENA.resolve():
        raise ValueError('load the confirmed arena master in a background process')
    sources = json.loads(VEHICLES.read_text(encoding='utf-8'))['records']
    by_name = {Path(row['path']).stem: row for row in sources}
    plans = PLANS
    if any(name not in by_name or by_name[name]['class'] != cls for cls, name, _, _ in plans):
        raise ValueError('required real-image vehicle sources are unavailable')
    scene = bpy.context.scene
    camera = scene.objects.get('Vehicle camera')
    if camera is None:
        raise ValueError('missing vehicle camera')
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == 'OPTIX']
    if not devices:
        raise RuntimeError('No Cycles OPTIX device available')
    for device in preferences.devices:
        device.use = device.type == 'OPTIX'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    out.mkdir(parents=True)
    for part in ('images', 'labels', 'instance_ids', 'instance_full'):
        (out / part).mkdir()
    records = []
    for index, (category, name, lateral, exposure) in enumerate(plans):
        camera.location = (lateral, -0.40, 0.16)
        target = Vector((0, 0.03, 0.085))
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.lens = 22
        source = by_name[name]
        item = {'artwork': source['path'], 'image': source['cropped_image'], 'source': source}
        # The confirmed arena floor is centred at z=-0.025 m. standee() adds
        # 0.004 m to its base, placing the cutout at the measured scene mesh top.
        actor = standee(item, (0, 0.02, -0.025), 0, False,
                        0.15 if category == 'upright' else 0.09, True)
        board = bpy.data.objects[actor['geometry']]
        material = board.data.materials[0]
        nodes, links = material.node_tree.nodes, material.node_tree.links
        texture = next(n for n in nodes if n.type == 'TEX_IMAGE')
        front_back = next(n for n in nodes if n.type == 'MIX_RGB')
        shader = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
        tone = nodes.new('ShaderNodeHueSaturation')
        saturation = 0.72 if exposure < 0 else 1.0
        tone.inputs['Saturation'].default_value = saturation
        links.new(texture.outputs['Color'], tone.inputs['Color'])
        links.new(tone.outputs['Color'], front_back.inputs[1])
        # A small texture-derived fill avoids near-black cutouts in the fitted
        # indoor scene; it is not a measured material or lighting calibration.
        links.new(tone.outputs['Color'], shader.inputs['Emission Color'])
        shader.inputs['Emission Strength'].default_value = 0.18
        stem = 'bike_arena_%02d' % (index + 1)
        image = out / 'images' / (stem + '.png')
        label = out / 'labels' / (stem + '.txt')
        mask = out / 'instance_ids' / (stem + '.png')
        boxes, colors, visible, full, full_paths = render_instance_masks(
            scene, [actor], mask, out / 'instance_full')
        x1, y1, x2, y2 = boxes[0]
        ratio = visible[0] / full[0]
        if x1 <= 0 or y1 <= 0 or x2 >= 640 or y2 >= 480 or ratio < .85:
            raise ValueError('pilot target is truncated or not sufficiently visible')
        cls = {'upright': 0, 'fallen': 1}[category]
        label.write_text(f'{cls} {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} '
                         f'{(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n', encoding='utf-8')
        scene.view_settings.exposure = exposure
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        records.append({'image': relative(image), 'label': relative(label),
                        'split': 'unassigned_pilot', 'class': category,
                        'bbox_xyxy': boxes[0], 'instance_id_rgb': colors[0],
                        'visible_mask_pixels': visible[0], 'isolated_mask_pixels': full[0],
                        'visible_fraction': ratio, 'isolated_mask': full_paths[0],
                        'instance_id_image': relative(mask), 'source': source,
                        'camera_location': list(camera.location), 'camera_target': list(target),
                        'camera_lens_mm': camera.data.lens, 'exposure_ev': exposure,
                        'foreground_saturation_gain': saturation,
                        'foreground_emission_strength': 0.18,
                        'review_status': 'pending_visual_and_label_review'})
        bpy.data.objects.remove(bpy.data.objects[actor['geometry']], do_unlink=True)
    manifest = {'status': 'pilot_visual_review_pending', 'source_scene': relative(ARENA),
                'image_size': [640, 480], 'records': records,
                'scope': 'Two fitted scene pilots using real photo cutouts on planes. '
                         'Camera and positions are fitted. Planes are not 3D vehicles; '
                         'contact, shadows, source alpha and labels require visual review. '
                         'Dark examples use fitted exposure and foreground saturation; '
                         'these values are simulations, not calibrated lighting.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'images': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
