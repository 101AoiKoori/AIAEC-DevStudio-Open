"""Install or render the nine stored cameras in the Waste Bins Photo scene."""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common import blender_workspace as workspace
from datasets.scripts.common.blender_masks import GEOMETRY_TYPES, configure_device, masks, read_rgba
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import bin_photo_orbit_diagnostic_dir
from datasets.scripts.waste_bin_state.bin_render_contract import (
    BIN_NAMES, MASK_COLORS, MIN_MOUTH_THICKNESS_PX, OPEN_POSES, attach_state_evidence, decode_mask, mouth_thickness,
)
from datasets.scripts.waste_bin_state.render_blender_waste_bin import (
    correct_hanging_hinge_clearance, create_mouth_proxies,
)
from lab.training_common.production_guard import generation_guard

SCENE = workspace.BINS_PHOTO_SCENE
TARGET_PROPERTY = 'aec_bin_photo_orbit_target'


def parse_bins(text):
    if text == 'none':
        return {}
    poses = {}
    for entry in text.split(','):
        name, _, pose = entry.partition('=')
        if name not in BIN_NAMES or name in poses or pose not in ('closed', 'hanging_back'):
            raise argparse.ArgumentTypeError('use color=closed or color=hanging_back')
        poses[name] = pose
    return poses


def install_cameras():
    scene = bpy.data.scenes[SCENE]
    bpy.context.window.scene = scene
    subjects = [obj for obj in scene.objects if obj.get('aec_source_name') in
                {name + ' body' for name in BIN_NAMES}]
    if len(subjects) != len(BIN_NAMES):
        raise ValueError('Waste Bins Photo must contain all four bodies')
    workspace.store_orbit_cameras(scene, subjects, TARGET_PROPERTY)
    bpy.context.preferences.filepaths.save_version = 0
    if bpy.ops.wm.save_as_mainfile(filepath=str(workspace.WORKSPACE_BLEND)) != {'FINISHED'}:
        raise RuntimeError('Could not save the bin orbit cameras')


def declaration(poses, probes, scene_ref, compare_lids=None):
    if compare_lids:
        cases = [(f'{view}_{pose}', view, {compare_lids: pose}, False)
                 for view, _, _ in workspace.ORBIT_VIEWS for pose in ('hanging_back', 'closed')]
    else:
        cases = [(view, view, poses, False) for view, _, _ in workspace.ORBIT_VIEWS]
    if probes:
        cases.extend([
            ('single_hanging', 'left_near', {'blue': 'hanging_back'}, False),
            ('neighbor_overlap', 'center_mid', {'red': 'closed', 'green': 'hanging_back'}, True),
            ('empty', 'center_mid', {}, False),
        ])
    asset = workspace.source_ref(workspace.WORKSPACE_BLEND, workspace.BIN_PHOTO)
    strata = [
        {'id': name, 'min_count': 1, 'max_count': 1, 'conditions': [],
         'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
         'failure_event': 'whole_bin_or_state_visibility_incomplete',
         'background': {'scene': scene_ref, 'scene_name': SCENE},
         'target': {'asset': asset, 'poses': settings}, 'parameters': {'view': view, 'overlap': overlap},
         'labels': {'format': 'yolo_detection', 'open': 0, 'closed': 1},
         'augmentation': {'mode': 'none'}, 'sources': [scene_ref, asset]}
        for name, view, settings, overlap in cases
    ]
    return {'schema_version': 1, 'model_key': 'waste_bin_state', 'target_split': 'test',
            'budget': len(strata), 'uncertainty_weight': 0, 'strata': strata,
            'independence_basis': 'one shared Blender scene and bin asset',
            'constraints': {'resolution': [640, 480], 'min_visible_fraction': .70, 'min_visible_pixels': 24},
            'purpose': 'Review stored orbit views, whole-bin masks and lid-state evidence'}


def measure_bins(scene, camera, out, stem, poses, parts, proxies):
    """Use the outline of the complete visible assembly for each box."""
    visible_parts = {}
    printed = []
    for name in poses:
        decals = {scene.objects[name + ' front marking'], scene.objects[name + ' marking border']}
        printed.extend(decals)
        visible_parts[name] = [obj for obj in parts[name] if obj not in decals]
    for obj in printed:
        obj.hide_render = True
    try:
        outline_path, outline = masks(scene, camera, visible_parts, out, stem, 'target', palette=MASK_COLORS)
        evidence_parts = {
            name: ([proxies[name]] if pose == 'hanging_back' else
                   [scene.objects[name + ' lid'], scene.objects[name + ' raised panel']])
            for name, pose in poses.items()
        }
        mouths = [proxies[name] for name, pose in poses.items() if pose == 'hanging_back']
        state_path, evidence = masks(scene, camera, evidence_parts, out, stem, 'state', mouths, palette=MASK_COLORS)
    finally:
        for obj in printed:
            obj.hide_render = False
    by_bin = {item['bin']: item for item in evidence}
    result = []
    for item in outline:
        name = item['bin']
        pose = poses[name]
        label = attach_state_evidence(item, by_bin[name], 'closed' if pose == 'closed' else 'open')
        if pose == 'hanging_back':
            opening = decode_mask(read_rgba(out / state_path), name)
            thickness = mouth_thickness(opening)
            label['state_evidence']['max_vertical_mouth_pixels'] = thickness
            if thickness < MIN_MOUTH_THICKNESS_PX:
                label['label_eligible'] = False
                label['rejection_reasons'].append('state_evidence:mouth_too_thin')
        result.append(label)
    return outline_path, state_path, result


def render(out, scene, run, samples):
    for marker in scene.timeline_markers:
        marker.camera = None
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    configure_device(scene, 'OPTIX')
    for folder in ('images', 'masks', 'labels_candidate'):
        (out / folder).mkdir(parents=True)
    pivots = {name: scene.objects[name + ' lid pivot - rotate X'] for name in BIN_NAMES}
    if any('hanging_back' in job['target']['poses'].values() for job in run.plan['strata']):
        correct_hanging_hinge_clearance(list(pivots.values()))
    parts = {name: [obj for obj in scene.objects if obj.type in GEOMETRY_TYPES and obj.name.startswith(name + ' ')]
             for name in BIN_NAMES}
    red_group = set(parts['red']) | {pivots['red']}
    red_roots = {obj: obj.location.copy() for obj in red_group if obj.parent not in red_group}
    proxies = create_mouth_proxies(scene)
    records = []
    for job in run.plan['strata']:
        if not job['planned_count']:
            continue
        stem, params, poses = job['id'], job['parameters'], job['target']['poses']
        camera = scene.objects[workspace.orbit_camera_name(SCENE, params['view'])]
        scene.camera = camera
        for obj, original in red_roots.items():
            obj.location = original + (Vector((.51, -.85, 0)) if params['overlap'] else Vector((0, 0, 0)))
        for name, pivot in pivots.items():
            pivot.rotation_euler.x = math.radians(OPEN_POSES.get(poses.get(name), 0))
            for obj in parts[name]:
                obj.hide_render = name not in poses
        bpy.context.view_layer.update()
        image = out / 'images' / f'{stem}.png'
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        mask, state_mask, labels = measure_bins(scene, camera, out, stem, poses, parts, proxies) if poses else (None, None, [])
        if scene.camera != camera:
            raise RuntimeError('Blender changed the requested camera')
        label_path = None
        if all(item['label_eligible'] for item in labels):
            label_path = out / 'labels_candidate' / f'{stem}.txt'
            lines = []
            for item in labels:
                x0, y0, x1, y1 = item['candidate_xyxy_normalized']
                cls = int(item['state'] == 'closed')
                lines.append(f'{cls} {(x0+x1)/2:.8f} {(y0+y1)/2:.8f} {x1-x0:.8f} {y1-y0:.8f}')
            label_path.write_text(''.join(line + '\n' for line in lines), encoding='utf8')
        actual = {'scene': SCENE, 'camera': camera.name, 'azimuth_deg': camera['azimuth_deg'],
                  'elevation_deg': camera['elevation_deg'], 'pitch_deg': camera['pitch_deg'],
                  'distance': camera['distance'], 'poses': poses, 'mask_visible': mask,
                  'mask_state_visible': state_mask, 'instances': labels}
        run.record_sample(stem, output=image, sources=job['sources'], parameters=params, actual=actual)
        records.append({'image': image.relative_to(out).as_posix(), 'id': stem,
                        'candidate_label': label_path.relative_to(out).as_posix() if label_path else None,
                        **actual})
    (out / 'manifest.json').write_text(json.dumps({'status': 'candidate', 'records': records},
                                                ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--install-cameras', action='store_true')
    mode.add_argument('--out', type=Path)
    parser.add_argument('--bins', type=parse_bins)
    parser.add_argument('--compare-lids', choices=BIN_NAMES, help='show both lid states of one bin at every view')
    parser.add_argument('--visibility-probes', action='store_true')
    parser.add_argument('--samples', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    blend = workspace.WORKSPACE_BLEND
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != blend:
        parser.error('load the unified workspace file in background Blender')
    if args.install_cameras:
        if args.bins is not None or args.compare_lids or args.visibility_probes:
            parser.error('view cases require --out')
        install_cameras()
        return
    if args.compare_lids and (args.bins is not None or args.visibility_probes):
        parser.error('--compare-lids cannot be combined with --bins or --visibility-probes')
    out = (ROOT / args.out).resolve()
    expected_out = bin_photo_orbit_diagnostic_dir(compare_lids=bool(args.compare_lids))
    if out != expected_out.resolve() or (out.exists() and any(out.iterdir())):
        parser.error(f'use the empty {expected_out.relative_to(ROOT).as_posix()} diagnostic directory')
    workspace.activate_scene(SCENE)
    asset_manifest = source_path('waste_bin_state', 'arena_bin_manifest')
    photo = ROOT / json.loads(asset_manifest.read_text(encoding='utf8'))['photo']
    poses = args.bins
    if poses is None:
        poses = parse_bins('blue=closed,red=hanging_back,green=closed,gray=hanging_back')
    plan = declaration(poses, args.visibility_probes, blend.relative_to(ROOT).as_posix(), args.compare_lids)
    code = [Path(__file__), ROOT / 'datasets/scripts/common/blender_workspace.py',
            ROOT / 'datasets/scripts/common/blender_masks.py',
            ROOT / 'datasets/scripts/waste_bin_state/bin_render_contract.py',
            ROOT / 'datasets/scripts/waste_bin_state/render_blender_waste_bin.py']
    config = ROOT / 'datasets/scripts/waste_bin_state/config'
    with generation_guard('waste_bin_state.photo_orbit', [photo], [out],
                          {'blender_version': bpy.app.version_string, 'samples': args.samples},
                          code_paths=code, procedural_inputs=[blend, asset_manifest, config / 'dataset.yaml',
                                                              config / 'source.yaml'], root=ROOT,
                          production_plan=plan, sample_output_paths=[out / 'images'],
                          auxiliary_output_paths=[out / 'masks']) as run:
        if not run.reused:
            render(out, bpy.context.scene, run, args.samples)


if __name__ == '__main__':
    main()
