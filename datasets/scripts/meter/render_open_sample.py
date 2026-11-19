"""Render the declared open-stage sample in an isolated background Blender.

Called by build_current_sample.py inside its publication/receipt transaction.
Never saves the shared blend file.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_workspace import activate_scene, STATION
from datasets.scripts.meter.meter_station_plan import restore_station_production_frame
from datasets.scripts.meter.render_meter_batch import KEYPOINTS, project, texture_xy
from datasets.scripts.meter.render_meter_candidates import (
    STATION_PIVOT, station_dial_masks, station_pose_labels, station_pose_rays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    restore_station_production_frame(bpy)
    activate_scene(STATION, source_blend=plan.get('source_blend'))
    scene = bpy.context.scene
    pivot = scene.objects['Station dial tilt pivot']
    hands = [scene.objects['Station temperature_needle_rotate_Z'], scene.objects['Station humidity_needle_rotate_Z']]
    for hand in hands:
        hand.animation_data_clear()
    markers = [scene.objects['KPT-meter-' + key] for key in KEYPOINTS]
    endpoints = json.loads((ROOT / plan['endpoint_config']).read_text())['endpoints']
    for key, marker in zip(KEYPOINTS[4:], markers[4:]):
        assert marker.parent == pivot
        assert (marker.location - (texture_xy(endpoints[key]) - STATION_PIVOT)).length < 1e-5
    for index, hand in enumerate(hands):
        assert markers[index*2].parent == hand and markers[index*2+1].parent == hand
        assert markers[index*2].location.length < 1e-5
        assert (markers[index*2+1].location - Vector((.76 if index == 0 else .31, 0, 0))).length < 1e-5

    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.get_devices()
    assert any(d.type == 'OPTIX' for d in preferences.devices)
    for device in preferences.devices:
        device.use = device.type == 'OPTIX'
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = plan['cycles_samples']
    scene.cycles.use_denoising = False  # retain subpixel ticks/print at real camera scale
    scene.cycles.pixel_filter_type = 'BOX'
    scene.render.threads_mode, scene.render.threads = 'FIXED', 2
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    scene.view_settings.exposure += plan.get('render_exposure_ev', 0.)
    # Keep a restrained ambient reflection. A separate small emitter supplies
    # a physically reflected patch, without painting white over the needle.
    for light in (o for o in scene.objects if o.type == 'LIGHT'):
        light.visible_glossy = False
    material = scene.objects['Station PDF_face_inpainted'].data.materials[0]
    texture = next(n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE')
    images = {name: bpy.data.images.load(str(ROOT / value), check_existing=True)
              for name, value in plan['face_variants'].items()}
    bpy.ops.mesh.primitive_plane_add(size=1)
    card = bpy.context.object
    card.name = 'Meter local reflection card'
    card.visible_camera = card.visible_diffuse = card.visible_shadow = False
    emit = bpy.data.materials.new('Meter local reflection emitter')
    emit.use_nodes = True
    nodes = emit.node_tree.nodes
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Strength'].default_value = 60
    output = nodes.new('ShaderNodeOutputMaterial')
    emit.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    card.data.materials.append(emit)
    card.hide_render = True
    out = args.output.resolve()
    for name in ('foreground', 'masks', 'render_records'):
        (out / name).mkdir(parents=True, exist_ok=True)
    states = [s for s in plan['strata'] if s['parameters']['branch'] == 'blender_render']
    if args.limit:
        states = states[:args.limit]
    masks = {}
    records = []
    camera_lenses = {}
    for number, state in enumerate(states):
        p = state['parameters']
        scene.camera = scene.objects[p['view']]
        camera_lenses.setdefault(p['view'], scene.camera.data.lens)
        scene.camera.data.lens = camera_lenses[p['view']] * plan.get('camera_lens_scale', 1.)
        texture.image = images[p['face_material']]
        hands[0].rotation_euler.z = math.radians(-147 - 241*p['temperature_fraction'])
        hands[1].rotation_euler.z = math.radians(44 - 264*p['humidity_fraction'])
        card.hide_render = True
        scene.cycles.seed = plan.get('render_seed',1918)  # matched noise within contrasts
        bpy.context.view_layer.update()
        if p['view'] not in masks:
            masks[p['view']] = station_dial_masks(scene, pivot, out, p['view'])
            # The older arena helper gates boxes on 97% silhouette agreement.
            # Here all projected geometry is known: retain the full dial box
            # and record partial visibility instead of dropping an oblique view.
            evidence = masks[p['view']]
            visible = evidence['visible']['bbox_xyxy_pixels']
            isolated = evidence['isolated']['bbox_xyxy_pixels']
            if visible and isolated:
                evidence['bbox_xyxy_pixels'] = [min(visible[0],isolated[0]),min(visible[1],isolated[1]),
                    max(visible[2],isolated[2]),max(visible[3],isolated[3])]
                evidence['bbox_status'] = 'known_synthetic_extent_with_separate_visibility'
        points = {key: project(scene, scene.camera, marker.matrix_world.translation)
                  for key, marker in zip(KEYPOINTS, markers)}
        for index, hand in enumerate(hands):
            expected = hand.matrix_world @ Vector((.76 if index == 0 else .31, 0, 0))
            assert (expected - markers[index*2+1].matrix_world.translation).length < 1e-5
        if p['glare'] != 'none':
            hit = markers[KEYPOINTS.index(p['glare'])].matrix_world.translation.copy()
            normal = (pivot.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
            hit += normal*.045
            view = (scene.camera.matrix_world.translation - hit).normalized()
            reflected = (2*normal.dot(view)*normal-view).normalized()
            card.location = hit + reflected*4
            card.rotation_euler = (hit-card.location).to_track_quat('Z', 'Y').to_euler()
            card.scale = (.70, .22, 1)
            card.hide_render = False
        bpy.context.view_layer.update()
        path = out / 'foreground' / (state['id'] + '.png')
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rays = station_pose_rays(scene, scene.camera, markers)
        visibility = station_pose_labels(scene, scene.camera, markers,
                    [points[k] for k in KEYPOINTS], rays, path)
        for key in KEYPOINTS:
            if p['glare'] != 'none':
                visibility[key]['visibility'] = min(1, visibility[key]['visibility'])
                visibility[key]['reason'] = 'known_synthetic_geometry_reflection_readability_not_certified'
        box = masks[p['view']]['bbox_xyxy_pixels']
        if not box or not all(pt[2] > 0 and 0 <= pt[0] < 1 and 0 <= pt[1] < 1 for pt in points.values()):
            raise ValueError(json.dumps({'state':state['id'],'camera':p['view'],'bbox':box,'points':points}))
        record = {'id': state['id'], 'parameters': p, 'keypoint_order': list(KEYPOINTS),
                  'keypoints_xy_depth_normalized_candidate': points,
                  'keypoint_visibility': visibility, 'keypoint_ray_evidence': rays,
                  'dial_raster_evidence': masks[p['view']], 'bbox_xyxy_pixels': box,
                  'foreground_rgba': 'foreground/' + path.name,
                  'display_roi_quad': [project(scene, scene.camera,
                       scene.objects['ROI-power-meter-'+key].matrix_world.translation)
                       for key in ('TL', 'TR', 'BR', 'BL')],
                  'geometry_checks': '8 named parent/local contracts, two independent rotations, projected tips and four texture endpoints checked',
                  'physical_reading_status': 'synthetic linear scale; physical registration not established'}
        (out/'render_records'/(state['id']+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
        print('METER_OPEN_PROGRESS', number+1, '/', len(states), state['id'], flush=True)
    (out/'render_records.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
