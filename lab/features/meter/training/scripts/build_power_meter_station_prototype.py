"""Make an editable 9011 station prototype and three single-frame previews.

Blender 5.2: blender -b SOURCE --python-exit-code 1 --python THIS --
--output NEW_DIRECTORY [--ground-angle-deg 65] [--device OPTIX].
Display characters are synthetic.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/meter/.")

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_meter_room import add_room_environment
from render_meter_batch import KEYPOINTS, project, texture_xy


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'datasets/assets/blender/meter/blender_9011_pbr_v2_20261110/meter_9011_pbr.blend'
ENDPOINTS = ROOT / 'lab/features/meter/training/config/generated_face_endpoints_v2.json'
DIAL_NAMES = ('metal_shell', 'PDF_face_inpainted', 'metal_bezel_bezier',
              'temperature_needle_rotate_Z', 'temperature_pivot_cap',
              'humidity_needle_rotate_Z', 'humidity_pivot_cap', 'glass_reflection_surface')
DIAL_PIVOT = Vector((0, -1.04, .02))
PLINTH_Y = -5.785
PLINTH_HEIGHT = .24
READING = '1234'
DIGIT_STROKES = {'1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg'}


def material(name, color, roughness, metallic=0.0, emission=0.0, specular=0.5):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Roughness'].default_value = roughness
    node.inputs['Metallic'].default_value = metallic
    node.inputs['Specular IOR Level'].default_value = specular
    if emission:
        node.inputs['Emission Color'].default_value = (*color, 1)
        node.inputs['Emission Strength'].default_value = emission
    return mat


def mesh(collection, name, points, faces, mat):
    data = bpy.data.meshes.new(name)
    data.from_pydata(points, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    data.materials.append(mat)
    return obj


def box(collection, name, center, size, mat, bevel=0):
    x, y, z = (dimension / 2 for dimension in size)
    points = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
              (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    faces = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    obj = mesh(collection, name, points, faces, mat)
    obj.location = center
    if bevel:
        modifier = obj.modifiers.new('Small manufactured edge', 'BEVEL')
        modifier.width = bevel
        modifier.segments = 3
    return obj


def cylinder(collection, name, center, radius, height, mat):
    count = 64
    points = [(radius * math.cos(i * math.tau / count), y,
               radius * math.sin(i * math.tau / count))
              for y in (-height / 2, height / 2) for i in range(count)]
    faces = [tuple(range(count)), tuple(reversed(range(count, 2 * count)))]
    faces.extend((i, count+i, count+(i+1) % count, (i+1) % count) for i in range(count))
    obj = mesh(collection, name, points, faces, mat)
    obj.location = center
    for face in obj.data.polygons[2:]:
        face.use_smooth = True
    return obj


def display_digit(collection, index, digit, x, y, z, mat):
    points, faces = [], []
    segments = {'a': (0, .205, .150, .027), 'b': (.091, .105, .027, .166),
                'c': (.091, -.105, .027, .166), 'd': (0, -.205, .150, .027),
                'e': (-.091, -.105, .027, .166), 'f': (-.091, .105, .027, .166),
                'g': (0, 0, .150, .027)}
    for key in DIGIT_STROKES[digit]:
        cx, cy, width, height = segments[key]
        u, v = width / 2, height / 2
        corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
        start = len(points)
        points.extend((x + cx + a + .09*(cy+b), y + cy+b, z) for a, b in corners)
        faces.append(tuple(range(start, start + 4)))
    return mesh(collection, f'GEO-power-meter-digit-{index}-{digit}', points, faces, mat)


def station(collection):
    housing = material('Power meter black polymer', (.006, .007, .009), .49, specular=.22)
    rim = material('Power meter dark window rim', (.026, .031, .036), .31)
    steel = material('Station support chrome', (.60, .61, .62), .24, 1.0)
    lcd = material('Power meter pale LCD', (.75, .85, .91), .31, emission=.67)
    ink = material('Power meter display segments', (.001, .002, .004), .81, specular=.02)
    root = bpy.data.objects.new('Power_meter_station_root', None)
    collection.objects.link(root)
    root.empty_display_size = .20
    root['display_value_example'] = READING
    root['display_unit_example'] = 'kWh'
    root['dimensions_basis'] = 'photo proportions in 9011 source scene units'
    box(collection, 'GEO-power-meter-cabinet', (.04, -3.48, -.40), (2.10, 4.44, 1.30), housing, .025)
    box(collection, 'GEO-power-meter-plinth', (.04, PLINTH_Y, -.33), (2.34, PLINTH_HEIGHT, 1.56), housing, .025)
    cylinder(collection, 'GEO-temperature-dial-support', (0, -1.22, -.05), .83, .12, steel)
    box(collection, 'GEO-temperature-dial-neck', (0, -1.11, -.01), (.43, .20, .20), steel, .015)

    left, right, top, bottom, depth = -.49, .76, -1.62, -2.61, .282
    box(collection, 'GEO-power-meter-window-rim',
        ((left+right)/2, (top+bottom)/2, .254), (1.34, 1.07, .038), rim, .014)
    roi = [(left, top, depth), (right, top, depth), (right, bottom, depth), (left, bottom, depth)]
    screen = mesh(collection, 'GEO-power-meter-screen-ROI', roi, [(3, 2, 1, 0)], lcd)
    screen['roi_vertex_order'] = 'top_left,top_right,bottom_right,bottom_left'
    screen['annotation_role'] = 'display quad for ROI and OCR geometry'
    for index, point in enumerate(roi):
        corner = bpy.data.objects.new(f'ROI-power-meter-{("TL", "TR", "BR", "BL")[index]}', None)
        collection.objects.link(corner)
        corner.empty_display_size = .05
        corner.location = point
    for index, digit in enumerate(READING):
        display_digit(collection, index, digit, -.225 + index*.248, -1.984, depth+.008, ink)
    font = bpy.data.curves.new('Power meter display unit', type='FONT')
    font.body = 'kWh'
    font.align_x = 'RIGHT'
    font.size = .17
    unit = bpy.data.objects.new('GEO-power-meter-unit', font)
    collection.objects.link(unit)
    unit.location = (.648, -2.453, depth+.008)
    font.materials.append(ink)
    for obj in collection.objects:
        if obj != root:
            obj.parent = root
    return roi


def aim(obj, target):
    forward = (Vector(target) - obj.location).normalized()
    right = forward.cross(Vector((0, 1, 0))).normalized()
    up = right.cross(forward).normalized()
    obj.rotation_euler = Matrix((right, up, -forward)).transposed().to_euler()


def add_area(collection, name, location, target, energy, size):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy, data.shape, data.size = energy, 'DISK', size
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = location
    aim(obj, target)


def camera(collection, name, location, target, lens):
    data = bpy.data.cameras.new(name)
    data.lens = lens
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = location
    aim(obj, target)
    return obj


def tilted_dial(scene, dial, ground_angle):
    collection = bpy.data.collections.new('Station 9011 dial assembly')
    scene.collection.children.link(collection)
    pivot = bpy.data.objects.new('Station dial tilt pivot', None)
    collection.objects.link(pivot)
    pivot.empty_display_type = 'ARROWS'
    pivot.empty_display_size = .22
    pivot.location = DIAL_PIVOT
    # +Y is up and the display faces +Z. Negative X rotation tips the top towards -Z.
    pivot.rotation_euler.x = math.radians(ground_angle - 90)
    pivot['ground_angle_deg_at_build'] = ground_angle
    pivot['orientation'] = 'Adjust local X rotation at the support contact pivot'

    copies = {}
    for source in dial:
        if source.parent is not None:
            raise ValueError(f'Expected an unparented calibrated dial object: {source.name}')
        obj = source.copy()
        obj.name = f'Station {source.name}'
        collection.objects.link(obj)
        obj.parent = pivot
        obj.matrix_parent_inverse = Matrix.Translation(-DIAL_PIVOT)
        copies[source.name] = obj

    def landmark(name, parent, position):
        marker = bpy.data.objects.new(f'KPT-meter-{name}', None)
        collection.objects.link(marker)
        marker.parent = parent
        marker.location = position
        marker.empty_display_size = .027
        return marker

    points = []
    for names, hand, tip in ((KEYPOINTS[:2], 'temperature_needle_rotate_Z', .76),
                             (KEYPOINTS[2:4], 'humidity_needle_rotate_Z', .31)):
        points.extend(landmark(name, copies[hand], (x, 0, 0)) for name, x in zip(names, (0, tip)))
    endpoints = json.loads(ENDPOINTS.read_text(encoding='utf-8'))['endpoints']
    points.extend(landmark(name, pivot, texture_xy(endpoints[name]) - DIAL_PIVOT) for name in KEYPOINTS[4:])
    return pivot, points


def draft_scene(equipment, device):
    scene = bpy.data.scenes.new('Station prototype previews')
    scene.collection.children.link(equipment)
    stage = bpy.data.collections.new('Preview room, lamps and cameras')
    scene.collection.children.link(stage)
    room_context = add_room_environment(scene, stage, box, material, add_area)
    front = camera(stage, 'Preview vehicle front', (-.78, -1.54, 14.5), (-.78, -1.18, 0), 35)
    angled = camera(stage, 'Preview close three quarter', (3.6, -.56, 10.25), (0, -2.35, 0), 35)
    side = camera(stage, 'Preview right side orthographic', (13, -2.43, -.38), (0, -2.43, -.38), 35)
    side.data.type = 'ORTHO'
    side.data.ortho_scale = 10.5
    scene.camera = front
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    if device == 'OPTIX':
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'OPTIX'
        preferences.get_devices()
        selected = [item for item in preferences.devices if item.type == 'OPTIX']
        if not selected:
            raise RuntimeError('No OptiX device available')
        for item in preferences.devices:
            item.use = item in selected
        scene.cycles.device = 'GPU'
    else:
        scene.cycles.device = 'CPU'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_percentage = 100
    return scene, front, angled, side, room_context


def _render(scene, camera_obj, path, size, roi, dial_points, assembly_matrix, top, base):
    scene.camera = camera_obj
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.filepath = str(path)
    bpy.context.view_layer.update()
    projected = [project(scene, camera_obj, point) for point in [*roi, *dial_points]]
    if any(p[2] <= 0 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in projected):
        raise ValueError(f'Projection falls outside preview {path.name}')
    rim = [project(scene, camera_obj, assembly_matrix @ Vector((1.04 * math.cos(i * math.tau / 128),
                                      1.04 * math.sin(i * math.tau / 128), .09))) for i in range(128)]
    projected_rim = ([min(p[axis] for p in rim) * size[axis] for axis in (0, 1)] +
                     [max(p[axis] for p in rim) * size[axis] for axis in (0, 1)])
    screen_quad = [p[:2] for p in projected[:4]]
    screen_area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(screen_quad, screen_quad[1:] + screen_quad[:1]))) / 2
    bpy.ops.render.render(write_still=True)
    return {'image': path.name, 'camera': camera_obj.name, 'size': size,
            'screen_roi_xy_normalized': screen_quad,
            'screen_quad_area_pixels': screen_area * size[0] * size[1],
            'dial_lean_line_xy_normalized': [project(scene, camera_obj, p)[:2] for p in (base, top)],
            'dial_rim_xyxy_pixels': projected_rim,
            'dial_points_xy_normalized': dict(zip(KEYPOINTS, [p[:2] for p in projected[4:]]))}


def main():
    raise RuntimeError("Retired production entry; use datasets/scripts/meter/ and its source.yaml")


def _legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('CPU', 'OPTIX'), default='OPTIX')
    parser.add_argument('--ground-angle-deg', type=float, default=65.0,
                        help='Angle of the display plane relative to ground, 60–70 degrees')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    from meter_production_guard import guarded, blender_media, reload_unpacked_media
    with guarded('build_power_meter_station_prototype', args, __file__, bpy.data.filepath, extra_inputs=[ENDPOINTS] + [ROOT / f'Archive/Assets/captures/site_capture_20261112/onboard_camera_images/vehicle_{i}.png' for i in ('0024','0100','0103','0104')], code_dependencies=[Path(__file__).with_name('render_meter_batch.py'), Path(__file__).with_name('power_meter_room.py')], media_probe=lambda: blender_media(bpy), media_reload=lambda: reload_unpacked_media(bpy)) as run:
        if run.reused:
            return
        if not 60 <= args.ground_angle_deg <= 70:
            parser.error('Ground angle must be between 60 and 70 degrees')
        output = args.output.resolve()
        base = (ROOT / 'datasets/assets/blender/meter').resolve()
        if output.exists() or not output.is_relative_to(base) or output == base:
            parser.error('Output must be a new directory under datasets/assets/blender/meter')
        if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != SOURCE.resolve():
            parser.error('Open the calibrated 9011 PBR source blend')
        original = bpy.context.scene
        dial = [original.objects[name] for name in DIAL_NAMES]
        equipment = bpy.data.collections.new('Power meter station prototype')
        roi = station(equipment)
        preview, front, angle, side, room_context = draft_scene(equipment, args.device)
        pivot, points = tilted_dial(preview, dial, args.ground_angle_deg)
        preview.frame_start, preview.frame_end = original.frame_start, original.frame_end
        bpy.context.window.scene = preview
        preview.frame_set(original.frame_current)
        bpy.context.view_layer.update()
        assembly_matrix = pivot.matrix_world @ Matrix.Translation(-DIAL_PIVOT)
        pivot_world = assembly_matrix @ DIAL_PIVOT
        top_world = assembly_matrix @ (DIAL_PIVOT + Vector((0, 2.08, 0)))
        up = top_world - pivot_world
        ground_angle = math.degrees(math.atan2(up.y, abs(up.z)))
        if abs(ground_angle - args.ground_angle_deg) > 1e-4 or up.z >= 0 or (pivot_world-DIAL_PIVOT).length > 1e-6:
            raise RuntimeError('Station dial tilt direction, angle or support contact does not match')
        world_points = [point.matrix_world.translation for point in points]
        output.mkdir(parents=True)
        images = [_render(preview, view, output / name, size, roi, world_points, assembly_matrix, top_world, pivot_world)
                  for view, name, size in ((front, 'prototype_vehicle_front.png', (640, 480)),
                                           (angle, 'prototype_three_quarter.png', (640, 480)))]
        # Keep the actual side elevation visible when the context wall sits between camera and station.
        room_context.hide_render = True
        try:
            side_image = _render(preview, side, output / 'prototype_right_side.png', (640, 480), roi, world_points,
                                assembly_matrix, top_world, pivot_world)
            side_image['room_context_visible'] = False
            images.append(side_image)
        finally:
            room_context.hide_render = False
        preview.camera = front
        preview.render.resolution_x, preview.render.resolution_y = (640, 480)
        preview.render.filepath = '//prototype_vehicle_front.png'
        bpy.ops.wm.save_as_mainfile(filepath=str(output / 'meter_9011_power_meter_station.blend'))
        report = {'status': 'editable visual prototype', 'source_blend': SOURCE.relative_to(ROOT).as_posix(),
                  'endpoint_config': ENDPOINTS.relative_to(ROOT).as_posix(),
                  'source_photos': [f'Archive/Assets/captures/site_capture_20261112/onboard_camera_images/vehicle_{id}.png'
                                    for id in ('0024', '0100', '0103', '0104')],
                  'unit_scale_basis': 'approximate proportions observed in car photos, not measured dimensions',
                  'dial_tilt': {'ground_angle_requested_deg': args.ground_angle_deg,
                                'ground_angle_measured_deg': ground_angle,
                                'pivot_world': list(pivot_world), 'top_world': list(top_world),
                                'front_direction': '+Z', 'up_direction': '+Y',
                                'source_scene': original.name, 'assembly_object': pivot.name},
                  'synthetic_display_example': {'text': READING, 'unit': 'kWh'},
                  'environment': 'procedural indoor track with sky contribution at window and interior fixtures',
                  'screen_roi_order': ['TL', 'TR', 'BR', 'BL'], 'previews': images}
        (output / 'prototype_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                       encoding='utf-8')
        print('POWER_METER_PROTOTYPE', output, flush=True)


if __name__ == '__main__':
    main()
