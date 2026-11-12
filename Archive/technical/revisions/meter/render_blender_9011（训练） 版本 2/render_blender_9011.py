"""Run inside Blender: build a PDF-inspired dual-hand meter and render samples.

The texture and calibration are derived from the user's PDF. Geometry and
visibility are approximate and require review against the physical meter.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

KEYPOINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
             'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')


def xy(pixel, z=0.0):
    return Vector(((pixel[0] - 627) / 627, (627 - pixel[1]) / 627, z))


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def material(name, color, metallic=0.0, roughness=.4):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    node = m.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Metallic'].default_value = metallic
    node.inputs['Roughness'].default_value = roughness
    return m


def disk_texture(texture_path):
    vertices = [(0, 0, .065)]
    n = 128
    vertices += [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n), .065)
                 for i in range(n)]
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    mesh = bpy.data.meshes.new('dial_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new('generated_face', mesh)
    bpy.context.collection.objects.link(obj)
    uv = mesh.uv_layers.new(name='face_uv')
    for loop in mesh.loops:
        x, y, _ = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = ((x + 1) / 2, (y + 1) / 2)
    m = material('generated face texture', (1, 1, 1), roughness=.65)
    node = m.node_tree.nodes.new('ShaderNodeTexImage')
    node.image = bpy.data.images.load(str(texture_path), check_existing=False)
    node.image.pack()
    m.node_tree.links.new(node.outputs['Color'], m.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    obj.data.materials.append(m)
    return obj


def hand(name, center, length, z, color):
    # Editable closed Bezier outline; local +X points toward the needle tip.
    w = .017 if name == 'temperature' else .012
    points = [(-.05, -w, 0), (0, -w * 1.5, 0), (length * .72, -w * .7, 0),
              (length, 0, 0), (length * .72, w * .7, 0), (0, w * 1.5, 0), (-.05, w, 0)]
    curve = bpy.data.curves.new(name + '_needle_bezier', 'CURVE')
    curve.dimensions = '2D'
    curve.fill_mode = 'BOTH'
    curve.extrude = .003
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points) - 1)
    for point, coord in zip(spline.bezier_points, points):
        point.co = coord
        point.handle_left_type = point.handle_right_type = 'VECTOR'
    spline.use_cyclic_u = True
    obj = bpy.data.objects.new(name + '_needle_rotate_Z', curve)
    bpy.context.collection.objects.link(obj)
    obj.location = (center.x, center.y, z)
    obj.data.materials.append(material(name + '_needle_black', color, metallic=.2, roughness=.34))
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=.036 if name == 'temperature' else .026,
                                        depth=.012, location=(center.x, center.y, z + .003))
    bpy.context.object.name = name + '_pivot_cap'
    bpy.context.object.data.materials.append(material(name + '_cap', (.025, .025, .025), metallic=.3))
    return obj


def make_glass():
    bpy.ops.mesh.primitive_circle_add(vertices=128, radius=.98, fill_type='NGON', location=(0, 0, .145))
    glass = bpy.context.object
    glass.name = 'glass_reflection_surface'
    m = bpy.data.materials.new('glass transparent plus specular approximation')
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = .20
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    glossy = nodes.new('ShaderNodeBsdfGlossy')
    glossy.inputs['Roughness'].default_value = .08
    links = m.node_tree.links
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(glossy.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs['Surface'])
    glass.data.materials.append(m)
    return glass


def area_light(name, location, energy, size):
    light = bpy.data.lights.new(name, 'AREA')
    light.energy = energy
    light.shape = 'DISK'
    light.size = size
    obj = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, (0, 0, 0))
    return obj


def angles_from_ticks(ticks, center):
    result = []
    for tick in ticks:
        p = xy(tick['end']) - center
        a = math.atan2(p.y, p.x)
        if result:
            while a > result[-1] + math.pi:
                a -= 2 * math.pi
            while a < result[-1] - math.pi:
                a += 2 * math.pi
        result.append(a)
    return result


def angle_at(angles, fraction):
    index = fraction * (len(angles) - 1)
    low = min(int(index), len(angles) - 2)
    t = index - low
    return (1 - t) * angles[low] + t * angles[low + 1]


def camera_point(scene, camera, point):
    view = world_to_camera_view(scene, camera, Vector(point))
    return (view.x, 1 - view.y, view.z)


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def screen_polygon(scene, camera, obj):
    return [camera_point(scene, camera, obj.matrix_world @ point.co)[:2]
            for point in obj.data.splines[0].bezier_points]


def load_render_pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return np.array(image.pixels[:], dtype=np.float32).reshape(512, 512, 4)
    finally:
        bpy.data.images.remove(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ticks', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(args.output)
    ticks = json.loads(Path(args.ticks).read_text(encoding='utf-8-sig'))
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.world.color = (.35, .35, .35)
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=1.04, depth=.12, location=(0, 0, 0))
    shell = bpy.context.object
    shell.name = 'metal_shell'
    shell.data.materials.append(material('metal', (.55, .56, .58), metallic=.85, roughness=.24))
    bevel = shell.modifiers.new('editable_shell_bevel', 'BEVEL')
    bevel.width = .022
    bevel.segments = 3
    disk_texture(out / 'face_texture.png')
    bpy.ops.curve.primitive_bezier_circle_add(radius=1.015, location=(0, 0, .09))
    bezel = bpy.context.object
    bezel.name = 'metal_bezel_bezier'
    bezel.data.bevel_depth = .025
    bezel.data.bevel_resolution = 4
    bezel.data.resolution_u = 32
    bezel.data.materials.append(material('bezel', (.65, .65, .65), metallic=.9, roughness=.2))
    temp_center, humid_center = xy(ticks['temperature_center']), xy(ticks['humidity_center'])
    temp = hand('temperature', temp_center, .76, .105, (.01, .01, .01))
    humid = hand('humidity', humid_center, .31, .12, (.01, .01, .01))
    glass = make_glass()
    bpy.ops.object.camera_add(location=(0, 0, 4.2))
    camera = bpy.context.object
    camera.name = 'training_camera'
    camera.data.lens = 48
    scene.camera = camera
    look_at(camera, (0, 0, 0))
    key = area_light('moving_softbox', (1, 1, 3), 400, 1.8)
    area_light('fill', (-2, -1, 3), 220, 3)
    # Save a self-contained, editable model with packed derived PDF texture.
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'meter_9011_approx.blend'))
    image_dir, label_dir = out / 'images' / 'train', out / 'labels' / 'train'
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    temperature_angles = angles_from_ticks(ticks['temperature'], temp_center)
    humidity_angles = angles_from_ticks(ticks['humidity'], humid_center)
    randomizer = random.Random(args.seed)
    manifest = {'purpose': 'PDF-inspired Blender smoke test; not a validated dataset',
                'source': 'AI-generated face, based on PDF and commercial reference; rights and fidelity unverified',
                'geometry': 'approximate generated image geometry, not measured dimensions',
                'calibration_status': 'illustrative equally spaced endpoints; generated markings and physical meter not verified',
                'keypoint_order': KEYPOINTS, 'seed': args.seed, 'samples': []}
    for index in range(args.count):
        tf, hf = randomizer.uniform(.05, .95), randomizer.uniform(.05, .95)
        temp.rotation_euler.z = angle_at(temperature_angles, tf)
        humid.rotation_euler.z = angle_at(humidity_angles, hf)
        camera.location = (randomizer.uniform(-.9, .9), randomizer.uniform(-.8, .8),
                           randomizer.uniform(3.8, 4.8))
        camera.data.lens = randomizer.uniform(44, 52)
        look_at(camera, (0, 0, 0))
        key.location = (randomizer.uniform(-1.8, 1.8), randomizer.uniform(-1.8, 1.8),
                        randomizer.uniform(2.0, 3.5))
        key.data.energy = randomizer.uniform(250, 650)
        key.data.size = randomizer.uniform(.5, 2.3)
        look_at(key, (0, 0, 0))
        scene.cycles.seed = index
        bpy.context.view_layer.update()
        # A clean comparison render estimates whether glass highlights obscure a point.
        glass.hide_render = True
        clean_path = out / 'render_work_clean.png'
        scene.render.filepath = str(clean_path)
        bpy.ops.render.render(write_still=True)
        clean = load_render_pixels(clean_path)
        clean_path.unlink()
        glass.hide_render = False
        name = 'meter_{:05d}'.format(index)
        scene.render.filepath = str(image_dir / (name + '.png'))
        bpy.ops.render.render(write_still=True)
        reflected = load_render_pixels(image_dir / (name + '.png'))
        tip_t = temp.matrix_world @ Vector((.76, 0, 0))
        tip_h = humid.matrix_world @ Vector((.31, 0, 0))
        points3 = [Vector(temp.location), tip_t, Vector(humid.location), tip_h,
                   xy(ticks['temperature'][0]['end'], .065), xy(ticks['temperature'][-1]['end'], .065),
                   xy(ticks['humidity'][0]['end'], .065), xy(ticks['humidity'][-1]['end'], .065)]
        points = [camera_point(scene, camera, p) for p in points3]
        outline = [camera_point(scene, camera, (1.04 * math.cos(i * math.tau / 128),
                                               1.04 * math.sin(i * math.tau / 128), .1))
                   for i in range(128)]
        x0, x1 = max(0., min(p[0] for p in outline)), min(1., max(p[0] for p in outline))
        y0, y1 = max(0., min(p[1] for p in outline)), min(1., max(p[1] for p in outline))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError('generated meter outside frame')
        tpoly, hpoly = screen_polygon(scene, camera, temp), screen_polygon(scene, camera, humid)
        keypoints, states, visibility_evidence = [], [], []
        for k, (x, y, depth) in enumerate(points):
            if not (depth > 0 and 0 <= x < 1 and 0 <= y < 1):
                keypoints.extend((0.0, 0.0, 0)); states.append(0)
                visibility_evidence.append({'reason': 'outside image'})
                continue
            px = min(511, max(0, int(x * 512)))
            py = min(511, max(0, int((1 - y) * 512)))
            delta = float(np.max(reflected[py, px, :3] - clean[py, px, :3]))
            neighborhood = reflected[max(0, py - 2):min(512, py + 3),
                                     max(0, px - 2):min(512, px + 3), :3]
            brightness = float(np.mean(neighborhood))
            # Humidity hand is above temperature hand. Both can cover scale marks.
            obscured = ((k in (0, 1) and point_in_polygon((x, y), hpoly)) or
                        (k >= 4 and (point_in_polygon((x, y), tpoly) or point_in_polygon((x, y), hpoly))) or
                        delta > .18 or brightness > .95)
            visible = 1 if obscured else 2
            keypoints.extend((x, y, visible)); states.append(visible)
            visibility_evidence.append({'glass_difference': delta, 'local_brightness': brightness,
                                        'reason': 'possibly obscured' if obscured else 'candidate visible'})
        label = [0, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0] + keypoints
        (label_dir / (name + '.txt')).write_text(' '.join(str(v) if isinstance(v, int) else '{:.6f}'.format(v) for v in label) + '\n', encoding='utf-8')
        manifest['samples'].append({'id': name, 'temperature_fraction': tf, 'humidity_fraction': hf,
                                    'camera': list(camera.location), 'lens_mm': camera.data.lens,
                                    'light': list(key.location), 'light_energy': key.data.energy,
                                    'light_size': key.data.size, 'keypoints': points, 'visibility': states,
                                    'values': {'temperature': -30 + tf * 80, 'humidity': hf * 100},
                                    'visibility_evidence': visibility_evidence,
                                    'visibility_note': 'approximate polygon, brightness and render-difference heuristic'})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
