"""Render blank bin fronts along eight saved Arena routes without saving the blend."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_workspace import STICKERS, WORKSPACE_BLEND, source_ref
from datasets.scripts.waste_sticker.configure_blender_orbit import STICKER_SCENE, sticker_objects

SCENE = 'Arena Waste Sticker Preview'
ROUTES = ('person', 'e_bike', 'fire', 'building_name_paper', 'sign',
          'waste_bin_state', 'waste_sticker', 'meter')
WIDTH, HEIGHT = 640, 480
SCALE = .10


def front_faces(papers):
    faces = {}
    for color, paper in papers.items():
        bottom_left, bottom_right, top_right, top_left = [v.co for v in paper.data.vertices]
        across = (bottom_right - bottom_left).normalized()
        up = (top_left - bottom_left).normalized()
        facing = across.cross(up).normalized()
        body = paper.parent
        polygon = min((p for p in body.data.polygons if len(p.vertices) == 4 and p.normal.dot(facing) > .95),
                      key=lambda p: p.center.y)
        corners = [body.data.vertices[index].co for index in polygon.vertices]
        by_height = sorted(corners, key=up.dot, reverse=True)
        top = sorted(by_height[:2], key=across.dot)
        bottom = sorted(by_height[2:], key=across.dot, reverse=True)
        center = sum(corners, Vector()) / 4
        # Keep paper off the beveled edge and leave a small tolerance for pixel rounding.
        faces[color] = (body, [center.lerp(p, .91) for p in top + bottom])
    return faces


def path_rays(camera, floor_z):
    origin = camera.matrix_world.translation
    rotation = camera.matrix_world.to_quaternion()
    view = camera.data.view_frame(scene=bpy.context.scene)
    left, right = min(v.x for v in view), max(v.x for v in view)
    bottom, top = min(v.y for v in view), max(v.y for v in view)
    for v in (.87, .80, .93, .72, .67):
        for u in (.50, .42, .58, .34, .66):
            local = Vector((left + u * (right - left), top - v * (top - bottom), view[0].z))
            direction = rotation @ local
            if direction.z < 0:
                hit = origin + (floor_z - origin.z) / direction.z * direction
                if max(abs(hit.x), abs(hit.y)) <= 1.76:
                    yield hit


def point_pixels(scene, camera, point):
    projection = world_to_camera_view(scene, camera, point)
    if projection.z <= 0 or not (.012 < projection.x < .988 and .018 < projection.y < .982):
        return None
    return round(projection.x * WIDTH), round((1 - projection.y) * HEIGHT)


def visible_surface(scene, camera, world, depsgraph):
    corners = [point_pixels(scene, camera, point) for point in world]
    if any(point is None for point in corners):
        return None
    x, y = zip(*corners)
    if max(x)-min(x) < 27 or max(y)-min(y) < 36:
        return None
    camera_position = camera.matrix_world.translation
    # Sample the interior and four corners, testing occlusion in the actual instanced scene.
    check = list(world)
    for ratio in (.25, .50, .75):
        across_left = world[0].lerp(world[3], ratio)
        across_right = world[1].lerp(world[2], ratio)
        check.extend(across_left.lerp(across_right, u) for u in (.25, .5, .75))
    for target in check:
        direction = target - camera_position
        distance = direction.length
        # Native classification decals sit on or immediately in front of the same panel.
        if scene.ray_cast(depsgraph, camera_position, direction.normalized(), distance=distance-.003)[0]:
            return None
    return [list(point) for point in corners]


def select_layout(scene, camera, placement, body_faces, fixed_markings, offset, floor_z, floor_offset, yaw_delta):
    before = camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
    camera_position = before.matrix_world.translation.copy()
    best = None
    for ground in list(path_rays(before, floor_z)):
        towards = camera_position - ground
        yaw = math.atan2(towards.x, -towards.y) + math.radians(yaw_delta)
        placement.matrix_world = (Matrix.Translation(ground + Vector((0, 0, floor_offset))) @
                                  Matrix.Rotation(yaw, 4, 'Z') @ Matrix.Scale(SCALE, 4))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_camera = camera.evaluated_get(depsgraph)
        poses = {
            item.object.original.name: item.matrix_world.copy() for item in depsgraph.object_instances
            if item.is_instance and item.parent and item.parent.original == placement
        }
        bins = []
        for color, (body, corners) in body_faces.items():
            if body.name not in poses:
                continue
            polygon = visible_surface(scene, evaluated_camera, [poses[body.name] @ pt for pt in corners], depsgraph)
            if not polygon:
                continue
            marking = fixed_markings.get(color)
            marking_points = []
            if marking and marking.name in poses:
                points = [point_pixels(scene, evaluated_camera, poses[marking.name] @ v.co) for v in marking.data.vertices]
                marking_points = list(dict.fromkeys(p for p in points if p is not None))
            bins.append({'bin_id': f'bin_{color}', 'bin_color': color, 'allowed_polygon': polygon,
                         'native_marking_polygon': [list(p) for p in marking_points],
                         'exclude_polygons': [], 'occluder_polygons': [],
                         'geometry_quality': 'blender_projected_unoccluded_front'})
        if bins and (best is None or len(bins) > len(best[0])):
            pose = {'ground_center_xyz': list(ground), 'yaw_degrees': math.degrees(yaw),
                    'scale': SCALE, 'instance_offset': list(offset)}
            best = bins, pose, placement.matrix_world.copy()
            if len(bins) == len(body_faces):
                break
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    output, manifest = args.output.resolve(), args.manifest.resolve()
    if (Path(bpy.data.filepath).resolve() != WORKSPACE_BLEND or
            not output.is_relative_to(ROOT/'datasets/raw/shared/backgrounds/rendered') or
            not manifest.is_relative_to(ROOT/'datasets/annotated/waste_bin')):
        raise ValueError('Expected unified blend and configured raw/annotation locations')
    if manifest.exists() or (output.exists() and any(output.iterdir())):
        raise FileExistsError('Output batch must be empty')
    source_hash = sha256(WORKSPACE_BLEND.read_bytes()).hexdigest()

    source_scene = bpy.data.scenes[STICKER_SCENE]
    bpy.context.window.scene = source_scene
    papers = sticker_objects(source_scene)
    faces = front_faces(papers)
    markings = {}
    for obj in bpy.data.collections[STICKERS].all_objects:
        name = obj.get('aec_source_name', obj.name)
        for color in papers:
            if name == color + ' front marking':
                markings[color] = obj
    offset = bpy.data.collections[STICKERS].instance_offset
    floor_offset = -SCALE * min(
        ((body.matrix_world @ vertex.co) - offset).z
        for body, _ in faces.values() for vertex in body.data.vertices)
    for paper in papers.values():
        paper.hide_render = paper.hide_viewport = True

    scene = bpy.data.scenes[SCENE]
    bpy.context.window.scene = scene
    placement = scene.objects['Waste Sticker Placement']
    floor = scene.objects['Arena | Arena floor 4.2 x 4.2 m']
    floor_z = max((floor.matrix_world @ Vector(corner)).z for corner in floor.bound_box)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.render.use_compositing = False
    scene.render.image_settings.file_format = 'PNG'
    jobs = []
    for route in ROUTES:
        camera = scene.objects['Arena | CAM_' + route]
        scene.camera = camera
        selected_frames = set()
        for slot, (anchor, yaw_delta) in enumerate(zip((15, 60, 107), (-15, 0, 15)), 1):
            best = None
            candidates = sorted(set(range(1, 121)) - selected_frames, key=lambda x: (abs(x-anchor), x))
            for frame in candidates:
                if best is not None and abs(frame-anchor) > 13:
                    break
                scene.frame_set(frame)
                layout = select_layout(scene, camera, placement, faces, markings, offset, floor_z, floor_offset, yaw_delta)
                if layout and (best is None or len(layout[0]) > len(best[1][0])):
                    best = frame, layout
                    if len(layout[0]) == len(faces):
                        break
            if best is None:
                raise ValueError(f'No visible front found: {route}, slot {slot}')
            frame, (bins, bin_pose, pose_matrix) = best
            selected_frames.add(frame)
            jobs.append((route, frame, camera, pose_matrix, bins, bin_pose))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for route, frame, camera, matrix_world, bins, bin_pose in jobs:
        scene.camera = camera
        scene.frame_set(frame)
        placement.matrix_world = matrix_world
        bpy.context.view_layer.update()
        image = output/f'{route}_f{frame:03d}.png'
        scene.render.filepath = str(image)
        matrix = [list(row) for row in camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world]
        bpy.ops.render.render(scene=scene.name, write_still=True)
        records.append({
            'image': image.relative_to(ROOT).as_posix(), 'image_size': [WIDTH, HEIGHT],
            'content_sha256': sha256(image.read_bytes()).hexdigest(),
            'source_blend': WORKSPACE_BLEND.relative_to(ROOT).as_posix(), 'source_blend_sha256': source_hash,
            'parent_group': WORKSPACE_BLEND.relative_to(ROOT).as_posix(),
            'scene': SCENE, 'route': route, 'frame': frame, 'camera': camera.name,
            'camera_matrix_world': matrix,
            'bin_asset': source_ref(WORKSPACE_BLEND, STICKERS), 'bin_pose': bin_pose, 'bins': bins,
            'source_kind': 'blender_render', 'item_stickers_hidden': True,
        })
        print('SURFACE', route, frame, len(bins), flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'schema_version': 1, 'records': records}, ensure_ascii=False, indent=2)+'\n',
                        encoding='utf-8')


if __name__ == '__main__':
    main()
