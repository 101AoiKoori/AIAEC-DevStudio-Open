"""Add editable low-mounted vehicle camera routes to the arena work copy.

Run in Blender with the project arena loaded. The first work copy was made
from shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend.
The caller saves the work copy; this script does not open or overwrite sources.
"""
import math
import bpy
from mathutils import Euler


ROUTES = {
    'person': [(-1.55, -1.25, .30), (-1.2, -.9, .30), (-.55, -.1, .29), (.05, .65, .30), (.65, 1.2, .31)],
    'e_bike': [(-1.65, -.75, .26), (-1.0, -.55, .27), (-.35, -.65, .27), (.4, -.65, .28), (1.45, -.3, .29)],
    'fire': [(1.30, -1.45, .32), (1.0, -.75, .33), (.8, .15, .33), (.9, .95, .34), (1.05, 1.7, .35)],
    'building_name_paper': [(-1.55, .85, .32), (-1.2, 1.25, .33), (-.6, 1.5, .34), (.2, 1.55, .35), (.9, 1.45, .36)],
    'sign': [(1.50, 1.25, .28), (.8, 1.0, .28), (.2, .5, .29), (-.3, -.1, .30), (-.45, -1.15, .30)],
    'waste_bin_state': [(-1.45, 1.45, .26), (-1.3, .8, .27), (-.85, .35, .28), (-1.05, -.3, .28), (-1.5, -1.0, .29)],
    'waste_sticker': [(1.5, -1.3, .25), (1.3, -.85, .26), (.8, -.3, .27), (.4, .3, .28), (.1, 1.2, .29)],
    'meter': [(1.45, .9, .34), (1.1, 1.3, .35), (.7, 1.55, .36), (.3, 1.65, .37), (-.15, 1.55, .38)],
}

INTENTS = {
    'person': 'Diagonal crossing of open central floor, towards back corner.',
    'e_bike': 'Crosswise travel through the southern open floor.',
    'fire': 'Travel along the eastern side towards the back wall.',
    'building_name_paper': 'Sweep parallel to the back wall and doorway.',
    'sign': 'Turn through the central area towards the front wall.',
    'waste_bin_state': 'Travel near the western wall towards the front.',
    'waste_sticker': 'Diagonal travel from front-right towards the back.',
    'meter': 'Sweep near the back wall in the opposite direction.',
}

def build_context():
    """Add unobtrusive route context using only the arena's existing materials."""
    scene = bpy.context.scene
    if bpy.data.collections.get('AEC_arena_route_context'):
        raise RuntimeError('Route context already exists.')
    collection = bpy.data.collections.new('AEC_arena_route_context')
    scene.collection.children.link(collection)

    def box(name, position, size, material):
        x, y, z = (v / 2 for v in size)
        vertices = [(-x, -y, -z), (-x, -y, z), (-x, y, -z), (-x, y, z),
                    (x, -y, -z), (x, -y, z), (x, y, -z), (x, y, z)]
        faces = [(0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4),
                 (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6)]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.materials.append(bpy.data.materials[material])
        obj = bpy.data.objects.new(name, mesh)
        obj.location = position
        collection.objects.link(obj)

    # Flat, non-target road markings laid on the existing dark road surface.
    for i in range(7):
        box('Crosswalk_%02d' % i, (-1.2 + .4 * i, -.05, .009),
            (.22, .65, .003), 'Door warm off-white')
    for side in (-1, 1):
        for i, y in enumerate((-1.55, -.9, .75, 1.4)):
            box('Lane_%s_%02d' % (side, i), (side * .68, y, .0095),
                (.035, .38, .004), 'Window and door frame')
    # Structural wall infill panels provide different north/south/east/west
    # context without adding any task targets or obstructing the vehicle paths.
    for i, y in enumerate((-1.45, -.45, .55, 1.55)):
        box('East_context_panel_%02d' % i, (2.085, y, .82),
            (.025, .72, .72), 'Curtain blue-grey fabric')
    for i, y in enumerate((-1.4, -.45, .5, 1.45)):
        box('West_context_panel_%02d' % i, (-2.085, y, .82),
            (.025, .76, .72), 'Door warm off-white')
    for i, x in enumerate((-1.4, -.45, .5, 1.45)):
        box('Front_context_panel_%02d' % i, (x, -2.085, .82),
            (.76, .025, .72), 'PBR surrounding concrete')


def build():
    scene = bpy.context.scene
    if bpy.data.collections.get('AEC_eight_camera_routes'):
        raise RuntimeError('Camera route collection already exists; use a fresh arena work copy.')
    collection = bpy.data.collections.new('AEC_eight_camera_routes')
    scene.collection.children.link(collection)

    # Complete the same enclosure as the three existing walls, including the
    # black kick panel. Copies share the source meshes and materials.
    for source_name, name, y in [('Back white wall', 'Front white wall', -2.25),
                                  ('Back black panel', 'Front black panel', -2.14)]:
        source = bpy.data.objects[source_name]
        obj = source.copy()
        obj.name = name
        obj.location.y = y
        collection.objects.link(obj)

    # The source arena contains one hidden, unused default cube at the origin.
    cube = bpy.data.objects.get('Cube')
    if cube and cube.type == 'MESH' and cube.hide_render and all(abs(v) < .001 for v in cube.location):
        bpy.data.objects.remove(cube, do_unlink=True)

    for index, (task, points) in enumerate(ROUTES.items()):
        curve = bpy.data.curves.new('Route ' + task, 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 32
        curve.path_duration = 120
        curve.use_path = True
        spline = curve.splines.new('BEZIER')
        spline.bezier_points.add(len(points) - 1)
        for bp, position in zip(spline.bezier_points, points):
            bp.co = position
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
        path = bpy.data.objects.new('PATH_' + task, curve)
        path['task'] = task
        path['description'] = INTENTS[task]
        path['frames'] = '1-120'
        collection.objects.link(path)

        rig = bpy.data.objects.new('RIG_' + task, None)
        rig.empty_display_type = 'ARROWS'
        collection.objects.link(rig)
        follow = rig.constraints.new('FOLLOW_PATH')
        follow.target = path
        follow.use_fixed_location = True
        follow.use_curve_follow = True
        follow.forward_axis = 'FORWARD_Y'
        follow.up_axis = 'UP_Z'
        for frame, value in ((1, 0.0), (120, 1.0)):
            follow.offset_factor = value
            follow.keyframe_insert(data_path='offset_factor', frame=frame)

        data = bpy.data.cameras.new('Camera ' + task)
        data.lens = (23, 25, 24, 27, 22, 25, 24, 26)[index]
        data.clip_start = .03
        camera = bpy.data.objects.new('CAM_' + task, data)
        collection.objects.link(camera)
        camera.parent = rig
        camera.rotation_mode = 'QUATERNION'
        base = Euler((math.pi / 2, 0, 0)).to_quaternion()
        # Local camera axes: x is pitch, y yaw, z roll.
        sign = -1 if index % 2 else 1
        for frame, angles in ((1, (-3, sign * -8, sign * 2)),
                              (60, (2 + index % 3, sign * 9, sign * -3)),
                              (120, (-4, sign * -5, sign * 2))):
            camera.rotation_quaternion = base @ Euler(tuple(math.radians(v) for v in angles)).to_quaternion()
            camera.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        camera['task'] = task
        camera['route'] = path.name
        camera['frames'] = '1-120'
    build_context()
    scene.frame_end = max(scene.frame_end, 120)
    return list(ROUTES)


if __name__ == '__main__':
    build()
