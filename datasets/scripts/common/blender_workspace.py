"""Select a render scene and its asset collections from the editable Blender file."""

from pathlib import Path
import math

from .paths import WORKSPACE

WORKSPACE_BLEND = WORKSPACE / 'datasets/assets/blender/aiaec_dataset.blend'
ARENA = 'Arena'
DETAILED = 'Arena Detailed'
BUILDINGS = 'Buildings'
BINS = 'Waste Bins'
BIN_PHOTO = 'AEC Waste Bins Photo'
BINS_PHOTO_SCENE = 'Waste Bins Photo'
STICKERS = 'AEC Waste Stickers'
STATION = 'Meter Station'
STATION_ASSETS = ('Meter Station | Power meter station prototype',
                  'Meter Station | Station 9011 dial assembly')
ORBIT_VIEWS = tuple(
    (f'{side}_{depth}', azimuth, distance)
    for depth, distance in (('far', 6.2), ('mid', 5.55), ('near', 5.05))
    for side, azimuth in (('left', -15), ('center', 0), ('right', 15))
)


def orbit_camera_name(scene_name, view):
    return f'{scene_name} | Orbit {view}'


def _set_orbit_pose(camera, target, azimuth, distance):
    from mathutils import Vector

    az = math.radians(azimuth)
    forward = Vector((-math.sin(az), math.cos(az), 0))
    camera.location = target - distance * forward
    camera.rotation_euler = forward.to_track_quat('-Z', 'Y').to_euler()


def _animate_front_scan(scene, target):
    import bpy

    camera = scene.objects[f'{scene.name} | Front scan camera']
    action_name = f'{scene.name} | Front scan orbit'
    previous_action = camera.animation_data.action if camera.animation_data else None
    camera.animation_data_clear()
    if previous_action and previous_action.name == action_name and previous_action.users == 0:
        bpy.data.actions.remove(previous_action)
    # Sweep each distance in alternating directions and move inward between sweeps.
    order = (0, 1, 2, 5, 4, 3, 6, 7, 8)
    span = scene.frame_end - scene.frame_start
    anchors = [(scene.frame_start + round(i * span / (len(order) - 1)), ORBIT_VIEWS[index])
               for i, index in enumerate(order)]
    for frame, (view, _, _) in anchors:
        scene.timeline_markers.new(f'Orbit {view}', frame=frame)
    for (first, (_, az0, dist0)), (last, (_, az1, dist1)) in zip(anchors, anchors[1:]):
        for frame in range(first, last + 1):
            fraction = (frame - first) / (last - first)
            _set_orbit_pose(camera, target, az0 + (az1 - az0) * fraction,
                            dist0 + (dist1 - dist0) * fraction)
            camera.keyframe_insert(data_path='location', frame=frame, group='Orbit')
            camera.keyframe_insert(data_path='rotation_euler', frame=frame, group='Orbit')
    action = camera.animation_data.action
    action.name = action_name
    strip = action.layers[0].strips[0]
    for curve in strip.channelbag(camera.animation_data.action_slot).fcurves:
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'
    scene.camera = camera
    scene.frame_set(anchors[4][0])


def store_orbit_cameras(scene, subjects, target_property):
    """Store the nine level vehicle views and the timeline that visits them."""
    import bpy
    from mathutils import Vector

    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for obj in subjects for corner in obj.bound_box]
    target = Vector(tuple((min(p[a] for p in corners) + max(p[a] for p in corners)) / 2
                          for a in range(3)))
    body_center_z = target.z
    bottom, rim = min(p.z for p in corners), max(p.z for p in corners)
    # Start just above the rim to keep the opening visible from the vehicle view.
    target.z = rim + .25 * (rim - bottom)
    collection_name = f'{scene.name} | Orbit cameras'
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        scene.collection.children.link(collection)
    for marker in list(scene.timeline_markers):
        if marker.name.startswith('Orbit '):
            scene.timeline_markers.remove(marker)
    scene[target_property] = list(target)
    orbit_cameras = []
    for view, azimuth, distance in ORBIT_VIEWS:
        name = orbit_camera_name(scene.name, view)
        camera = collection.objects.get(name)
        if camera is None:
            camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            collection.objects.link(camera)
        _set_orbit_pose(camera, target, azimuth, distance)
        camera['aec_source_name'] = name
        camera['azimuth_deg'] = azimuth
        camera['distance'] = distance
        camera['elevation_deg'] = math.degrees(math.atan2(target.z - body_center_z, distance))
        camera['pitch_deg'] = 0.0
        orbit_cameras.append(camera)
    for camera in list(collection.objects):
        if (camera.type == 'CAMERA' and camera not in orbit_cameras
                and camera.get('aec_source_name') == camera.name):
            data = camera.data
            bpy.data.objects.remove(camera, do_unlink=True)
            if data.users == 0:
                bpy.data.cameras.remove(data)
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    _animate_front_scan(scene, target)
    # Match the timeline camera to the fixed render views, including the lens shift.
    for camera in [scene.camera, *orbit_cameras]:
        camera.data.type = 'PERSP'
        camera.data.sensor_fit = 'HORIZONTAL'
        camera.data.lens = 35
        camera.data.sensor_width = 36
        camera.data.shift_x = camera.data.shift_y = 0
    for area in bpy.context.window.screen.areas:
        if area.type == 'VIEW_3D' and area.spaces.active.region_3d:
            area.spaces.active.region_3d.view_perspective = 'CAMERA'


def source_ref(path, selector):
    return f'{path.relative_to(WORKSPACE).as_posix()}#{selector}'


def activate_scene(name, assets=(), *, source_blend=None):
    """Prepare an isolated in-memory scene for a background renderer.

    Only the selected scene and explicitly requested asset objects survive.
    The legacy object names are restored for renderers that use Blender's
    global object lookup. The editable file stays on disk throughout.
    """
    import bpy

    expected = Path(source_blend).resolve() if source_blend else WORKSPACE_BLEND
    if not expected.is_relative_to(WORKSPACE) or expected.suffix.lower() != '.blend':
        raise ValueError('Production Blender source must be an explicit workspace .blend')
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != expected:
        raise ValueError(f'Open {WORKSPACE_BLEND.name} in background Blender for production')
    scene = bpy.data.scenes[name]
    collections = tuple(bpy.data.collections[key] for key in assets)
    keep = set(scene.objects)
    for collection in collections:
        keep.update(collection.all_objects)
        # The caller chooses where these collections enter the render scene.
        collection.use_fake_user = True

    names = {obj: obj.get('aec_source_name', obj.name.split(' | ', 1)[-1]) for obj in keep}
    if len(set(names.values())) != len(names):
        raise ValueError(f'Object names conflict in the render inputs for {name}')
    bpy.context.window.scene = scene
    bpy.data.batch_remove([other for other in bpy.data.scenes if other != scene])
    bpy.data.batch_remove([obj for obj in bpy.data.objects if obj not in keep])
    bpy.data.orphans_purge(do_recursive=True)
    for obj, original in names.items():
        obj.name = original
    for collection in scene.collection.children_recursive:
        if 'aec_source_name' in collection:
            collection.name = collection['aec_source_name']
    for image in bpy.data.images:
        if 'aec_source_name' in image:
            image.name = image['aec_source_name']
    bpy.context.view_layer.update()
    return collections
