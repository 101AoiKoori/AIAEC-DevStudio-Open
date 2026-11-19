"""Camera poses on the configured spherical strip in front of the buildings."""
import math


def orange_peel_pose(config, index, center_x, building_span=0):
    azimuths = config['azimuth_deg']
    anchors = [(azimuth, height) for row, height in enumerate(config['camera_height_m'])
               for azimuth in (azimuths if row % 2 == 0 else azimuths[::-1])]
    if not 0 <= index <= len(anchors) - 1:
        raise ValueError('Camera trajectory index is outside its configured strip')
    left = min(int(index), len(anchors) - 2)
    weight = index - left
    yaw, height = [a + (b - a) * weight for a, b in zip(anchors[left], anchors[left + 1])]
    radius = config['radius_m'] + building_span * config['span_distance_factor']
    elevation = math.degrees(math.asin((height - config['target_height_m']) / radius))
    az, el = math.radians(yaw), math.radians(elevation)
    target = [center_x, 0, config['target_height_m']]
    position = [center_x + radius * math.cos(el) * math.sin(az),
                -radius * math.cos(el) * math.cos(az), target[2] + radius * math.sin(el)]
    return {'position_m': position, 'target_m': target, 'azimuth_deg': yaw,
            'elevation_deg': elevation, 'radius_m': radius, 'trajectory_index': index,
            'focal_length_mm': config['focal_length_mm'], 'view_profile': 'orange_peel',
            'distortion': 'not modelled'}


def apply_pose(camera, pose):
    from mathutils import Vector

    camera.location = pose['position_m']
    camera.rotation_euler = (Vector(pose['target_m']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = pose['focal_length_mm']
    camera.data.clip_start = .001
    camera.data.dof.use_dof = False


def store_camera(scene, config, centers):
    """Save an editable continuous scan of the same strip used by production."""
    import bpy

    scene.use_fake_user = True
    name = 'Fire orange peel camera'
    camera = scene.objects.get(name)
    if camera is None:
        camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        scene.collection.objects.link(camera)
    camera.animation_data_clear()
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 36
    camera.data.shift_x = camera.data.shift_y = 0
    frames = config['frames']
    anchors = len(config['azimuth_deg']) * len(config['camera_height_m'])
    center = sum(centers) / len(centers)
    span = max(centers) - min(centers)
    for frame in range(1, frames + 1):
        pose = orange_peel_pose(config, (frame - 1) * (anchors - 1) / (frames - 1), center, span)
        apply_pose(camera, pose)
        camera.keyframe_insert('location', frame=frame)
        camera.keyframe_insert('rotation_euler', frame=frame)
    camera['aec_source_name'] = name
    camera['configuration'] = 'datasets/scripts/fire/config/source.yaml#camera_trajectory'
    scene.camera = camera
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.frame_set((frames + 1) // 2)
    return camera
