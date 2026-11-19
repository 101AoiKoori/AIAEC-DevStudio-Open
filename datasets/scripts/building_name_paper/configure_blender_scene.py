"""Configure the saved Buildings scene and its four front hemisphere cameras.

Call configure() inside the loaded workspace. The caller owns saving and rendering.
Camera azimuth is negative towards the building's left side (world -X).
"""
import math

from datasets.scripts.common.config import load_task_config

SCENE = 'Buildings'


def source_name(obj):
    return obj.get('aec_source_name', obj.name.split(' | ', 1)[-1])


def camera_name(key):
    return 'Camera - ' + key


def set_camera_pose(camera, target, azimuth, elevation, radius, lens):
    from mathutils import Vector

    yaw, pitch = math.radians(azimuth), math.radians(elevation)
    target = Vector(target)
    camera.location = target + radius * Vector((
        math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)))
    previous = camera.rotation_euler.copy()
    camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler('XYZ', previous)
    camera.data.type = 'PERSP'
    camera.data.lens = lens
    camera.data.sensor_width = 36
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.clip_start, camera.data.clip_end = .03, 100
    camera.data.dof.use_dof = False


def _material(name, color, roughness, specular):
    import bpy

    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    shader = nodes.new('ShaderNodeBsdfPrincipled')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Specular IOR Level'].default_value = specular
    shader.inputs['Metallic'].default_value = 0
    shader.inputs['Coat Weight'].default_value = 0
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    material.diffuse_color = (*color, 1)
    return material


def configure():
    import bpy

    _, config = load_task_config('building_name_paper')
    settings = config['blender_view']
    scene = bpy.data.scenes[SCENE]
    scene.use_fake_user = True
    light = settings['lighting']
    world_name = 'Buildings | Diffuse environment'
    world = bpy.data.worlds.get(world_name) or bpy.data.worlds.new(world_name)
    world.use_nodes = True
    world.node_tree.nodes.clear()
    background = world.node_tree.nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (*light['world_color'], 1)
    background.inputs['Strength'].default_value = light['world_strength']
    output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])
    scene.world = world
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
    paper = _material('Buildings | Matte name paper', (.82, .82, .79),
                      light['paper_roughness'], light['paper_specular'])
    ink = _material('Buildings | Matte name ink', (.012, .012, .012), 1, 0)
    groups, centers = {}, {}
    for key, name in config['blender_collections'].items():
        collection = next(c for c in scene.collection.children if source_name(c) == name)
        groups[key] = collection
        collection.hide_render = collection.hide_viewport = False
        board = next(o for o in collection.all_objects if source_name(o).startswith('Name paper - '))
        text = next(o for o in collection.all_objects if source_name(o).startswith('Editable text - '))
        for obj, material in ((board, paper), (text, ink)):
            if 'name_original_y' not in obj:
                obj['name_original_y'] = obj.location.y
            obj.location.y = obj['name_original_y'] - .0012
            obj.data.materials.clear()
            obj.data.materials.append(material)
        centers[key] = (text.location.x, 0, settings['target_height_m'])

    routes = bpy.data.collections.get('Buildings | Name camera routes')
    if routes is None:
        routes = bpy.data.collections.new('Buildings | Name camera routes')
        scene.collection.children.link(routes)
    routes['aec_source_name'] = 'Name camera routes'
    existing = {source_name(o): o for o in scene.objects if o.type == 'CAMERA'}
    camera_specs = {key: (centers[key], settings['single']) for key in groups}
    camera_specs['ABC'] = ((sum(p[0] for p in centers.values()) / 3, 0, settings['target_height_m']), settings['global'])
    frame_count = 3 * settings['frames_per_sweep'] + 2 * settings['transition_frames']
    cameras = {}
    for key, (target, spec) in camera_specs.items():
        name = camera_name(key)
        camera = existing.get(name)
        if camera is None and key == 'ABC':
            camera = existing.get('Reference camera')
        if camera is None:
            camera = bpy.data.objects.new('Buildings | ' + name, bpy.data.cameras.new('Buildings | ' + name))
            routes.objects.link(camera)
        camera.name = 'Buildings | ' + name
        camera['aec_source_name'] = name
        camera['building_keys'] = key
        camera['target_world_m'] = target
        camera['azimuth_degrees'] = spec['azimuth_degrees']
        camera['camera_height_m'] = spec['camera_height_m']
        camera['view_layer'] = 'Buildings - ' + key
        old_action = camera.animation_data.action if camera.animation_data else None
        camera.animation_data_clear()
        if old_action is not None and not old_action.users:
            bpy.data.actions.remove(old_action)
        frame = 0
        for band, (radius, height) in enumerate(zip(spec['radius_m'], spec['camera_height_m'])):
            elevation = math.degrees(math.asin((height - target[2]) / radius))
            left, right = spec['azimuth_degrees']
            if band:
                yaw = right if band % 2 else left
                for step in range(settings['transition_frames']):
                    blend = (step + 1) / (settings['transition_frames'] + 1)
                    blend = blend * blend * (3 - 2 * blend)
                    previous_radius, previous_height = spec['radius_m'][band - 1], spec['camera_height_m'][band - 1]
                    blended_radius = previous_radius + blend * (radius - previous_radius)
                    blended_height = previous_height + blend * (height - previous_height)
                    blended_elevation = math.degrees(math.asin((blended_height - target[2]) / blended_radius))
                    frame += 1
                    set_camera_pose(camera, target, yaw, blended_elevation, blended_radius, spec['lens_mm'])
                    camera.keyframe_insert('location', frame=frame)
                    camera.keyframe_insert('rotation_euler', frame=frame)
            for step in range(settings['frames_per_sweep']):
                fraction = step / (settings['frames_per_sweep'] - 1)
                fraction = fraction * fraction * (3 - 2 * fraction)
                if band % 2:
                    fraction = 1 - fraction
                yaw = left + fraction * (right - left)
                frame += 1
                set_camera_pose(camera, target, yaw, elevation, radius, spec['lens_mm'])
                camera.keyframe_insert('location', frame=frame)
                camera.keyframe_insert('rotation_euler', frame=frame)
        camera.animation_data.action.name = 'Buildings | Orbit ' + key
        action = camera.animation_data.action
        for curve in action.layers[0].strips[0].channelbag(camera.animation_data.action_slot).fcurves:
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
        cameras[key] = camera

    # Render layers isolate each single-building camera while retaining all three in ABC.
    main = scene.view_layers.get('Buildings - ABC')
    if main is None:
        main = scene.view_layers[0]
        main.name = 'Buildings - ABC'
    for key in camera_specs:
        layer_name = 'Buildings - ' + key
        layer = scene.view_layers.get(layer_name) or scene.view_layers.new(layer_name)
        layer.use = key == 'ABC'
        for building_key, collection in groups.items():
            layer.layer_collection.children[collection.name].exclude = key != 'ABC' and building_key != key
    scene.camera = cameras['ABC']
    scene.frame_start, scene.frame_end = 1, frame_count
    scene.render.fps = 24
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.frame_set(47)
    return {'scene': scene.name, 'cameras': {key: obj.name for key, obj in cameras.items()},
            'frames': frame_count, 'lighting': 'world_environment_cycles_diffuse'}


def set_buildings(scene, active, papers=None):
    """Toggle whole building collections; optional paper keys control paper and ink together."""
    import bpy

    active = set(active)
    if not active <= set('ABC') or not active:
        raise ValueError('Select one, two or three buildings from A/B/C')
    if papers is not None and not set(papers) <= active:
        raise ValueError('Visible name papers must belong to enabled buildings')
    _, config = load_task_config('building_name_paper')
    for key, name in config['blender_collections'].items():
        collection = next(c for c in scene.collection.children if source_name(c) == name)
        collection.hide_render = collection.hide_viewport = key not in active
        if papers is not None:
            for obj in list(collection.all_objects):
                if source_name(obj).startswith(('Name paper - ', 'Editable text - ')):
                    obj.hide_render = key not in papers
                    obj.hide_viewport = key not in papers
    bpy.context.view_layer.update()


def select_camera(scene, key, frame):
    """Choose the camera and its matching render layer, then evaluate its route."""
    import bpy

    camera = next(o for o in scene.objects if o.type == 'CAMERA' and o.get('building_keys') == key)
    layer = scene.view_layers[camera['view_layer']]
    scene.camera = camera
    for other in scene.view_layers:
        other.use = other == layer
    scene.frame_set(frame)
    if bpy.context.window:
        bpy.context.window.scene = scene
        bpy.context.window.view_layer = layer
    layer.update()
    return camera, layer
