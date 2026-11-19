"""Apply reviewed photo-camera poses and Cycles contact shadows in Blender."""
import numpy as np
import bpy
import math
from mathutils import Matrix, Vector


def capture_scene(scene):
    """Capture only this background process's working state; never save the blend."""
    return {'matrices': {o: o.matrix_world.copy() for o in scene.objects},
            'lights': {o: (o.data.energy, tuple(o.data.color), o.data.size if o.data.type == 'AREA' else None)
                       for o in scene.objects if o.type == 'LIGHT'},
            'exposure': scene.view_settings.exposure,
            'world': [(n, float(n.inputs['Strength'].default_value)) for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND']}


def reset_scene(scene, baseline):
    for obj, matrix in baseline['matrices'].items():
        obj.matrix_world = matrix.copy()
    for obj, (energy, color, size) in baseline['lights'].items():
        obj.data.energy, obj.data.color = energy, color
        if size is not None:
            obj.data.size = size
    for node, strength in baseline['world']:
        node.inputs['Strength'].default_value = strength
    scene.view_settings.exposure = baseline['exposure']
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGB'
    scene.camera.data.shift_x = scene.camera.data.shift_y = 0
    scene.camera.data.sensor_fit = 'HORIZONTAL'
    floor = bpy.data.objects.get('Presentation floor')
    if floor:
        floor.hide_render = False
    catcher = bpy.data.objects.get('Fire photo shadow catcher')
    if catcher:
        catcher.hide_render = True


def prepare_photo(scene, groups, active, review, rng, baseline):
    camera = review['camera']
    size = np.asarray(camera['image_size'], dtype=float)
    if len(active) != 1:
        raise ValueError('Current photo fits support one added building per reviewed floor placement')
    key = active[0]
    group = groups[key]
    # Flush location/scale changes made while placing each paper before reading
    # matrix_world. Otherwise Blender returns the previous pose for those papers.
    bpy.context.view_layer.update()
    originals={obj:obj.matrix_world.copy() for obj in group['collection'].all_objects}
    points=[matrix@Vector(v) for obj,matrix in originals.items() if obj.type=='MESH' and not obj.hide_render for v in obj.bound_box]
    bounds=[[min(p[a] for p in points),max(p[a] for p in points)] for a in range(3)]
    bounds[0]=[x-group['x'] for x in bounds[0]]
    from datasets.scripts.fire.photo_geometry import choose_photo_placement
    placement=choose_photo_placement(review,bounds,rng)
    location=np.asarray(placement['ground_position_m']);model_scale=placement['model_scale'];yaw=placement['model_yaw_rad']
    transform=Matrix(placement['matrix'])@Matrix.Translation(Vector((-group['x'],0,0)))
    for obj,matrix in originals.items():
        obj.matrix_world = transform @ matrix
    # Generated sticker meshes are linked to the building collection, and must
    # follow the same rigid transform as the facade to preserve physical contact.
    rotation = np.asarray(camera['world_to_cv_rotation'])
    translation = np.asarray(camera['world_to_cv_translation'])
    eye = -rotation.T @ translation
    cv_to_blender = np.diag([1., -1., -1.])
    pose = np.eye(4)
    pose[:3, :3] = rotation.T @ cv_to_blender
    pose[:3, 3] = eye
    scene.camera.matrix_world = Matrix(pose.tolist())
    width, height = scene.render.resolution_x, scene.render.resolution_y
    intrinsic = np.asarray(camera['intrinsic_matrix'])
    sx, sy = np.asarray([width, height])/size
    scene.camera.data.sensor_fit = 'HORIZONTAL'
    scene.camera.data.sensor_width = 36
    scene.camera.data.lens = float(intrinsic[0, 0]*sx/width*36)
    scene.camera.data.shift_x = float((width/2-intrinsic[0, 2]*sx)/width)
    scene.camera.data.shift_y = float((intrinsic[1, 2]*sy-height/2)/width)
    scene.camera.data.clip_start = .001
    scene.camera.data.clip_end = 100
    if scene.camera.animation_data:
        scene.camera.animation_data_clear()
    floor = bpy.data.objects.get('Presentation floor')
    if floor:
        floor.hide_render = True
    catcher = bpy.data.objects.get('Fire photo shadow catcher')
    if catcher is None:
        mesh = bpy.data.meshes.new('Fire photo floor mesh')
        mesh.from_pydata([(-5,-5,-.0002),(5,-5,-.0002),(5,5,-.0002),(-5,5,-.0002)], [], [(0,1,2,3)])
        catcher = bpy.data.objects.new('Fire photo shadow catcher', mesh)
        scene.collection.objects.link(catcher)
        mat = bpy.data.materials.new('Fire photo neutral floor')
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        node = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(node.outputs['BSDF'],output.inputs['Surface'])
        node.inputs['Base Color'].default_value = (.7,.7,.7,1)
        node.inputs['Roughness'].default_value = .95
        catcher.data.materials.append(mat)
    catcher.hide_render = False
    catcher.is_shadow_catcher = True
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.exposure = float(review['exposure_ev'])
    center = Vector((float(location[0]), float(location[1]), .2))
    floor_rgb = np.asarray(review['floor_rgb_srgb'])
    color = np.clip(floor_rgb/max(float(floor_rgb.max()), 1e-6), .85, 1)
    for name, offset, energy_factor in [('Broad softbox', review['key_position_offset_m'], 1.),
                                        ('Soft fill', [.5,.2,1.], .8)]:
        light = bpy.data.objects[name]
        light.location = center + Vector(offset)
        light.rotation_euler = (center-light.location).to_track_quat('-Z','Y').to_euler()
        light.data.energy = baseline['lights'][light][0]*energy_factor
        light.data.color = tuple(float(c) for c in color)
        if light.data.type == 'AREA':
            light.data.size = review['key_size_m']
    bpy.context.view_layer.update()
    return {'position_m': list(scene.camera.location), 'focal_length_mm': scene.camera.data.lens,
            'view_profile': 'photo_corner_pnp', 'photo_frame_id': review['frame_id'],
            'photo_reprojection_rms_px': camera['reprojection_rms_px'],
            'photo_calibration_size': camera['image_size'], 'calibration_status': camera['calibration_status'],
            'ground_anchor_px': placement['ground_anchor_px'], 'ground_position_m': location.tolist(),
            'model_scale': model_scale, 'model_yaw_rad': yaw,
            'footprint_photo_px':placement['footprint_photo_px'],'body_bbox_photo_px':placement['body_bbox_photo_px'],
            'placement_attempts':placement['placement_attempts'],
            'world_transform': [list(row) for row in transform],
            'shadow': {'method': 'cycles_physical_shadow_catcher_combined_alpha', 'floor_z_m': -.0002,
                       'key_size_m': review['key_size_m'], 'photo_light_evidence': review['light_direction_evidence']}}
