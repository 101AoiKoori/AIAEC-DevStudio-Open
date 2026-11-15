"""Shared Blender device, projection and palette-based visibility passes."""
import math
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META"}

def configure_device(scene, device_type):
    if device_type == 'CPU':
        scene.cycles.device = 'CPU'
        return []
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = device_type
    prefs.get_devices()
    selected = []
    for device in prefs.devices:
        device.use = device.type == device_type
        if device.use:
            selected.append(device.name)
    if not selected:
        raise RuntimeError('no ' + device_type + ' render device available; use --device CPU')
    scene.cycles.device = 'GPU'
    return selected

def read_rgba(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = 'Non-Color'
        width, height = image.size
        if width <= 0 or height <= 0 or image.channels != 4:
            raise ValueError('mask PNG must contain four channels')
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        # Blender image pixels start at bottom-left, labels use image top-left.
        return pixels.reshape((height, width, 4))[::-1].copy()
    finally:
        bpy.data.images.remove(image)

def targets_inside_frame(scene, camera, targets, margin=0.0):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in targets:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            point = world_to_camera_view(scene, camera, evaluated.matrix_world @ Vector(corner))
            points.append((point.x, point.y, point.z))
    return projected_points_fit(points, margin)

def projected_points_fit(points, margin=0.0):
    """Reject geometry outside the camera, including behind-camera corners."""
    if not 0 <= margin < .5:
        raise ValueError('margin must be in [0, .5)')
    points = list(points)
    return bool(points) and all(math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
                                and z > 0 and margin <= x <= 1 - margin
                                and margin <= y <= 1 - margin for x, y, z in points)

def mask_visibility(visible, isolated, inside):
    """Use per-pixel intersection; Workbench edge coverage differs by pass."""
    visible_count, isolated_count = int(visible.sum()), int(isolated.sum())
    fraction = int(np.logical_and(visible, isolated).sum()) / isolated_count if isolated_count else None
    reasons = []
    if not inside:
        reasons.append('clipped_or_behind_camera')
    if not isolated_count:
        reasons.append('no_isolated_pixels')
    if visible_count < 24:
        reasons.append('too_few_visible_pixels')
    if fraction is not None and fraction < .70:
        reasons.append('occluded_below_threshold')
    box = None
    if visible_count:
        y, x = np.nonzero(visible)
        h, w = visible.shape
        box = [int(x.min()) / w, int(y.min()) / h, (int(x.max()) + 1) / w, (int(y.max()) + 1) / h]
    return {'candidate_xyxy_normalized': box, 'visible_pixels': visible_count,
            'isolated_pixels': isolated_count, 'visible_fraction': fraction,
            'in_frame': bool(inside), 'label_eligible': not reasons, 'rejection_reasons': reasons,
            'visibility_verified': 'workbench_intersection_estimate'}

def decode_mask(rgba, name, palette):
    """Decode solid emission colors, conservatively excluding antialiased edges."""
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4 or not np.isfinite(array).all():
        raise ValueError('mask image must be finite H×W×4')
    if name not in palette:
        raise ValueError('unknown palette key')
    mask = array[:, :, 3] >= .75
    for channel, value in enumerate(palette[name]):
        mask &= array[:, :, channel] >= .75 if value else array[:, :, channel] <= .25
    return mask

def masks(scene, camera, parts, output, stem, tag, force_visible=(), *, palette):
    """Visible and isolated Workbench object-color passes; opaque silhouettes."""
    objects = [o for o in scene.objects if o.type in GEOMETRY_TYPES]
    hidden = {o: o.hide_render for o in objects}
    colors = {o: tuple(o.color) for o in objects}
    layer = bpy.context.view_layer
    original_override = layer.material_override
    original_engine, original_compositor = scene.render.engine, scene.render.use_compositing
    original_film, original_mode = scene.render.film_transparent, scene.render.image_settings.color_mode
    original_view = (scene.view_settings.view_transform, scene.view_settings.look,
                     scene.view_settings.exposure, scene.view_settings.gamma)
    shade = scene.display.shading
    old_shading = (shade.light, shade.color_type, shade.show_shadows, shade.show_cavity)
    target = {o for group in parts.values() for o in group}
    force = set(force_visible)
    try:
        scene.render.engine = 'BLENDER_WORKBENCH'
        scene.render.use_compositing = False
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure, scene.view_settings.gamma = 0, 1
        shade.light, shade.color_type = 'FLAT', 'OBJECT'
        shade.show_shadows = shade.show_cavity = False
        layer.material_override = None
        for o in objects:
            o.color = (0, 0, 0, 1)
            if o in force:
                o.hide_render = False
        for name, group in parts.items():
            for o in group:
                o.color = (*palette[name], 1)
        directory = output / 'masks'
        path = directory / (stem + '_' + tag + '_visible.png')
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        visible = read_rgba(path)
        results = []
        for name, group in parts.items():
            selected = set(group)
            for o in objects:
                o.hide_render = o not in selected or (hidden[o] and o not in force)
            bpy.context.view_layer.update()
            path = directory / (stem + '_' + tag + '_' + name + '_isolated.png')
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            isolated = read_rgba(path)
            result = mask_visibility(decode_mask(visible, name, palette), decode_mask(isolated, name, palette),
                                     targets_inside_frame(scene, camera, group))
            result.update(bin=name, isolated_mask=path.relative_to(output).as_posix())
            results.append(result)
        return 'masks/' + stem + '_' + tag + '_visible.png', results
    finally:
        for o in objects:
            o.hide_render, o.color = hidden[o], colors[o]
        scene.render.engine, scene.render.use_compositing = original_engine, original_compositor
        scene.render.film_transparent, scene.render.image_settings.color_mode = original_film, original_mode
        scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure, scene.view_settings.gamma = original_view
        shade.light, shade.color_type, shade.show_shadows, shade.show_cavity = old_shading
        layer.material_override = original_override
        bpy.context.view_layer.update()
