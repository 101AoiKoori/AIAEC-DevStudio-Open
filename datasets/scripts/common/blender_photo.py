"""Camera rays and diffuse receivers for Blender foregrounds on photographs.

Extracted from the existing whole-bin photo renderer. Image coordinates are
top-left pixels; world Z is up and the unrolled camera looks along world +Y.
"""
import math

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def project(scene, camera, point, size):
    projected = world_to_camera_view(scene, camera, point)
    return [projected.x * size[0], (1 - projected.y) * size[1]]


def ground_ray(camera, point, size, focal):
    x, y = point
    w, h = size
    local = Vector(((x - w / 2) / focal, (h / 2 - y) / focal, -1))
    direction = camera.rotation_euler.to_matrix() @ local
    if direction.z >= -1e-6:
        raise ValueError('Requested footpoint is above the inferred horizon')
    return camera.location - direction * (camera.location.z / direction.z)


def set_plate_camera(scene, camera, resolution, height_m, horizon_y, focal_px):
    if min(*resolution, height_m, focal_px) <= 0 or not all(
            math.isfinite(v) for v in (*resolution, height_m, horizon_y, focal_px)):
        raise ValueError('Invalid photo camera parameters')
    width, height = resolution
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 36
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.lens = focal_px * camera.data.sensor_width / width
    depression = math.atan2(height / 2 - horizon_y, focal_px)
    camera.location = (0, 0, height_m)
    camera.rotation_euler = (math.pi / 2 - depression, 0, 0)
    camera.data.clip_start = .001
    camera.data.dof.use_dof = False
    scene.camera = camera
    bpy.context.view_layer.update()
    return {'height_m': height_m, 'horizon_y_px': horizon_y, 'focal_px': focal_px,
            'downwards_angle_deg': math.degrees(depression),
            'scope': 'estimated ground horizon, assumed focal length and camera height; not calibration'}


def aim_area(light, location, target, energy, size):
    light.location = location
    light.rotation_euler = (Vector(target) - light.location).to_track_quat('-Z', 'Y').to_euler()
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size


def add_receiver():
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 10, -.001))
    receiver = bpy.context.object
    receiver.name = 'Temporary real plate shadow receiver'
    material = bpy.data.materials.new('Diffuse receiver')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs[0].default_value = (.37, .37, .37, 1)
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(diffuse.outputs[0], output.inputs[0])
    receiver.data.materials.append(material)
    return receiver


def save_render(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def footprint_inside(footprint, polygon):
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    for x, y in footprint:
        cross = [(b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) for a, b in edges]
        if min(cross) < 0 < max(cross):
            return False
    return True
