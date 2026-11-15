"""Shared Blender planar artwork and instance-mask operations."""
import math
from pathlib import Path
import bpy
import numpy as np
ROOT = Path(__file__).resolve().parents[3]

def principled_node(mat):
    """Create and connect a shader when Blender does not add default nodes."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next((node for node in nodes if node.bl_idname == 'ShaderNodeBsdfPrincipled'), None)
    if shader is None:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        output = next((node for node in nodes if node.bl_idname == 'ShaderNodeOutputMaterial'), None)
        if output is None:
            output = nodes.new('ShaderNodeOutputMaterial')
        links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    return shader

def material(name, texture):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.surface_render_method = 'DITHERED'
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = principled_node(mat)
    bsdf.inputs['Roughness'].default_value = .8
    image = nodes.new('ShaderNodeTexImage')
    image.image = bpy.data.images.load(str(texture), check_existing=True)
    image.interpolation = 'Linear'
    geo = nodes.new('ShaderNodeNewGeometry')
    mix = nodes.new('ShaderNodeMixRGB')
    mix.blend_type = 'MIX'
    mix.inputs[2].default_value = (.91,.91,.91,1)
    links.new(geo.outputs['Backfacing'], mix.inputs[0])
    links.new(image.outputs['Color'], mix.inputs[1])
    links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(image.outputs['Alpha'], bsdf.inputs['Alpha'])
    return mat

def base_material():
    mat = bpy.data.materials.new('MAT-person-acrylic-approx')
    mat.use_nodes = True
    mat.surface_render_method = 'DITHERED'
    bsdf = principled_node(mat)
    bsdf.inputs['Base Color'].default_value = (.7,.75,.75,1)
    bsdf.inputs['Roughness'].default_value = .2
    bsdf.inputs['Alpha'].default_value = .18
    return mat

def standee(item, xyz, yaw, front=True, height=.15, vehicle=False):
    source = ROOT / item['image']
    im = bpy.data.images.load(str(source), check_existing=True)
    h = height
    w = h * im.size[0] / im.size[1]
    mesh = bpy.data.meshes.new('GEO-' + source.stem)
    mesh.from_pydata([(-w/2,0,0),(w/2,0,0),(w/2,0,h),(-w/2,0,h)], [], [(0,1,2,3)])
    mesh.update()
    uv = mesh.uv_layers.new(name='Artwork')
    for loop, point in zip(mesh.polygons[0].loop_indices, ((0,0),(1,0),(1,1),(0,1))):
        uv.data[loop].uv = point
    obj = bpy.data.objects.new('GEO-standee-' + source.stem, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (xyz[0],xyz[1],xyz[2]+.004)
    obj.rotation_euler.z = yaw + (0 if front or vehicle else math.pi)
    obj.data.materials.append(material('MAT-' + source.stem, source))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(xyz[0],xyz[1],xyz[2]+.002))
    foot = bpy.context.object
    foot.name = 'GEO-base-' + source.stem
    foot.dimensions = (max(.045,w*.9),.035,.004)
    foot.rotation_euler.z = yaw
    foot.data.materials.append(base_material())
    if vehicle:
        bpy.data.objects.remove(foot, do_unlink=True)
        return {'artwork':item['artwork'],'texture':item['image'],'class':'vehicle_occluder',
                'front':False,'location':list(xyz),'yaw_radians':yaw,'height_m':h,
                'geometry':obj.name,'base':None,'source':item['source']}
    return {'artwork':item['artwork'],'texture':item['image'],'class':item['class'],
            'front':front,'location':list(xyz),'yaw_radians':yaw,'height_m':h,
            'white_border_radius_px':item['border_radius_px'],
            'geometry':obj.name,'base':foot.name}

def mask_material(texture, color):
    mat = bpy.data.materials.new('MAT-person-mask')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(texture), check_existing=True)
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1)
    mix = nodes.new('ShaderNodeMixShader')
    links.new(tex.outputs['Alpha'], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])
    return mat

def render_instance_masks(scene, objects, path, full_dir=None):
    """Render color IDs with the same silhouette and occlusion order."""
    colors = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
    boards = [bpy.data.objects[item['geometry']] for item in objects]
    saved_visibility = {o.name:o.hide_render for o in scene.objects}
    saved_materials = [o.data.materials[0] for o in boards]
    saved_settings = (scene.render.film_transparent, scene.cycles.samples,
                      scene.cycles.use_denoising, scene.view_settings.view_transform,
                      scene.view_settings.exposure, scene.render.filepath)
    try:
        for obj in scene.objects:
            if obj.type == 'MESH' and obj not in boards:
                obj.hide_render = True
        for obj, item, color in zip(boards, objects, colors):
            obj.data.materials.clear()
            obj.data.materials.append(mask_material(ROOT/item['texture'], color))
        scene.render.film_transparent = True
        scene.cycles.samples = 4
        scene.cycles.use_denoising = False
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((480,640,4))
        pixels = np.flipud(pixels)
        palette = np.asarray(colors[:len(objects)], dtype=np.float32)
        distance = ((pixels[:,:,:3,None] - palette.T[None,None,:,:]) ** 2).sum(axis=2)
        ids = distance.argmin(axis=2)
        confidence = distance.min(axis=2) < .32
        boxes, visible_counts = [], []
        for i in range(len(objects)):
            ys, xs = np.nonzero((ids == i) & confidence & (pixels[:,:,3] > .45))
            if len(xs) == 0:
                raise ValueError('instance has no visible mask')
            boxes.append([int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1])
            visible_counts.append(int(len(xs)))
        bpy.data.images.remove(image)
        full_counts, full_paths = [], []
        if full_dir is not None:
            for i, board in enumerate(boards):
                for j, other in enumerate(boards):
                    other.hide_render = i != j
                full_path = full_dir / (path.stem + f'_{i:02d}.png')
                scene.render.filepath = str(full_path)
                bpy.ops.render.render(write_still=True)
                full_image = bpy.data.images.load(str(full_path), check_existing=False)
                full_pixels = np.asarray(full_image.pixels[:],dtype=np.float32).reshape((480,640,4))
                full_pixels = np.flipud(full_pixels)
                full_rgb = full_pixels[:,:,:3]
                delta = ((full_rgb - palette[i]) ** 2).sum(axis=2)
                isolated = (delta < .32) & (full_pixels[:,:,3] > .45)
                count = int(np.count_nonzero(isolated))
                bpy.data.images.remove(full_image)
                if not count:
                    raise ValueError('instance has no isolated mask')
                # Mixed anti-aliased colors may resemble another instance ID.
                # A pixel can only belong to this instance inside its isolated silhouette.
                matched = (ids == i) & confidence & (pixels[:,:,3] > .45) & isolated
                yy, xx = np.nonzero(matched)
                if not len(xx):
                    raise ValueError('instance has no visible mask')
                boxes[i] = [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
                visible_counts[i] = int(len(xx))
                full_counts.append(count)
                full_paths.append(full_path.relative_to(ROOT).as_posix())
            for board in boards:
                board.hide_render = False
        return boxes, colors[:len(objects)], visible_counts, full_counts, full_paths
    finally:
        for obj, mat in zip(boards, saved_materials):
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        for obj in scene.objects:
            if obj.name in saved_visibility:
                obj.hide_render = saved_visibility[obj.name]
        (scene.render.film_transparent,scene.cycles.samples,scene.cycles.use_denoising,
         scene.view_settings.view_transform,scene.view_settings.exposure,scene.render.filepath) = saved_settings
