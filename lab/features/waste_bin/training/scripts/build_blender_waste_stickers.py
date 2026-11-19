"""Build an editable four-bin sticker asset; no images are rendered.

Run from the workspace root with Blender opening the photo-marking parent::

    blender -b datasets/assets/blender/waste_bin/blender_photo_markings_v4/waste_bin_photo_markings.blend \
      --python-exit-code 1 --python lab/features/waste_bin/training/scripts/build_blender_waste_stickers.py

The new blend and task plan must not exist. The 24 source images are packed.
One material-index keyframe per bin and frame makes changing frames work in a
fresh Blender session without enabling automatic Python execution. This builds
a source asset, not a guarded training-data generation or rendering entry.
"""
import json
import random
import re
from pathlib import Path

import bpy
ROOT = Path(__file__).resolve().parents[5]
PARENT = 'datasets/assets/blender/waste_bin/blender_photo_markings_v4/waste_bin_photo_markings.blend'
OUTPUT = 'datasets/assets/blender/waste_bin/waste_sticker_animated_v4.blend'
PLAN = 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json'
TOOLKIT = 'Archive/Assets/working_sources/digital/waste_sticker/toolkit'
REFERENCE_PHOTO = 'Archive/Assets/captures/car_real_capture_20261112/P22_TRASH.png'
VIDEO = 'Archive/Assets/captures/car_real_capture_20261112/V_20261112_152304.mp4'
FRAMES = 'Archive/Assets/captures/car_real_capture_20261112/frames_20261112_152304'
COLORS = ('blue', 'red', 'green', 'gray')
SEED = 20261114
FRAME_COUNT = 240


def source_classes():
    text = (ROOT / TOOLKIT / 'configs/classes.yaml').read_text(encoding='utf8')
    names = re.findall(r'^\s*-\s*\{name:\s*([^,]+),\s*id:', text, flags=re.MULTILINE)
    names = [name.strip() for name in names]
    tasks = json.loads((ROOT / 'config/perception_tasks.json').read_text(encoding='utf8'))['tasks']
    items = next(task['waste_items'] for task in tasks if task['id'] == 'waste_bin')
    if len(names) != 24 or len(set(names)) != 24 or set(names) != set(items):
        raise ValueError('24 artwork names must match the task mapping')
    return names, items


def make_material(name, path, class_id):
    image = bpy.data.images.load(str(path), check_existing=True)
    image.colorspace_settings.name = 'sRGB'
    if not image.packed_file:
        image.pack()
    material = bpy.data.materials.new(f'waste_sticker_{class_id:02d}_{name}')
    material.use_nodes = True
    material['class_id'] = class_id
    material['source_artwork'] = path.relative_to(ROOT).as_posix()
    nodes = material.node_tree.nodes
    output = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output), None)
    if output is None:
        output = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL'), None)
    if output is None:
        output = nodes.new('ShaderNodeOutputMaterial')
    surface = output.inputs['Surface']
    if surface.is_linked:
        shader = surface.links[0].from_node
        if shader.type != 'BSDF_PRINCIPLED':
            raise ValueError('material output must use a Principled BSDF')
    else:
        shader = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED'), None)
        if shader is None:
            shader = nodes.new('ShaderNodeBsdfPrincipled')
        material.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    shader.inputs['Roughness'].default_value = .75
    texture = nodes.new('ShaderNodeTexImage')
    texture.image = image
    texture.interpolation = 'Linear'
    # Composite any source transparency onto white paper, without modifying it.
    mix = nodes.new('ShaderNodeMixRGB')
    mix.blend_type = 'MIX'
    mix.inputs[1].default_value = (1, 1, 1, 1)
    material.node_tree.links.new(texture.outputs['Alpha'], mix.inputs[0])
    material.node_tree.links.new(texture.outputs['Color'], mix.inputs[2])
    material.node_tree.links.new(mix.outputs[0], shader.inputs['Base Color'])
    return material


def make_paper(color, materials):
    body = bpy.data.objects[color + ' body']
    marking = bpy.data.objects[color + ' front marking']
    if body.type != 'MESH' or marking.type != 'MESH':
        raise ValueError('expected the source bin body and marking meshes')
    # Source vertices were constructed in the common parent coordinate system.
    # Transform through world into body coordinates to avoid assuming its origin.
    bpy.context.view_layer.update()
    inverse = body.matrix_world.inverted()
    points = [inverse @ (body.matrix_world @ vertex.co) for vertex in body.data.vertices]
    z0, z1 = min(p.z for p in points), max(p.z for p in points)
    lower = [p for p in points if abs(p.z-z0) < 1e-5]
    upper = [p for p in points if abs(p.z-z1) < 1e-5]
    if len(lower) < 2 or len(upper) < 2 or z1 <= z0:
        raise ValueError('bin body does not expose the expected tapered rings')
    center_x = (min(p.x for p in points) + max(p.x for p in points))/2
    body_width = max(p.x for p in points) - min(p.x for p in points)
    width = body_width * .30
    height = body_width * .285
    low = z0 + .015
    high = low + height
    marking_low = min((inverse @ (marking.matrix_world @ v.co)).z for v in marking.data.vertices)
    if high >= marking_low - .01:
        raise ValueError('paper would intersect the existing category marking')
    front0, front1 = min(p.y for p in lower), min(p.y for p in upper)
    def front(z):
        return front0 + (front1-front0)*(z-z0)/(z1-z0) - .006
    vertices = [(center_x-width/2, front(low), low), (center_x+width/2, front(low), low),
                (center_x+width/2, front(high), high), (center_x-width/2, front(high), high)]
    mesh = bpy.data.meshes.new(color + ' item sticker paper mesh')
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name='paper UV')
    for loop, point in zip(mesh.polygons[0].loop_indices, ((0,0), (1,0), (1,1), (0,1))):
        uv.data[loop].uv = point
    for material in materials:
        mesh.materials.append(material)
    obj = bpy.data.objects.new(color + ' item sticker paper', mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = body
    obj.matrix_parent_inverse.identity()
    obj['bin_id'] = 'source_' + color
    obj['bin_color'] = color
    obj['paper_dimensions_source_units'] = [width, height]
    obj['placement_basis'] = 'P22 visual reference and source bin geometry; approximate'
    return obj


def main():
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


if __name__ == '__main__':
    main()
