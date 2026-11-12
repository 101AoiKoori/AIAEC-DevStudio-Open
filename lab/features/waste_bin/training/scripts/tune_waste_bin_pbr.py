"""Tune the existing four-bin Blender scene and save a separate editable copy.

Run with Blender 5.2: blender -b SOURCE.blend --python-exit-code 1 --python
THIS_SCRIPT -- NEW_OUTPUT_DIRECTORY. Source geometry, light and camera animation
are retained; generated previews are for visual comparison, not dataset labels.
"""
import sys
from pathlib import Path

import bpy


def principled(name):
    material = bpy.data.materials[name]
    node = material.node_tree.nodes.get('Principled BSDF')
    if node is None:
        raise ValueError(f'Expected Principled BSDF in {name}')
    return material, node


def plastic_microtexture(material, shader, roughness, distance):
    shader.inputs['Metallic'].default_value = 0.0
    shader.inputs['IOR'].default_value = 1.47
    shader.inputs['Roughness'].default_value = roughness
    nodes, links = material.node_tree.nodes, material.node_tree.links
    texture = nodes.new('ShaderNodeTexNoise')
    texture.name = 'PBR injection molded microtexture'
    texture.inputs['Scale'].default_value = 210.0
    texture.inputs['Detail'].default_value = 2.0
    texture.inputs['Roughness'].default_value = 0.55
    bump = nodes.new('ShaderNodeBump')
    bump.name = 'PBR subtle plastic relief'
    bump.inputs['Strength'].default_value = 0.08
    bump.inputs['Distance'].default_value = distance
    links.new(texture.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], shader.inputs['Normal'])


def main():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    if len(args) != 1:
        raise ValueError('Expected one new output directory')
    output = Path(args[0]).resolve()
    source = Path(bpy.data.filepath).resolve()
    workspace = Path.cwd().resolve()
    if not source.is_relative_to(workspace) or not output.is_relative_to(workspace):
        raise ValueError('Source and output must be inside the workspace')
    if output.exists():
        raise FileExistsError(output)
    scene = bpy.context.scene
    if not scene.camera:
        raise ValueError('Source scene has no camera')
    # Linear colors tuned against the available real-bin photograph under the
    # original scene lights; they are editable, not measurements of albedo.
    colors = {
        'blue': (0.006, 0.035, 0.48, 1.0),
        'red': (0.50, 0.012, 0.007, 1.0),
        'green': (0.004, 0.36, 0.008, 1.0),
        'gray': (0.24, 0.25, 0.21, 1.0),
    }
    for name in ('blue', 'red', 'green', 'gray'):
        material, shader = principled(name + ' adjustable body color')
        shader.inputs['Base Color'].default_value = colors[name]
        plastic_microtexture(material, shader, 0.18, 0.00012)
        _, marking = principled(name + ' photographed marking')
        marking.inputs['Roughness'].default_value = 0.5
    material, lid = principled('Black lid - editable')
    plastic_microtexture(material, lid, 0.17, 0.00015)
    output.mkdir(parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'waste_bin_pbr.blend'))
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output / 'preview.png')
    bpy.ops.render.render(write_still=True)
    print('PBR_BIN_SAVED', output)


if __name__ == '__main__':
    main()
