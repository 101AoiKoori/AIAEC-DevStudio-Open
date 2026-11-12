"""Tune an existing animated meter scene and save a separate editable copy.

Run with Blender 5.2: blender -b SOURCE.blend --python-exit-code 1 --python
THIS_SCRIPT -- NEW_OUTPUT_DIRECTORY. The packed face, existing lights, camera
and needle animation stay in place. The thin clear cover has Fresnel reflections.
"""
import sys
from pathlib import Path

import bpy


def shader(name):
    material = bpy.data.materials[name]
    node = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if node is None:
        raise ValueError(f'Expected Principled BSDF in {name}')
    return node


def thin_cover():
    cover = bpy.data.objects['glass_reflection_surface']
    material = bpy.data.materials.new('PBR clear thin meter cover')
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    mix = nodes.new('ShaderNodeMixShader')
    clear = nodes.new('ShaderNodeBsdfTransparent')
    reflection = nodes.new('ShaderNodeBsdfAnisotropic')
    reflection.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    reflection.inputs['Roughness'].default_value = 0.035
    fresnel = nodes.new('ShaderNodeFresnel')
    fresnel.inputs['IOR'].default_value = 1.49
    # The model has one cover plane, rather than a closed solid. A shadow ray
    # should pass through that plane rather than treating its reflection as an
    # opaque blocker between the existing lamps and the printed face.
    light_path = nodes.new('ShaderNodeLightPath')
    shadow_mix = nodes.new('ShaderNodeMixShader')
    links.new(fresnel.outputs['Fac'], mix.inputs[0])
    links.new(clear.outputs['BSDF'], mix.inputs[1])
    links.new(reflection.outputs['BSDF'], mix.inputs[2])
    links.new(light_path.outputs['Is Shadow Ray'], shadow_mix.inputs[0])
    links.new(mix.outputs['Shader'], shadow_mix.inputs[1])
    links.new(clear.outputs['BSDF'], shadow_mix.inputs[2])
    links.new(shadow_mix.outputs['Shader'], output.inputs['Surface'])
    cover.data.materials[0] = material


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
    for name, roughness in (('metal', 0.23), ('bezel', 0.14)):
        node = shader(name)
        node.inputs['Metallic'].default_value = 0.96
        node.inputs['Roughness'].default_value = roughness
    for name in ('temperature_needle_black', 'humidity_needle_black'):
        node = shader(name)
        node.inputs['Metallic'].default_value = 0.1
        node.inputs['Roughness'].default_value = 0.3
    thin_cover()
    output.mkdir(parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'meter_9011_pbr.blend'))
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output / 'preview.png')
    bpy.ops.render.render(write_still=True)
    print('PBR_METER_SAVED', output)


if __name__ == '__main__':
    main()
