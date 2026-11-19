"""Replace red/green bin markings with alpha derived from the supplied photograph.

Run in Blender with the existing PBR scene open. The scene copy and manifest are
saved under datasets/assets/blender/waste_bin; the input image is packed unmodified.
The pixel rectangles and threshold are fitted estimates, not measured artwork.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[5]
PHOTO = ROOT / 'Archive/Assets/working_sources/legacy/waste_bin/照片/四色桶正面_闭盖_分类标志_001.jpg'
SOURCE = ROOT / 'datasets/assets/blender/waste_bin/blender_pbr_v2_20261110/waste_bin_pbr.blend'
# Image coordinates are top-left origin. The existing geometric border is retained.
RECTANGLES = {'red': (704, 335, 829, 522), 'green': (1182, 384, 1302, 572)}


def make_material(name, photo):
    mat = bpy.data.materials.new(name + ' photo-derived marking')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.9, .9, .9, 1)
    shader.inputs['Roughness'].default_value = .55
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = photo
    tex.extension = 'CLIP'
    separate = nodes.new('ShaderNodeSeparateColor')
    links.new(tex.outputs['Color'], separate.inputs['Color'])
    mins = []
    for channel in ('Green', 'Blue'):
        node = nodes.new('ShaderNodeMath')
        node.operation = 'MINIMUM'
        links.new(separate.outputs['Red'] if not mins else mins[-1].outputs[0], node.inputs[0])
        links.new(separate.outputs[channel], node.inputs[1])
        mins.append(node)
    ramp = nodes.new('ShaderNodeMapRange')
    ramp.clamp = True
    ramp.inputs['From Min'].default_value = .30
    ramp.inputs['From Max'].default_value = .62
    links.new(mins[-1].outputs[0], ramp.inputs['Value'])
    links.new(ramp.outputs['Result'], shader.inputs['Alpha'])
    return mat


def _generate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    output = args.output.resolve()
    owner = (ROOT / 'datasets/assets/blender/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner or output.exists():
        parser.error('output must be a new directory under datasets/assets/blender/waste_bin')
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != SOURCE.resolve() or not PHOTO.is_file():
        parser.error('open the workspace PBR source scene and provide the source photo')
    image = bpy.data.images.load(str(PHOTO), check_existing=True)
    image.colorspace_settings.name = 'sRGB'
    image.pack()
    width, height = image.size
    if (width, height) != (2275, 1279):
        raise ValueError('Unexpected source photo dimensions')
    for color, (x1, y1, x2, y2) in RECTANGLES.items():
        obj = bpy.data.objects[color + ' front marking']
        obj.data = obj.data.copy()
        uv = obj.data.uv_layers.active
        if uv is None:
            raise ValueError(color + ': missing marking UV')
        for loop in uv.data:
            u, v = loop.uv
            loop.uv = ((x1 + u * (x2 - x1)) / width,
                       1 - (y2 - v * (y2 - y1)) / height)
        obj.data.materials.clear()
        obj.data.materials.append(make_material(color, image))
    output.mkdir(parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'waste_bin_photo_markings.blend'))
    record = {'status': 'candidate_requires_visual_review', 'source_scene': SOURCE.relative_to(ROOT).as_posix(),
              'photo': PHOTO.relative_to(ROOT).as_posix(), 'photo_size': [width, height],
              'photo_license': 'user-supplied source; redistribution terms not established',
              'crop_xyxy': RECTANGLES, 'alpha': 'minimum linear RGB mapped from 0.30 to 0.62',
              'unchanged_markings': ['blue', 'gray'], 'split': 'unassigned'}
    (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args, _ = parser.parse_known_args(raw)
    if bpy.data.is_dirty:
        raise ValueError('save or discard scene changes before producing a guarded copy')
    output = args.output.resolve()
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media = {PHOTO}
    for image in bpy.data.images:
        if image.source == 'FILE':
            if not image.filepath:
                raise ValueError('scene image source path is missing')
            path = Path(bpy.path.abspath(image.filepath)).resolve()
            if not path.is_file():
                raise ValueError('scene image source missing: ' + str(path))
            media.add(path)
    with generation_guard('waste_bin.apply_photo_markings:' + str(output.relative_to(ROOT)),
                          sorted(media), [output], {'argv': raw, 'blender_version': bpy.app.version_string},
                          code_paths=[Path(__file__).resolve()],
                          procedural_inputs=[SOURCE], root=ROOT) as run:
        if run.reused:
            return
        _generate()


if __name__ == '__main__':
    main()
