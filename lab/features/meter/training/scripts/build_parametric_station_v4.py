"""Build a station copy with explicitly generated dial ticks and endpoint markers.

Run inside Blender with the existing station scene loaded. The output directory
 must be new and live under datasets/assets/blender/meter. No source scene is changed.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/meter/. Production remains paused.")

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_meter_batch import texture_xy

ROOT = Path(__file__).resolve().parents[5]
PIVOT = Vector((0, -1.04, .02))
SIZE = 1254


def main():
    raise RuntimeError("Retired production entry; use datasets/scripts/meter/ and its source.yaml")


def _legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    from meter_production_guard import guarded, blender_media, reload_unpacked_media
    with guarded('build_parametric_station_v4', args, __file__, bpy.data.filepath, extra_inputs=[], code_dependencies=[Path(__file__).with_name('render_meter_batch.py')], media_probe=lambda: blender_media(bpy), media_reload=lambda: reload_unpacked_media(bpy)) as run:
        if run.reused:
            return
        out = args.output.resolve()
        owner = (ROOT / 'datasets/assets/blender/meter').resolve()
        if not out.is_relative_to(owner) or out == owner or out.exists():
            parser.error('output must be a new directory under datasets/assets/blender/meter')
        scene = bpy.context.scene
        if scene.name != 'Station prototype previews':
            raise ValueError('expected the station preview scene')
        if not bpy.data.filepath:
            raise ValueError('expected a saved workspace source scene')
        mother = Path(bpy.data.filepath).resolve()
        if not mother.is_relative_to(ROOT) or not mother.is_file():
            raise ValueError('source scene must be an existing workspace file')
        pivot = scene.objects['Station dial tilt pivot']
        face = scene.objects['Station PDF_face_inpainted']
        specs = {
            'temperature': ('Station temperature_needle_rotate_Z', .83, -147., -388., 80),
            'humidity': ('Station humidity_needle_rotate_Z', .36, 44., -220., 50),
        }
        # Image coordinates have a top-left origin; image buffer has a bottom-left origin.
        pixels = np.ones((SIZE, SIZE, 4), dtype=np.float32)
        pixels[:, :, :3] = (.88, .89, .85)
        endpoints, ticks = {}, {}
        for name, (hand_name, radius, start, end, count) in specs.items():
            hand = scene.objects[hand_name]
            # Both hands and face carry the same parent-inverse translation; their
            # local coordinates therefore share the original face coordinate frame.
            center = hand.location.copy()
            tick_list = []
            for i in range(count + 1):
                angle = math.radians(start + (end - start) * i / count)
                direction = Vector((math.cos(angle), math.sin(angle), 0))
                outer = center + radius * direction
                inner = center + (radius - (.045 if i % 5 == 0 else .025)) * direction
                # Rasterize using the same exact outer coordinates later used by markers.
                x0, y0 = (inner.x + 1) * 627, (1 - inner.y) * 627
                x1, y1 = (outer.x + 1) * 627, (1 - outer.y) * 627
                steps = max(2, int(math.hypot(x1-x0, y1-y0) * 2))
                xs = np.clip(np.rint(np.linspace(x0, x1, steps)).astype(int), 0, SIZE-1)
                ys = np.clip(np.rint(np.linspace(y0, y1, steps)).astype(int), 0, SIZE-1)
                width = 2 if i % 5 == 0 else 1
                for dx in range(-width, width+1):
                    for dy in range(-width, width+1):
                        pixels[SIZE-1-np.clip(ys+dy, 0, SIZE-1), np.clip(xs+dx, 0, SIZE-1), :3] = .035
                tick_list.append({'inner_px': [x0, y0], 'outer_px': [x1, y1]})
                if i in (0, count):
                    key = name + ('_min' if i == 0 else '_max')
                    endpoints[key] = [x1, y1]
                    scene.objects['KPT-meter-' + key].location = texture_xy(endpoints[key]) - PIVOT
            ticks[name] = tick_list
        out.mkdir(parents=True)
        image = bpy.data.images.new('Parametric meter face', width=SIZE, height=SIZE, alpha=True)
        image.pixels.foreach_set(pixels.ravel())
        image.filepath_raw = str(out / 'face.png')
        image.file_format = 'PNG'
        image.save()
        # Load saved bytes so renderer's packed-image identity check remains meaningful.
        bpy.data.images.remove(image)
        image = bpy.data.images.load(str(out / 'face.png'), check_existing=False)
        image.pack()
        nodes = [n for m in face.data.materials if m and m.use_nodes for n in m.node_tree.nodes
                 if n.type == 'TEX_IMAGE' and n.image]
        if len(nodes) != 1:
            raise ValueError('expected one face image node')
        nodes[0].image = image
        config = {'source_blend': (out / 'station.blend').relative_to(ROOT).as_posix(),
                  'mother_blend': mother.relative_to(ROOT).as_posix(),
                  'source_face': (out / 'face.png').relative_to(ROOT).as_posix(),
                  'image_size': [SIZE, SIZE], 'coordinate_system': 'top-left pixels; generated outer tick endpoints',
                  'status': 'parameterized synthetic tick geometry; physical meter correspondence and point visibility unknown',
                  'endpoints': endpoints, 'ticks': ticks,
                  'scale_spec': {k: {'radius': v[1], 'start_deg': v[2], 'end_deg': v[3], 'intervals': v[4]} for k,v in specs.items()}}
        (out / 'endpoints.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf8')
        bpy.ops.wm.save_as_mainfile(filepath=str(out / 'station.blend'), copy=True)
        print('PARAMETRIC_STATION_READY', out, flush=True)


if __name__ == '__main__':
    main()
