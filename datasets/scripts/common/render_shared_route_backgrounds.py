"""Render empty Arena views from the eight saved camera routes.

Run with Blender --background aiaec_dataset.blend --python this_file -- [options].
The batch manifest records the exact scene, route, frame and camera transform;
task-specific placement and split decisions are made during frame ingestion.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import bpy


TASKS = ('person', 'e_bike', 'fire', 'building_name_paper', 'sign',
         'waste_bin_state', 'waste_sticker', 'meter')
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'datasets/assets/blender/aiaec_dataset.blend'
OUTPUT = ROOT / 'datasets/raw/shared/backgrounds/rendered/aiaec_eight_routes_20261117'
MANIFEST = ROOT / 'datasets/annotated/common/sources/aiaec_eight_routes_20261117/manifest.json'


def main():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count-per-route', type=int, default=24)
    parser.add_argument('--scene', choices=('Arena', 'Arena Detailed', 'Arena Indoor Soft Light'),
                        default='Arena')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--frames', help='Explicit comma-separated frames; fractional subframes supported')
    parser.add_argument('--target-split', choices=('train','val','test'), default='train')
    parser.add_argument('--lens-scale', type=float, default=1.)
    parser.add_argument('--exposure-ev', type=float, default=0.)
    options = parser.parse_args(args)
    if not 1 <= options.count_per_route <= 120:
        raise ValueError('count-per-route must be 1..120')
    output = options.output.resolve()
    manifest = options.manifest.resolve()
    meter_work = ROOT / 'datasets/annotated/meter/full_production'
    if not output.is_relative_to(ROOT / 'datasets/raw/shared/backgrounds/rendered') and not output.is_relative_to(meter_work):
        raise ValueError('output must be in the shared rendered raw background pool')
    if not manifest.is_relative_to(ROOT / 'datasets/annotated/common/sources') and not manifest.is_relative_to(meter_work):
        raise ValueError('manifest must be in annotated/common/sources')
    if manifest.exists():
        raise FileExistsError(manifest)
    scene = bpy.data.scenes[options.scene]
    routes = bpy.data.collections['AEC_eight_camera_routes']
    if routes.name not in {collection.name for collection in scene.collection.children}:
        # The Detailed variant does not save these routes. Link them only in the
        # background process; the source .blend is never saved by this script.
        scene.collection.children.link(routes)
    # Blender starts on the file's last active scene. Both animation evaluation
    # and rendering must use the selected scene or frames can look identical.
    bpy.context.window.scene = scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.view_settings.exposure += options.exposure_ev
    if not .8 <= options.lens_scale <= 1.2:
        raise ValueError('lens-scale must be 0.8..1.2')
    frames = [1 + round(i * 119 / (options.count_per_route - 1))
              for i in range(options.count_per_route)] if options.count_per_route > 1 else [60]
    if options.frames:
        frames = [float(value) for value in options.frames.split(',')]
        if any(not 1 <= frame <= 120 for frame in frames):
            raise ValueError('explicit frames must be inside 1..120')
    if len(set(frames)) != len(frames):
        raise ValueError('sampled frames are not unique')
    source = Path(bpy.data.filepath).resolve()
    source_digest = sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for task in TASKS:
        camera = bpy.data.objects['Arena | CAM_' + task]
        path = bpy.data.objects['Arena | PATH_' + task]
        rig = bpy.data.objects['Arena | RIG_' + task]
        if camera.parent != rig or rig.constraints[0].target != path or camera.name not in scene.objects:
            raise ValueError('route is not connected in selected scene: ' + task)
        scene.camera = camera
        camera.data.lens *= options.lens_scale
        for frame in frames:
            scene.frame_set(int(frame), subframe=frame-int(frame))
            evaluated = camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
            matrix = [[round(float(v), 8) for v in row] for row in evaluated.matrix_world]
            frame_name = str(frame).replace('.','p') if options.frames else f'{frame:03}'
            image = output / f'{task}_f{frame_name}.png'
            if image.exists():
                raise FileExistsError(image)
            scene.render.filepath = str(image)
            bpy.ops.render.render(scene=scene.name, write_still=True)
            records.append({
                'image': image.relative_to(ROOT).as_posix(),
                'content_sha256': sha256(image.read_bytes()).hexdigest(),
                'source_blend': source.relative_to(ROOT).as_posix(),
                'source_blend_sha256': source_digest,
                'scene': scene.name, 'route': task, 'frame': frame,
                'camera': camera.name, 'camera_matrix_world': matrix,
                'lens_mm': camera.data.lens, 'split': options.target_split,
                'image_size': [640, 480], 'source_kind': 'blender_render',
                'parent_group': SOURCE.relative_to(ROOT).as_posix(),
            })
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'schema_version': 1, 'purpose': 'shared_background_candidates',
                                    'visible_scene_meshes':[o.name for o in scene.objects if o.type=='MESH' and not o.hide_render],
                                    'records': records}, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')
    print('ROUTE_BACKGROUND_BATCH', len(records), str(manifest))


if __name__ == '__main__':
    main()
