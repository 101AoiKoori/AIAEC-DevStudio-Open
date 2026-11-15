"""Render diagnostic multi-view images from a waste-bin Blender scene.

Run with Blender --background MODEL --python-exit-code 1 --python THIS -- OUT.
Views are orthographic diagnostics, not registrations to the perspective photos.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _generate():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    if len(args)!=1:
        raise ValueError('provide a new output directory')
    out=Path(args[0]).resolve()
    out.mkdir(parents=True,exist_ok=False)
    scene=bpy.context.scene
    for obj in scene.objects:
        if obj.name.startswith(('red ', 'green ', 'gray ')) or obj.name in (
                'Ground','Background wall','Front road boundary'):
            obj.hide_render=True
    camera=scene.camera
    camera.animation_data_clear()
    camera.data.type='ORTHO'
    camera.data.ortho_scale=1.8
    x=-1.68
    target=Vector((x,0,.65))
    views={'front':(x,-5,.65),'side':(x+5,0,.65),
           'back':(x,5,.65),'top':(x,0,5)}
    scene.render.resolution_x=scene.render.resolution_y=1024
    scene.render.resolution_percentage=100
    scene.cycles.samples=8
    scene.render.image_settings.file_format='PNG'
    pivot=bpy.data.objects['blue lid pivot - rotate X']
    for state,angle in [('closed',0),('open',-270)]:
        pivot.rotation_euler.x=math.radians(angle)
        for name,position in views.items():
            camera.location=position
            camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
            scene.render.filepath=str(out/(state+'_'+name+'.png'))
            bpy.ops.render.render(write_still=True)


def _guard_inputs():
    if bpy.data.is_dirty:
        raise ValueError('loaded Blender scene has unsaved changes; use a saved workspace copy')
    scene = Path(bpy.data.filepath).resolve()
    procedural, media = {scene}, set()
    sidecar = scene.parent / 'manifest.json'
    if sidecar.is_file():
        import json
        procedural.add(sidecar)
        details = json.loads(sidecar.read_text(encoding='utf8'))
        if details.get('source_scene'):
            procedural.add(Path.cwd() / details['source_scene'])
        if details.get('photo'):
            media.add(Path.cwd() / details['photo'])
    for image in bpy.data.images:
        if image.source == 'FILE':
            if not image.filepath:
                raise ValueError('scene image has no source path: ' + image.name)
            path = Path(bpy.path.abspath(image.filepath)).resolve()
            if path.is_file():
                media.add(path)
            else:
                raise ValueError('scene image source missing: ' + str(path))
    return sorted(media), sorted(procedural)


def main():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    if len(args) != 1:
        raise ValueError('provide a new output directory')
    root = Path.cwd().resolve()
    out = Path(args[0]).resolve()
    sys.path.insert(0, str(root))
    from lab.training_common.production_guard import generation_guard
    media, procedural = _guard_inputs()
    with generation_guard('waste_bin.render_reference:' + str(out.relative_to(root)),
                          media, [out], {'argv': args, 'blender_version': bpy.app.version_string},
                          code_paths=[Path(__file__).resolve()], procedural_inputs=procedural, root=root) as run:
        if run.reused:
            return
        _generate()


if __name__=='__main__':
    main()
