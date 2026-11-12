"""Render a same-source synthetic YOLO Pose candidate set from an existing scene.

Run in Blender with --background INPUT.blend --python-exit-code 1 --python THIS --
--output NEW_DIR --geometry FILE --count 40 --seed 9011. No .blend changes are saved.
The 4 scale endpoints remain unlabelled until their locations are verified.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

KEYPOINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
             'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')


def project(scene, camera, point):
    p = world_to_camera_view(scene, camera, Vector(point))
    return (float(p.x), float(1 - p.y), float(p.z))


def texture_xy(point):
    return Vector(((point[0] - 627) / 627, (627 - point[1]) / 627, .065))


def look_at(camera, roll):
    forward = -camera.location.normalized()
    right = forward.cross(Vector((0, 1, 0))).normalized()
    up = right.cross(forward).normalized()
    camera.rotation_euler = (Matrix((right, up, -forward)).transposed() @
                             Matrix.Rotation(roll, 3, 'Z')).to_euler()


def in_polygon(p, polygon):
    x, y = p
    result = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
            result = not result
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--geometry', required=True)
    ap.add_argument('--count', type=int, default=40, help='A multiple of ten, at least 10')
    ap.add_argument('--seed', type=int, default=9011)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--samples', type=int, default=32)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.count < 10 or args.count % 10 or not 256 <= args.resolution <= 2048 or args.samples < 1:
        ap.error('count must be a positive multiple of ten >= 10; resolution 256..2048; samples >= 1')
    out = Path(args.output).resolve()
    if out.exists():
        ap.error('output exists; refusing to overwrite')
    if not bpy.data.filepath:
        ap.error('open the source Blender model first')
    geometry = json.loads(Path(args.geometry).read_text(encoding='utf-8-sig'))
    scene = bpy.context.scene
    camera = bpy.data.objects['training_camera']
    hands = [bpy.data.objects['temperature_needle_rotate_Z'], bpy.data.objects['humidity_needle_rotate_Z']]
    lights = [o for o in scene.objects if o.type == 'LIGHT']
    base_lights = {o.name: {'energy':o.data.energy, 'location':list(o.location)} for o in lights}
    for obj in [camera] + hands:
        obj.animation_data_clear()
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.samples = args.samples
    for split in ('train','val','test'):
        (out/'images'/split).mkdir(parents=True)
        (out/'labels'/split).mkdir(parents=True)
    rng = random.Random(args.seed)
    indices = list(range(args.count))
    # Two independent stratified lists avoid correlating the two needle values.
    t_bins, h_bins = indices[:], indices[:]
    rng.shuffle(t_bins); rng.shuffle(h_bins)
    splits = ['train']*(args.count*8//10) + ['val']*(args.count//10) + ['test']*(args.count//10)
    rng.shuffle(splits)
    manifest = {'version':1, 'status':'synthetic candidate; not real-world validation',
        'source_model':bpy.data.filepath, 'source':'one AI-generated dial; permission/provenance and fidelity unverified',
        'keypoint_order':KEYPOINTS, 'endpoint_status':'unverified: last 4 points are unlabelled (0,0,0)',
        'visibility_status':'approximate; brightness/needle overlap heuristic; manual review required',
        'split_status':'distinct parameter draws, but all splits share one model and texture; not an independent test',
        'seed':args.seed, 'count':args.count, 'split_ratio':[8,1,1], 'resolution':args.resolution,
        'cycles_samples':args.samples, 'original_lights':base_lights, 'samples':[]}
    for index, split in enumerate(splits):
        fraction_t = (t_bins[index] + rng.random()) / args.count
        fraction_h = (h_bins[index] + rng.random()) / args.count
        hands[0].rotation_euler.z = math.radians(-147 - 241*fraction_t)
        hands[1].rotation_euler.z = math.radians(44 - 264*fraction_h)
        camera.location = (rng.uniform(-.8,.8), rng.uniform(-.65,.65), rng.uniform(3.8,4.7))
        roll = math.radians(rng.uniform(-8,8))
        look_at(camera, roll)
        lamp_values = {}
        for light in lights:
            base = base_lights[light.name]
            light.data.energy = base['energy']*rng.uniform(.75,1.25)
            light.location = Vector(base['location']) + Vector((rng.uniform(-.3,.3),rng.uniform(-.3,.3),0))
            lamp_values[light.name] = {'energy':float(light.data.energy),'location':list(light.location)}
        scene.cycles.seed = args.seed + index
        bpy.context.view_layer.update()
        outline = [project(scene, camera, (1.05*math.cos(i*math.tau/128),1.05*math.sin(i*math.tau/128),z))
                   for z in (-.06,.15) for i in range(128)]
        if min(p[2] for p in outline) <= 0:
            raise RuntimeError('Object behind camera')
        x0,y0 = min(p[0] for p in outline), min(p[1] for p in outline)
        x1,y1 = max(p[0] for p in outline), max(p[1] for p in outline)
        if min(x0,y0,1-x1,1-y1) < .025:
            raise RuntimeError('Sample %d exceeds frame safety margin' % index)
        name = 'meter_%05d' % index
        path = out/'images'/split/(name+'.png')
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        points = [hands[0].matrix_world@Vector((0,0,0)),hands[0].matrix_world@Vector((.76,0,0)),
                  hands[1].matrix_world@Vector((0,0,0)),hands[1].matrix_world@Vector((.31,0,0)),
                  texture_xy(geometry['temperature'][0]['end']),texture_xy(geometry['temperature'][-1]['end']),
                  texture_xy(geometry['humidity'][0]['end']),texture_xy(geometry['humidity'][-1]['end'])]
        points = [project(scene,camera,p) for p in points]
        # The higher humidity hand can occlude the temperature hand.
        polygon = [project(scene,camera,hands[1].matrix_world @ p.co)[:2]
                   for p in hands[1].data.splines[0].bezier_points]
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = image.pixels[:]
        def brightness(p):
            x = min(args.resolution-1, max(0, int(p[0]*args.resolution)))
            y = min(args.resolution-1, max(0, int((1-p[1])*args.resolution)))
            i = 4*(y*args.resolution+x)
            return max(pixels[i:i+3])
        states = []
        for k,p in enumerate(points):
            if k >= 4:
                states.append(0)
            elif not (0 <= p[0] < 1 and 0 <= p[1] < 1 and p[2] > 0):
                states.append(0)
            elif (k in (0,1) and in_polygon(p[:2], polygon)) or brightness(p) > .97:
                states.append(1)
            else:
                states.append(2)
        bpy.data.images.remove(image)
        row = [0, (x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0]
        for p,v in zip(points,states):
            row.extend([p[0],p[1],v] if v else [0.,0.,0])
        (out/'labels'/split/(name+'.txt')).write_text(' '.join(str(v) if isinstance(v,int) else '%.6f'%v for v in row)+'\n', encoding='utf8')
        manifest['samples'].append({'id':name,'split':split,'scene_id':'draw_%05d'%index,
            'camera_location':list(camera.location),'camera_rotation_euler':list(camera.rotation_euler),
            'roll_rad':roll,'temperature_fraction':fraction_t,'humidity_fraction':fraction_h,
            'temperature_angle_rad':hands[0].rotation_euler.z,'humidity_angle_rad':hands[1].rotation_euler.z,
            'lights':lamp_values,'bbox_xyxy':[x0,y0,x1,y1], 'projected_candidates':points, 'visibility':states})
        print('BATCH_PROGRESS',index+1,'/',args.count,flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (out/'data.yaml').write_text("path: " + json.dumps(out.as_posix()) + "\ntrain: images/train\nval: images/val\ntest: images/test\nkpt_shape: [8, 3]\nnames:\n  0: meter\n",encoding='utf8')
    print('BATCH_COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
