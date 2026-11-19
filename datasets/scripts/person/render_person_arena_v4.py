"""Render a bounded person-standee pilot in the existing fitted arena scene.

Run with Blender --background <arena.blend> --python this_file -- --output <batch> --production-plan <plan>.
The scene camera and object placement are fitted, not field calibration. Outputs
are source-linked render candidates and do not carry complete training labels.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.blender_workspace import ARENA, DETAILED, activate_scene
from datasets.scripts.common.paths import sample_dir, split_dir
from datasets.scripts.person.person_blender_plan import validate_production_plan, select_occlusion_sources
_SOURCE = load_task_config('person')[1]
TEXTURES = (ROOT / _SOURCE['role_records']['foreground_texture']).parent
VEHICLE_MANIFEST = ROOT / _SOURCE['role_records']['vehicle_occluder']
SCENE_SOURCE = source_path('person', 'blender_original_scene')
USER_MASTER = source_path('person', 'blender_master_scene')
PERSON_WORKING_SCENE = source_path('person', 'blender_working_scene')



from datasets.scripts.common.blender_standees import principled_node, material, base_material, standee, mask_material, render_instance_masks

















def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int)
    ap.add_argument('--seed', type=int, default=20261113)
    ap.add_argument('--scene', choices=('original', 'user-master'), default='original')
    ap.add_argument('--camera-preset', choices=('master', 'person-near'), default='master')
    ap.add_argument('--camera-jitter', type=float, default=0.0)
    ap.add_argument('--position-jitter', type=float, default=0.0)
    ap.add_argument('--mixed-back', action='store_true')
    ap.add_argument('--occlusion', action='store_true')
    ap.add_argument('--vehicle-occlusion', action='store_true')
    ap.add_argument('--render-device', choices=('cpu', 'optix'), default='optix')
    ap.add_argument('--production-plan', type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output = (ROOT / args.output).resolve()
    if args.count is not None or args.mixed_back or args.occlusion or args.vehicle_occlusion:
        raise ValueError('count and target states must be declared in the production plan')
    if not 0 <= args.camera_jitter <= .05 or not 0 <= args.position_jitter <= .025:
        raise ValueError('render jitter is outside the supported range')
    plan_path = (ROOT / args.production_plan).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('production plan must be an existing workspace file')
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    validate_production_plan(declaration, args)
    if args.output not in {sample_dir('person').resolve(), split_dir('person', declaration['target_split']).resolve()}:
        raise ValueError('output must be the person sample or declared split')
    if type(declaration.get('budget')) is not int or not 1 <= declaration['budget'] <= 128:
        raise ValueError('render budget must be between 1 and 128')
    needs_vehicle = any(state['target']['occlusion'] == 'vehicle' for state in declaration['strata'])
    texture_manifest = TEXTURES / 'manifest.json'
    textures = json.loads(texture_manifest.read_text(encoding='utf8'))['records']
    scene_source = USER_MASTER if args.scene == 'user-master' else SCENE_SOURCE
    working_scene = PERSON_WORKING_SCENE if args.scene == 'user-master' else SCENE_SOURCE
    if Path(bpy.data.filepath).resolve() != working_scene:
        raise ValueError('expected source arena scene')
    procedural = list(dict.fromkeys((scene_source, working_scene, texture_manifest, plan_path,
        ROOT / ('datasets/annotated/person/v4_user_master_production_plan_20261113.json' if args.scene == 'user-master' else
                'Archive/technical/revisions/arena/blender_sources_20261116/arena_4m2_pbr_20261112/detail_provenance.json'))))
    media = {ROOT / row[key] for row in textures for key in ('artwork', 'image')}
    if needs_vehicle:
        vehicle_manifest = VEHICLE_MANIFEST
        procedural.append(vehicle_manifest)
        for row in json.loads(vehicle_manifest.read_text(encoding='utf8'))['records']:
            media.update(ROOT / row[key] for key in ('cropped_image', 'path', 'source_frame', 'source_video'))
    for image in bpy.data.images:
        if image.source != 'FILE' or image.packed_file:
            continue
        if not image.filepath:
            raise ValueError('scene image has no traceable source: ' + image.name)
        path = Path(bpy.path.abspath(image.filepath)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('scene image source must be an existing workspace file: ' + image.name)
        media.add(path)
    procedural.extend([ROOT/'datasets/scripts/person/config/source.yaml', ROOT/'datasets/scripts/person/config/dataset.yaml'])
    parameters = {'seed': args.seed, 'scene': args.scene, 'camera_preset': args.camera_preset,
                  'camera_jitter': args.camera_jitter, 'position_jitter': args.position_jitter,
                  'render_device': args.render_device}
    with generation_guard('person.render_arena_v4:' + args.output.relative_to(ROOT).as_posix(),
                          list(media) + procedural, [args.output], parameters,
                          code_paths=[Path(__file__), ROOT/'datasets/scripts/person/person_blender_plan.py', ROOT/'datasets/scripts/common/blender_standees.py', ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py', ROOT/'datasets/scripts/common/blender_workspace.py'], procedural_inputs=procedural, root=ROOT,
                          production_plan=declaration, sample_output_paths=[args.output/'images'],
                          auxiliary_output_paths=[args.output/'instance_ids', args.output/'instance_full',
                                                  args.output/'instance_without_vehicle', args.output/'rejected']) as run:
        if run.reused:
            print('reused indexed production output', flush=True)
            return
        activate_scene(ARENA if args.scene == 'user-master' else DETAILED)
        _produce(args, run)


def _produce(args, run):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output is not empty')
    source_scene = USER_MASTER if args.scene == 'user-master' else SCENE_SOURCE
    working_scene = PERSON_WORKING_SCENE if args.scene == 'user-master' else SCENE_SOURCE
    if Path(bpy.data.filepath).resolve() != working_scene.resolve():
        raise ValueError('expected source arena scene')
    if not 0 <= args.camera_jitter <= .05:
        raise ValueError('camera jitter must be between 0 and 0.05 m')
    if not 0 <= args.position_jitter <= .025:
        raise ValueError('position jitter must be between 0 and 0.025 m')
    textures = json.loads((TEXTURES/'manifest.json').read_text(encoding='utf8'))['records']
    scene = bpy.context.scene
    cam = scene.objects.get('Vehicle camera')
    if cam is None: raise ValueError('missing source vehicle camera')
    scene.camera = cam
    master_camera = {'location':list(cam.location),'rotation_euler':list(cam.rotation_euler),
                     'lens_mm':cam.data.lens}
    target = Vector((0,.03,.085))
    use_near_camera = args.scene == 'original' or args.camera_preset == 'person-near'
    if use_near_camera:
        cam.location = (0,-.40,.16)
        cam.rotation_euler = (target - cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.lens = 22
    camera_origin = cam.location.copy()
    camera_rotation = cam.rotation_euler.copy()
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    if args.render_device == 'optix':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        devices = [device for device in prefs.devices if device.type == 'OPTIX']
        if not devices:
            raise RuntimeError('No OptiX GPU available; rendering was not started')
        for device in prefs.devices:
            device.use = device.type == 'OPTIX'
        scene.cycles.device = 'GPU'
    else:
        scene.cycles.device = 'CPU'
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('images','labels','instance_ids','instance_full','instance_without_vehicle','rejected'):
        (out/folder).mkdir(parents=True)
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    communities = [x for x in textures if x['class'] == 'community']
    noncommunities = [x for x in textures if x['class'] == 'noncommunity']
    plan = run.plan
    scheduled = [state for state in plan['strata'] for _ in range(state['planned_count'])]
    vehicles = json.loads(VEHICLE_MANIFEST.read_text(encoding='utf8'))['records'] if any(state['target']['occlusion'] == 'vehicle' for state in scheduled) else []
    records = []
    rejected = []
    for index, state in enumerate(scheduled):
        rng = random.Random(args.seed + index * 1009)
        target_state = state['target']
        args.occlusion = target_state['occlusion'] == 'person'
        args.vehicle_occlusion = target_state['occlusion'] == 'vehicle'
        args.mixed_back = target_state['white_back_class'] == 'noncommunity'
        if args.camera_jitter:
            cam.location = (camera_origin.x+rng.uniform(-args.camera_jitter,args.camera_jitter),
                            camera_origin.y+rng.uniform(-args.camera_jitter/2,args.camera_jitter/2),
                            camera_origin.z+rng.uniform(-args.camera_jitter/3,args.camera_jitter/3))
            if use_near_camera:
                cam.rotation_euler = (target-cam.location).to_track_quat('-Z','Y').to_euler()
            else:
                cam.rotation_euler = camera_rotation
        positive_count = target_state['front_count']
        back_count = 1
        non_count = target_state['noncommunity_front']
        use_noncommunity_back = args.mixed_back
        if use_noncommunity_back:
            non_count = 1
            non = rng.sample(noncommunities,2)
            selected = non + rng.sample(communities,positive_count-non_count)
            back_item = non[-1]
        else:
            selected = rng.sample(noncommunities,non_count) + rng.sample(communities,positive_count-non_count+back_count)
            back_item = selected[-1]
        rng.shuffle(selected)
        positions = np.linspace(-.23,.23,len(selected))
        if args.position_jitter:
            positions += np.array([rng.uniform(-args.position_jitter,args.position_jitter) for _ in selected])
        overlap_pair = None
        if args.occlusion:
            front_indices = [j for j,item in enumerate(selected) if item is not back_item]
            overlap_pair = next(((a,b) for a,b in zip(front_indices,front_indices[1:]) if b == a+1),None)
            if overlap_pair is None:
                raise ValueError('no adjacent front pair for occlusion')
            widths = []
            for j in overlap_pair:
                art = bpy.data.images.load(str(ROOT/selected[j]['image']),check_existing=True)
                widths.append(.15 * art.size[0] / art.size[1])
            separation = sum(widths)/2 - .40*min(widths)
            positions[overlap_pair[1]] = positions[overlap_pair[0]] + separation
        objects = []
        for j, item in enumerate(selected):
            # One distinct community artwork is presented as a white back.
            front = item is not back_item
            depth = rng.uniform(-.01,.08) if args.position_jitter else rng.uniform(.01,.04)
            if overlap_pair is not None and j in overlap_pair:
                depth = .01 if j == overlap_pair[0] else .05
            objects.append(standee(item,(float(positions[j]),depth,0),rng.uniform(-.06,.06),front))
        if args.vehicle_occlusion:
            target_person, vehicle = select_occlusion_sources(objects, vehicles, target_state, rng, index)
            vehicle_item = {'artwork':vehicle['path'],'image':vehicle['cropped_image'],'source':vehicle}
            objects.append(standee(vehicle_item,(target_person['location'][0]+rng.uniform(-.025,.025),-.025,0),0,False,
                                   rng.uniform(.035,.055),True))
        stem = f'person_arena_{index+1:05d}'
        image = out/'images'/(stem+'.png')
        mask = out/'instance_ids'/(stem+'.png')
        scene.view_settings.exposure = state['augmentation']['exposure_ev']
        try:
            boxes, palette, visible_counts, full_counts, full_paths = render_instance_masks(
                scene,objects,mask,(out/'instance_full') if (args.occlusion or args.vehicle_occlusion) else None)
            if args.vehicle_occlusion:
                _, _, without_counts, _, _ = render_instance_masks(scene,objects[:-1],out/'instance_without_vehicle'/(stem+'.png'),out/'instance_full')
        except ValueError as exc:
            if str(exc) not in ('instance has no visible mask','instance has no isolated mask'):
                raise
            if mask.exists():
                mask.replace(out/'rejected'/(stem+'_ids.png'))
            rejected.append({'name':stem,'reason':str(exc)})
            for item in objects:
                bpy.data.objects.remove(bpy.data.objects[item['geometry']],do_unlink=True)
                if item['base']: bpy.data.objects.remove(bpy.data.objects[item['base']],do_unlink=True)
            print('rejected',index+1,str(exc),flush=True)
            raise RuntimeError(f"state {state['id']} did not produce the planned image: {exc}") from exc
        rows = []
        rejection = None
        for j, (item, box, color) in enumerate(zip(objects,boxes,palette)):
            a,b,c,d = box
            if a <= 0 or b <= 0 or c >= 640 or d >= 480 or (item['class'] != 'vehicle_occluder' and d-b < 100):
                rejection = 'instance truncated or too small'
            item['visible_bbox_xyxy'] = box
            item['instance_id_rgb'] = color
            if args.occlusion or args.vehicle_occlusion:
                item['visible_mask_pixels'] = visible_counts[j]
                item['isolated_mask_pixels'] = full_counts[j]
                item['visible_fraction'] = min(1.0,visible_counts[j]/full_counts[j])
                item['isolated_mask'] = full_paths[j]
                if item['front'] and item['visible_fraction'] < .80:
                    rejection = 'front visibility below 80 percent'
                if args.vehicle_occlusion and j < len(objects)-1:
                    item['vehicle_occluded_fraction'] = max(0, (without_counts[j]-visible_counts[j])/full_counts[j])
            if item['front']:
                cls = {'community':0,'noncommunity':1}[item['class']]
                rows.append(f'{cls} {(a+c)/1280:.8f} {(b+d)/960:.8f} {(c-a)/640:.8f} {(d-b)/480:.8f}')
        if args.vehicle_occlusion:
            losses = ([target_person['vehicle_occluded_fraction']] if target_state.get('occluded_person_class') else
                      [o['vehicle_occluded_fraction'] for o in objects if o['front']])
            if not any(.05 <= loss <= .20 for loss in losses):
                rejection = 'no front person with 5–20 percent vehicle occlusion'
        elif args.occlusion:
            losses = [1-o['visible_fraction'] for o in objects if o['front']]
            if not any(.05 <= loss <= .20 for loss in losses):
                rejection = 'no front person with 5–20 percent visible-mask occlusion'
        else:
            for i, left in enumerate(boxes):
                for right in boxes[i+1:]:
                    if min(left[2],right[2]) > max(left[0],right[0]) and min(left[3],right[3]) > max(left[1],right[1]):
                        rejection = 'overlap requires separate visibility review'
        if rejection:
            mask.replace(out/'rejected'/(stem+'_ids.png'))
            rejected.append({'name':stem,'reason':rejection})
            for item in objects:
                bpy.data.objects.remove(bpy.data.objects[item['geometry']],do_unlink=True)
                if item['base']: bpy.data.objects.remove(bpy.data.objects[item['base']],do_unlink=True)
            print('rejected',index+1,rejection,flush=True)
            raise RuntimeError(f"state {state['id']} did not produce the planned image: {rejection}")
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        label = out/'labels'/(stem+'.txt')
        label.write_text('\n'.join(rows)+'\n',encoding='utf8')
        record = {'image':image.relative_to(ROOT).as_posix(),'label':label.relative_to(ROOT).as_posix(),
                        'instance_id_image':mask.relative_to(ROOT).as_posix(),'split':'train_candidate',
                        'objects':objects,'camera_location':list(cam.location),'camera_rotation_euler':list(cam.rotation_euler),
                        'camera_lens_mm':cam.data.lens,'occlusion':args.occlusion,'vehicle_occlusion':args.vehicle_occlusion,
                        'exposure_ev':scene.view_settings.exposure,'semantic_review':'pending',
                        'seed':args.seed+index*1009,'state_id':state['id']}
        run.record_sample(state['id'], output=image, sources=state['sources'],
                          parameters=state['parameters'], actual=record)
        records.append(record)
        for item in objects:
            bpy.data.objects.remove(bpy.data.objects[item['geometry']],do_unlink=True)
            if item['base']: bpy.data.objects.remove(bpy.data.objects[item['base']],do_unlink=True)
        print('completed',index+1,flush=True)
    data = {'status':'train_candidates_semantic_review_pending','source_scene':source_scene.relative_to(ROOT).as_posix(),
            'working_scene':working_scene.relative_to(ROOT).as_posix(),
            'source_scene_reference':('datasets/annotated/person/v4_user_master_production_plan_20261113.json' if args.scene == 'user-master' else 'Archive/technical/revisions/arena/blender_sources_20261116/arena_4m2_pbr_20261112/detail_provenance.json'),
            'texture_manifest':(TEXTURES/'manifest.json').relative_to(ROOT).as_posix(),
            'image_size':[640,480],'split':'train_candidate','seed':args.seed,'count':len(records),
            'camera_jitter_m':args.camera_jitter,
            'position_jitter_m':args.position_jitter,
            'render_device':args.render_device,
            'plan':plan,'source_refs':run.source_refs,
            'camera_preset':'person-near' if use_near_camera else 'master',
            'master_camera':master_camera,
            'requested_count':len(scheduled),'rejected':rejected,
            'camera':{'location':list(cam.location),'initial_location':list(camera_origin),'initial_rotation_euler':list(camera_rotation),
                      'target':list(target) if use_near_camera else None,'lens_mm':cam.data.lens,
                      'sensor_fit':cam.data.sensor_fit,'sensor_width_mm':cam.data.sensor_width,
                      'sensor_height_mm':cam.data.sensor_height,'status':'visual_fit_not_calibrated'},
            'production_receipt_operation':'person.render_arena_v4:' + out.relative_to(ROOT).as_posix(),
            'records':records,'scope':'Flat silhouette and white rim from supplied digital artwork; rear white and acrylic-like bases are approximations. Arena, camera, lighting and standee sizes are fitted. Labels derive from rendered color IDs and require visual review.'}
    (out/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(out.relative_to(ROOT)),'images':len(records)}))


if __name__ == '__main__':
    main()
