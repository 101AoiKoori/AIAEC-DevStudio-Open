"""Build a separate, freely parameterized color/shape contrast scene and preview.

Run in background Blender with the shared mother open. This entry never saves
the shared mother and never changes the current 3600-image export collection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.fire.fire_blender_labels import visible_bbox
from datasets.scripts.fire.render_blender_fire_dataset import material_override, render_id_pass
from lab.training_common.production_guard import generation_guard

CONFIG = ROOT / 'datasets/scripts/fire/config/hard_negative_preview.json'
COLORS = {'red': (.8, .012, .005, 1), 'green': (.015, .45, .025, 1),
          'blue': (.012, .045, .55, 1), 'orange': (1, .17, .008, 1),
          'gray': (.22, .24, .25, 1)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def material(name, color, roughness=.5, emission=0):
    m = bpy.data.materials.new('HN | ' + name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    p = m.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    output = m.node_tree.nodes.new('ShaderNodeOutputMaterial')
    m.node_tree.links.new(p.outputs['BSDF'], output.inputs['Surface'])
    p.inputs['Base Color'].default_value = color
    p.inputs['Roughness'].default_value = roughness
    p.inputs['Emission Color'].default_value = color
    p.inputs['Emission Strength'].default_value = emission
    return m


def box(name, location, dimensions, mat, bevel=.001):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    o = bpy.context.object
    o.name = 'HN | ' + name
    o.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if bevel:
        mod = o.modifiers.new('Rounded edges', 'BEVEL'); mod.width = bevel; mod.segments = 3
    return o


def clone_objects(objects, collection, transform, prefix):
    clones = []
    # Freeze world matrices before dropping source parents and animation.
    matrices = [(o, o.matrix_world.copy()) for o in objects]
    for original, world in matrices:
        o = original.copy()
        if original.data is not None:
            o.data = original.data.copy()
        o.name = 'HN | ' + prefix + ' | ' + original.name
        o.animation_data_clear(); o.constraints.clear(); o.parent = None
        collection.objects.link(o); o.matrix_world = transform @ world
        o.hide_render = False; o.hide_viewport = False
        o['source_object'] = original.name
        clones.append(o)
    return clones


def bounds(objects):
    points = [o.matrix_world @ v.co for o in objects for v in o.data.vertices]
    return Vector(tuple(min(p[i] for p in points) for i in range(3))), Vector(tuple(max(p[i] for p in points) for i in range(3)))


def freeze_objects(objects):
    """Evaluate in the owning scene before detaching animated asset parents.

    Curve bound_box values can include default/control-point extents; measuring
    their evaluated mesh is necessary for correct physical size and contact.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    frozen = []
    for obj in objects:
        evaluated = obj.evaluated_get(deps)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=deps)
        clone = bpy.data.objects.new('HN captured | ' + obj.name, mesh)
        clone.matrix_world = evaluated.matrix_world.copy()
        frozen.append(clone)
    return frozen


def build_scene(cfg):
    """Copy reusable assets; no mutation of the source objects or materials."""
    if bpy.data.scenes.get(cfg['scene_name']):
        raise ValueError('Preview scene already exists; use a fresh background mother')
    bpy.context.window.scene = bpy.data.scenes['Fire Buildings']
    bpy.context.view_layer.update()
    source_objects = list(bpy.data.collections[cfg['building_collection']].all_objects)
    originals = freeze_objects([o for o in source_objects if o.type in {'MESH', 'CURVE'}])
    bpy.context.window.scene = bpy.data.scenes[cfg['bin_scene']]
    bpy.context.view_layer.update()
    bins = freeze_objects([o for o in bpy.context.scene.objects
        if '| red ' in o.name and o.type in {'MESH', 'CURVE', 'FONT'}])
    scene = bpy.data.scenes.new(cfg['scene_name']); bpy.context.window.scene = scene
    scene.unit_settings.system = 'METRIC'
    scene.render.engine = 'CYCLES'; scene.cycles.samples = cfg['samples']
    scene.cycles.use_denoising = True; scene.cycles.max_bounces = 6
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
    if not any(d.use for d in prefs.devices):
        raise RuntimeError('No OPTIX device available for this preview')
    scene.cycles.device = 'GPU'
    scene.render.resolution_x, scene.render.resolution_y = cfg['image_size']
    scene.render.resolution_percentage = 100; scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'; scene.view_settings.view_transform = 'AgX'
    scene.render.film_transparent = False; scene.render.use_compositing = False
    scene.world = bpy.data.worlds.new('HN | neutral indoor world'); scene.world.use_nodes = True
    scene.world.node_tree.nodes.clear()
    bg = scene.world.node_tree.nodes.new('ShaderNodeBackground')
    world_output = scene.world.node_tree.nodes.new('ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(bg.outputs[0], world_output.inputs['Surface'])
    bg.inputs['Color'].default_value = (.3, .32, .35, 1); bg.inputs['Strength'].default_value = .45
    collection = scene.collection
    geometry = [o for o in originals if o.type in {'MESH', 'CURVE', 'FONT'} and 'paper' not in o.name.lower()]
    low, high = bounds(geometry)
    offset = Vector((-(low.x + high.x) / 2, -(low.y + high.y) / 2, -low.z))
    building = clone_objects(geometry, collection, Matrix.Translation(offset), 'building A')
    template = next(o for o in originals if 'Fire paper' in o.name)
    paper = clone_objects([template], collection, Matrix.Translation(offset), 'designated sticker')[0]
    paper.data.materials.clear()
    for m in template.data.materials:
        paper.data.materials.append(m.copy())
    original_vertices = [v.co.copy() for v in paper.data.vertices]
    pmin, pmax = bounds([paper]); pcenter = (pmin + pmax) / 2
    paper_original_height = pmax.z - pmin.z
    # Reuse the complete red bin, including lid/hinges/marking, at model-scene scale.
    low, high = bounds(bins); scale = cfg['variants'][1]['height_m'] / (high.z - low.z)
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(Vector((-(low.x+high.x)/2, -(low.y+high.y)/2, -low.z)))
    bin_parts = clone_objects(bins, collection, transform, 'bin')
    for o in bin_parts:
        for slot in o.material_slots:
            if slot.material:
                slot.material = slot.material.copy()
    floor = material('floor', (.19, .20, .22, 1), .82)
    box('floor', (0, 0, -.006), (8, 8, .01), floor)
    box('backdrop', (0, .7, 1), (8, .02, 2), material('wall', (.48, .49, .5, 1), .9))
    black = material('signal housing', (.009, .012, .014, 1), .4)
    signal = [box('signal base', (0, 0, .008), (.08, .07, .016), black),
              box('signal pole', (0, 0, .07), (.008, .009, .14), black),
              box('signal housing', (0, 0, .18), (.055, .03, .08), black)]
    lamps = []
    for z in (.16, .20):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1, location=(0, -.018, z))
        o = bpy.context.object; o.name = 'HN | signal lens'; o.scale = (.019, .005, .019)
        o.data.materials.append(material('signal lens', COLORS['red'], .26))
        lamps.append(o)
    signal += lamps
    cone_mat = material('cone colored body', COLORS['orange'], .5)
    cone = [box('cone base', (0, 0, .004), (.09, .09, .008), black)]
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=.034, radius2=.006, depth=.092, location=(0, 0, .054))
    o = bpy.context.object; o.name = 'HN | cone body'; o.data.materials.append(cone_mat); cone.append(o)
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=.023, radius2=.018, depth=.016, location=(0, 0, .064))
    o = bpy.context.object; o.name = 'HN | cone reflective band'
    o.data.materials.append(material('reflective band', (.8, .8, .75, 1), .3)); cone.append(o)
    cam = bpy.data.objects.new('HN | contrast camera', bpy.data.cameras.new('HN | camera'))
    collection.objects.link(cam); scene.camera = cam; cam.data.lens = cfg['camera']['lens_mm']
    data = bpy.data.lights.new('HN | broad indoor key', 'AREA'); data.energy = 18; data.shape = 'DISK'; data.size = .9
    light = bpy.data.objects.new(data.name, data); collection.objects.link(light); light.location = (-.4, -.5, 1.0)
    light.rotation_euler = (Vector((0, 0, .2)) - light.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()
    return {'scene': scene, 'building': building, 'paper': paper, 'paper_vertices': original_vertices,
            'paper_center': pcenter, 'paper_height': paper_original_height,
            'props': {'signal_lamp': signal, 'bin': bin_parts, 'cone': cone},
            'lamp_objects': lamps, 'cone_material': cone_mat, 'camera': cam, 'light': light,
            'prop_matrices': {o.name: o.matrix_world.copy() for o in signal + bin_parts + cone}}


def set_variant(rig, cfg, kind, color, index):
    rng = random.Random(cfg['seed'] + index)
    scene = rig['scene']; prop = rig['props'][kind]
    placement = cfg['prop_placement']
    shift = Matrix.Translation((rng.uniform(*placement['x_m']), rng.uniform(*placement['y_m']), 0))
    rotation = Matrix.Rotation(math.radians(rng.uniform(*placement['yaw_deg'])), 4, 'Z')
    # Always scale from the saved base pose, not a prior variant's scaled pose.
    base_points = [rig['prop_matrices'][o.name] @ v.co for o in prop for v in o.data.vertices]
    base_height = max(p.z for p in base_points)-min(p.z for p in base_points)
    factor = cfg['variants'][index]['height_m'] * rng.uniform(*placement['height_factor']) / base_height
    transform = shift @ rotation @ Matrix.Scale(factor, 4)
    for key, objects in rig['props'].items():
        for obj in objects:
            obj.hide_render = key != kind
            if key == kind:
                obj.matrix_world = transform @ rig['prop_matrices'][obj.name]
    appearance = cfg['appearance']; value = rng.uniform(*appearance['color_value'])
    rgba = tuple(v * value for v in COLORS[color][:3]) + (1,)
    roughness = rng.uniform(*appearance['roughness'])
    if kind == 'signal_lamp':
        for i, lamp in enumerate(rig['lamp_objects']):
            p = next(n for n in lamp.data.materials[0].node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            # A plausible two-aspect signal: red above, green below; only one lit.
            lit = (color == 'red' and i == 1) or (color == 'green' and i == 0)
            base = rgba if lit else (.018, .02, .018, 1)
            p.inputs['Base Color'].default_value = base
            p.inputs['Emission Color'].default_value = base
            p.inputs['Emission Strength'].default_value = rng.uniform(2, 6) if lit else 0
    elif kind == 'cone':
        p = next(n for n in rig['cone_material'].node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        p.inputs['Base Color'].default_value = rgba; p.inputs['Roughness'].default_value = roughness
    else:
        for obj in prop:
            if any(t in obj.name for t in ['marking', 'hinge', 'support']):
                continue
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    p = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                    if p:
                        p.inputs['Base Color'].default_value = rgba; p.inputs['Roughness'].default_value = roughness
    az = math.radians(rng.uniform(*cfg['camera']['azimuth_deg'])); distance = rng.uniform(*cfg['camera']['distance_m'])
    cam = rig['camera']; cam.location = (.06 + math.sin(az) * distance, -math.cos(az) * distance, rng.uniform(*cfg['camera']['height_m']))
    cam.rotation_euler = (Vector((.06, 0, .2)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.view_settings.exposure = rng.uniform(*appearance['exposure_ev'])
    rig['light'].data.energy = rng.uniform(*appearance['key_energy_w'])
    paper = rig['paper']; size = .4 * rng.uniform(*cfg['sticker_height_fraction'])
    center = Vector((rng.uniform(*cfg['sticker_anchor']['x_m']), -.0632, rng.choice(cfg['sticker_anchor']['height_m'])))
    # Flatten the source paper onto the actual A front plane, retaining its UVs.
    inverse = paper.matrix_world.inverted(); source_center = rig['paper_center']
    for v, original in zip(paper.data.vertices, rig['paper_vertices']):
        world = paper.matrix_world @ original
        target = center + Vector(((world.x-source_center.x)*size/rig['paper_height'], 0, (world.z-source_center.z)*size/rig['paper_height']))
        v.co = inverse @ target
    paper.data.update(); bpy.context.view_layer.update()
    ground_min, ground_max = bounds(prop)
    plane_error = max(abs((paper.matrix_world @ v.co).y+.0632) for v in paper.data.vertices)
    if abs(ground_min.z) > .0001 or plane_error > .000001:
        raise ValueError('Prop contact or sticker wall plane is inconsistent')
    return {'camera_position_m': list(cam.location), 'camera_lens_mm': cam.data.lens,
            'exposure_ev': scene.view_settings.exposure, 'key_energy_w': rig['light'].data.energy,
            'prop_transform': [list(row) for row in transform], 'color': color,
            'prop_height_m': ground_max.z-ground_min.z, 'ground_error_m': abs(ground_min.z),
            'sticker_wall_plane_error_m': plane_error,
            'color_value': value, 'roughness': roughness, 'sticker_height_m': size}


def preview(cfg, replace=False):
    output = ROOT / cfg['output']; mother = ROOT / cfg['source_scene']
    if Path(bpy.data.filepath).resolve() != mother.resolve() or not bpy.app.background:
        raise ValueError('Run in background Blender with the configured mother open')
    existing = {p for p in output.rglob('*') if p.is_file()} if output.exists() else set()
    if existing:
        if not replace or not (output/'manifest.json').is_file():
            raise ValueError('Preview exists; use --replace after inspecting its manifest')
        previous = json.loads((output/'manifest.json').read_text(encoding='utf8'))
        generated = {output/'manifest.json', output/'contact_sheet.jpg'}
        for row in previous['records']:
            generated.update(output/row[key] for key in ('image', 'label', 'instance_pass'))
            generated.add(output/'overlays'/Path(row['image']).name)
        if existing-generated or any(not p.resolve().is_relative_to(output.resolve()) for p in generated):
            raise ValueError('Preview contains unowned or external files; preserve them')
    digest = hashlib.sha256(mother.read_bytes()).hexdigest()
    states = []
    for index, variant in enumerate(cfg['variants']):
        for color in variant['colors']:
            for count in cfg['target_counts']:
                params = {'kind': variant['kind'], 'color': color, 'count': count, 'seed': cfg['seed']+index}
                states.append({'id': f'{variant["kind"]}_{color}_{count}', 'failure_event': 'color_shortcut',
                    'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                    'exposure_weight': 1, 'failure_cost': 1, 'conditions': ['color_shape_contrast'],
                    'background': {'scene': cfg['scene_name']}, 'target': {'count': count, 'building': 'A'},
                    'labels': {'class_id': 0, 'scope': 'entire_output'}, 'augmentation': cfg['appearance'],
                    'sources': [cfg['source_scene']], 'parameters': params})
    plan = {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
        'purpose': cfg['purpose'], 'budget': len(states), 'uncertainty_weight': .5,
        'independence_basis': cfg['split_basis'], 'constraints': {'scope': 'development_preview'}, 'strata': states}
    snapshot = ROOT / cfg['scene_snapshot']
    from datasets.scripts.common.config import load_task_config
    _, source = load_task_config('fire')
    artwork = source['target_artwork']
    with generation_guard('fire.hard_negative_preview', [mother, ROOT/artwork], [output, snapshot],
            {'config': cfg}, procedural_inputs=[CONFIG, CONFIG.with_name('source.yaml'), CONFIG.with_name('dataset.yaml')], code_paths=[Path(__file__),
            ROOT/'datasets/scripts/fire/render_blender_fire_dataset.py', ROOT/'datasets/scripts/fire/fire_blender_labels.py'],
            production_plan=plan, sample_output_paths=[output/'images'],
            auxiliary_output_paths=[output/'instance_pass', output/'overlays'],
            replace_outputs=True, replace_verifier=lambda workspace, state, receipt: {'preserved_paths': []}) as run:
        if run.reused:
            return
        for path in existing:
            path.unlink()
        rig = build_scene(cfg); scene = rig['scene']; mask = material_override(); records = []
        for sub in ('images', 'labels', 'instance_pass', 'overlays'):
            (output/sub).mkdir(parents=True, exist_ok=True)
        for state in states:
            p = state['parameters']; kind = p['kind']; index = next(i for i,v in enumerate(cfg['variants']) if v['kind']==kind)
            actual = set_variant(rig, cfg, kind, p['color'], index)
            rig['paper'].hide_render = not p['count']; bpy.context.view_layer.update()
            image_path = output/'images'/(state['id']+'.png'); label_path = output/'labels'/(state['id']+'.txt')
            scene.render.filepath = str(image_path); bpy.ops.render.render(write_still=True)
            targets = [{'object': rig['paper'], 'id': 1}] if p['count'] else []
            distractors = [{'object': obj, 'id': i+2} for i,obj in enumerate(rig['props'][kind])]
            ids = render_id_pass(scene, targets+distractors, output/'instance_pass'/(state['id']+'.png'), mask)
            instances = []
            if p['count']:
                bbox, area = visible_bbox(ids==1)
                if bbox is None or area < 50:
                    raise ValueError('Preview target is not sufficiently visible')
                instances.append({'id': 1, 'xyxy': bbox, 'visible_pixels': area, 'building': 'A'})
            distractor_box, area = visible_bbox(ids>=2)
            if distractor_box is None or area < 100:
                raise ValueError('Preview distractor is not visible')
            label_path.write_text(''.join(yolo_row(i['xyxy'], *cfg['image_size'])+'\n' for i in instances), encoding='utf8')
            row = {'image': image_path.relative_to(output).as_posix(), 'label': label_path.relative_to(output).as_posix(),
                'instance_pass': f'instance_pass/{state["id"]}.png', 'branch': 'blender_render', 'split': 'train',
                'source_group': cfg['source_scene'], 'parents': [cfg['source_scene']], 'source_blend_sha256': digest,
                'scene_snapshot': cfg['scene_snapshot'], 'pair_id': f'{kind}_{p["color"]}', 'color_control_group': kind,
                'distractor': {'kind': kind, 'color': p['color'], 'xyxy': distractor_box, 'visible_pixels': area,
                    'object_ids': [t['id'] for t in distractors],
                    'same_color_hard_negative': p['color'] in ('red', 'orange'),
                    'hard_negative_type': {'signal_lamp':'red_signal_lamp', 'bin':'red_orange_bin',
                        'cone':'other_red_orange_object'}[kind] if p['color'] in ('red', 'orange') else None},
                'instances': instances, 'building_counts': {'A': {'visible_count': len(instances), 'scope': 'visible_output'}},
                'actual': actual, 'development_contact': True, 'status': 'candidate_requires_visual_review'}
            records.append(row)
            run.record_sample(state['id'], output=image_path, sources=state['sources'], parameters=p, actual=row)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        bpy.data.libraries.write(str(snapshot), {scene}, path_remap='ABSOLUTE', fake_user=True)
        write_json(output/'manifest.json', {'task': 'fire', 'status': 'development_preview', 'plan': plan,
            'source_blend_sha256': digest, 'images': len(records), 'records': records})
        run.set_result({'images': len(records), 'positive': sum(bool(r['instances']) for r in records), 'scope': 'preview_only'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace', action='store_true', help='Replace only manifest-owned preview outputs')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    preview(json.loads(CONFIG.read_text(encoding='utf8')), args.replace)
