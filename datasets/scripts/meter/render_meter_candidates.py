"""Render one-source meter images with eight *candidate* projected landmarks.

Run in Blender with --background SOURCE.blend --python-exit-code 1 --python THIS --
--output NEW_DIRECTORY --endpoints CONFIG [--count 10]. The optional room pilot
saves a separate editable scene in its output; the source scene is unchanged.
All frames share a source group; this program does not create train/val/test splits.
Visibility, physical-meter agreement and scale readings are not established.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK_HELPERS = ROOT / 'datasets/scripts/meter'
sys.path.insert(0, str(ROOT))
import bpy
from mathutils import Matrix, Vector
from datasets.scripts.meter.render_meter_batch import KEYPOINTS, look_at, project, texture_xy
from datasets.scripts.meter.meter_station_plan import allocated_station_states, validate_station_declaration
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import sample_dir, split_dir

STATION_SOURCE = source_path('meter', 'station_source')
STATION_PIVOT = Vector((0, -1.04, .02))


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def station_tone_compositor(scene):
    """Bounded simulated saturation; independent from landmark visibility."""
    tree = bpy.data.node_groups.new('V4 station tone compositor', 'CompositorNodeTree')
    tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    tone = tree.nodes.new('CompositorNodeHueSat')
    output = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layers.outputs['Image'], tone.inputs['Image'])
    tree.links.new(tone.outputs['Image'], output.inputs['Image'])
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    return tone


def landmark_ray_evidence(scene, camera, markers):
    """Report geometry in front of each point; glass is not a visibility proof."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation.copy()
    evidence = {}
    for name, marker in zip(KEYPOINTS, markers):
        point = marker.matrix_world.translation.copy()
        projected = project(scene, camera, point)
        in_frame = projected[2] > 0 and 0 <= projected[0] <= 1 and 0 <= projected[1] <= 1
        ray = point - origin
        if not in_frame or ray.length <= 1e-5:
            evidence[name] = {'in_frame': False, 'first_hit': None, 'ray_status': 'outside_frame_or_behind'}
            continue
        hit, location, normal, face, obj, matrix = scene.ray_cast(
            depsgraph, origin, ray.normalized(), distance=max(0.0, ray.length - 1e-4))
        evidence[name] = {'in_frame': True, 'first_hit': obj.name if hit and obj else None,
                          'first_hit_distance': float((location - origin).length) if hit else None,
                          'ray_status': 'geometry_intersection' if hit else 'no_intersection_before_point',
                          'visibility_conclusion': 'unknown; geometry does not measure pixel readability or reflection'}
    return evidence


def station_pose_rays(scene, camera, markers):
    """Trace the known cover and target surfaces; any other hit is an obstruction."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation.copy()
    results = {}
    for name, marker in zip(KEYPOINTS, markers):
        point = marker.matrix_world.translation.copy()
        xy = project(scene, camera, point)
        if xy[2] <= 0 or not (0 <= xy[0] < 1 and 0 <= xy[1] < 1):
            results[name] = {'in_frame': False, 'hits': [], 'blocked': False}
            continue
        delta = point - origin
        direction = delta.normalized()
        start = origin.copy()
        hits, blocked, target_hit = [], False, False
        family = 'temperature' if name.startswith('temperature') else 'humidity'
        own = set()
        if name.endswith('pivot'):
            own = {'Station ' + family + '_pivot_cap', 'Station ' + family + '_needle_rotate_Z'}
        elif name.endswith('tip'):
            own = {'Station ' + family + '_needle_rotate_Z'}
        for _ in range(32):
            remaining = delta.length - (start - origin).length - 1e-4
            if remaining <= 0:
                break
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, start, direction, distance=remaining)
            if not hit:
                break
            material_names = [m.name for m in obj.data.materials if m] if hasattr(obj.data, 'materials') else []
            cover = obj.data.materials[0] if material_names == ['PBR clear thin meter cover'] else None
            transparent = (obj.name == 'Station glass_reflection_surface' and cover is not None
                           and cover.use_nodes and any(n.type == 'BSDF_TRANSPARENT' for n in cover.node_tree.nodes)
                           and any(link.from_node.type == 'BSDF_TRANSPARENT' for link in cover.node_tree.links))
            target = obj.name in own
            hits.append({'object': obj.name, 'distance': float((location-origin).length),
                         'role': 'transparent_cover' if transparent else ('target' if target else 'obstruction')})
            target_hit |= target
            if not (transparent or target):
                blocked = True
                break
            start = location + direction * .001
        else:
            blocked = True  # A bounded trace that cannot finish is never evidence of visibility.
        results[name] = {'in_frame': True, 'hits': hits, 'blocked': blocked, 'target_hit': target_hit}
    return results


def station_pose_labels(scene, camera, markers, points, rays, image_path):
    """Assign geometric visibility, using image contrast only for recognizable pivot caps."""
    import numpy as np
    image = bpy.data.images.load(str(image_path), check_existing=False)
    try:
        w, h = image.size
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
        gray = pixels @ np.array([.2126, .7152, .0722], dtype=np.float32)
        result = {}
        for name, marker, point in zip(KEYPOINTS, markers, points):
            ray = rays[name]
            if not ray['in_frame']:
                result[name] = {'visibility': 0, 'reason': 'outside_frame_or_behind'}
                continue
            visibility, reason = 1, 'geometric_position_weak_or_occluded'
            detail = {}
            cap = 'Station ' + ('temperature' if name.startswith('temperature') else 'humidity') + '_pivot_cap'
            cap_hit = any(hit['object'] == cap and hit['role'] == 'target' for hit in ray['hits'])
            if name.endswith('pivot') and not ray['blocked'] and cap_hit:
                radius = .036 if name.startswith('temperature') else .026
                world_radius = marker.parent.matrix_world.to_3x3() @ Vector((radius, 0, 0))
                edge = project(scene, camera, marker.matrix_world.translation + world_radius)
                diameter = 2 * math.hypot((edge[0] - point[0]) * w, (edge[1] - point[1]) * h)
                x, y = int(point[0] * w), int(point[1] * h)
                patch = gray[max(0, y-3):min(h, y+4), max(0, x-3):min(w, x+4)]
                center = gray[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
                contrast = float(np.percentile(patch, 90) - np.median(center))
                detail = {'cap_diameter_px': diameter, 'local_contrast_linear': contrast}
                if diameter >= 3.0 and contrast >= .08:
                    visibility, reason = 2, 'unblocked_target_cap_and_local_contrast'
            if ray['blocked']:
                reason = 'geometry_obstructed'
            result[name] = {'visibility': visibility, 'reason': reason, **detail}
        return result
    finally:
        bpy.data.images.remove(image)


def station_dial_masks(scene, pivot, output, name):
    """Rasterize visible and isolated dial surfaces without changing the source scene.

    The workbench pass treats glass as opaque and uses simplified flat lighting;
    it is geometric visibility evidence, not proof of photoreal contrast.
    """
    import numpy as np

    objects = list(scene.objects)
    target = {o for o in objects if o.parent == pivot and o.type in {'MESH', 'CURVE'}}
    if len(target) < 5:
        raise ValueError('station dial geometry is incomplete')
    old_colors = {o: tuple(o.color) for o in objects}
    old_hidden = {o: o.hide_render for o in objects}
    old_engine, old_transparent = scene.render.engine, scene.render.film_transparent
    old_mode, old_compositing = scene.render.image_settings.color_mode, scene.render.use_compositing
    shade = scene.display.shading
    old_shading = (shade.light, shade.color_type, shade.show_shadows, shade.show_cavity)
    masks, pixel_masks = {}, {}
    try:
        for o in objects:
            o.color = (1, 1, 1, 1) if o in target else (0, 0, 0, 1)
        scene.render.engine = 'BLENDER_WORKBENCH'
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.use_compositing = False
        shade.light, shade.color_type = 'FLAT', 'OBJECT'
        shade.show_shadows = shade.show_cavity = False
        for kind in ('visible', 'isolated'):
            for o in objects:
                o.hide_render = old_hidden[o] or (kind == 'isolated' and o not in target and o.type != 'CAMERA')
            path = output / 'masks' / (name + '_' + kind + '.png')
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(path), check_existing=False)
            try:
                w, h = image.size
                pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(h, w, 4)
                mask = (pixels[:, :, :3].max(axis=2) > .05) & (pixels[:, :, 3] > .5)
                pixel_masks[kind] = mask
                ys, xs = np.where(mask)
                bbox = None if not len(xs) else [int(xs.min()), int(h - 1 - ys.max()),
                                                  int(xs.max()) + 1, int(h - ys.min())]
                masks[kind] = {'path': 'masks/' + path.name, 'pixels': int(len(xs)), 'bbox_xyxy_pixels': bbox}
            finally:
                bpy.data.images.remove(image)
    finally:
        for o in objects:
            o.color = old_colors[o]
            o.hide_render = old_hidden[o]
        scene.render.engine, scene.render.film_transparent = old_engine, old_transparent
        scene.render.image_settings.color_mode, scene.render.use_compositing = old_mode, old_compositing
        shade.light, shade.color_type, shade.show_shadows, shade.show_cavity = old_shading
    visible, isolated = masks['visible'], masks['isolated']
    b = isolated['bbox_xyxy_pixels']
    fraction = (int(np.logical_and(pixel_masks['visible'], pixel_masks['isolated']).sum()) /
                isolated['pixels']) if isolated['pixels'] else 0.0
    valid = bool(b and min(b[0], b[1], 640 - b[2], 480 - b[3]) >= 2 and
                 b[2] - b[0] >= 45 and b[3] - b[1] >= 40 and fraction >= .97)
    v = visible['bbox_xyxy_pixels']
    union_box = [min(b[0], v[0]), min(b[1], v[1]), max(b[2], v[2]), max(b[3], v[3])] if valid and v else None
    return {'visible': visible, 'isolated': isolated, 'visible_fraction_candidate': fraction,
            'bbox_xyxy_pixels': union_box,
            'bbox_status': 'candidate_from_isolated_raster_and_visible_fraction' if valid else 'unknown',
            'method': 'workbench flat object colors; black other geometry; isolated comparison; opaque glass',
            'limitations': 'synthetic geometry evidence only; no proof of photoreal image contrast or real-meter fidelity'}


def add_station_room_context(scene):
    """Add approximate cables and shelving informed by P18, not measured geometry."""
    def material(name, color, roughness):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = (*color, 1)
        bsdf.inputs['Roughness'].default_value = roughness
        return mat

    rubber = material('MAT-station-cable-rubber', (.025, .026, .026), .76)
    pale_cable = material('MAT-station-cable-gray', (.14, .14, .13), .72)
    steel = material('MAT-station-shelving', (.17, .18, .17), .8)
    backing = material('MAT-station-shelf-backing', (.095, .098, .09), .9)

    def cable(name, points, radius, mat):
        curve = bpy.data.curves.new(name + '-curve', 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 8
        curve.bevel_depth = radius
        curve.bevel_resolution = 3
        spline = curve.splines.new('BEZIER')
        spline.bezier_points.add(len(points) - 1)
        for point, position in zip(spline.bezier_points, points):
            point.co = position
            point.handle_left_type = point.handle_right_type = 'AUTO'
        obj = bpy.data.objects.new(name, curve)
        scene.collection.objects.link(obj)
        obj.data.materials.append(mat)

    # The scene's Y axis is vertical, and the floor lies at approximately -5.9.
    # These are visual stand-ins for the P18 cables, not reconstructed wiring.
    cable('GEO-station-gray-foreground-cable',
          [(-5.4, -1.4, 5), (-5.4, -3.0, 0), (-6.0, -4.8, -5),
           (-4.5, -5.80, -5.7), (-2.8, -5.82, -5.6)], .055, pale_cable)
    cable('GEO-station-floor-cable-loop',
          [(-5.2, -5.79, -5.7), (-4.0, -5.78, -4.9), (-2.5, -5.78, -5.0),
           (-1.9, -5.78, -5.7), (-3.0, -5.78, -5.8), (-4.7, -5.79, -5.4),
           (-5.9, -5.79, -4.5)], .035, rubber)
    cable('GEO-station-rising-cable',
          [(-3.0, -5.78, -5.4), (-3.3, -5.0, -4.5), (-3.6, -4.1, -3.4),
           (-3.5, -3.2, -2.0), (-2.8, -1.9, -.8), (-.9, -3.3, .3)], .035, rubber)

    def cube(name, center, size, mat):
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = size
        obj.data.materials.append(mat)
        return obj

    cube('GEO-station-shelf-back', (-4.15, -.35, -9.18), (3.6, 3.6, .12), backing)
    for x in (-5.9, -4.15, -2.4):
        cube('GEO-station-shelf-upright', (x, -.35, -9.08), (.075, 3.65, .18), steel)
    for y in (-2.05, -1.2, -.35, .5, 1.35):
        cube('GEO-station-shelf-level', (-4.15, y, -9.08), (3.6, .07, .18), steel)
    return {'reference': 'datasets/raw/car_real_capture_20261112/P18_GAUGE.png',
            'method': 'manually fitted 3D cables and shelving; no photo composited',
            'cables': ['GEO-station-gray-foreground-cable', 'GEO-station-floor-cable-loop',
                       'GEO-station-rising-cable'], 'shelf': 'GEO-station-shelf-back',
            'measurements': 'visual approximations; geometry and lighting not physically measured'}


def render_station_v4(args, out, config, endpoint_data, run=None):
    """Produce a small, unsplit pair with projections and unknown visibility."""
    station_source = Path(args.station_source).resolve() if args.station_source else STATION_SOURCE.resolve()
    station_roots = ((ROOT / 'datasets/annotated/meter').resolve(),
                     (ROOT / 'datasets/assets/blender/meter').resolve())
    if not any(station_source.is_relative_to(root) for root in station_roots) or not station_source.is_file():
        raise ValueError('station-v4 source must be an existing blend under meter assets or annotations')
    if not bpy.data.filepath or Path(bpy.data.filepath).resolve() != station_source:
        raise ValueError('station-v4 requires the specified 65-degree station blend')
    maximum_count = 10000 if args.station_batch_v4 else 8
    if not 1 <= args.count <= maximum_count or not 1 <= args.samples <= 32:
        raise ValueError(f'station-v4 requires 1..{maximum_count} images and 1..32 samples')
    scene = bpy.data.scenes['Station prototype previews']
    if bpy.context.scene != scene or scene.render.engine != 'CYCLES':
        raise ValueError('station-v4 requires the active Cycles station scene')
    pivot = scene.objects['Station dial tilt pivot']
    ground_angle = 90 + math.degrees(pivot.rotation_euler.x)
    if not 60 <= ground_angle <= 70 or pivot.rotation_euler.x >= 0:
        raise ValueError('station dial lean is outside the 20..30 degree range')
    face = scene.objects['Station PDF_face_inpainted']
    if face.parent != pivot or endpoint_data['image_size'] != [1254, 1254]:
        raise ValueError('station face or endpoint reference does not match')
    source_face = (ROOT / endpoint_data['source_face']).resolve()
    if not source_face.is_relative_to(ROOT) or not source_face.is_file():
        raise ValueError('calibration face is missing or outside the workspace')
    image_nodes = [node for material in face.data.materials if material and material.use_nodes
                   for node in material.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image]
    if (len(image_nodes) != 1 or not image_nodes[0].image.packed_file or
            bytes(image_nodes[0].image.packed_file.data) != source_face.read_bytes()):
        raise ValueError('station face differs from the calibrated generated texture')
    if args.face_override:
        derived = Path(args.face_override).resolve()
        permitted = source_path('meter', 'face_override')
        if station_source != (ROOT / 'datasets/assets/blender/meter/v4_parametric_station_v2_20261113/station.blend').resolve() or derived != permitted:
            raise ValueError('face override must be the traced v4 thick-tick face for the parametric station')
        image = bpy.data.images.load(str(derived), check_existing=False)
        if tuple(image.size) != (1254, 1254):
            raise ValueError('derived face dimensions differ from the calibration')
        image_nodes[0].image = image
    for loop in face.data.loops:
        vertex = face.data.vertices[loop.vertex_index].co
        uv = face.data.uv_layers.active.data[loop.index].uv
        if max(abs(vertex.x - (2 * uv.x - 1)), abs(vertex.y - (2 * uv.y - 1)), abs(vertex.z - .065)) > 1e-5:
            raise ValueError('station face geometry or UV differs from the calibrated source')
    endpoints = endpoint_data['endpoints']
    needle_sweeps = endpoint_data.get('needle_sweep_deg', {
        'temperature': (-147, -241), 'humidity': (44, -264)})
    for family in ('temperature', 'humidity'):
        start, sweep = needle_sweeps[family]
        if not math.isfinite(start) or not math.isfinite(sweep) or not -360 < sweep < 0:
            raise ValueError(f'Invalid clockwise needle sweep for {family}')
    markers = [scene.objects['KPT-meter-' + name] for name in KEYPOINTS]
    hands = [scene.objects['Station temperature_needle_rotate_Z'], scene.objects['Station humidity_needle_rotate_Z']]
    expected_parents = [hands[0], hands[0], hands[1], hands[1]] + [pivot] * 4
    expected_local = [Vector((0, 0, 0)), Vector((.76, 0, 0)), Vector((0, 0, 0)), Vector((.31, 0, 0))]
    expected_local += [texture_xy(endpoints[name]) - STATION_PIVOT for name in KEYPOINTS[4:]]
    if any(marker.parent != parent or (marker.location - expected).length > 1e-5
           for marker, parent, expected in zip(markers, expected_parents, expected_local)):
        raise ValueError('station landmark marker parent or coordinates differ from calibration')
    roi = [scene.objects['ROI-power-meter-' + suffix] for suffix in ('TL', 'TR', 'BR', 'BL')]
    if any(point.parent != scene.objects['Power_meter_station_root'] for point in roi):
        raise ValueError('power-meter ROI markers do not belong to the station')
    camera = scene.objects['Preview vehicle front']
    scene.camera = camera
    for obj in [camera] + hands:
        obj.animation_data_clear()
    if args.station_camera_z is not None:
        if not 8 <= args.station_camera_z <= 20:
            raise ValueError('station camera Z must be between 8 and 20 scene units')
        camera.location.z = args.station_camera_z
    room_context = None
    if args.station_room_v4:
        if args.count > 2 and not args.station_batch_v4:
            raise ValueError('station-room-v4 is limited to two review samples')
        camera.location.x -= 1.45
        room_context = add_station_room_context(scene)
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.threads_mode, scene.render.threads = 'FIXED', 2
    devices_used = ['CPU']
    if args.device == 'CPU':
        scene.cycles.device = 'CPU'
    else:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        try:
            preferences.compute_device_type = args.device
            preferences.get_devices()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError('requested device is unavailable: %s' % exc) from exc
        selected = [d for d in preferences.devices if d.type == args.device]
        if not selected:
            raise ValueError('requested %s device is unavailable' % args.device)
        for device in preferences.devices:
            device.use = device in selected
        scene.cycles.device = 'GPU'
        devices_used = [d.name for d in selected]
    scene.cycles.samples = args.samples
    base_location = camera.location.copy()
    base_rotation = camera.rotation_euler.copy()
    lights = [obj for obj in scene.objects if obj.type == 'LIGHT']
    base_lights = {obj.name: {'energy': float(obj.data.energy), 'location': obj.location.copy()} for obj in lights}
    if args.station_batch_v4:
        if not args.preflight and not any('window' in obj.name.lower() for obj in lights):
            raise ValueError('planned side lighting requires a window light in the station scene')
        tone = station_tone_compositor(scene)
        distances = (7.5, 8.5, 10.0) if args.station_pose_v4 else (9.0, 10.8, 13.5)
        states = ([] if args.preflight else allocated_station_states(
            run.plan, count=args.count, source=station_source.relative_to(ROOT).as_posix(),
            allow_occlusion=args.station_occlusion_v41))
    scene.cycles.seed = args.seed

    def geometry(allow_partial=False):
        bpy.context.view_layer.update()
        assembly = pivot.matrix_world @ Matrix.Translation(-STATION_PIVOT)
        rim = [project(scene, camera, assembly @ Vector((1.04 * math.cos(i * math.tau / 128),
                         1.04 * math.sin(i * math.tau / 128), .09))) for i in range(128)]
        points = [project(scene, camera, obj.matrix_world.translation) for obj in markers]
        screen = [project(scene, camera, obj.matrix_world.translation) for obj in roi]
        all_points = rim + points + screen
        if not allow_partial and any(p[2] <= 0 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in all_points):
            raise ValueError('station projection is behind or outside the camera frame')
        bbox = [min(p[0] for p in rim), min(p[1] for p in rim),
                max(p[0] for p in rim), max(p[1] for p in rim)]
        return bbox, points, screen

    bbox, points, screen = geometry()
    if args.preflight:
        result = {'resolution': [640, 480], 'ground_angle_deg': ground_angle,
                  'bbox_xyxy': bbox, 'keypoints': dict(zip(KEYPOINTS, points)), 'display_quad': screen}
        print('STATION_V4_PREFLIGHT_OK', json.dumps(result), flush=True)
        return result
    # Reject infeasible planned cameras before creating any output directory.
    for allocated in states:
        params = allocated['parameters']
        pivot.rotation_euler.x = math.radians(params['ground_angle_deg'] - 90)
        camera.location = Vector((base_location.x + params['camera_x_offset'], base_location.y, params['camera_z']))
        camera.rotation_euler = base_rotation.copy()
        camera.rotation_euler.y += math.radians(params['camera_yaw_deg'])
        planned_box, _, planned_screen = geometry(allow_partial=True)
        if any(q[2] <= 0 or not (.02 <= q[0] <= .98 and .02 <= q[1] <= .98) for q in planned_screen):
            raise ValueError('planned camera does not keep the lower display inside its required margin')
        if (planned_box[0] < 2 / 640 or planned_box[1] < 2 / 480 or
                planned_box[2] > 638 / 640 or planned_box[3] > 478 / 480 or
                planned_box[2] - planned_box[0] < 45 / 640 or
                planned_box[3] - planned_box[1] < 40 / 480):
            raise ValueError('planned camera cannot meet the required dial box frame and size')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir()
    (out / 'annotations').mkdir()
    if args.station_dial_masks:
        (out / 'masks').mkdir()
    if args.station_pose_v4:
        (out / 'labels').mkdir()
    occluder = None
    if args.station_occlusion_v41:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=.035)
        occluder = bpy.context.object
        occluder.name = 'V41 controlled opaque occluder'
        material = bpy.data.materials.new('V41 opaque occluder material')
        material.diffuse_color = (.035, .035, .035, 1)
        material.use_nodes = True
        material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.035, .035, .035, 1)
        occluder.data.materials.append(material)
    manifest = {'version': 1, 'status': 'running', 'source_blend': station_source.relative_to(ROOT).as_posix(),
        'mother_scene': 'Station prototype previews', 'source_assets': [endpoint_data['source_face'], endpoint_data['source_blend']],
        'endpoint_config': config.relative_to(ROOT).as_posix(), 'resolution': [640, 480],
        'keypoint_order': KEYPOINTS, 'screen_roi_order': ['TL', 'TR', 'BR', 'BL'],
        'seed': args.seed, 'cycles_samples': args.samples, 'device': args.device,
        'selected_devices': devices_used, 'cpu_threads': 2,
        'camera_parameters': 'fitted prototype, not measured camera calibration', 'ground_angle_deg': ground_angle,
        'camera_z_override_scene_units': args.station_camera_z,
        'lights': [{'name': obj.name, 'energy': float(obj.data.energy), 'location': list(obj.location)}
                   for obj in scene.objects if obj.type == 'LIGHT'],
        'status_scope': 'unsplit synthetic candidates; same scene and texture; not real truth or independent test',
        'landmark_status': 'geometric projections only; point visibility, reflection and occlusion unknown',
        'face_calibration_status': endpoint_data['status'],
        'face_override': Path(args.face_override).resolve().relative_to(ROOT).as_posix() if args.face_override else None,
        'requested_count': args.count, 'completed_count': 0,
        'samples': []}
    if args.station_batch_v4:
        manifest.update({'mode': 'station_v4_state_batch', 'sample_records': 'samples.jsonl',
            'state_offset': args.state_offset, 'planned_state_combinations': len(states),
            'state_dimensions': {'temperature_bin': 5, 'humidity_bin': 5, 'ground_angle_deg': [60, 65, 70],
                'camera_z_fitted': list(distances), 'camera_x_offset': [-1.0, 0.0, 1.0],
                'exposure_ev': [-.5, 0.0], 'saturation': [.65, 1.0], 'side_light_shift': [-1, 1]},
            'label_policy': '8 geometric projections and ray evidence only; unknown visibility; no pose labels',
            'samples': None, 'state_counts': {}})
    if args.station_dial_masks:
        manifest['dial_mask_policy'] = ('visible and isolated workbench silhouettes; box only when >=97% visible, '
            'fully in frame and large enough; candidates require photoreal image review; no pose labels')
    if args.station_pose_v4:
        manifest['label_policy'] = ('synthetic geometric coordinates; v0 outside, v1 weak or geometrically '
            'occluded, v2 pivot cap only with unobstructed target ray and local image contrast; '
            'glass reflections can reduce contrast; no real-meter validation')
        manifest['landmark_status'] = 'synthetic geometry with per-point ray and conservative image evidence'
        manifest['dial_mask_policy'] = ('visible and isolated workbench silhouettes; pose labels only when '
            'box is >=97% visible, fully in frame and large enough')
        manifest['pose_label_count'] = 0
    if args.station_occlusion_v41:
        manifest['controlled_occlusion'] = ('four deterministic strata: none, temperature tip, '
            'humidity tip, temperature minimum; opaque sphere radius 0.035 scene units; '
            'ray evidence and box mask determine usable labels')
    if room_context is not None:
        manifest['room_context'] = room_context
        manifest['camera_x_adjustment_scene_units'] = -1.45
    manifest_path = out / 'manifest.json'
    save_manifest(manifest_path, manifest)
    try:
        for index in range(args.count):
            if args.station_batch_v4:
                absolute_index = index
                allocated = states[index]
                parameters = allocated['parameters']
                fraction_t, fraction_h = parameters['temperature_fraction'], parameters['humidity_fraction']
                angle, distance = parameters['ground_angle_deg'], parameters['camera_z']
                offset, exposure = parameters['camera_x_offset'], parameters['exposure_ev']
                saturation, light_side = parameters['saturation'], parameters['light_side']
                tb, hb = min(int(fraction_t * 5), 4), min(int(fraction_h * 5), 4)
                for hand, family, fraction in zip(hands, ('temperature', 'humidity'), (fraction_t, fraction_h)):
                    start, sweep = needle_sweeps[family]
                    hand.rotation_euler.z = math.radians(start + sweep * fraction)
                pivot.rotation_euler.x = math.radians(angle - 90)
                camera.location = Vector((base_location.x + offset, base_location.y, distance))
                camera.rotation_euler = base_rotation.copy()
                camera.rotation_euler.y += math.radians(parameters['camera_yaw_deg'])
                scene.view_settings.exposure = exposure
                tone.inputs['Saturation'].default_value = saturation
                for light in lights:
                    original = base_lights[light.name]
                    light.data.energy = original['energy'] * (.72 if exposure < 0 else .95)
                    light.location = original['location'].copy()
                    if 'window' in light.name.lower():
                        light.location.x += light_side * .55
                scene.cycles.seed = (args.seed + absolute_index) % 2147483647
                bbox, points, screen = geometry(allow_partial=True)
                if args.station_pose_v4 and any(q[2] <= 0 or not (.02 <= q[0] <= .98 and .02 <= q[1] <= .98)
                                                for q in screen):
                    raise ValueError('planned camera does not keep the lower display inside its required margin')
                occlusion = parameters['occlusion']
                if occluder is not None:
                    occluder.hide_render = occlusion == 'none'
                    occluder.hide_set(occlusion == 'none')
                    if occlusion != 'none':
                        point = markers[KEYPOINTS.index(occlusion)].matrix_world.translation.copy()
                        occluder.location = point + (camera.matrix_world.translation - point).normalized() * .12
                    bpy.context.view_layer.update()
                ray_evidence = landmark_ray_evidence(scene, camera, markers)
                pose_rays = station_pose_rays(scene, camera, markers) if args.station_pose_v4 else None
                state_record = {'temperature_bin': tb, 'humidity_bin': hb,
                    'temperature_fraction_candidate': fraction_t, 'humidity_fraction_candidate': fraction_h,
                    'ground_angle_deg': angle, 'camera_z_fitted_scene_units': distance,
                    'camera_x_offset_scene_units': offset, 'exposure_ev': exposure,
                    'saturation': saturation, 'side_light_shift_scene_units': light_side * .55,
                    'state_index': absolute_index, 'planned_state_id': allocated['id']}
                if occluder is not None:
                    state_record['controlled_occlusion'] = occlusion
                state_key = f't{tb}_h{hb}_ground{angle}_distance{distance}_offset{offset}_ev{exposure}_sat{saturation}_light{light_side}'
                manifest['state_counts'][state_key] = manifest['state_counts'].get(state_key, 0) + 1
            else:
                offset = 0.0 if args.count == 1 else -.18 + .36 * index / (args.count - 1)
                camera.location = base_location + Vector((offset, 0, 0))
                bbox, points, screen = geometry()
            name = 'station_%04d' % index
            scene.render.filepath = str(out / 'images' / (name + '.png'))
            bpy.ops.render.render(write_still=True)
            if args.station_room_v4 and index == 0:
                preview_scene = ROOT / 'datasets/assets/blender/batch_scenes/meter' / out.name / 'preview_scene.blend'
                preview_scene.parent.mkdir(parents=True, exist_ok=True)
                bpy.ops.wm.save_as_mainfile(filepath=str(preview_scene), copy=True)
                manifest['editable_preview_scene'] = preview_scene.relative_to(ROOT).as_posix()
            dial_mask = station_dial_masks(scene, pivot, out, name) if args.station_dial_masks else None
            sample = {'id': name, 'image': 'images/' + name + '.png', 'annotation_status': 'candidate',
                'image_size': [640, 480], 'split': 'unassigned', 'camera_offset_x': offset,
                'camera_location': list(camera.location), 'camera_rotation_euler': list(camera.rotation_euler),
                'camera_lens_mm_fitted': float(camera.data.lens), 'camera_sensor_width_mm_fitted': float(camera.data.sensor_width),
                'needle_angle_rad': {name: float(obj.rotation_euler.z) for name, obj in zip(('temperature', 'humidity'), hands)},
                'bbox_xyxy_normalized_candidate': bbox,
                'keypoints_xy_depth_normalized_candidate': dict(zip(KEYPOINTS, points)),
                'keypoint_visibility': {name: 'unknown' for name in KEYPOINTS},
                'display_roi_quad_xy_depth_normalized_candidate': screen,
                'display_roi_visibility': 'unknown', 'display_text': 'synthetic 1234 kWh',
                'image_processing': {'resize': 'none; native 640x480 render', 'compression': 'PNG lossless', 'blur': 'none added'}}
            if args.station_batch_v4:
                sample.update({'state': state_record, 'keypoint_ray_evidence': ray_evidence,
                    'bbox_frame_intersection_candidate': [max(0.0, bbox[0]), max(0.0, bbox[1]),
                                                           min(1.0, bbox[2]), min(1.0, bbox[3])],
                    'bbox_visibility': 'unknown; geometry projection only',
                    'screen_roi_role': 'auxiliary ROI; independent from eight-point pose',
                    'light_geometry': [{'name': obj.name, 'energy': float(obj.data.energy),
                                        'location': list(obj.location)} for obj in lights],
                    'reflection_status': 'glass and environment rendered; pixel visibility not established'})
            if dial_mask is not None:
                sample['dial_raster_evidence'] = dial_mask
            if args.station_pose_v4:
                pose = station_pose_labels(scene, camera, markers, points, pose_rays, out / 'images' / (name + '.png'))
                sample['keypoint_ray_evidence'] = pose_rays
                sample['keypoint_visibility'] = pose
                sample['reflection_status'] = 'transparent cover traversed; reflection may weaken projected features'
                box = dial_mask['bbox_xyxy_pixels']
                if box:
                    x1, y1, x2, y2 = box
                    values = [0, (x1+x2)/1280, (y1+y2)/960, (x2-x1)/640, (y2-y1)/480]
                    for point_name, point in zip(KEYPOINTS, points):
                        visibility = pose[point_name]['visibility']
                        values.extend((point[0], point[1], visibility) if visibility else (0, 0, 0))
                    (out / 'labels' / (name + '.txt')).write_text(
                        ' '.join(str(v) if isinstance(v, int) else f'{v:.8f}' for v in values) + '\n', encoding='utf8')
                    sample['annotation_status'] = 'synthetic_pose_geometric'
                    sample['pose_label'] = 'labels/' + name + '.txt'
                    manifest['pose_label_count'] += 1
                else:
                    raise ValueError('planned candidate pose has no eligible bounding box')
            (out / 'annotations' / (name + '.json')).write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            if args.station_batch_v4:
                with (out / 'samples.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
            else:
                manifest['samples'].append({'image': sample['image'], 'annotation': 'annotations/' + name + '.json'})
            if args.station_batch_v4:
                run.record_sample(allocated['id'], output=out / 'images' / (name + '.png'),
                                  sources=allocated['sources'], parameters=parameters,
                                  actual={'source_blend': station_source.relative_to(ROOT).as_posix(),
                                          'endpoint_config': config.relative_to(ROOT).as_posix(),
                                          'seed': scene.cycles.seed, 'state': state_record,
                                          'camera_location': list(camera.location),
                                          'camera_rotation_euler': list(camera.rotation_euler),
                                          'lights': sample['light_geometry'],
                                          'bbox_xyxy_normalized_candidate': bbox,
                                          'keypoints_xy_depth_normalized_candidate': dict(zip(KEYPOINTS, points)),
                                          'pose_label': sample['pose_label'],
                                          'keypoint_visibility': sample['keypoint_visibility']})
            manifest['completed_count'] = index + 1
            if not args.station_batch_v4 or (index + 1) % 25 == 0 or index + 1 == args.count:
                save_manifest(manifest_path, manifest)
            print('STATION_V4_PROGRESS', index + 1, '/', args.count, flush=True)
        manifest['status'] = 'complete_synthetic_pose_candidates' if args.station_pose_v4 else 'complete_candidates'
    except BaseException as exc:
        manifest['status'] = 'failed_or_interrupted'
        manifest['error'] = '%s: %s' % (type(exc).__name__, exc)
        save_manifest(manifest_path, manifest)
        raise
    save_manifest(manifest_path, manifest)
    print('STATION_V4_COMPLETE', out, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--endpoints', default=str(source_path('meter', 'endpoint_config')))
    ap.add_argument('--count', type=int, default=10)
    ap.add_argument('--seed', type=int, default=9011)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--device', choices=('CPU', 'OPTIX', 'CUDA'), default='OPTIX')
    ap.add_argument('--preflight', action='store_true', help='Check model and projection without writing or rendering')
    ap.add_argument('--station-v4', action='store_true', help='Render native 640x480 candidates from the station prototype')
    ap.add_argument('--station-batch-v4', action='store_true',
                    help='Cycle balanced dual-needle, viewpoint and tone states in the station')
    ap.add_argument('--station-dial-masks', action='store_true',
                    help='Produce visible and isolated synthetic dial masks and conservative box candidates')
    ap.add_argument('--station-pose-v4', action='store_true',
                    help='Export geometry-grounded synthetic pose labels for box-qualified samples')
    ap.add_argument('--station-occlusion-v41', action='store_true',
                    help='Add deterministic small opaque occluder strata to pose production')
    ap.add_argument('--state-offset', type=int, default=0, help='Start index in the deterministic batch state plan')
    ap.add_argument('--station-source', default=str(STATION_SOURCE), help='Workspace station blend path from source.yaml; explicit overrides are recorded')
    ap.add_argument('--face-override', help='Traced thick-tick face for the parametric station')
    ap.add_argument('--station-room-v4', action='store_true', help='Add bounded P18-inspired room context to a station pilot')
    ap.add_argument('--station-camera-z', type=float, help='Fitted station-camera Z in scene units, 8..20')
    ap.add_argument('--production-plan', help='Workspace JSON declaration for station pose production')
    ap.add_argument('--parent-allocation', help='Prior-only scene quota record for this pilot')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    from datasets.scripts.meter.meter_production_guard import guarded, blender_media, reload_unpacked_media, rebind_station_face
    out_candidate = Path(args.output).resolve()
    declaration = None
    plan_path = None
    plan_bytes = None
    pending_face_binding = None
    if not args.preflight:
        if not (args.station_v4 and args.station_batch_v4 and args.station_dial_masks and args.station_pose_v4):
            ap.error('production supports planned station pose batches only')
        if args.station_room_v4 or args.station_camera_z is not None or args.state_offset:
            ap.error('room, camera-z override and state-offset are not supported by planned production')
        if not args.production_plan:
            ap.error('--production-plan is required before sample generation')
        plan_path = Path(args.production_plan).resolve()
        if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
            ap.error('production plan must be an existing workspace JSON file')
        source = Path(bpy.data.filepath).resolve()
        if not source.is_relative_to(ROOT):
            ap.error('station blend must be inside the workspace')
        plan_bytes = plan_path.read_bytes()
        declaration = json.loads(plan_bytes.decode('utf-8-sig'))
        if args.parent_allocation:
            parent = Path(args.parent_allocation).resolve()
            if not parent.is_relative_to(ROOT) or not parent.is_file():
                ap.error('parent allocation must be an existing workspace JSON')
            parent_data = json.loads(parent.read_text(encoding='utf-8-sig'))
            quotas = {s['id']: s['planned_count'] for s in parent_data['allocation']['strata']}
            if quotas.get('station') != args.count:
                ap.error('station count differs from parent allocation')
        validate_station_declaration(declaration, count=args.count, source=source.relative_to(ROOT).as_posix(),
                                     allow_occlusion=args.station_occlusion_v41)
        parametric_station = (ROOT / 'datasets/assets/blender/meter/v4_parametric_station_v2_20261113/station.blend').resolve()
        if source == parametric_station:
            endpoint_path = Path(args.endpoints).resolve()
            if not endpoint_path.is_relative_to(ROOT) or not endpoint_path.is_file():
                ap.error('endpoint configuration must be an existing workspace file')
            endpoint_info = json.loads(endpoint_path.read_text(encoding='utf-8-sig'))
            expected_face = (parametric_station.parent / 'face.png').resolve()
            if (ROOT / endpoint_info.get('source_face', '')).resolve() != expected_face:
                ap.error('parametric station endpoints must name its current face')
            pending_face_binding = expected_face
    elif args.production_plan:
        ap.error('preflight does not generate samples and does not accept a production plan')
    extra_outputs = ([ROOT / 'datasets/assets/blender/batch_scenes/meter' / out_candidate.name / 'preview_scene.blend']
                     if args.station_room_v4 else [])
    def reload_meter_media():
        if pending_face_binding is not None:
            rebind_station_face(bpy, pending_face_binding)
        reload_unpacked_media(bpy)
    with guarded('render_meter_candidates', args, __file__, bpy.data.filepath,
                 extra_inputs=[args.endpoints, ROOT/'datasets/scripts/meter/config/dataset.yaml', ROOT/'datasets/scripts/meter/config/source.yaml'] + ([args.station_source] if args.station_source else []) + ([args.face_override] if args.face_override else []) + ([args.parent_allocation] if args.parent_allocation else []) + ([plan_path] if plan_path else []),
                 code_dependencies=[TASK_HELPERS/'render_meter_batch.py', TASK_HELPERS/'power_meter_room.py', TASK_HELPERS/'meter_station_plan.py', ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py'],
                 outputless=args.preflight, media_probe=lambda: blender_media(bpy, face_source=pending_face_binding),
                 media_reload=reload_meter_media, extra_outputs=extra_outputs,
                 production_plan=declaration,
                 sample_output_paths=[out_candidate / 'images'] if declaration else (),
                 auxiliary_output_paths=[out_candidate / 'masks'] if declaration else ()) as run:
        if plan_path is not None and plan_path.read_bytes() != plan_bytes:
            raise ValueError('production plan changed while entering the guard')
        if run.reused:
            if args.preflight:
                result = run.value
                if result['kind'] == 'station':
                    print('STATION_V4_PREFLIGHT_OK', json.dumps(result['summary']), flush=True)
                else:
                    print('PREFLIGHT_OK', result['device'], result['devices_used'], json.dumps(result['points']), flush=True)
            return
        if not 1 <= args.count <= 10000 or not 256 <= args.resolution <= 2048 or not 1 <= args.samples <= 4096:
            ap.error('count 1..10000, resolution 256..2048, samples 1..4096 required')
        out = Path(args.output).resolve()
        if not args.preflight and out not in {sample_dir('meter').resolve(), split_dir('meter', 'train').resolve()}:
            ap.error('output must be the meter sample or train directory')
        if not args.preflight and out.exists() and any(out.iterdir()):
            ap.error('output exists; refusing to overwrite')
        config = Path(args.endpoints).resolve()
        if not config.is_relative_to(ROOT):
            ap.error('endpoint configuration must be inside the workspace')
        endpoint_data = json.loads(config.read_text(encoding='utf-8-sig'))
        if args.station_room_v4 and not args.station_v4:
            ap.error('--station-room-v4 requires --station-v4')
        if args.station_batch_v4 and not args.station_v4:
            ap.error('--station-batch-v4 requires --station-v4')
        if args.station_dial_masks and not args.station_batch_v4:
            ap.error('--station-dial-masks requires --station-batch-v4')
        if args.station_pose_v4 and not (args.station_batch_v4 and args.station_dial_masks):
            ap.error('--station-pose-v4 requires --station-batch-v4 and --station-dial-masks')
        if args.station_occlusion_v41 and not args.station_pose_v4:
            ap.error('--station-occlusion-v41 requires --station-pose-v4')
        if args.state_offset < 0 or (args.state_offset and not args.station_batch_v4):
            ap.error('state-offset must be non-negative and requires --station-batch-v4')
        if args.station_source and not args.station_v4:
            ap.error('--station-source requires --station-v4')
        if args.station_v4:
            if args.preflight and not any(arg == '--count' or arg.startswith('--count=') for arg in sys.argv):
                args.count = 2
            if any(arg == '--resolution' or arg.startswith('--resolution=') for arg in sys.argv):
                ap.error('station-v4 uses fixed native 640x480; omit --resolution')
            summary = render_station_v4(args, out, config, endpoint_data, run=run)
            if args.preflight:
                run.set_result({'kind': 'station', 'summary': summary})
            return
        if args.station_camera_z is not None:
            ap.error('--station-camera-z requires --station-v4')
        source = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
        expected_source = (ROOT / endpoint_data['source_blend']).resolve()
        if source != expected_source:
            ap.error('endpoint calibration applies only to the specified source blend')
        if endpoint_data['image_size'] != [1254, 1254]:
            ap.error('unexpected endpoint image dimensions')
        face = bpy.data.objects['PDF_face_inpainted']
        image_nodes = [node for material in face.data.materials if material and material.use_nodes
                       for node in material.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image]
        reference_face = (ROOT / endpoint_data['source_face']).resolve()
        if (not reference_face.is_relative_to(ROOT) or len(image_nodes) != 1 or
                not image_nodes[0].image.packed_file or
                bytes(image_nodes[0].image.packed_file.data) != reference_face.read_bytes()):
            ap.error('packed face does not match the calibrated source image')
        if any(abs(face.matrix_world[i][j] - (1.0 if i == j else 0.0)) > 1e-6
               for i in range(4) for j in range(4)):
            ap.error('face transform differs from calibrated projection')
        uv_data = face.data.uv_layers.active.data
        for loop_index, loop in enumerate(face.data.loops):
            vertex = face.data.vertices[loop.vertex_index].co
            uv = uv_data[loop_index].uv
            if max(abs(vertex.x - (2 * uv.x - 1)), abs(vertex.y - (2 * uv.y - 1)),
                   abs(vertex.z - .065)) > 1e-5:
                ap.error('face UV or geometry differs from calibrated projection')
        endpoints = endpoint_data['endpoints']
        if any(len(endpoints[k]) != 2 or not all(0 <= v < 1254 for v in endpoints[k]) for k in KEYPOINTS[4:]):
            ap.error('invalid endpoint coordinates')
        scene = bpy.context.scene
        camera = bpy.data.objects['training_camera']
        hands = [bpy.data.objects['temperature_needle_rotate_Z'], bpy.data.objects['humidity_needle_rotate_Z']]
        lights = [o for o in scene.objects if o.type == 'LIGHT']
        base_lights = {o.name: {'energy': o.data.energy, 'location': list(o.location)} for o in lights}
        # In-memory changes only. Blender source and its original actions are never saved.
        for obj in [camera] + hands:
            obj.animation_data_clear()
        scene.camera = camera
        scene.render.resolution_x = scene.render.resolution_y = args.resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.cycles.samples = args.samples
        if scene.render.engine != 'CYCLES':
            ap.error('source must use the Cycles renderer')
        devices_used = ['CPU']
        if args.device == 'CPU':
            scene.cycles.device = 'CPU'
        else:
            preferences = bpy.context.preferences.addons['cycles'].preferences
            try:
                preferences.compute_device_type = args.device
                preferences.get_devices()
            except (TypeError, ValueError, RuntimeError) as exc:
                ap.error('requested device is unavailable: %s' % exc)
            selected = [d for d in preferences.devices if d.type == args.device]
            if not selected:
                ap.error('requested %s device is unavailable' % args.device)
            for device in preferences.devices:
                device.use = device in selected
            scene.cycles.device = 'GPU'
            devices_used = [d.name for d in selected]
        if args.preflight:
            camera.location = (0, 0, 4.2)
            look_at(camera, 0)
            bpy.context.view_layer.update()
            points3 = [hands[0].matrix_world @ Vector((0, 0, 0)), hands[0].matrix_world @ Vector((.76, 0, 0)),
                       hands[1].matrix_world @ Vector((0, 0, 0)), hands[1].matrix_world @ Vector((.31, 0, 0))]
            points3 += [texture_xy(endpoints[k]) for k in KEYPOINTS[4:]]
            points = [project(scene, camera, p) for p in points3]
            if any(p[2] <= 0 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in points):
                raise RuntimeError('Preflight projection falls outside the camera')
            projected = dict(zip(KEYPOINTS, points))
            run.set_result({'kind': 'standard', 'device': args.device, 'devices_used': devices_used, 'points': projected})
            print('PREFLIGHT_OK', args.device, devices_used, json.dumps(projected), flush=True)
            return
        out.mkdir(parents=True, exist_ok=True)
        (out / 'images').mkdir()
        manifest = {
            'version': 1, 'status': 'running', 'requested_count': args.count,
            'source_blend': endpoint_data['source_blend'],
            'endpoint_config': config.relative_to(ROOT).as_posix(),
            'source_group': 'meter_9011_pbr_v2_generated_face',
            'split_status': 'unsplit; all draws share one model and face; not independent evaluation',
            'landmark_status': 'geometric projection candidates; four face ticks visually calibrated on generated texture; visibility unknown; not real meter truth',
            'face_calibration_status': endpoint_data['status'],
            'keypoint_order': KEYPOINTS, 'seed': args.seed, 'resolution': args.resolution,
            'cycles_samples': args.samples, 'requested_device': args.device,
            'cycles_device': scene.cycles.device, 'selected_devices': devices_used, 'original_lights': base_lights,
            'sample_records': 'samples.jsonl', 'completed_count': 0}
        manifest_path = out / 'manifest.json'
        save_manifest(manifest_path, manifest)
        rng = random.Random(args.seed)
        bins_t, bins_h = list(range(args.count)), list(range(args.count))
        rng.shuffle(bins_t)
        rng.shuffle(bins_h)
        try:
            for index in range(args.count):
                ft, fh = (bins_t[index] + rng.random()) / args.count, (bins_h[index] + rng.random()) / args.count
                hands[0].rotation_euler.z = math.radians(-147 - 241 * ft)
                hands[1].rotation_euler.z = math.radians(44 - 264 * fh)
                for camera_attempt in range(1, 65):
                    camera.location = (rng.uniform(-.8, .8), rng.uniform(-.65, .65), rng.uniform(3.8, 4.7))
                    roll = math.radians(rng.uniform(-8, 8))
                    look_at(camera, roll)
                    bpy.context.view_layer.update()
                    outline = [project(scene, camera, (1.05 * math.cos(i * math.tau / 128),
                               1.05 * math.sin(i * math.tau / 128), z))
                               for z in (-.06, .15) for i in range(128)]
                    x0, y0 = min(p[0] for p in outline), min(p[1] for p in outline)
                    x1, y1 = max(p[0] for p in outline), max(p[1] for p in outline)
                    if min(p[2] for p in outline) > 0 and min(x0, y0, 1-x1, 1-y1) >= .025:
                        break
                else:
                    raise RuntimeError('Sample %d failed full-frame margin after 64 camera attempts' % index)
                lamp_values = {}
                for light in lights:
                    base = base_lights[light.name]
                    light.data.energy = base['energy'] * rng.uniform(.75, 1.25)
                    light.location = Vector(base['location']) + Vector((rng.uniform(-.3, .3), rng.uniform(-.3, .3), 0))
                    lamp_values[light.name] = {'energy': float(light.data.energy), 'location': list(light.location)}
                scene.cycles.seed = (args.seed + index) % 2147483647
                bpy.context.view_layer.update()
                points3 = [hands[0].matrix_world @ Vector((0, 0, 0)), hands[0].matrix_world @ Vector((.76, 0, 0)),
                           hands[1].matrix_world @ Vector((0, 0, 0)), hands[1].matrix_world @ Vector((.31, 0, 0))]
                points3 += [texture_xy(endpoints[k]) for k in KEYPOINTS[4:]]
                candidates = [project(scene, camera, p) for p in points3]
                filename = 'meter_%05d.png' % index
                scene.render.filepath = str(out / 'images' / filename)
                bpy.ops.render.render(write_still=True)
                sample = {
                    'id': 'meter_%05d' % index, 'image': 'images/' + filename,
                    'camera_location': list(camera.location), 'camera_rotation_euler': list(camera.rotation_euler),
                    'camera_attempts': camera_attempt,
                    'temperature_angle_rad': hands[0].rotation_euler.z, 'humidity_angle_rad': hands[1].rotation_euler.z,
                    'lights': lamp_values, 'bbox_xyxy_candidate': [x0, y0, x1, y1],
                    'keypoints_xy_depth_candidates': candidates, 'visibility': [None] * 8}
                with (out / 'samples.jsonl').open('a', encoding='utf8') as records:
                    records.write(json.dumps(sample, ensure_ascii=False) + '\n')
                manifest['completed_count'] = index + 1
                save_manifest(manifest_path, manifest)
                print('CANDIDATE_PROGRESS', index + 1, '/', args.count, flush=True)
            manifest['status'] = 'complete_candidates'
        except BaseException as exc:
            manifest['status'] = 'failed_or_interrupted'
            manifest['error'] = '%s: %s' % (type(exc).__name__, exc)
            save_manifest(manifest_path, manifest)
            raise
        save_manifest(manifest_path, manifest)
        print('CANDIDATE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
