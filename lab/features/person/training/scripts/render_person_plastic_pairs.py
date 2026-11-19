"""Render paired plastic-standee scenes in an isolated Blender background process."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_standees import material, principled_node
from datasets.scripts.common.blender_workspace import ARENA, DETAILED, WORKSPACE_BLEND, activate_scene

WIDTH, HEIGHT = 640, 480
CLASSES = ("community", "noncommunity")
LAYOUTS = ("side", "oblique_back", "edge", "mixed", "oblique_front", "front")
COUNTS = {"train": 60, "val": 18}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_scene(split, device):
    activate_scene(ARENA if split == "train" else DETAILED)
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 7
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 3
    if device == "OPTIX":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = device
        prefs.get_devices()
        if not any(item.type == device for item in prefs.devices):
            raise RuntimeError("OptiX device is unavailable")
        for item in prefs.devices:
            item.use = item.type == device
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    world = bpy.data.worlds.new("Plastic standee sky")
    world.use_nodes = True
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(58)
    sky.sun_rotation = math.radians(35)
    sky.sun_disc = False
    background = world.node_tree.nodes.get("Background")
    background.inputs["Strength"].default_value = .22
    world.node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
    scene.world = world
    camera = scene.objects["Vehicle camera"]
    camera.animation_data_clear()
    camera.parent = None
    for constraint in list(camera.constraints):
        camera.constraints.remove(constraint)
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.shift_x = camera.data.shift_y = 0
    camera.data.clip_start = .012
    camera.data.dof.use_dof = False
    scene.camera = camera
    return scene, camera


def area(scene, name, size, width, power):
    data = bpy.data.lights.new(name, "AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y, data.energy = size, width, power
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return obj


def plastic():
    mat = bpy.data.materials.new("White molded plastic")
    mat.use_nodes = True
    shader = principled_node(mat)
    shader.inputs["Base Color"].default_value = (.77, .79, .81, 1)
    shader.inputs["Metallic"].default_value = 0
    shader.inputs["IOR"].default_value = 1.46
    shader.inputs["Specular IOR Level"].default_value = .5
    return mat


def make_board(scene, item, thickness, height, pos, yaw, back):
    image = ROOT / item["image"]
    width_px, height_px = item["size"]
    width = height * width_px / height_px
    bevel = .00048
    curve = bpy.data.curves.new("Closed alpha | " + image.stem, "CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.extrude = thickness / 2 - bevel
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    for contour in item["contours"]:
        spline = curve.splines.new("POLY")
        spline.points.add(len(contour) - 1)
        for point, (x, y) in zip(spline.points, contour):
            point.co = ((x / width_px - .5) * width, (1 - y / height_px) * height, 0, 1)
        spline.use_cyclic_u = True
        spline.use_smooth = True

    # Local +Z becomes the printed face normal, global -Y at zero yaw.
    board = bpy.data.objects.new("Board | " + image.stem, curve)
    scene.collection.objects.link(board)
    board.rotation_euler = (math.pi / 2, 0, yaw)
    lowest = max(point[1] for outline in item["contours"] for point in outline)
    board.location = pos - Vector((0, 0, (1 - lowest / height_px) * height))
    curve.materials.append(back)

    # The isolated face carries the alpha artwork and shares the parent outline coordinates.
    mesh = bpy.data.meshes.new("Print | " + image.stem)
    z = thickness / 2 + .00012
    mesh.from_pydata([(-width / 2, 0, z), (width / 2, 0, z),
                    (width / 2, height, z), (-width / 2, height, z)], [], [(0, 1, 2, 3)])
    uv = mesh.uv_layers.new()
    for loop, coords in zip(mesh.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
        uv.data[loop].uv = coords
    print_layer = bpy.data.objects.new("Print | " + image.stem, mesh)
    scene.collection.objects.link(print_layer)
    print_layer.parent = board
    print_layer.data.materials.append(bpy.data.materials.get("Artwork | " + image.stem)
                                      or material("Artwork | " + image.stem, image))
    return board, print_layer


def box_for(scene, camera, board):
    evaluated = board.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        points = [world_to_camera_view(scene, camera, board.matrix_world @ vertex.co)
                  for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()
    if not points or min(point.z for point in points) <= 0:
        raise ValueError("Board is behind the camera or has no evaluated vertices")
    return [min(p.x for p in points) * WIDTH, (1 - max(p.y for p in points)) * HEIGHT,
            max(p.x for p in points) * WIDTH, (1 - min(p.y for p in points)) * HEIGHT]


def clipped(box):
    return [min(WIDTH, max(0, box[0])), min(HEIGHT, max(0, box[1])),
            min(WIDTH, max(0, box[2])), min(HEIGHT, max(0, box[3]))]


def yolo_row(obj):
    a, b, c, d = obj["box_xyxy"]
    return " ".join([str(CLASSES.index(obj["class"])),
                     *(f"{v:.8f}" for v in ((a+c)/(2*WIDTH), (b+d)/(2*HEIGHT), (c-a)/WIDTH, (d-b)/HEIGHT))])


def place(index, scene, camera, art, back, rng):
    layout = LAYOUTS[index % len(LAYOUTS)]
    # Each held-out render uses the detailed room and disjoint printed identities.
    scene_heading = (0, -math.pi / 2, math.pi / 2)[(index // 3) % 3]
    sine, cosine = math.sin(scene_heading), math.cos(scene_heading)
    wall_depth = (1.45, 1.66, .65, 1.55, .95)[(index // 2) % 5]
    side = rng.uniform(-.55, .55)

    def world(x, y, z):
        return Vector((cosine*(x+side)-sine*(y+wall_depth), sine*(x+side)+cosine*(y+wall_depth), z))

    distance = rng.uniform(.28, .49) if layout != "front" else rng.uniform(.25, .43)
    camera.location = world(0, -distance, rng.uniform(.098, .151))
    camera.data.lens = rng.uniform(20, 24)
    aim(camera, world(0, .01, rng.uniform(.057, .091)))
    scene.view_settings.exposure = rng.uniform(-.48, .12)
    scene.cycles.seed = 9021 + index

    plane_width = distance * camera.data.sensor_width / camera.data.lens
    positive_class = CLASSES[(index // 6) % 2]
    neg_class = "noncommunity" if index % 5 == 0 else "community"
    sign = (-1, 1)[(index // 6) % 2]
    side_angle = rng.uniform(88, 94)
    oblique_back = rng.uniform(106, 144)
    oblique_front = rng.uniform(31, 53)
    angle = {"side": side_angle, "oblique_back": oblique_back, "edge": rng.uniform(97, 140),
             "mixed": rng.choice([side_angle, oblique_back]), "oblique_front": oblique_front,
             "front": rng.uniform(0, 24)}[layout]
    negative = (neg_class, sign * angle, sign * .49 if layout == "edge" else rng.uniform(-.25, .25), False)
    if layout == "mixed":
        opposite = CLASSES[1 - CLASSES.index(positive_class)]
        descriptions = [(positive_class, -20, -.30, True), (opposite, 37, .30, True),
                        (neg_class, sign * angle, 0, False)]
    elif layout in ("oblique_front", "front"):
        descriptions = [(positive_class, sign * angle, rng.uniform(-.25, .25), True)]
    else:
        descriptions = [negative]
        if layout in ("side", "oblique_back") and (index // 6) % 2:
            descriptions.insert(0, (neg_class, -sign * rng.uniform(98, 140), -.31, False))
            descriptions[1] = (neg_class, sign * angle, .31, False)
    objects, evidence = [], []
    for slot, (kind, angle_from_front, x, positive) in enumerate(descriptions):
        item = rng.choice(art[kind])
        pos = world(x * plane_width, rng.uniform(-.011, .04), 0)
        yaw = math.atan2(camera.location.x - pos.x, pos.y - camera.location.y)
        yaw += math.radians(angle_from_front)
        thickness = rng.uniform(.0046, .0087)
        height = rng.uniform(.135, .156)
        board, front = make_board(scene, item, thickness, height, pos, yaw, back)
        objects.extend([board, front])
        # Pure side may expose <4 degrees of print, below the design visibility limit.
        bpy.context.view_layer.update()
        bbox = box_for(scene, camera, board)
        xyxy = clipped(bbox)
        if xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 45:
            raise ValueError(f"{layout} board outside frame: {bbox}")
        if positive and bbox != xyxy:
            raise ValueError(f"{layout} positive unexpectedly cropped: {bbox}")
        evidence.append({"object_id": f"{index:03d}-{slot}", "class": kind,
                         "texture": item["image"], "artwork": item["artwork"], "positive": positive,
                         "view_deg_from_print": angle_from_front, "thickness_m": thickness,
                         "bevel_m": .00048, "height_m": height, "position_m": list(pos), "yaw_rad": yaw,
                         "box_xyxy": xyxy, "unclipped_box_xyxy": bbox})

    for i, first in enumerate(evidence):
        a, b, c, d = first["box_xyxy"]
        for second in evidence[i+1:]:
            e, f, g, h = second["box_xyxy"]
            if min(c, g) > max(a, e) and min(d, h) > max(b, f):
                raise ValueError(f"{layout} contains overlapping board projections: {evidence}")
    return layout, objects, evidence, world


def render(args):
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    scene, camera = setup_scene(args.split, args.device)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "labels"):
        (out / folder).mkdir()
    back = plastic()
    shader = principled_node(back)
    overhead = area(scene, "Local ceiling panel", .32, .24, 10)
    strip = area(scene, "Long diffuse ceiling strip", .24, .045, 4)
    receivers = bpy.data.collections.new("Plastic reflection receivers")
    scene.collection.children.link(receivers)
    strip.light_linking.receiver_collection = receivers
    # Keep the lamps fixed between pairs. Vary the dielectric specular response of the board.
    strip.data.diffuse_factor = 0
    strip.data.transmission_factor = 0
    strip.data.volume_factor = 0
    rows = []
    count = len(LAYOUTS) if args.preview else COUNTS[args.split]
    for index in range(count):
        rng = random.Random((78016 if args.split == "train" else 11481) + 1009*index)
        layout, objects, evidence, world = place(index, scene, camera, plan[args.split], back, rng)
        overhead.location = world(.05, -.055, .41)
        overhead.data.energy = rng.uniform(4, 9)
        aim(overhead, world(0, 0, .05))
        subject_slot = next((i for i, obj in enumerate(evidence) if not obj["positive"]), 0)
        subject = objects[subject_slot * 2]
        receivers.objects.link(subject)
        target = subject.location + Vector((0, 0, .081))
        board_normal = subject.rotation_euler.to_matrix() @ Vector((0, 0, 1))
        to_camera = (camera.location - target).normalized()
        # The side edge and oblique broad back are separate reflecting surfaces.
        tangent = subject.rotation_euler.to_matrix() @ Vector((1, 0, 0))
        if layout == "side" or (layout == "mixed" and abs(evidence[subject_slot]["view_deg_from_print"]) < 96):
            normal = tangent if tangent.dot(to_camera) > 0 else -tangent
        else:
            normal = board_normal if board_normal.dot(to_camera) > 0 else -board_normal
        reflection = 2 * normal.dot(to_camera) * normal - to_camera
        # A real area source spans the ideal reflection direction, with the center lifted enough to clear the floor.
        strip.location = target + reflection * .18 + Vector((0, 0, .025))
        aim(strip, target)
        strip.data.size_y = rng.uniform(.035, .065)
        strip.data.energy = rng.uniform(1.8, 3.7)
        roughness = rng.uniform(.23, .34)
        shader.inputs["Roughness"].default_value = roughness
        for condition, specular in (("soft", .12), ("highlight", .50)):
            shader.inputs["Specular IOR Level"].default_value = specular
            stem = f"{args.split}_{index:03d}_{layout}_{condition}"
            scene.render.filepath = str(out / "images" / (stem + ".png"))
            label = out / "labels" / (stem + ".txt")
            labels = [yolo_row(obj) for obj in evidence if obj["positive"]]
            bpy.ops.render.render(write_still=True)
            label.write_text("".join(row + "\n" for row in labels), encoding="utf-8")
            rows.append({"image": relative(Path(scene.render.filepath)), "label": relative(label),
                         "split": args.split, "pair": f"{args.split}_{index:03d}", "layout": layout,
                         "light": condition, "plastic_specular_ior_level": specular,
                         "strip_power_w": strip.data.energy,
                         "strip_size_m": [strip.data.size, strip.data.size_y],
                         "strip_location_m": list(strip.location), "overhead_power_w": overhead.data.energy,
                         "roughness": roughness, "exposure": scene.view_settings.exposure,
                         "camera_location_m": list(camera.location), "camera_euler": list(camera.rotation_euler),
                         "focal_mm": camera.data.lens, "cycles_seed": scene.cycles.seed, "objects": evidence})
        if index == 3 and not args.preview:
            bpy.ops.file.pack_all()
            use_preview = bpy.context.preferences.filepaths.save_version
            try:
                bpy.context.preferences.filepaths.save_version = 0
                bpy.ops.wm.save_as_mainfile(filepath=str(out / "paired_plastic_standees.blend"), copy=True)
            finally:
                bpy.context.preferences.filepaths.save_version = use_preview
        for obj in objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        print(f"PAIR {args.split} {index+1}/{count} {layout}", flush=True)
    write_json(out / "render_manifest.json", {
        "source_blend": relative(WORKSPACE_BLEND),
        "source_scene": ARENA if args.split == "train" else DETAILED,
        "source_blend_sha256": hashlib.sha256(WORKSPACE_BLEND.read_bytes()).hexdigest(),
        "contour_plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "blender": bpy.app.version_string, "device": args.device, "samples": scene.cycles.samples,
        "global_illumination": {"sky_strength": .22, "diffuse_bounces": 4, "glossy_bounces": 3},
        "geometry_level": "alpha-derived decorative board, board thickness and lighting visually approximated",
        "supervision": "procedural geometry; clipped board projection, no added foot support",
        "records": rows,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=tuple(COUNTS))
    parser.add_argument("--device", default="OPTIX", choices=("OPTIX", "CPU"))
    parser.add_argument("--preview", action="store_true")
    render(parser.parse_args(sys.argv[sys.argv.index("--") + 1:]))
