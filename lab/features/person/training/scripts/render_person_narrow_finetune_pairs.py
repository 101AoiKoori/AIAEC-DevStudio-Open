"""Render width-matched front and white-back pairs in the saved low-exposure scene."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_person_plastic_pairs import aim, box_for, make_board, plastic, principled_node, yolo_row

PAIR_COUNTS = {"train": 64, "val": 18}
ANGLES = {"train": (14, 15, 16, 16, 17, 18, 20, 21, 22, 22, 23, 24, 28, 34, 43, 59),
          "val": (14, 16, 17, 20, 22, 24, 27, 36, 54)}
CAMERAS = {"train": ((-.33, 1.21), (.17, .96), (-.62, 1.12), (.48, 1.27)),
           "val": ((-.96, .83), (.77, 1.14), (.28, .62))}
CLASS_NAMES = ("community", "noncommunity")
WIDTH, HEIGHT = 640, 480


def heading(camera, pos):
    return math.atan2(camera.location.x - pos.x, pos.y - camera.location.y)


def footprint(camera, image_x, depth):
    focal_px = camera.data.lens * WIDTH / camera.data.sensor_width
    return camera.location.x + (image_x - WIDTH / 2) * depth / focal_px, camera.location.y + depth, 0


def pose(board, position, yaw):
    board.location.x, board.location.y = position[:2]
    board.rotation_euler.z = yaw
    bpy.context.view_layer.update()


def box(scene, camera, board):
    result = box_for(scene, camera, board)
    x1, y1, x2, y2 = result
    if not (0 < x1 < x2 < WIDTH and 0 < y1 < y2 < HEIGHT and y2-y1 > 75):
        raise ValueError(f"Unusable projection: {result}")
    return result


def width(xyxy):
    return xyxy[2] - xyxy[0]


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0],b[0]) and min(a[3], b[3]) > max(a[1],b[1])


def init_scene():
    scene = bpy.context.scene
    if scene.name != "Arena Indoor Soft Light" or not math.isclose(scene.view_settings.exposure, -.3, abs_tol=1e-5):
        raise ValueError("Expected the audited low-exposure diagnostic scene")
    for obj in list(scene.objects):
        if obj.name.startswith(("Board | ", "Print | ")):
            bpy.data.objects.remove(obj, do_unlink=True)
    strip = scene.objects["Diagnostic specular strip"]
    # Keep the artist's lamps and exposure. Use the fixed diagnostic source on the current target only.
    receivers = strip.light_linking.receiver_collection
    scene.render.use_persistent_data = True
    scene.cycles.samples = 20
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    if not any(device.type == "OPTIX" for device in prefs.devices):
        raise RuntimeError("OptiX is unavailable")
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    return scene, scene.camera, receivers


def make_pair(scene, camera, receivers, split, index, artwork):
    group = index % len(CAMERAS[split])
    rng = random.Random((352681 if split == "train" else 930062) + index*807)
    x, y = CAMERAS[split][group]
    camera.location = (x+rng.uniform(-.05,.05), y+rng.uniform(-.04,.04), rng.uniform(.127,.170))
    camera.data.lens = rng.choice((20., 22., 24.))
    aim(camera, camera.location + Vector((0, 1, rng.uniform(-.027,.035))))
    angle = ANGLES[split][(index // len(CAMERAS[split]) + group) % len(ANGLES[split])]
    side_sign = (-1, 1)[(index + index // len(CAMERAS[split])) % 2]
    screen_x = rng.uniform(362, 482) if side_sign > 0 else rng.uniform(152, 282)
    depth = rng.uniform(.35,.49)
    position = Vector(footprint(camera, screen_x, depth))
    direction = heading(camera, position)
    back_yaw = direction + math.radians(side_sign*(90+angle))
    thickness = rng.uniform(.0052,.0081)
    height = rng.uniform(.140,.155)
    class_name = CLASS_NAMES[(index // 2 + group) % 2]
    item = rng.choice(artwork[class_name])
    target_material = plastic()
    shader = principled_node(target_material)
    shader.inputs["Roughness"].default_value = rng.uniform(.23,.35)
    shader.inputs["Specular IOR Level"].default_value = rng.choice((.12,.25,.50,.75,1.))
    target, print_layer = make_board(scene, item, thickness, height, position, back_yaw, target_material)
    receivers.objects.link(target)
    scene.cycles.seed = 6101 + index
    bpy.context.view_layer.update()
    back_box = box(scene,camera,target)
    # Compare the projected physical width of the same board at near-front yaw angles.
    # The minimum printed-face exposure avoids treating nearly invisible print as a positive.
    front_choices = []
    for front_angle in range(17,70):
        yaw = direction + math.radians(side_sign*(90-front_angle))
        pose(target, position, yaw)
        delta = abs(width(box(scene,camera,target))-width(back_box))
        front_choices.append((delta, front_angle, yaw))
    delta, front_angle, front_yaw = min(front_choices)
    if delta > 7:
        raise ValueError(f"Width match exceeds 7 px for {index}: {delta:.2f}")
    pose(target,position,front_yaw)
    front_box = box(scene,camera,target)

    # Introduce some fully visible opposite-side positives and some neutral white context.
    companions = []
    objects = [target,print_layer]
    if index % 4 != 0:
        companion_class = CLASS_NAMES[1-CLASS_NAMES.index(class_name)]
        companion_item = rng.choice(artwork[companion_class])
        at = Vector(footprint(camera, 640-screen_x, depth+rng.uniform(.01,.05)))
        printed = index % 4 in (1,2)
        companion_yaw = heading(camera,at)+math.radians((rng.uniform(-32,32) if printed else 118))
        other, other_art = make_board(scene,companion_item,.006,.141,at,companion_yaw,target_material)
        objects += [other, other_art]
        bpy.context.view_layer.update()
        other_box = box(scene,camera,other)
        if overlaps(other_box,back_box) or overlaps(other_box,front_box):
            raise ValueError("Diagnostic boards overlap")
        companions.append({"class":companion_class,"box_xyxy":other_box,"printed":printed,
                           "artwork":companion_item["artwork"],"texture":companion_item["image"]})
    common = {"split":split,"pair":f"{split}_{index:03d}","group":f"{split}_camera_family_{group}",
              "texture":item["image"],"artwork":item["artwork"],"class":class_name,
              "camera_location_m":list(camera.location),"camera_angles_rad":list(camera.rotation_euler),
              "focal_mm":camera.data.lens,"board_position_m":list(position),"height_m":height,
              "thickness_m":thickness,"back_angle_deg":angle,"front_angle_deg":front_angle,"side_sign":side_sign,
              "specular_ior_level":shader.inputs["Specular IOR Level"].default_value,
              "roughness":shader.inputs["Roughness"].default_value,"width_delta_px":delta,
              "companions":companions}
    return objects, target, position, common, ((False,back_yaw,back_box),(True,front_yaw,front_box))


def render(args):
    art = json.loads(args.contours.read_text(encoding="utf8"))
    source = Path(bpy.data.filepath).resolve()
    scene,camera,receivers = init_scene()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for split,count in PAIR_COUNTS.items():
        directory = args.output / split
        for name in ("images","labels"):
            (directory/name).mkdir(parents=True)
        for i in range(count):
            objects,target,position,common,conditions = make_pair(scene,camera,receivers,split,i,art[split])
            for positive,yaw,xyxy in conditions:
                pose(target,position,yaw)
                state = "front" if positive else "white_back"
                stem = common["pair"]+"_"+state
                scene.render.filepath = str(directory/"images"/(stem+".png"))
                bpy.ops.render.render(write_still=True)
                label = directory/"labels"/(stem+".txt")
                targets = [row for row in common["companions"] if row["printed"]]
                if positive:
                    targets = targets + [{"class":common["class"],"box_xyxy":xyxy}]
                label.write_text("".join(yolo_row(obj)+"\n" for obj in targets), encoding="utf8")
                rows.append({**common,"state":state,"target_printed":positive,"target_box_xyxy":xyxy,
                             "image":Path(scene.render.filepath).relative_to(ROOT).as_posix(),
                             "label":label.relative_to(ROOT).as_posix(),"boxes":targets})
            for obj in objects:
                bpy.data.objects.remove(obj,do_unlink=True)
            print(f"PAIR {split} {i+1}/{count}, angles {common['back_angle_deg']}/{common['front_angle_deg']}, delta {common['width_delta_px']:.2f}",flush=True)
    manifest = {"records":rows,"source":source.relative_to(ROOT).as_posix(),
                "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "scene":scene.name,"exposure_ev":scene.view_settings.exposure,"world":scene.world.name,
                "contour_plan":args.contours.relative_to(ROOT).as_posix(),
                "contour_plan_sha256":hashlib.sha256(args.contours.read_bytes()).hexdigest(),
                "blender":bpy.app.version_string,"samples":scene.cycles.samples,"supports_added":False,
                "camera_families":CAMERAS,"conditions_deg":ANGLES}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contours",required=True,type=lambda p:Path(p).resolve())
    parser.add_argument("--output",required=True,type=lambda p:Path(p).resolve())
    render(parser.parse_args(sys.argv[sys.argv.index("--")+1:]))
