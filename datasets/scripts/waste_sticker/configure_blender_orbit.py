"""Bake per-bin paper placement and nine orbit cameras into the unified blend."""
import argparse
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_workspace import ORBIT_VIEWS, STICKERS, WORKSPACE_BLEND, orbit_camera_name, store_orbit_cameras
STICKER_SCENE = "Waste Stickers"
COLORS = ("blue", "red", "green", "gray")
PAPER_SCALE = (1.10, 1.42)
MIN_POSITION_JUMP = .24
FACE_MARGIN = .025


def camera_name(view):
    return orbit_camera_name(STICKER_SCENE, view)

def sticker_objects(scene):
    by_source = {obj.get("aec_source_name", obj.name.split(" | ", 1)[-1]): obj
                 for obj in scene.objects}
    return {color: by_source[f"{color} item sticker paper"] for color in COLORS}


def sticker_bin_objects(objects):
    groups = {color: [] for color in COLORS}
    for obj in objects:
        name = obj.get("aec_source_name", obj.name)
        color = name.split(" ", 1)[0]
        if color in groups:
            groups[color].append(obj)
    return groups


def store_bin_collections():
    asset = bpy.data.collections[STICKERS]
    for color, objects in sticker_bin_objects(asset.all_objects).items():
        name = f"{STICKER_SCENE} | {color.capitalize()} bin"
        collection = asset.children.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            asset.children.link(collection)
        for obj in objects:
            if obj.name in asset.objects:
                collection.objects.link(obj)
                asset.objects.unlink(obj)


def front_outline(body, center, across, up, normal):
    face = min((polygon for polygon in body.data.polygons
                if len(polygon.vertices) == 4 and polygon.normal.dot(normal) > .95),
               key=lambda polygon: polygon.center.y)
    vertices = [body.data.vertices[index].co - center for index in face.vertices]
    outline = [Vector((across.dot(vertex), up.dot(vertex))) for vertex in vertices]
    edges = [(start, end - start) for start, end in zip(outline, outline[1:] + outline[:1])]
    return outline, edges, vertices[0]


def bake_placement(scene, papers, seed):
    rng = random.Random(seed)
    for color, paper in papers.items():
        if paper.animation_data and not paper.get("aec_sticker_placement"):
            raise ValueError(f"{paper.name} has a separate object animation")
        bottom_left, bottom_right, top_right, top_left = [v.co.copy() for v in paper.data.vertices]
        center = (bottom_left + top_right) / 2
        across = (bottom_right - bottom_left).normalized()
        up = (top_left - bottom_left).normalized()
        normal = across.cross(up).normalized()
        corners = [v.co.copy() - center for v in paper.data.vertices]
        outline, edges, front_corner = front_outline(paper.parent, center, across, up, normal)
        left, right = min(p.x for p in outline), max(p.x for p in outline)
        low, high = min(p.y for p in outline), max(p.y for p in outline)
        # Keep the paper in front of the permanent artwork and its raised border.
        standoff = normal * (front_corner.dot(normal) + .018)
        for label in ("front marking", "marking border"):
            sign = scene.objects[f"Waste Stickers | {color} {label}"]
            key = "aec_sticker_orbit_sign_offset"
            if key in sign:
                sign.location -= Vector(sign[key])
                del sign[key]
        paper.animation_data_clear()
        paper.rotation_mode = "QUATERNION"
        previous = None
        for frame in range(scene.frame_start, scene.frame_end + 1):
            for _ in range(1000):
                scale, angle = rng.uniform(*PAPER_SCALE), rng.uniform(-8, 8)
                if previous and (abs(scale - previous[2]) < .045 or abs(angle - previous[3]) < 2.5):
                    continue
                rotation = Quaternion(normal, math.radians(angle))
                rotated = [rotation @ (corner * scale) for corner in corners]
                shape = [Vector((across.dot(point), up.dot(point))) for point in rotated]
                u = rng.uniform(left + FACE_MARGIN - min(p.x for p in shape),
                                right - FACE_MARGIN - max(p.x for p in shape))
                v = rng.uniform(low + FACE_MARGIN - min(p.y for p in shape),
                                high - FACE_MARGIN - max(p.y for p in shape))
                if previous and math.hypot(u - previous[0], v - previous[1]) < MIN_POSITION_JUMP:
                    continue
                offset = Vector((u, v))
                if all(edge.cross(point + offset - start) >= FACE_MARGIN * edge.length
                       for start, edge in edges for point in shape):
                    break
            else:
                raise ValueError(f"{paper.name}: no usable placement for frame {frame}")
            placed_center = center + across * u + up * v + standoff
            paper.location = placed_center - rotation @ (center * scale)
            paper.rotation_quaternion = rotation
            paper.scale = (scale,) * 3
            for channel in ("location", "rotation_quaternion", "scale"):
                paper.keyframe_insert(channel, frame=frame, group="Sticker placement")
            previous = (u, v, scale, angle)
        action = paper.animation_data.action
        action.name = f"Waste Stickers | {color} paper placement"
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for key in curve.keyframe_points:
                            key.interpolation = "CONSTANT"
        paper["aec_sticker_placement"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--camera-only", action="store_true", help="save camera motion using the existing sticker frames")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != WORKSPACE_BLEND:
        raise ValueError("Load the unified blend in a background Blender session")
    scene = bpy.data.scenes[STICKER_SCENE]
    bpy.context.window.scene = scene
    papers = sticker_objects(scene)
    if not args.camera_only:
        bake_placement(scene, papers, args.seed)
    store_bin_collections()
    store_orbit_cameras(scene, [paper.parent for paper in papers.values()], "aec_sticker_orbit_target")
    bpy.context.preferences.filepaths.save_version = 0
    if bpy.ops.wm.save_as_mainfile(filepath=str(WORKSPACE_BLEND)) != {"FINISHED"}:
        raise RuntimeError("Could not save the Waste Stickers setup")


if __name__ == "__main__":
    main()
