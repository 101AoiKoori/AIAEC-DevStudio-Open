"""Render a small unapproved plate-standee pilot from a task-owned scene copy.

Run only in a background Blender process with --threads 2 and a compatible GPU.  The scene is never
saved.  A successful render is a review candidate, not a training label.
"""

import json
import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector



import sys
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from lab.features.plate.training.scripts.plate_guard_inputs import blender_external_dependencies

def main():
    scene_dependencies = blender_external_dependencies(bpy) - {
        ROOT / "datasets/exports/plate/v4_far_composite_pilot_20261113/cutout.png",
        ROOT / "datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png",
        ROOT / "datasets/raw/site_capture_20261112/videos/full_scene.mp4",
        ROOT / "datasets/assets/blender/plate/v4_arena_vehicle_20261113/arena_plate_working.blend",
    }
    with generation_guard('plate_render_plate_arena_pilot', [ROOT / "datasets/exports/plate/v4_far_composite_pilot_20261113/cutout.png", ROOT / "datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png", ROOT / "datasets/raw/site_capture_20261112/videos/full_scene.mp4", ROOT / "datasets/assets/blender/plate/v4_arena_vehicle_20261113/arena_plate_working.blend"] + sorted(scene_dependencies), [ROOT / "datasets/exports/plate/v4_arena_pilot_20261113"], {"script": 'render_plate_arena_pilot.py'}, code_paths=[Path(__file__), ROOT / "lab/features/plate/training/scripts/plate_guard_inputs.py"], procedural_inputs=[ROOT / "datasets/assets/blender/plate/v4_arena_vehicle_20261113/arena_plate_working.blend"] + sorted(scene_dependencies), root=ROOT) as run:
        if run.reused:
            return
        WORKING = ROOT / "datasets/assets/blender/plate/v4_arena_vehicle_20261113/arena_plate_working.blend"
        CUTOUT = ROOT / "datasets/exports/plate/v4_far_composite_pilot_20261113/cutout.png"
        OUTPUT = ROOT / "datasets/exports/plate/v4_arena_pilot_20261113"
        SOURCE = "datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png"
        SOURCE_BOX = (65.0, 56.0, 111.0, 74.0)
        IMAGE_WIDTH, IMAGE_HEIGHT = 175.0, 130.0
        ALPHA_BOTTOM = 122.0


        def relative(path):
            return path.relative_to(ROOT).as_posix()


        def projected_box(scene, camera, origin, right, width, ground, height):
            points = []
            for x in (SOURCE_BOX[0], SOURCE_BOX[2]):
                for y in (SOURCE_BOX[1], SOURCE_BOX[3]):
                    point = origin + right * ((x / IMAGE_WIDTH - 0.5) * width)
                    point.z = ground + (ALPHA_BOTTOM - y) / IMAGE_HEIGHT * height
                    points.append(world_to_camera_view(scene, camera, point))
            if any(p.z <= 0 for p in points):
                return None
            x1, x2 = min(p.x for p in points), max(p.x for p in points)
            y1, y2 = min(p.y for p in points), max(p.y for p in points)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                return None
            return (x1, y1, x2, y2)


        if Path(bpy.data.filepath).resolve() != WORKING.resolve():
            raise RuntimeError("Open only the task-owned arena copy")
        if OUTPUT.exists():
            raise RuntimeError("Pilot destination already exists; do not overwrite")
        if not CUTOUT.is_file():
            raise RuntimeError("Vehicle cutout is missing")

        scene = bpy.context.scene
        camera = scene.camera
        if camera is None or camera.type != "CAMERA":
            raise RuntimeError("Active camera is missing")
        scene.render.resolution_x, scene.render.resolution_y = 640, 480
        scene.render.resolution_percentage = 100
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 2
        scene.render.engine = "CYCLES"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                preferences.compute_device_type = backend
                preferences.get_devices()
                gpu_devices = [d for d in preferences.devices if d.type != "CPU"]
                if gpu_devices:
                    for device in preferences.devices:
                        device.use = device.type != "CPU"
                    break
            except (TypeError, ValueError):
                continue
        else:
            raise RuntimeError("No compatible GPU compute device available; no CPU fallback")
        scene.cycles.device = "GPU"
        scene.cycles.samples = 16
        scene.render.image_settings.file_format = "PNG"

        material = bpy.data.materials.new("plate_pilot_real_frame_cutout")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        shader = nodes.get("Principled BSDF")
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(str(CUTOUT), check_existing=True)
        material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        material.node_tree.links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "BLEND"

        forward = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        forward.z = 0
        if forward.length < 1e-6:
            raise RuntimeError("Camera horizontal direction is undefined")
        forward.normalize()
        camera_right = Vector((forward.y, -forward.x, 0)).normalized()
        camera_position = camera.matrix_world.translation.copy()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        base_exposure = scene.view_settings.exposure

        OUTPUT.mkdir(parents=True)
        (OUTPUT / "images").mkdir()
        (OUTPUT / "labels_candidate").mkdir()
        records = []
        try:
            for index, (yaw_degrees, target_width, dark) in enumerate(
                ((0, 26, False), (-15, 30, False), (15, 35, True), (0, 39, True))
            ):
                angle = math.radians(yaw_degrees)
                right = Vector((
                    camera_right.x * math.cos(angle) - camera_right.y * math.sin(angle),
                    camera_right.x * math.sin(angle) + camera_right.y * math.cos(angle),
                    0,
                ))
                options = []
                for distance in (1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8):
                    pos = camera_position + forward * distance
                    if abs(pos.x) > 1.95 or abs(pos.y) > 1.95:
                        continue
                    hit, point, normal, _, hit_object, _ = scene.ray_cast(
                        depsgraph, Vector((pos.x, pos.y, 2.0)), Vector((0, 0, -1)), distance=3.0
                    )
                    if not hit or normal.z < 0.9 or abs(point.z) > 0.1:
                        continue
                    for width in (0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58):
                        height = width * IMAGE_HEIGHT / IMAGE_WIDTH
                        origin = Vector((pos.x, pos.y, point.z))
                        box = projected_box(scene, camera, origin, right, width, point.z, height)
                        if box is None:
                            continue
                        pixels = (box[2] - box[0]) * 640
                        if 25 <= pixels <= 40:
                            options.append((abs(pixels - target_width), distance, width, point.z, box, origin, hit_object.name))
                if not options:
                    raise RuntimeError("No fitted ground placement for pilot %d" % index)
                _, distance, width, ground, box, origin, ground_object = min(options, key=lambda x: x[0])
                height = width * IMAGE_HEIGHT / IMAGE_WIDTH
                bottom = ground - (IMAGE_HEIGHT - ALPHA_BOTTOM) / IMAGE_HEIGHT * height
                left = origin - right * (width / 2)
                right_point = origin + right * (width / 2)
                vertices = [(left.x, left.y, bottom), (right_point.x, right_point.y, bottom),
                            (right_point.x, right_point.y, bottom + height), (left.x, left.y, bottom + height)]
                mesh = bpy.data.meshes.new("plate_pilot_mesh_%02d" % index)
                mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
                mesh.update()
                uv = mesh.uv_layers.new(name="UVMap")
                for loop, coord in zip(mesh.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
                    uv.data[loop].uv = coord
                obj = bpy.data.objects.new("plate_pilot_%02d" % index, mesh)
                scene.collection.objects.link(obj)
                obj.data.materials.append(material)
                scene.view_settings.exposure = base_exposure - (0.7 if dark else 0)
                name = "plate_arena_pilot_%02d" % index
                image = OUTPUT / "images" / (name + ".png")
                label = OUTPUT / "labels_candidate" / (name + ".txt")
                scene.render.filepath = str(image)
                bpy.ops.render.render(write_still=True)
                x1, y1, x2, y2 = box
                label.write_text("0 %.8f %.8f %.8f %.8f\n" % (
                    (x1 + x2) / 2, 1 - (y1 + y2) / 2, x2 - x1, y2 - y1
                ), encoding="utf-8")
                records.append({
                    "image": relative(image), "label": relative(label), "split": "unassigned",
                    "source_frame": SOURCE, "source_time_seconds": 131.000533,
                    "source_crop": relative(CUTOUT), "source_plate_box_in_crop": SOURCE_BOX,
                    "arena_copy": relative(WORKING), "camera": camera.name,
                    "camera_matrix_world": [list(row) for row in camera.matrix_world],
                    "resolution": [640, 480], "ground_object": ground_object,
                    "ground_point": [origin.x, origin.y, ground], "distance_fitted": distance,
                    "standee_width_fitted": width, "yaw_degrees": yaw_degrees,
                    "plate_width_px_projected": (x2 - x1) * 640,
                    "exposure": scene.view_settings.exposure, "occlusion": "not_verified",
                    "review_status": "pilot_unreviewed_not_trainable",
                })
                bpy.data.objects.remove(obj, do_unlink=True)
            status = "pilot_rendered_review_pending"
        except BaseException as exc:
            status = "failed_or_interrupted"
            (OUTPUT / "FAILED.txt").write_text(str(exc), encoding="utf-8")
            raise
        finally:
            (OUTPUT / "manifest.json").write_text(json.dumps({
                "status": status, "source_refs": run.source_refs, "records": records,
                "scope": "Fitted scene and real-frame cutout; physical size is not measured. "
                         "Cutout edge, ground contact, visibility, lighting and labels require visual review.",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
