"""Inspect the task-owned arena copy in a background Blender process."""

import json

import bpy


def vector(values):
    return [round(float(value), 6) for value in values]


scene = bpy.context.scene
objects = []
for obj in scene.objects:
    row = {
        "name": obj.name,
        "type": obj.type,
        "location": vector(obj.location),
        "rotation_euler": vector(obj.rotation_euler),
        "dimensions": vector(obj.dimensions),
        "hide_render": bool(obj.hide_render),
        "collections": [collection.name for collection in obj.users_collection],
    }
    if obj.type == "CAMERA":
        row["camera"] = {
            "lens": float(obj.data.lens),
            "sensor_width": float(obj.data.sensor_width),
            "type": obj.data.type,
        }
    objects.append(row)

result = {
    "blend_file": bpy.data.filepath,
    "blender_version": bpy.app.version_string,
    "active_camera": scene.camera.name if scene.camera else None,
    "render": {
        "width": scene.render.resolution_x,
        "height": scene.render.resolution_y,
        "percentage": scene.render.resolution_percentage,
        "engine": scene.render.engine,
    },
    "objects": objects,
    "images": [
        {"name": image.name, "path": image.filepath, "packed": bool(image.packed_file)}
        for image in bpy.data.images
    ],
}
print("PLATE_ARENA_INSPECTION=" + json.dumps(result, ensure_ascii=False))
