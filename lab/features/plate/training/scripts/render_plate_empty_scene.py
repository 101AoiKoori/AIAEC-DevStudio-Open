"""Render candidate plate images in a generated empty scene using Cycles GPU."""
import argparse
import json, math, random
from pathlib import Path
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


import sys
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New task-owned dataset directory')
    parser.add_argument('--count', type=int, default=432, help='Number of candidate frames (1..2000)')
    parser.add_argument('--seed', type=int, default=20261113)
    argv = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else (argv or [])
    args = parser.parse_args(argv)
    out = (ROOT / args.output).resolve()
    export_root = (ROOT / 'datasets/exports/plate').resolve()
    if out == export_root or not out.is_relative_to(export_root):
        parser.error('--output must be a new directory inside datasets/exports/plate')
    if out == export_root / 'v4_empty_scene_gpu_20261113':
        parser.error('Existing 108-image batch is reserved; choose a different output')
    if not 1 <= args.count <= 2000:
        parser.error('--count must be between 1 and 2000')
    params = {'count': args.count, 'seed': args.seed, 'resolution': [640, 480],
              'plate_size_m': [0.36, 0.24], 'profiles': ['normal', 'dark', 'low_saturation'],
              'yaw_limit_degrees': 18, 'samples': 16}
    with generation_guard('plate_render_empty_scene:' + out.relative_to(ROOT).as_posix(), [ROOT / "datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png", ROOT / "datasets/raw/site_capture_20261112/videos/full_scene.mp4"], [out], params, code_paths=[Path(__file__)], procedural_inputs=[], root=ROOT) as run:
        if run.reused:
            return
        TEXTURE = ROOT / "datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png"
        OUT = out
        if OUT.exists():
            raise RuntimeError("Output exists; preserve previous run")
        if not TEXTURE.is_file():
            raise RuntimeError("Photographic source frame missing")
        rng = random.Random(args.seed)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene=bpy.context.scene
        scene.render.engine="CYCLES"
        prefs=bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type="OPTIX"
        prefs.get_devices()
        devices=[d for d in prefs.devices if d.type=="OPTIX"]
        if not devices:
            raise RuntimeError("No OPTIX GPU available; CPU fallback disabled")
        for d in prefs.devices:
            d.use=d.type=="OPTIX"
        scene.cycles.device="GPU"
        scene.cycles.samples=16
        scene.cycles.use_denoising=True
        scene.render.resolution_x=640
        scene.render.resolution_y=480
        scene.render.resolution_percentage=100
        scene.render.image_settings.file_format="PNG"
        scene.world.color=(0.12,0.12,0.12)

        def material(name,color):
            m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
            m.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value=(*color,1)
            return m

        def cube(name,loc,scale,mat):
            bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
            o=bpy.context.object; o.name=name; o.scale=scale; o.data.materials.append(mat); return o

        floor=cube("floor",(0,0,-.025),(12,12,.05),material("concrete",(.19,.19,.19)))
        # The support dimensions are illustrative; only the face dimensions were supplied.
        board=cube("support",(0,0,.25),(.44,.025,.50),material("support_gray",(.25,.25,.25)))
        mesh=bpy.data.meshes.new("plate_mesh")
        mesh.from_pydata([(-.18,-.014,.23),(.18,-.014,.23),(.18,-.014,.47),(-.18,-.014,.47)],[],[(0,1,2,3)])
        mesh.update(); uv=mesh.uv_layers.new(name="UVMap")
        for i,xy in zip(mesh.polygons[0].loop_indices,((195/640,91/360),(241/640,91/360),(241/640,109/360),(195/640,109/360))): uv.data[i].uv=xy
        plate=bpy.data.objects.new("plate_36x24cm",mesh);scene.collection.objects.link(plate)
        mat=bpy.data.materials.new("real_plate_crop");mat.use_nodes=True
        nodes=mat.node_tree.nodes; tex=nodes.new("ShaderNodeTexImage");tex.image=bpy.data.images.load(str(TEXTURE)); saturation=nodes.new("ShaderNodeHueSaturation"); mat.node_tree.links.new(tex.outputs["Color"], saturation.inputs["Color"]); mat.node_tree.links.new(saturation.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"]); plate.data.materials.append(mat)
        bpy.ops.object.light_add(type="AREA", location=(0,-3,4));light=bpy.context.object;light.data.energy=550;light.data.shape="DISK";light.data.size=4
        bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera;camera.data.lens=22;camera.data.sensor_width=36
        OUT.mkdir(parents=True);(OUT/"images").mkdir();(OUT/"labels_candidate").mkdir()
        records=[];skipped=[]
        try:
         for i in range(args.count):
            distance=(1.4,1.8,2.2,2.6,3.0,3.5)[(i//3)%6] + rng.uniform(-.06,.06)
            yaw=math.radians((-18,-12,-6,0,6,12,18)[(i//18)%7] + rng.uniform(-.8,.8))
            camera.location=(math.sin(yaw)*distance,-math.cos(yaw)*distance, (.32,.42,.52)[(i//126)%3] + rng.uniform(-.012,.012))
            camera.rotation_euler=(Vector((0,0,.35))-camera.location).to_track_quat("-Z","Y").to_euler()
            profile = ('normal','dark','low_saturation')[i%3]
            light.data.energy={'normal':650,'dark':350,'low_saturation':600}[profile]
            scene.view_settings.exposure={'normal':0.0,'dark':-.7,'low_saturation':0.0}[profile]
            saturation.inputs['Saturation'].default_value = .4 if profile == 'low_saturation' else 1.0
            bpy.context.view_layer.update()
            corners=[world_to_camera_view(scene,camera,plate.matrix_world @ v.co) for v in plate.data.vertices]
            x1,x2=min(v.x for v in corners),max(v.x for v in corners);y1,y2=min(v.y for v in corners),max(v.y for v in corners)
            if any(v.z<=0 for v in corners) or not (0<=x1<x2<=1 and 0<=y1<y2<=1):
                skipped.append({"index":i,"reason":"plate outside frame"});continue
            name="plate_empty_%04d"%i; im=OUT/"images"/(name+".png"); lab=OUT/"labels_candidate"/(name+".txt")
            scene.render.filepath=str(im); bpy.ops.render.render(write_still=True)
            lab.write_text("0 %.8f %.8f %.8f %.8f\n"%((x1+x2)/2,1-(y1+y2)/2,x2-x1,y2-y1),encoding="utf-8")
            records.append({"image":im.relative_to(ROOT).as_posix(),"label":lab.relative_to(ROOT).as_posix(),"split":"unassigned","parent_image":"datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png","source_video":"datasets/raw/site_capture_20261112/videos/full_scene.mp4","source_time_seconds":131.000533,"source_crop_xyxy":[195,251,241,269],"source_crop_basis":"original frame pixels","generation_method":"blender_cycles_optix_empty_scene_real_photo_crop","plate_width_m":.36,"plate_height_m":.24,"support_size":"illustrative","source_crop_px":[46,18],"camera_distance_m":distance,"camera_yaw_degrees":math.degrees(yaw),"review_status":"candidate_needs_visual_label_review","illumination_profile":profile,"scene_exposure":scene.view_settings.exposure,"texture_saturation":saturation.inputs["Saturation"].default_value,"support_dimensions_m":[.44,.025,.50],"support_dimensions_status":"illustrative","texture_mapping":"46x18 photo pixels mapped to 36x24 cm; aspect ratio changes"})
            (OUT/"manifest.json").write_text(json.dumps({"status":"running","parameters":params,"source_refs":run.source_refs,"records":records,"skipped":skipped},ensure_ascii=False,indent=2),encoding="utf-8")
         status="rendered_review_pending"
        except BaseException:
         status="interrupted_or_failed"
         raise
        finally:
         (OUT/"manifest.json").write_text(json.dumps({"status":status,"parameters":params,"source_refs":run.source_refs,"records":records,"skipped":skipped},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == "__main__":
    main()
