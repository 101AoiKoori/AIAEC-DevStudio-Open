"""Build an editable, approximate four-bin Blender scene and preview it.

Run: blender --background --factory-startup --python-exit-code 1 --python
     lab/features/waste_bin/training/scripts/build_blender_waste_bin.py -- OUTPUT_DIR
One scene unit represents 10 cm. Width and height follow user measurements;
depth, lid details and camera distance remain approximate.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PART_Z_SCALE = 1.2


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def cube(name, location, size, mat, bevel=0.012, parent=None):
    location=(location[0],location[1],location[2]*PART_Z_SCALE)
    size=(size[0],size[1],size[2]*PART_Z_SCALE)
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new('Rounded molded edges', 'BEVEL')
        mod.width, mod.segments = bevel, 3
        obj.modifiers.new('Weighted normals', 'WEIGHTED_NORMAL')
    obj.data.materials.append(mat)
    if parent:
        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


def tube(name, points, radius, mat):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new('POLY')
    spline.points.add(len(points)-1)
    for point, coord in zip(spline.points, points):
        point.co = (coord[0],coord[1],coord[2]*PART_Z_SCALE,1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def rounded_rectangle(cx, cy, half_w, half_d, radius, z):
    points=[]
    for px,py,start in ((cx+half_w-radius,cy+half_d-radius,0),
                        (cx-half_w+radius,cy+half_d-radius,90),
                        (cx-half_w+radius,cy-half_d+radius,180),
                        (cx+half_w-radius,cy-half_d+radius,270)):
        for step in range(9):
            angle=math.radians(start+step*90/8)
            points.append((px+radius*math.cos(angle),py+radius*math.sin(angle),z))
    return points


def rim_band(x, mat, name):
    """A flat, rounded flange with a real opening and editable modifiers."""
    outer=rounded_rectangle(x,0,.425,.40,.065,.925)
    inner=rounded_rectangle(x,0,.35,.34,.05,.925)
    count=len(outer)
    faces=[(i,(i+1)%count,(i+1)%count+count,i+count) for i in range(count)]
    mesh=bpy.data.meshes.new(name+' flange mesh')
    mesh.from_pydata([(a,b,c*PART_Z_SCALE) for a,b,c in outer+inner],[],faces)
    mesh.update()
    obj=bpy.data.objects.new(name+' flat rounded flange',mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    solid=obj.modifiers.new('Molded lip thickness','SOLIDIFY')
    solid.thickness=.033
    bevel=obj.modifiers.new('Lip edge radius','BEVEL')
    bevel.width=.008; bevel.segments=3
    obj.modifiers.new('Lip weighted normals','WEIGHTED_NORMAL')
    return obj


def shell(x, mat, name):
    # Open-top tapered shell: outer wall, rim, inner wall and floor.
    def ring(w, d, z):
        return [(x-w, -d, z*PART_Z_SCALE), (x+w, -d, z*PART_Z_SCALE),
                (x+w, d, z*PART_Z_SCALE), (x-w, d, z*PART_Z_SCALE)]
    verts = ring(.30,.28,.03)+ring(.375,.37,.92)+ring(.35,.34,.92)+ring(.275,.25,.07)
    faces = []
    for i in range(3):
        for j in range(4):
            a=i*4+j; b=i*4+(j+1)%4
            faces.append((a,b,b+4,a+4))
    faces.append((12,13,14,15))
    mesh=bpy.data.meshes.new(name+' shell')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj=bpy.data.objects.new(name+' body',mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    mod=obj.modifiers.new('Soft plastic corners','BEVEL')
    mod.width=.018; mod.segments=3
    obj.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    rim_band(x,mat,name)
    outline=rounded_rectangle(x,0,.353,.343,.05,.932)
    tube(name+' inner rim bead',outline+[outline[0]],.006,mat)


def create_bin(x, name, color, black):
    mat=material(name+' adjustable body color',color,.28)
    shell(x,mat,name)
    tube(name+' hinge axle',[(x-.30,.43,.95),(x+.30,.43,.95)],.025,mat)
    pivot=bpy.data.objects.new(name+' lid pivot - rotate X',None)
    bpy.context.collection.objects.link(pivot)
    pivot.location=(x,.43,.96*PART_Z_SCALE)
    bpy.context.view_layer.update()
    # Heights include the 1.2 construction scale and equal 1 cm and 0.5 cm.
    cube(name+' lid',(x,0,.976667),(.75,.78,.083333),black,.035,pivot)
    cube(name+' raised panel',(x,-.015,1.039167),(.55,.52,.041667),black,.018,pivot)
    cube(name+' front tab',(x,-.405,.951),(.20,.07,.025),black,.012,pivot)
    lid_seam=rounded_rectangle(x,0,.368,.38,.055,.938)
    seam=tube(name+' lid lower seam',lid_seam+[lid_seam[0]],.004,black)
    seam.parent=pivot
    seam.matrix_parent_inverse=pivot.matrix_world.inverted()
    for dx in (-.22,.22):
        cube(name+' hinge strap',(x+dx,.37,.98),(.10,.18,.07),black,.018,pivot)
    for dx in (-.31,0,.31):
        cube(name+' hinge support',(x+dx,.416,.91),(.035,.13,.11),mat,.012)
    for side in (-1,1):
        for yy in (-.18,.14):
            cube(name+' side lip clip',(x+side*.423,yy,.90),(.035,.12,.055),mat,.009)
    return pivot


def add_decal(x, name, path):
    """Place a packed transparent marking on the tapered front surface."""
    image=bpy.data.images.load(str(path.resolve()),check_existing=True)
    image.pack()
    mat=bpy.data.materials.new(name+' photographed marking')
    mat.use_nodes=True
    nodes=mat.node_tree.nodes
    shader=nodes.get('Principled BSDF')
    texture=nodes.new('ShaderNodeTexImage')
    texture.image=image
    mat.node_tree.links.new(texture.outputs['Color'],shader.inputs['Base Color'])
    mat.node_tree.links.new(texture.outputs['Alpha'],shader.inputs['Alpha'])
    shader.inputs['Roughness'].default_value=.55
    def front(z):
        return -(.28+(.37-.28)*(z-.036)/(1.104-.036))-.006
    low,high=.30,.74
    verts=[(x-.18,front(low),low),(x+.18,front(low),low),
           (x+.18,front(high),high),(x-.18,front(high),high)]
    mesh=bpy.data.meshes.new(name+' decal mesh')
    mesh.from_pydata(verts,[],[(0,1,2,3)])
    mesh.update()
    uv=mesh.uv_layers.new(name='marking UV')
    for loop,coordinate in zip(mesh.polygons[0].loop_indices,[(0,0),(1,0),(1,1),(0,1)]):
        uv.data[loop].uv=coordinate
    obj=bpy.data.objects.new(name+' front marking',mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    border_material=bpy.data.materials.get('Marking border')
    if border_material is None:
        border_material=material('Marking border',(.92,.92,.90),.55)
    bottom,top=.285,.755
    border=tube(name+' marking border',[(x-.19,front(bottom)-.002,bottom/PART_Z_SCALE),
                (x+.19,front(bottom)-.002,bottom/PART_Z_SCALE),
                (x+.19,front(top)-.002,top/PART_Z_SCALE),
                (x-.19,front(top)-.002,top/PART_Z_SCALE),
                (x-.19,front(bottom)-.002,bottom/PART_Z_SCALE)],.003,border_material)
    border.data.resolution_u=1


def look_at(obj, target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def _generate():
    global PART_Z_SCALE
    parser=argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--decals',type=Path)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    black=material('Black lid - editable',(.012,.014,.018),.24)
    pivots=[]
    for i,(name,col) in enumerate([('blue',(.012,.07,.8)),('red',(.8,.025,.012)),
                                   ('green',(.02,.55,.035)),('gray',(.32,.34,.32))]):
        pivots.append(create_bin((i-1.5)*1.12,name,col,black))
        if args.decals is not None:
            add_decal((i-1.5)*1.12,name,args.decals/(name+'.png'))
    # Reference width ~8.5 cm and height ~11 cm; scene unit is 10 cm.
    root=bpy.data.objects.new('Measured height scale',None)
    bpy.context.collection.objects.link(root)
    for obj in list(bpy.context.scene.objects):
        if obj != root and obj.parent is None:
            obj.parent=root
    PART_Z_SCALE=1.0
    floor=material('Floor',(.115,.12,.13),.58)
    cube('Ground',(0,0,-.06),(20,20,.1),floor,.01)
    markings=material('Road marking',(.75,.75,.73),.7)
    tube('Front road boundary',[(-7,-2.1,-.004),(-3,-2.1,-.004),
                                (0,-2.0,-.004),(3,-2.1,-.004),(7,-2.1,-.004)],
         .018,markings)
    wall=material('Backdrop',(.32,.34,.35),.8)
    cube('Background wall',(0,4,2.0),(20,.1,4),wall,.01)
    for name,loc,power,size in [('Key',(-3,-4,6),900,5),('Fill',(4,-1,4),500,4)]:
        bpy.ops.object.light_add(type='AREA',location=loc)
        lamp=bpy.context.object; lamp.name=name; lamp.data.energy=power; lamp.data.shape='DISK'; lamp.data.size=size
        look_at(lamp,(0,0,.5))
    bpy.ops.object.camera_add(location=(0,-5.4,1.35))
    camera=bpy.context.object; camera.name='Front scan camera'; camera.data.lens=35
    bpy.context.scene.camera=camera
    # Smooth straight pass; yaw offsets can be layered by editing Z rotation.
    for frame,x in [(1,-1.4),(60,0),(120,1.4)]:
        camera.location=(x,-5.4,1.35)
        look_at(camera,(x,0,.78))
        camera.keyframe_insert('location',frame=frame)
        camera.keyframe_insert('rotation_euler',frame=frame)
    scene=bpy.context.scene
    if args.decals is not None and (args.decals/'attribution.json').is_file():
        scene['decal_attribution']=json.dumps(json.loads((args.decals/'attribution.json').read_text(encoding='utf8')),
                                                ensure_ascii=False)
    scene.frame_end=120
    scene.render.engine='CYCLES'; scene.cycles.samples=24
    scene.render.resolution_x=scene.render.resolution_y=1024
    scene.render.resolution_percentage=100
    scene.world.color=(.16,.16,.16)
    scene.frame_set(60)
    for i,p in enumerate(pivots):
        p.rotation_euler.x=math.radians(-270 if i%2==0 else 0)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'waste_bin_reference.blend'))
    scene.render.image_settings.file_format='PNG'
    scene.render.filepath=str(out/'preview.png')
    bpy.ops.render.render(write_still=True)
    # Close inspection render; the saved scene retains its scan animation.
    camera.animation_data_clear()
    camera.location=(-.40,-2.7,1.75)
    camera.data.lens=48
    look_at(camera,(-.56,0,1.05))
    scene.render.filepath=str(out/'preview_detail.png')
    bpy.ops.render.render(write_still=True)


def main():
    raw=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--decals',type=Path)
    args,_=parser.parse_known_args(raw)
    if bpy.data.is_dirty:
        raise ValueError('save or discard scene changes before producing a guarded copy')
    workspace=Path(__file__).resolve().parents[5]
    output=Path(args.output).resolve()
    media=set() if args.decals is None else {(args.decals/(name+'.png')).resolve()
                                              for name in ('blue','red','green','gray')}
    for image in bpy.data.images:
        if image.source == 'FILE':
            if not image.filepath:
                raise ValueError('scene image source path is missing')
            path=Path(bpy.path.abspath(image.filepath)).resolve()
            if not path.is_file():
                raise ValueError('scene image source missing: '+str(path))
            media.add(path)
    attribution=[]
    if args.decals is not None and (args.decals/'attribution.json').is_file():
        attribution=[(args.decals/'attribution.json').resolve()]
    if bpy.data.filepath:
        attribution.append(Path(bpy.data.filepath).resolve())
        scene_source='loaded_blend'
    else:
        if '--factory-startup' not in sys.argv:
            raise ValueError('an untitled scene requires --factory-startup for a reproducible source')
        scene_source='factory_startup'
    sys.path.insert(0,str(workspace))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin.build_scene:'+str(output.relative_to(workspace)),
                          sorted(media),[output],{'argv':raw,'blender_version':bpy.app.version_string,
                                                 'scene_source':scene_source},
                          code_paths=[Path(__file__).resolve()],
                          procedural_inputs=attribution or [Path(__file__).resolve()],root=workspace) as run:
        if run.reused:
            return
        _generate()


if __name__=='__main__':
    main()
