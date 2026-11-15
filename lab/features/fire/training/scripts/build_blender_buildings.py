"""Build three editable, approximate competition building references in Blender.

Run from the workspace root with Blender 5.2::

    blender --background --factory-startup --python-exit-code 1 \
      --python lab/features/fire/training/scripts/build_blender_buildings.py -- \
      --font datasets/assets/blender/fire/blender_buildings_v13/NotoSansSC.ttf \
      --output datasets/assets/blender/fire/blender_buildings_new --seed 431

Physical dimensions are estimates derived from photographs and the user's
average building dimensions.  This is a research reference, not a measured CAD
survey or a labelled training dataset. Existing outputs are never overwritten.
"""
import argparse
import math
import random
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


def parse_args():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--font', required=True, help='workspace-local TTF with sibling OFL.txt')
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--rect-depth', type=float, default=.125,
                        help='A/B depth in metres; photo-derived visual estimate, not a measurement')
    parsed = parser.parse_args(args)
    parsed.output = (ROOT / parsed.output).resolve()
    parsed.font = (ROOT / parsed.font).resolve()
    if not parsed.output.is_relative_to(ROOT / 'datasets' / 'assets' / 'blender' / 'fire'):
        parser.error('output must be under datasets/assets/blender/fire')
    if not 1 <= parsed.samples <= 128:
        parser.error('samples must be between 1 and 128')
    if not .05 <= parsed.rect_depth <= .35:
        parser.error('rect-depth must be between 0.05 and 0.35 m')
    if not parsed.font.is_relative_to(ROOT) or not parsed.font.is_file():
        parser.error('font must be an existing workspace-local file')
    if not (parsed.font.parent / 'OFL.txt').is_file():
        parser.error('the font license OFL.txt must be beside the font')
    return parsed


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def paper_material(name, image=None):
    mat = material(name, (0.88, 0.87, 0.82), 0.86)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Thin Wall'].default_value = True
    bsdf.inputs['Transmission Weight'].default_value = 0.035
    bsdf.inputs['Subsurface Weight'].default_value = 0.015
    noise = nodes.new('ShaderNodeTexNoise')
    noise.label = 'Fine paper fibers (procedural)'
    noise.inputs['Scale'].default_value = 620
    noise.inputs['Detail'].default_value = 2.5
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.12
    bump.inputs['Distance'].default_value = 0.000018
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    rough=nodes.new('ShaderNodeMapRange')
    rough.label='Small paper roughness variation'
    rough.inputs['To Min'].default_value=.80
    rough.inputs['To Max'].default_value=.89
    links.new(noise.outputs['Fac'],rough.inputs['Value'])
    links.new(rough.outputs['Result'],bsdf.inputs['Roughness'])
    if image:
        texture = nodes.new('ShaderNodeTexImage')
        texture.label = 'Provided printed flame artwork'
        texture.image = image
        # Fine, low-contrast regular print dots are an approximation informed
        # by the supplied close photo, not a measured screen frequency.
        uv=nodes.new('ShaderNodeTexCoord')
        rotate=nodes.new('ShaderNodeVectorRotate')
        rotate.rotation_type='EULER_XYZ'
        rotate.inputs['Center'].default_value=(.5,.5,0)
        rotate.inputs['Rotation'].default_value=(0,0,math.radians(25))
        links.new(uv.outputs['UV'],rotate.inputs['Vector'])
        waves=[]
        for axis in ('X','Y'):
            wave=nodes.new('ShaderNodeTexWave')
            wave.wave_type='BANDS'; wave.bands_direction=axis
            wave.inputs['Scale'].default_value=60
            links.new(rotate.outputs['Vector'],wave.inputs['Vector'])
            waves.append(wave)
        product=nodes.new('ShaderNodeMath'); product.operation='MULTIPLY'
        links.new(waves[0].outputs['Fac'],product.inputs[0])
        links.new(waves[1].outputs['Fac'],product.inputs[1])
        dots=nodes.new('ShaderNodeMath'); dots.operation='GREATER_THAN'
        dots.inputs[1].default_value=.78
        links.new(product.outputs[0],dots.inputs[0])
        hsv=nodes.new('ShaderNodeSeparateColor'); hsv.mode='HSV'
        links.new(texture.outputs['Color'],hsv.inputs['Color'])
        mask=nodes.new('ShaderNodeMath'); mask.operation='MULTIPLY'
        links.new(dots.outputs[0],mask.inputs[0])
        # In Blender 5.2 this socket retains the identifier 'Green' in HSV mode.
        links.new(hsv.outputs[1],mask.inputs[1])
        strength=nodes.new('ShaderNodeMath'); strength.operation='MULTIPLY'
        strength.inputs[1].default_value=.20
        links.new(mask.outputs[0],strength.inputs[0])
        printed=nodes.new('ShaderNodeMixRGB'); printed.blend_type='MULTIPLY'
        printed.label='Subtle printed ink dot structure'
        printed.inputs[2].default_value=(.65,.65,.65,1)
        links.new(strength.outputs[0],printed.inputs[0])
        links.new(texture.outputs['Color'],printed.inputs[1])
        links.new(printed.outputs['Color'],bsdf.inputs['Base Color'])
    return mat


def mesh_object(name, vertices, faces, collection, mat):
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def box_mesh(vertices, faces, center, dimensions):
    x, y, z = center
    dx, dy, dz = (v / 2 for v in dimensions)
    i = len(vertices)
    vertices.extend([(x-dx,y-dy,z-dz),(x+dx,y-dy,z-dz),
                     (x+dx,y+dy,z-dz),(x-dx,y+dy,z-dz),
                     (x-dx,y-dy,z+dz),(x+dx,y-dy,z+dz),
                     (x+dx,y+dy,z+dz),(x-dx,y+dy,z+dz)])
    faces.extend([tuple(i+j for j in f) for f in ((0,3,2,1),(4,5,6,7),
                  (0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))])


def box(name, collection, mat, center, dimensions):
    vertices, faces = [], []
    box_mesh(vertices, faces, center, dimensions)
    return mesh_object(name, vertices, faces, collection, mat)


def wall(name, collection, mat, start, end, holes, height=.4, thickness=.0016):
    """Generate a wall with actual rectangular apertures, not dark textures."""
    dx, dy = end[0]-start[0], end[1]-start[1]
    width = math.hypot(dx, dy)
    tx, ty = dx / width, dy / width
    nx, ny = -ty, tx
    holes = [(max(0,a), min(width,b), max(0,c), min(height,d))
             for a,b,c,d in holes if b>a and d>c]
    xs = sorted({0., width, *(v for a,b,c,d in holes for v in (a,b))})
    zs = sorted({0., height, *(v for a,b,c,d in holes for v in (c,d))})
    vertices, faces = [], []
    for x0,x1 in zip(xs,xs[1:]):
        for z0,z1 in zip(zs,zs[1:]):
            xm,zm=(x0+x1)/2,(z0+z1)/2
            if any(a < xm < b and c < zm < d for a,b,c,d in holes):
                continue
            i=len(vertices)
            for n in (-thickness/2, thickness/2):
                for u,z in ((x0,z0),(x1,z0),(x1,z1),(x0,z1)):
                    vertices.append((start[0]+tx*u+nx*n,start[1]+ty*u+ny*n,z))
            faces.extend([tuple(i+j for j in f) for f in
                          ((0,3,2,1),(4,5,6,7),(0,1,5,4),
                           (1,2,6,5),(2,3,7,6),(3,0,4,7))])
    return mesh_object(name, vertices, faces, collection, mat)


def split_window(holes, bars, x0, x1, z0, z1, divisions=2):
    holes.append((x0,x1,z0,z1))
    # narrow physical frame bars occupy the aperture itself
    for i in range(1, divisions):
        x=x0+(x1-x0)*i/divisions
        bars.append((x-.00065,x+.00065,z0,z1))
    bars.append((x0,x1,z0+(z1-z0)*.67-.00055,z0+(z1-z0)*.67+.00055))


def apply_bars(name, collection, mat, start, end, bars):
    dx,dy=end[0]-start[0],end[1]-start[1]
    w=math.hypot(dx,dy); tx,ty=dx/w,dy/w
    vertices, faces = [], []
    for a,b,c,d in bars:
        u=(a+b)/2
        center=(start[0]+tx*u,start[1]+ty*u,(c+d)/2)
        if abs(tx)>.5:
            size=(b-a,.0018,d-c)
        else:
            size=(.0018,b-a,d-c)
        box_mesh(vertices,faces,center,size)
    if vertices:
        mesh_object(name, vertices, faces, collection, mat)


def roof_and_base(coll, mat, x, is_l=False):
    if is_l:
        box('Roof_long',coll,mat,(x,-.052,.399),(.2,.096,.002))
        box('Roof_wing',coll,mat,(x+.054,.048,.399),(.092,.104,.002))
    else:
        box('Roof',coll,mat,(x,0,.399),(.2,.2,.002))
    # fine mould lines in the actual roof panels
    for i in range(1,10):
        xx=x-.1+.02*i
        if is_l and xx < x+.008:
            y0,y1=-.1,-.004
        else:
            y0,y1=-.1,.1
        box('Roof_grid_%02d'%i,coll,mat,(xx,(y0+y1)/2,.4002),(.00025,y1-y0,.00025))
    for i in range(1,10):
        yy=-.1+.02*i
        x0 = x+.008 if is_l and yy > -.004 else x-.1
        box('Roof_cross_grid_%02d'%i,coll,mat,((x0+x+.1)/2,yy,.4002),(x+.1-x0,.00025,.00025))


def building_c(x, coll, mat):
    # Morning: front of L is a wall with small square openings.
    holes=[]
    for row in range(7):
        z=.043+row*.05
        for u in (.025,.073,.091,.109,.157):
            holes.append((u,u+.009,z,z+.009))
        holes.append((.091,.100,z+.012,z+.021))
    wall('C_front_small_holes',coll,mat,(x-.1,-.1),(x+.1,-.1),holes)
    seam=material('C moulded panel seams',(.46,.46,.42),.8)
    vertices,faces=[],[]
    for row in range(1,7):
        box_mesh(vertices,faces,(x,-.10087,row*.05),(.2,.00012,.00012))
    mesh_object('C front horizontal panel seams',vertices,faces,coll,seam)
    # The photographed right face has six rows of one three-pane window.
    holes=[]; bars=[]
    for j in range(6):
        z=.073+j*.05
        holes.append((.067,.133,z,z+.020))
        for u in (.089,.111):
            bars.append((u-.0006,u+.0006,z,z+.020))
    wall('C_right_outer',coll,mat,(x+.1,-.1),(x+.1,.1),holes)
    apply_bars('C_right_three_pane_mullions',coll,mat,(x+.1,-.1),(x+.1,.1),bars)
    # Inscribed double door below the right windows, kept shut like the photo.
    for y in (-.024,0,.024):
        box('C_right_door_vertical',coll,seam,(x+.10088,y,.029),(.00012,.00012,.039))
    for z in (.0095,.0485):
        box('C_right_door_horizontal',coll,seam,(x+.10088,0,z),(.00012,.048,.00012))
    for name,start,end,windows in [
        ('C_back_wing',(x+.1,.1),(x+.008,.1),[(.014,.038),(.052,.076)]),
        ('C_inner_wing',(x+.008,.1),(x+.008,-.004),[(.012,.034),(.056,.078)])]:
        holes=[]; bars=[]
        for j in range(6):
            for a,b in windows:
                split_window(holes,bars,a,b,.05+j*.05,.08+j*.05,2)
        wall(name,coll,mat,start,end,holes)
        apply_bars(name+'_mullions',coll,mat,start,end,bars)
    holes=[]; bars=[]
    split_window(holes,bars,.018,.096,.004,.041,6)
    for j in range(6):
        split_window(holes,bars,.014,.10,.06+j*.05,.087+j*.05,6)
    wall('C_inner_long',coll,mat,(x+.008,-.004),(x-.1,-.004),holes)
    apply_bars('C_inner_mullions',coll,mat,(x+.008,-.004),(x-.1,-.004),bars)
    canopy=box('C inner entrance canopy',coll,mat,(x-.046,.003,.048),(.102,.014,.0028))
    bevel=canopy.modifiers.new('Small rounded canopy edge','BEVEL')
    bevel.width=.0012; bevel.segments=3
    wall('C_left_outer',coll,mat,(x-.1,-.004),(x-.1,-.1),[(.016+i*.034,.025+i*.034,.05+j*.05,.059+j*.05) for i in range(2) for j in range(7)])
    roof_and_base(coll,mat,x,True)


def building_b(x, coll, mat):
    holes=[]; bars=[]
    for row in range(5):
        z=.080+row*.052
        for u in (.015,.049,.128,.163):
            split_window(holes,bars,u,u+.023,z,z+.027,2)
    holes.extend([(.083,.117,.05,.34),(.008,.192,.341,.376)])
    for a,b in ((.015,.055),(.075,.125),(.145,.185)):
        split_window(holes,bars,a,b,.006,.038,3)
    for z in [.052+i*.017 for i in range(18)]:
        bars.append((.083,.117,z,z+.0012))
    for u in [.09+i*.008 for i in range(4)]:
        bars.append((u,u+.0012,.05,.34))
    for u in [.015+i*.011 for i in range(17)]:
        bars.append((u,u+.0012,.341,.376))
    bars.append((.008,.192,.359,.360))
    wall('B_front_glazed_openings',coll,mat,(x-.1,-.1),(x+.1,-.1),holes)
    apply_bars('B_front_window_and_stair_mullions',coll,mat,(x-.1,-.1),(x+.1,-.1),bars)
    for name,start,end in [('left',(x-.1,-.1),(x-.1,.1)),('right',(x+.1,-.1),(x+.1,.1)),('back',(x-.1,.1),(x+.1,.1))]:
        holes=[(.083,.117,.05,.34),(.008,.192,.341,.376),(.075,.125,.005,.038)]
        bars=[(.083,.117,.052+i*.017,.0532+i*.017) for i in range(17)]
        bars += [(.09+i*.008,.0912+i*.008,.05,.34) for i in range(4)]
        bars += [(.015+i*.011,.0162+i*.011,.341,.376) for i in range(17)]
        bars.append((.008,.192,.359,.360))
        wall('B_'+name,coll,mat,start,end,holes)
        apply_bars('B_'+name+'_mullions',coll,mat,start,end,bars)
    roof_and_base(coll,mat,x)


def building_a(x, coll, mat):
    holes=[]; bars=[]
    # Horizontal window bands and side gaps; central thin horizontal mullions.
    for row in range(7):
        z=.064+row*.045
        for a,b in ((.012,.045),(.050,.080),(.084,.116),(.120,.150),(.155,.188)):
            holes.append((a,b,z,z+.011))
            for k in range(1,4):
                u=a+(b-a)*k/4
                bars.append((u-.0005,u+.0005,z,z+.011))
        bars.extend([(.084,.116,z+.003,z+.004),
                     (.084,.116,z+.007,z+.008)])
        # Long narrow apertures down the two ends between the window bands.
        if row < 6:
            holes.extend([(.008,.019,z+.012,z+.043),
                          (.181,.192,z+.012,z+.043)])
    # Two rows of three narrow four-pane groups above the entrance.
    for z in (.041,.051):
        for a,b in ((.015,.065),(.075,.125),(.135,.185)):
            holes.append((a,b,z,z+.005))
            for i in range(1,4):
                u=a+(b-a)*i/4
                bars.append((u-.00045,u+.00045,z,z+.005))
    holes.append((.025,.175,.004,.037))
    wall('A_front_window_bands',coll,mat,(x-.1,-.1),(x+.1,-.1),holes)
    apply_bars('A_front_window_mullions',coll,mat,(x-.1,-.1),(x+.1,-.1),bars)
    # Slightly inset moulded horizontal grooves between the major window bands.
    seam=material('A fine panel seams',(.40,.40,.37),.82)
    vertices,faces=[],[]
    for row in range(6):
        for sub in range(1,5):
            z=.075+row*.045+sub*.006
            box_mesh(vertices,faces,(x,-.10087,z),(.161,.00012,.00014))
    mesh_object('A front horizontal panel grooves',vertices,faces,coll,seam)
    # The photographs show a chamfered upper edge on the entrance.
    for sign in (-1,1):
        xx=x+sign*.075; yy=-.10085
        points=[(xx,yy,.037),(xx-sign*.014,yy,.037),(xx,yy,.023)]
        vertices=points+[(a,b+.0018,c) for a,b,c in points]
        mesh_object('A entrance chamfer',vertices,
                    [(0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)],coll,mat)
    for side,start,end in [('back',(x-.1,.1),(x+.1,.1)),('left',(x-.1,-.1),(x-.1,.1)),('right',(x+.1,-.1),(x+.1,.1))]:
        hs=[]; bs=[]
        for row in range(7):
            z=.055+row*.045
            groups=((.02,.07,4),(.085,.115,2),(.13,.18,4)) if side=='back' else ((.025,.065,2),(.105,.18,4))
            for a,b,count in groups:
                hs.append((a,b,z,z+.006))
                for k in range(1,count):
                    u=a+(b-a)*k/count
                    bs.append((u-.0005,u+.0005,z,z+.006))
        wall('A_'+side,coll,mat,start,end,hs)
        apply_bars('A_'+side+'_mullions',coll,mat,start,end,bs)
    roof_and_base(coll,mat,x)


def paper_mesh(name, coll, mat, x, y, z, width, height, curl=.00035):
    vertices=[]; faces=[]; nx=12; nz=12
    for j in range(nz+1):
        for i in range(nx+1):
            u=i/nx; v=j/nz
            # The central glue area is nearly flush; corners lift unequally.
            # The restrained edge ripple is a visual approximation, not a
            # measurement of paper thickness or deformation.
            edge=max(abs(2*u-1),abs(2*v-1))
            curl_profile=(.11*(2*u-1)**2+.09*(2*v-1)**2
                          +.72*u**3*v**3+.26*(1-u)**3*(1-v)**3)
            ripple=.000035*math.sin(3.1*math.pi*u+1.2)*math.sin(2.7*math.pi*v)*edge**2
            lift=max(0,curl*curl_profile+ripple)
            # Subtle irregularities imitate a hand-cut edge without altering
            # the stated paper size by more than a fraction of a millimetre.
            edge_x=(.000025*abs(math.sin(j*1.7))*(1 if i==0 else -1)
                    if i in (0,nx) else 0)
            edge_z=(.000025*abs(math.sin(i*1.9))*(1 if j==0 else -1)
                    if j in (0,nz) else 0)
            vertices.append((x+(u-.5)*width+edge_x,y-lift,z+(v-.5)*height+edge_z))
    for j in range(nz):
        for i in range(nx):
            k=j*(nx+1)+i
            faces.append((k,k+1,k+nx+2,k+nx+1))
    obj=mesh_object(name,vertices,faces,coll,mat)
    uv=obj.data.uv_layers.new(name='Paper UV')
    for poly in obj.data.polygons:
        for idx in poly.loop_indices:
            vi=obj.data.loops[idx].vertex_index
            uv.data[idx].uv=(vi%(nx+1)/nx,vi//(nx+1)/nz)
    mod=obj.modifiers.new('Paper thickness 0.10 mm','SOLIDIFY')
    mod.thickness=.00010
    return obj


def name_label(coll, x, front_y, shell_factor, label, paper, ink, font):
    paper_y=front_y-.0008*shell_factor-.00008
    label_paper=paper_mesh('Name paper - '+label,coll,paper,x,paper_y,.382,.122,.032,.00035)
    curve=bpy.data.curves.new('Editable building name - '+label,'FONT')
    curve.body=label; curve.align_x='CENTER'; curve.align_y='CENTER'; curve.font=font
    curve.size=.073
    obj=bpy.data.objects.new('Editable text - '+label,curve); coll.objects.link(obj)
    obj.rotation_euler=(math.pi/2,0,0); obj.location=(x,paper_y-.0006,.382)
    curve.materials.append(ink)
    bpy.context.view_layer.update()
    if obj.dimensions.x > .115:
        obj.scale *= .115/obj.dimensions.x
    ink_fit=obj.modifiers.new('Printed ink follows the paper surface','SHRINKWRAP')
    ink_fit.target=label_paper
    ink_fit.wrap_method='NEAREST_SURFACEPOINT'
    ink_fit.wrap_mode='ABOVE_SURFACE'
    ink_fit.offset=.000006


def fire_stickers(coll, x, front_y, shell_factor, material, rng):
    # This is a configurable example layout, not a measured competition truth.
    xs=[-.065,-.025,.022,.061]
    rows=[.115,.165,.215,.265,.315]
    for index,(xx,zz) in enumerate(rng.sample([(a,b) for a in xs for b in rows],3),1):
        # Window mullions protrude slightly beyond the outer wall surface;
        # leave a small clearance so printed paper stays in front of them.
        paper_y=front_y-.00093*shell_factor-.00008
        obj=paper_mesh('Fire paper %02d'%index,coll,material,x+xx,paper_y,zz,.025,.035,
                       rng.uniform(.00040,.00070))
        obj['source_art']='datasets/raw/fire/team_assets/image1.png'
        obj['placement']='illustrative; glue fixed, no visible adhesive'


def add_light(name, energy, size, location, target):
    data=bpy.data.lights.new(name,'AREA'); data.energy=energy; data.shape='DISK'; data.size=size
    obj=bpy.data.objects.new(name,data); bpy.context.scene.collection.objects.link(obj)
    obj.location=location; obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def _produce(args):
    artwork=ROOT/'datasets/raw/fire/team_assets/image1.png'
    if not artwork.is_file():
        raise FileNotFoundError('fire image missing')
    args.output.mkdir(parents=True,exist_ok=False)
    font_path=args.output/'NotoSansSC.ttf'
    shutil.copy2(args.font,font_path)
    shutil.copy2(args.font.parent/'OFL.txt',args.output/'OFL.txt')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC'
    shell=material('Moulded warm grey plastic',(.67,.67,.62),.61)
    ink=material('Matte printed black ink',(.012,.012,.011),.88)
    paper=paper_material('Paper - subtle fiber and transmission')
    fire_img=bpy.data.images.load(str(artwork),check_existing=True)
    fire=paper_material('Fire print on white paper',fire_img)
    font=bpy.data.fonts.load(str(font_path))
    rng=random.Random(args.seed)
    items=[('A','开心大厦',-.27,building_a),('B','锦绣大厦',0,building_b),('C','晨光大厦',.27,building_c)]
    building_cols=[]
    for key,label,x,builder in items:
        coll=bpy.data.collections.new(key+' - '+label); scene.collection.children.link(coll)
        building_cols.append(coll)
        builder(x,coll,shell)
        shell_factor=1.0
        if key in ('A','B'):
            # Photographed roofs appear rectangular. Compress only the shell;
            # the independently constructed paper retains its own dimensions
            # and deformation. This ratio is an image-based estimate only.
            shell_factor=args.rect_depth/.2
            for obj in coll.objects:
                if obj.type=='MESH':
                    for vertex in obj.data.vertices:
                        vertex.co.y *= shell_factor
                obj.location.y *= shell_factor
        front_y=-args.rect_depth/2 if key in ('A','B') else -.1
        name_label(coll,x,front_y,shell_factor,label,paper,ink,font)
        fire_stickers(coll,x,front_y,shell_factor,fire,rng)
    floor=material('Neutral presentation floor',(.15,.15,.15),.78)
    box('Presentation floor',scene.collection,floor,(0,0,-.002),(1.5,1,.004))
    world=bpy.data.worlds.new('Studio world'); scene.world=world; world.use_nodes=True
    world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.72,.75,.78,1)
    world.node_tree.nodes.get('Background').inputs['Strength'].default_value=.34
    add_light('Broad softbox',14,.8,(0,-1,1.2),(0,0,.2))
    add_light('Soft fill',4,.7,(-.8,-.2,.8),(0,0,.2))
    data=bpy.data.cameras.new('Reference camera'); cam=bpy.data.objects.new('Reference camera',data)
    scene.collection.objects.link(cam); scene.camera=cam; data.type='PERSP'; data.lens=52
    data.clip_start=.001
    scene.render.engine='CYCLES'; scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True
    scene.render.resolution_x=600; scene.render.resolution_y=800; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.view_settings.view_transform='AgX'
    scene['geometry_note']=(f'A/B depth {args.rect_depth:.3f} m is a visual estimate from roof photos; '
                            'user supplied average long/wide 0.20 m and height 0.40 m. '
                            'C overall approx 0.20 x 0.20 m. Dimensions are not measured CAD.')
    scene['paper_note']='Fire paper 0.025 x 0.035 m; flame artwork is a color texture on the paper PBR material.'
    scene['name_editing']='Edit the named FONT objects inside each building collection.'
    cam.location=(0,-1.55,.62)
    cam.rotation_euler=(Vector((0,0,.2))-cam.location).to_track_quat('-Z','Y').to_euler()
    # All source data are packed so the blend opens without its local input paths.
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/'three_buildings.blend'))
    for (key,label,x,_),coll in zip(items,building_cols):
        for other in building_cols:
            other.hide_render=(other!=coll)
        for view,offset in [('front',(0,-.82,.28)),('oblique',(.40,-.72,.36)),
                            ('back',(-.40,.72,.38)),('top',(0,-.08,1.08))]:
            cam.location=(x+offset[0],offset[1],offset[2])
            cam.rotation_euler=(Vector((x,0,.20))-cam.location).to_track_quat('-Z','Y').to_euler()
            scene.render.filepath=str(args.output/(key+'_'+view+'.png'))
            bpy.ops.render.render(write_still=True)
    # Detail view of a real-size fire paper, including its edge and wall contact.
    for other in building_cols:
        other.hide_render=(other!=building_cols[2])
    sticker=next(obj for obj in building_cols[2].objects if obj.name.startswith('Fire paper'))
    target=sum((v.co for v in sticker.data.vertices),Vector())/len(sticker.data.vertices)
    cam.location=target+Vector((.070,-.110,.023))
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    data.lens=85
    scene.render.filepath=str(args.output/'C_fire_paper_close.png')
    bpy.ops.render.render(write_still=True)
    cam.location=target+Vector((.090,-.040,.010))
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    data.lens=85
    scene.render.filepath=str(args.output/'C_fire_paper_edge.png')
    bpy.ops.render.render(write_still=True)
    label=next(obj for obj in building_cols[2].objects if obj.name.startswith('Name paper'))
    target=sum((v.co for v in label.data.vertices),Vector())/len(label.data.vertices)
    cam.location=target+Vector((.09,-.21,.035))
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    data.lens=38
    scene.render.filepath=str(args.output/'C_name_paper_close.png')
    bpy.ops.render.render(write_still=True)
    sticker=next(obj for obj in building_cols[2].objects if obj.name.startswith('Fire paper'))
    target=sum((v.co for v in sticker.data.vertices),Vector())/len(sticker.data.vertices)
    target+=Vector((0,0,-.004))
    cam.location=target+Vector((.006,-.043,.004))
    cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
    data.lens=120
    scene.render.resolution_x=1000; scene.render.resolution_y=1000
    scene.cycles.samples=min(128,max(args.samples,80))
    scene.render.filepath=str(args.output/'C_fire_ink_macro.png')
    bpy.ops.render.render(write_still=True)
    # Save genuinely separate editable scenes without mutating the combined
    # scene: reopen the saved source, remove the other two groups, recenter the
    # chosen building and save under a new name.
    combined_path=str(args.output/'three_buildings.blend')
    names=[key+' - '+label for key,label,_,_ in items]
    for key,label,x,_ in items:
        bpy.ops.wm.open_mainfile(filepath=combined_path)
        keep=bpy.data.collections[key+' - '+label]
        for other_name in names:
            if other_name == keep.name:
                continue
            other=bpy.data.collections[other_name]
            for obj in list(other.objects):
                bpy.data.objects.remove(obj,do_unlink=True)
            bpy.data.collections.remove(other)
        for obj in keep.objects:
            obj.location.x-=x
        cam=bpy.data.objects['Reference camera']
        cam.location=(.32,-.79,.35)
        cam.rotation_euler=(Vector((0,0,.2))-cam.location).to_track_quat('-Z','Y').to_euler()
        bpy.context.scene.camera=cam
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output/(key+'_building.blend')))
    print('BUILT:',args.output/'three_buildings.blend')


def main():
    args = parse_args()
    artwork = ROOT / 'datasets/raw/fire/team_assets/image1.png'
    procedural = [args.font, args.font.parent / 'OFL.txt']
    with generation_guard(
        'fire.build_blender_buildings:' + args.output.relative_to(ROOT).as_posix(),
        [artwork, *procedural], [args.output],
        {'seed': args.seed, 'samples': args.samples, 'rect_depth': args.rect_depth},
        code_paths=[Path(__file__)], procedural_inputs=procedural,
    ) as run:
        if not run.reused:
            _produce(args)


if __name__ == '__main__':
    main()
