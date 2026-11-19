"""Prepare and publish paired meter material/view/reading/reflection samples."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import image_digest
from datasets.scripts.common import recoverable_publish as publisher
from datasets.scripts.meter.compose_meter_real_background_v4 import place_foreground
from datasets.scripts.meter.backgrounds import select_backgrounds, photometric_schedule, adjust, pixel_stats, placement
from lab.training_common.production_guard import generation_guard, receipt_completed

POINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
          'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')
COLORS = ('#ff3434','#ff9728','#1abb52','#24dcdf','#ee39cc','#b258ed','#447aff','#a3b927')
OPERATION = 'meter.build_current_sample'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def direct_sources(recipe):
    rows=[json.loads(line) for line in (ROOT/recipe['direct_review_manifest']).read_text(encoding='utf8').splitlines()]
    selected=[]
    for index,row in enumerate(rows):
        if row['source_kind'] in recipe['direct_capture_kinds']:
            path=ROOT/row['path']
            if not path.is_file():raise FileNotFoundError(path)
            selected.append((index,row))
    return selected


def prepare(source, recipe):
    """Generate technical face variants without changing any tick/endpoint pixel."""
    axes=yaml.safe_load((ROOT/'config/combination_axes.yaml').read_text(encoding='utf8'))['tasks']['meter']['axes']
    for axis,levels in recipe['photometry'].items():
        if set(levels)!=set(axes[axis]):raise ValueError('Recipe differs from canonical meter axis: '+axis)
    base = Image.open(ROOT/source['station_face']).convert('RGB')
    assert base.size == (1254,1254)
    black = np.asarray(base).max(axis=2) < 80
    fontpath = Path('C:/Windows/Fonts/arialbd.ttf')
    font = ImageFont.truetype(str(fontpath), 42)
    title = ImageFont.truetype(str(fontpath), 30)
    face_report = {}
    for variant, path in recipe['face_variants'].items():
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for family, center, start, sweep, low, high, radius in (
                ('temperature',(623.,646.),-147.,-241.,-30,50,520.),
                ('humidity',(631.,948.),44.,-264.,0,100,225.)):
            def point(fraction, r):
                angle=math.radians(start+sweep*fraction)
                return center[0]+r*math.cos(angle),center[1]-r*math.sin(angle)
            if variant == 'colored':
                bands = [(0,.55,'#284e88'),(.55,.75,'#7b8d76'),(.75,1,'#993b40')] if family == 'temperature' else [(0,.35,'#356b4d'),(.35,.7,'#87917b'),(.7,1,'#99494b')]
                for first,last,color in bands:
                    draw.line([point(first+(last-first)*i/100,radius*.90) for i in range(101)],fill=color,width=58 if family=='temperature' else 25)
            values = range(low,high+1,10 if family=='temperature' else 20)
            for value in values:
                xy=point((value-low)/(high-low),radius*.77)
                draw.text(xy,str(value),font=font,anchor='mm',fill='#30373a')
        draw.text((623,430),'TEMPERATURE  °C',font=title,anchor='mm',fill='#30373a')
        draw.text((631,866),'%RH',font=title,anchor='mm',fill='#30373a')
        # The existing parametric black tick pixels are immutable in both looks.
        pixels=np.array(image);pixels[black]=np.asarray(base)[black];image=Image.fromarray(pixels)
        target=ROOT/path;target.parent.mkdir(parents=True,exist_ok=True);image.save(target)
        assert np.array_equal(np.array(image)[black],np.array(base)[black])
        face_report[variant]={'image':path,'sha256':image_digest(path),'preserved_tick_pixels':int(black.sum())}
    write(ROOT/recipe['face_evidence'],{'base':source['station_face'],'variants':face_report,
          'range_reference':'docs/research/METER_VISION.md','printed_ranges':{'temperature':[-30,50],'humidity':[0,100]},
          'scope':'Procedural real-inspired printed face. Same synthetic geometry; not a scan or physical tick registration.'})
    backgrounds=select_backgrounds(recipe['background_selection'])
    write(ROOT/recipe['background_review'],{'backgrounds':backgrounds,'task':'meter',
          'basis':'Current eligible meter composite_background decisions from shared pool and its extensions.',
          'selection':recipe['background_selection'],
          'placement':'Real frames use reviewed contact footprints; empty rendered scenes use near-floor placement. Uniform scale/translation; physical scale and perspective are approximate.',
          'crop':'640x360 center-cropped to 480x360 then resized to 640x480; no anisotropic stretch.',
          'use':'unassigned review sample; source train identity retained'})
    views=read(ROOT/recipe['camera_inventory'])['views']
    assert len(views)==27 and len({v['view'] for v in views})==27
    foreground=[]
    levels=recipe['reading_fractions']
    for i,view in enumerate(views):
        for j in range(2):
            pair=((2*i+j)*7)%36
            for material in ('white','colored'):
                p={'branch':'blender_render','view':view['view'],'face_material':material,
                   'temperature_fraction':levels[pair//6],'humidity_fraction':levels[pair%6],
                   'glare':'none','distance_m':view['distance_m'],'yaw_deg':view['yaw_deg'],'elevation_deg':view['elevation_deg']}
                foreground.append(p)
        if view['view'].endswith('_level'):
            pair=(2*i*7)%36
            for material in ('white','colored'):
                foreground.append({'branch':'blender_render','view':view['view'],'face_material':material,
                    'temperature_fraction':levels[pair//6],'humidity_fraction':levels[pair%6],
                    'glare':('temperature_tip','humidity_tip','temperature_max')[(i//3)%3],
                    'distance_m':view['distance_m'],'yaw_deg':view['yaw_deg'],'elevation_deg':view['elevation_deg']})
    states=[]
    copies=recipe['composites_per_foreground']
    schedule=photometric_schedule(recipe,((len(foreground)+1)//2)*copies)
    for i,p in enumerate(foreground):
        sid=f'meter_open_{i:04d}'
        sources=[source['station_source'],recipe['face_variants'][p['face_material']]]
        def state(name,params,inputs):
            return {'id':name,'failure_event':'dual_needle_pose_or_reflection','min_count':1,'max_count':1,
                'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,'conditions':[],
                'background':{'kind':params['branch']},'target':{'kind':'dual_needle_meter'},
                'labels':{'kind':'synthetic_geometric_pose','kpt_shape':[8,3]},
                'augmentation':{'kind':'same_geometry_material_and_glare_contrasts'},'parameters':params,'sources':inputs}
        states.append(state(sid,p,sources))
        for copy_index in range(copies):
            slot=(i//2)*copies+copy_index
            bg=backgrounds[slot%len(backgrounds)]
            q={**p,'branch':'blender_background_composite' if bg['source_kind']=='blender_render' else 'real_background_composite',
               'foreground_id':sid,'background_record_id':bg['record_id'],**schedule[slot]}
            states.append(state(sid+f'_composite_{copy_index}',q,sources+[bg['image']]))
    for index,row in direct_sources(recipe):
        complete=row.get('pose_annotation',{}).get('complete',False)
        p={'branch':'direct_task_pose' if complete else 'direct_task_review','source_record_index':index,'source_image':row['path'],'content_sha256':image_digest(row['path'])}
        entry=state(f'meter_capture_{index:04d}',p,[row['path']])
        entry['target']={'kind':'dual_needle_meter' if complete else 'meter_direct_review'}
        entry['labels']={'kind':'direct_visual_pose' if complete else 'partial_visual_points','kpt_shape':[8,3]}
        states.append(entry)
    plan={'schema_version':1,'model_key':'meter','target_split':'unassigned','budget':len(states),
          'constraints':{'pose_resolution':[640,480],'direct_resolution':'native','kpt_shape':[8,3],'label_policy':'synthetic_and_direct_visual_pose_with_partial_review','foreground_rgba':True},
          'uncertainty_weight':0,'purpose':'Review matched materials, 27 camera views, independent needle grid and local glare with eight-point overlays.',
          'independence_basis':'One Blender station, real/Blender background parents and original capture histories retained. Capture candidates are review-only; no independent evaluation or formal release.',
          'cycles_samples':recipe['cycles_samples'],'render_seed':recipe['render_seed'],'face_variants':recipe['face_variants'],
          'endpoint_config':source['endpoint_config'],'strata':states}
    write(ROOT/source['sample_plan'],plan)
    print('PREPARED',len(foreground),'foregrounds',len(states),'sample images',flush=True)


def owned_files(output):
    manifest=output/'manifest.json'
    if not manifest.exists():return set()
    data=read(manifest)
    if 'owned_files' in data:return set(data['owned_files'])
    names={'manifest.json','samples.jsonl'}
    samples=output/data['samples']
    def collect(value):
        if isinstance(value,dict):
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
        elif isinstance(value,str) and (output/value).is_file():names.add(value)
    for line in samples.read_text().splitlines():
        row=json.loads(line);collect(row);names.add('annotations/'+row['id']+'.json')
    return names


def label_values(record):
    x1,y1,x2,y2=record['bbox_xyxy_pixels']
    values=[0,(x1+x2)/1280,(y1+y2)/960,(x2-x1)/640,(y2-y1)/480]
    for key in POINTS:
        x,y,_=record['keypoints_xy_depth_normalized_candidate'][key]
        v=record['keypoint_visibility'][key]['visibility']
        values.extend([x,y,v] if v else [0,0,0])
    return values


def overlay(image,values):
    image=image.convert('RGB').copy();d=ImageDraw.Draw(image)
    cx,cy,w,h=values[1:5];box=((cx-w/2)*640,(cy-h/2)*480,(cx+w/2)*640,(cy+h/2)*480)
    d.rectangle(box,outline='yellow',width=2)
    for i in range(8):
        x,y,v=values[5+3*i:8+3*i]
        if v:
            x,y=x*640,y*480;d.ellipse((x-2,y-2,x+2,y+2),fill=COLORS[i]);d.text((x+4,y-8),str(i),fill=COLORS[i])
    return image


def finish(stage,plan,recipe):
    rendered={r['id']:r for r in read(stage/'render_records.json')}
    backgrounds={r['record_id']:r for r in read(ROOT/recipe['background_review'])['backgrounds']}
    for name in ('images','review_images','labels','annotations','overlays','crops'):(stage/name).mkdir(exist_ok=True)
    records=[];geometry_pairs={};reflection_changes=[]
    for state in plan['strata']:
        p=state['parameters'];sid=state['id']
        if p['branch'].startswith('direct_task_'):continue
        fid=p.get('foreground_id',sid);raw=rendered[fid]
        fg=Image.open(stage/raw['foreground_rgba']).convert('RGBA');values=label_values(raw);transform=None;photometry=None
        if p['branch'] in ('real_background_composite','blender_background_composite'):
            bg=backgrounds[p['background_record_id']]
            assert image_digest(bg['image'])==bg['content_sha256']
            image=ImageOps.fit(Image.open(ROOT/bg['image']).convert('RGB'),(640,480),method=Image.Resampling.LANCZOS).convert('RGBA')
            recipe_placement,support=placement(bg,fg)
            fg,values,transform=place_foreground(fg,values,recipe_placement)
            transform['contact_support_xyxy']=support
            levels=recipe['photometry']
            sat=levels['saturation'][p['saturation']]
            gain=levels['exposure'][p['exposure']]
            relative=levels['light_relation'][p['light_relation']]
            background_before=pixel_stats(image)
            image=adjust(image,saturation=sat,brightness=gain)
            fg=adjust(fg,saturation=sat,brightness=gain*relative)
            photometry={'saturation_factor':sat,'exposure_gain':gain,'foreground_relative_gain':relative,
                'background_before':background_before,'background_after':pixel_stats(image),
                'foreground_after':pixel_stats(fg,np.asarray(fg.getchannel('A'))>127),
                'scope':'RGB augmentation; alpha and geometry unchanged; readability not certified'}
            # Contact shadow is an explicit 2-D compositing approximation.
            bounds=fg.getchannel('A').getbbox();shadow=Image.new('RGBA',image.size);d=ImageDraw.Draw(shadow)
            foot=bounds[3]
            d.ellipse((bounds[0]-4,foot-5,bounds[2]+4,foot+4),fill=(0,0,0,55))
            image=Image.alpha_composite(image,shadow.filter(ImageFilter.GaussianBlur(4)))
            image=Image.alpha_composite(image,fg).convert('RGB')
        else:
            image=Image.alpha_composite(Image.new('RGBA',(640,480),(178,178,178,255)),fg).convert('RGB')
        assert len(values)==29 and all(math.isfinite(x) for x in values)
        assert all(v in (0,1,2) for v in values[7::3])
        label=' '.join(f'{v:.8f}' for v in values)+'\n'
        image.save(stage/'images'/(sid+'.png'));(stage/'labels'/(sid+'.txt')).write_text(label)
        annotated=overlay(image,values);annotated.save(stage/'overlays'/(sid+'.png'))
        cx,cy,w,h=values[1:5];box=(max(0,int((cx-w/2)*640)-8),max(0,int((cy-h/2)*480)-8),min(640,int((cx+w/2)*640)+8),min(480,int((cy+h/2)*480)+8))
        ImageOps.pad(annotated.crop(box),(256,256),color='white').save(stage/'crops'/(sid+'.png'))
        row={**raw,'id':sid,'image':'images/'+sid+'.png','pose_label':'labels/'+sid+'.txt','annotation':'annotations/'+sid+'.json',
             'overlay':'overlays/'+sid+'.png','crop':'crops/'+sid+'.png','parameters':p,'split':'unassigned','source_split':bg['split'] if transform else None,
             'label_values':values,'image_size':[640,480],'transform':transform,'background':backgrounds.get(p.get('background_record_id')),
             'sources':state['sources'],'foreground_id':fid,'photometry':photometry}
        row['foreground_geometry']={'bbox_xyxy_pixels':raw['bbox_xyxy_pixels'],
            'display_roi_quad':raw['display_roi_quad'],'coordinate_frame':'original_foreground_render'}
        row['bbox_xyxy_pixels']=[(cx-w/2)*640,(cy-h/2)*480,(cx+w/2)*640,(cy+h/2)*480]
        if transform:
            scale=transform['scale'];dx,dy=transform['offset_xy']
            row['display_roi_quad']=[[q[0]*scale+dx/640,q[1]*scale+dy/480,q[2]] for q in raw['display_roi_quad']]
        row['raster_coordinate_frame']='original_foreground_render; see foreground_geometry and transform'
        # Rebuild coordinates from the published label, including the real-image transform.
        row['keypoints_xy_depth_normalized_candidate']={key:[values[5+3*i],values[6+3*i],raw['keypoints_xy_depth_normalized_candidate'][key][2]] for i,key in enumerate(POINTS)}
        row['keypoint_visibility']={key:{**raw['keypoint_visibility'][key],'visibility':int(values[7+3*i])} for i,key in enumerate(POINTS)}
        if transform:
            for evidence in row['keypoint_visibility'].values():
                evidence['reason']='known_synthetic_position_after_2d_placement; readability_not_reassessed'
        (stage/row['annotation']).write_text(json.dumps(row,indent=2)+'\n');records.append(row)
        if not transform:
            key=(p['view'],p['temperature_fraction'],p['humidity_fraction'],p['glare'])
            coords=np.array([values[5+3*i:7+3*i] for i in range(8)])
            if key in geometry_pairs:assert np.max(np.abs(coords-geometry_pairs[key]))<1e-7
            geometry_pairs[key]=coords
    with (stage/'samples.jsonl').open('w',encoding='utf8') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    for row in records:
        p=row['parameters']
        if p['branch']!='blender_render' or p['glare']=='none':continue
        clean=next(r for r in records if r['parameters']=={**p,'glare':'none'})
        a=np.array(Image.open(stage/row['image'])).astype(float);b=np.array(Image.open(stage/clean['image'])).astype(float)
        x,y=row['keypoints_xy_depth_normalized_candidate'][p['glare']][:2];x,y=int(x*640),int(y*480)
        difference=(a-b)[max(0,y-8):y+9,max(0,x-12):x+13]
        reflection_changes.append({'id':row['id'],'target':p['glare'],'max_brightness_increase':float(difference.max()),'mean_brightness_change':float(difference.mean())})
    synthetic_records=list(records);pose_records=list(records)
    direct=dict(direct_sources(recipe))
    for state in plan['strata']:
        p=state['parameters']
        if not p['branch'].startswith('direct_task_'):continue
        sid=state['id'];original=direct[p['source_record_index']]
        assert image_digest(original['path'])==p['content_sha256']
        complete=p['branch']=='direct_task_pose'
        directory='images' if complete else 'review_images'
        image=Image.open(ROOT/original['path']).convert('RGB');image.save(stage/directory/(sid+'.png'))
        marked=image.copy();d=ImageDraw.Draw(marked);box=original.get('dial_bbox')
        if box:d.rectangle(box,outline='orange',width=max(2,image.width//300))
        for i,point in enumerate(original['keypoints']):
            if point['xy'] is not None:
                x,y=point['xy'];radius=max(2,image.width//650)
                d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=COLORS[i]);d.text((x+radius+2,y),str(i),fill=COLORS[i])
        marked.save(stage/'overlays'/(sid+'.png'))
        ImageOps.pad(marked.crop(box) if box else marked,(256,256),color='white').save(stage/'crops'/(sid+'.png'))
        row={'id':sid,'image':directory+'/'+sid+'.png','annotation':'annotations/'+sid+'.json','overlay':'overlays/'+sid+'.png','crop':'crops/'+sid+'.png',
             'parameters':p,'sources':state['sources'],'image_size':list(image.size),'split':'unassigned',
             'source_image':original['path'],'source_record_index':p['source_record_index'],'source_manifest':recipe['direct_review_manifest'],
             'source_use_history':original.get('task_use_status',[]),'source_kind':original['source_kind'],
             'candidate_dial_bbox':box,'observed_keypoints':original['keypoints'],'pose_label':None,'training_eligible':complete,
             'pose_annotation':original.get('pose_annotation'),'transform':None,
             'supervision_scope':'direct AI visual 2D annotation; incomplete clipped image retains null points'}
        if complete:
            width,height=image.size;x1,y1,x2,y2=box
            values=[0,(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height]
            for point in original['keypoints']:
                x,y=point['xy'];assert 0<=x<width and 0<=y<height
                values.extend([x/width,y/height,int(point['visibility'])])
            row.update(pose_label='labels/'+sid+'.txt',label_values=values,keypoint_order=list(POINTS),bbox_xyxy_pixels=box,
                keypoints_xy_depth_normalized_candidate={key:[point['xy'][0]/width,point['xy'][1]/height,None] for key,point in zip(POINTS,original['keypoints'])},
                keypoint_visibility={key:{'visibility':point['visibility'],'reason':'direct_visual_annotation'} for key,point in zip(POINTS,original['keypoints'])})
            (stage/row['pose_label']).write_text(' '.join(f'{v:.8f}' for v in values)+'\n')
            pose_records.append(row)
        write(stage/row['annotation'],row);records.append(row)
    with (stage/'samples.jsonl').open('w',encoding='utf8') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    summary={'images':len(records),'pose_labels':len(pose_records),'task_real_pose_images':len(pose_records)-len(synthetic_records),'task_real_review_images':len(records)-len(pose_records),'transparent_foregrounds':len(rendered),'branches':dict(Counter(r['parameters']['branch'] for r in records)),
        'camera_views':len({r['parameters']['view'] for r in synthetic_records}),'needle_pairs':len({(r['parameters']['temperature_fraction'],r['parameters']['humidity_fraction']) for r in synthetic_records}),
        'materials':dict(Counter(r['parameters']['face_material'] for r in synthetic_records)),
        'glare_images':sum(r['parameters']['glare']!='none' for r in synthetic_records),'real_backgrounds':sum(r['source_kind']!='blender_render' for r in backgrounds.values()),
        'blender_backgrounds':sum(r['source_kind']=='blender_render' for r in backgrounds.values()),
        'visibility_per_point':{key:dict(Counter(str(r['keypoint_visibility'][key]['visibility']) for r in pose_records)) for key in POINTS},
        'background_records_used':len({r['parameters']['background_record_id'] for r in synthetic_records if r.get('transform')}),
        'photometric_axes':{axis:dict(Counter(r['parameters'][axis] for r in synthetic_records if r.get('transform'))) for axis in ('saturation','exposure','light_relation')},
        'geometry_material_pair_max_error':0.0,'label_width':29,'kpt_shape':[8,3]}
    write(stage/'checks.json',{'summary':summary,'local_reflection_pixel_checks':reflection_changes,
          'keypoint_order':list(POINTS),'visibility_policy':'v1 known synthetic position, weak/obstructed; v2 only qualified cap contrast. Real unknown locations are not invented.',
          'scope':'Render/composite review only. No physical reading accuracy or independent test claim.'})
    # One image covers task captures, point detail and all generation branches.
    sheet=Image.new('RGB',(1600,4*280),'white');d=ImageDraw.Draw(sheet)
    direct_rows=[r for r in records if r['parameters']['branch']=='direct_task_pose']
    branches=[('REAL POSE',direct_rows),('8 POINT DETAIL',direct_rows),
              ('BLENDER RENDER',[r for r in records if r['parameters']['branch']=='blender_render']),
              ('REAL BACKGROUND',[r for r in records if r['parameters']['branch']=='real_background_composite']),
              ('BLENDER BACKGROUND',[r for r in records if r['parameters']['branch']=='blender_background_composite'])]
    for column,(title,items) in enumerate(branches):
        for number,fraction in enumerate((0,.35,.7,1)):
            row=items[round(fraction*(len(items)-1))]
            path=row['crop'] if column==1 else row['overlay']
            pic=ImageOps.pad(Image.open(stage/path).convert('RGB'),(320,240),color=(244,244,244))
            sheet.paste(pic,(column*320,number*280+40))
            d.text((column*320+6,number*280+5),title,fill='black')
            d.text((column*320+6,number*280+21),row['id'],fill='#555555')
    sheet.save(stage/'contact_sheet.jpg',quality=94)
    return records,summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--reuse-render-from',type=Path,help='Reuse verified matching RGBA renders; no Blender run')
    args=ap.parse_args()
    _,source=load_task_config('meter');recipe=read(ROOT/source['sample_recipe'])
    if args.prepare:prepare(source,recipe);return
    plan=read(ROOT/source['sample_plan']);output=ROOT/'datasets/samples/meter'
    metadata=[ROOT/source[k] for k in ('sample_plan','sample_recipe','endpoint_config')]+[ROOT/'datasets/scripts/meter/config/source.yaml',ROOT/'datasets/scripts/meter/config/dataset.yaml',ROOT/'config/combination_axes.yaml',ROOT/recipe['background_review'],ROOT/recipe['face_evidence'],ROOT/recipe['camera_inventory'],ROOT/recipe['direct_review_manifest'],ROOT/recipe['shared_pool_manifest']]
    backgrounds=read(ROOT/recipe['background_review'])['backgrounds']
    media=[ROOT/source['station_source'],*[ROOT/v for v in recipe['face_variants'].values()],*[ROOT/r['image'] for r in backgrounds]]
    media.extend(ROOT/r['path'] for _,r in direct_sources(recipe))
    code=[Path(__file__),Path(__file__).with_name('render_open_sample.py'),Path(__file__).with_name('render_meter_candidates.py'),Path(__file__).with_name('render_meter_batch.py'),Path(__file__).with_name('meter_station_plan.py'),Path(__file__).with_name('compose_meter_real_background_v4.py'),ROOT/'datasets/scripts/common/blender_workspace.py',Path(publisher.__file__),Path(__file__).with_name('backgrounds.py')]
    cached=None
    if args.reuse_render_from:
        cache=args.reuse_render_from.resolve()
        cached=read(cache/'render_records.json')
        prior=read(cache/'manifest.json')['generation_plan']
        for key in ('render_seed','cycles_samples','face_variants','endpoint_config'):
            if prior[key]!=plan[key]:raise ValueError('Render cache input differs: '+key)
        expected={s['id']:s['parameters'] for s in plan['strata'] if s['parameters']['branch']=='blender_render'}
        if {r['id']:r['parameters'] for r in cached}!=expected:raise ValueError('Render cache state mismatch')
        # Declare every reused raster and its geometry as production inputs.
        cache_files=[p for name in ('foreground','masks','render_records') for p in (cache/name).rglob('*') if p.is_file()]
        media.extend(p for p in cache_files if p.suffix.lower() in ('.png','.jpg','.exr'))
        metadata.extend(p for p in cache_files if p.suffix.lower() not in ('.png','.jpg','.exr'))
        metadata.extend([cache/'render_records.json',cache/'manifest.json'])
    with publisher.sample_lock(output):
        committed=lambda token:receipt_completed(ROOT,OPERATION,token)
        publisher.recover(output,committed);owned=owned_files(output)
        preserved=[rel(p) for p in output.rglob('*') if p.is_file() and p.relative_to(output).as_posix() not in owned]
        archive=ROOT/recipe.get('archive_before_replace','Archive/technical/datasets/meter/sample_before_material_variants_20261118')
        if output.exists() and not archive.exists():shutil.copytree(output,archive)
        try:
            with generation_guard(OPERATION,media,[output],{'plan':source['sample_plan']},root=ROOT,procedural_inputs=metadata,code_paths=code,
                    production_plan=plan,sample_output_paths=[output/'images',output/'review_images'],
                    auxiliary_output_paths=[output/name for name in ('foreground','masks','overlays','crops','contact_sheet.jpg')],
                    replace_outputs=True,replace_verifier=lambda *_:{'preserved_paths':preserved}) as run:
                if run.reused:print(json.dumps(read(output/'manifest.json')['summary']));return
                stage=publisher.prepare(output,run.token);publisher.copy_unowned(output,stage,owned)
                if cached is not None:
                    for name in ('foreground','masks','render_records'):
                        shutil.copytree(cache/name,stage/name,dirs_exist_ok=True)
                    shutil.copy2(cache/'render_records.json',stage/'render_records.json')
                else:
                    log=ROOT/'tmp/meter_open_sample_render.log'
                    with log.open('w',encoding='utf8') as stream:
                        subprocess.run([recipe['blender_executable'],'--background',str(ROOT/source['station_source']),'--python-exit-code','1','--python',str(code[1]),'--','--plan',str(ROOT/source['sample_plan']),'--output',str(stage)],stdout=stream,stderr=subprocess.STDOUT,check=True)
                records,summary=finish(stage,plan,recipe)
                generated=[p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file() and rel(output/p.relative_to(stage)) not in preserved]
                write(stage/'manifest.json',{'task':'meter','status':'review_only_paired_material_camera_reading_glare','split':'unassigned','plan':source['sample_plan'],
                    'generation_plan':plan,'samples':'samples.jsonl','summary':summary,'owned_files':generated+['manifest.json'],
                    'source_refs':run.source_refs,'scope':'Complete task captures use direct AI visual pose annotations; clipped unresolved capture stays partial. Parent source use retained; no formal release or independent test.'})
                publisher.publish(output,stage,run.token,keep_root=True)
                states={s['id']:s for s in run.plan['strata']}
                for row in records:
                    state=states[row['id']];run.record_sample(state['id'],output=output/row['image'],sources=state['sources'],parameters=state['parameters'],actual=row)
                run.set_result(summary)
            publisher.recover(output,committed)
        except BaseException:
            publisher.recover(output,committed);raise
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
