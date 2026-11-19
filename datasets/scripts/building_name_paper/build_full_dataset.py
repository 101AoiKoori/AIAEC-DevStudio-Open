"""Build and audit the current 3000/300/300 paper-detection dataset."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.building_name_paper import full_plan as full,hybrid_sample as hybrid
from datasets.scripts.building_name_paper import compose_rooftop_papers as rooftop
from datasets.scripts.building_name_paper import package_detection_dataset as package
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.config import source_path,load_task_config
from datasets.scripts.common.images import read_image,write_image,box as mask_box
from datasets.scripts.common.package_direct_real import check_yolo
from datasets.scripts.common import recoverable_publish as publisher
from datasets.scripts.common.paths import dataset_dir
from datasets.scripts.common.source_history import read_source_history
from lab.training_common.production_guard import generation_guard,receipt_completed

OUT=dataset_dir(full.TASK)
MASKS=full.WORK/'visible_masks'


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


class Composer:
    def __init__(self,manifest=None):
        self.recipe=full.config()
        self.plates={p['id']:full.render_plate(p) for p in full.plates()}
        self.manifests=[Path(manifest)] if manifest else [ROOT/p for p in self.recipe['layer_manifests']]
        self.records={}
        for path in self.manifests:
            data=full.read(path)
            if not manifest and data['status']!='complete':raise ValueError('Physical layer production is incomplete')
            for row in data['records']:
                if row['status']=='accepted' and row['plate_id'] in self.plates:
                    self.records[row['id']]={**row,'layer_root':relative(path.parent),
                        'generation_split':row['split'],'split':self.plates[row['plate_id']]['split']}

    @lru_cache(maxsize=8)
    def layers(self,key):
        row=self.records[key];plate=self.plates[row['plate_id']];w,h=row['image_size']
        layer_root=ROOT/row['layer_root']
        on=read_image(layer_root/row['foreground'],cv2.IMREAD_UNCHANGED)
        off=read_image(layer_root/row['paper_off_foreground'],cv2.IMREAD_UNCHANGED)
        shadowed=read_image(layer_root/row['shadowed_receiver'])
        lit=read_image(layer_root/row['lit_receiver'],cv2.IMREAD_UNCHANGED)
        palette=cv2.cvtColor(read_image(layer_root/row['visible_mask'],cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA)
        masks={}
        for obj in row['instances']:
            b=obj['building_id'];selected=palette[:,:,3]>=192
            for ch,value in enumerate(hybrid.PALETTE[b]):
                selected &= palette[:,:,ch]>=192 if value else palette[:,:,ch]<=63
            if int(selected.sum())!=obj['visibility']['visible_pixels'] or mask_box(selected)!=obj['xyxy']:
                raise ValueError('Template paper-mask evidence differs: '+key+'/'+b)
            masks[b]=selected
        photo=read_image(ROOT/row['source_photo'])
        photo=cv2.resize(photo,(w,h),interpolation=cv2.INTER_AREA)
        support=np.zeros((h,w),np.uint8)
        cv2.fillPoly(support,[np.rint(plate['ground_polygon']).astype(np.int32)],1)
        for a,b,c,d in plate['forbidden_footprint_xyxy']:
            support[max(0,round(b)):min(h,round(d)),max(0,round(a)):min(w,round(c))]=0
        return on,off,shadowed,lit,photo,support,masks

    def compose(self,state):
        key=state['parameters']['template_id'];record=self.records[key]
        on,off,shadowed,lit,photo,support,all_masks=self.layers(key)
        papers=state['target']['papers'];selected=np.zeros(photo.shape[:2],np.uint8)
        targets={b:all_masks[b] for b in papers}
        for mask in targets.values():selected |= mask.astype(np.uint8)
        # Include antialiased edges and the small facade shadow of a matte paper.
        patch=cv2.dilate(selected,np.ones((5,5),np.uint8))>0
        foreground=off.copy();foreground[patch]=on[patch]
        if state['branch']=='blender_render':
            floor_alpha=lit[:,:,3:4].astype(np.float32)/255
            world=np.array([[[183,183,183]]],np.float32)
            base=np.rint(lit[:,:,:3]*floor_alpha+world*(1-floor_alpha)).astype(np.uint8)
            support=(lit[:,:,3]>=192).astype(np.uint8)
            gain=np.ones(3)
        else:
            base=photo
            pixels=photo[photo.shape[0]//2:].astype(np.float32)/255
            reference=np.percentile(hybrid.linear(pixels).reshape(-1,3),95,axis=0)
            paper_pixels=on[:,:,:3][np.logical_or.reduce(list(all_masks.values()))]
            white=np.percentile(hybrid.linear(paper_pixels.astype(np.float32)/255),85,axis=0)
            gain=np.clip(reference/np.maximum(white,.01),.8,1.25)
        image,shade,alpha=hybrid.compose_layers(base,foreground,shadowed,lit[:,:,:3],
            strength=self.recipe['lighting']['shadow_strength'],gain=gain,
            blur_sigma=self.recipe['lighting']['foreground_blur_sigma_px'],shadow_support=support)
        image,targets,transform=augment(image,targets,state['parameters']['seed'],self.recipe['augmentation'])
        objects=[{'building_id':b,'xyxy':mask_box(mask),'visible_pixels':int(mask.sum())} for b,mask in targets.items()]
        ids=np.zeros(image.shape[:2],np.uint8)
        for b,mask in targets.items():ids[mask]=ord(b)-64
        return image,objects,ids,{'camera':record['camera'],'layer_id':key,
            'layer_manifest':record['layer_root']+'/manifest.json','paper_presence':papers,
            'current_source_record_id':self.plates[record['plate_id']]['record_id'],
            'lighting':record['lighting'],'linear_bgr_gain':gain.tolist(),'augmentation':transform,
            'template_paper_masks':{b:record['visible_mask'] for b in papers},
            'shadow_pixels_below_95pct':int(((shade<.95)&(alpha<.1)).sum()),
            'source_photo':record['source_photo'],'source_group':record['source_group']}


def augment(image,masks,seed,config):
    rng=np.random.default_rng(seed);h,w=image.shape[:2]
    selected=masks
    matrix=np.array([[1.,0,0],[0,1.,0]])
    for attempt in range(6):
        scale=float(rng.uniform(*config['scale']))
        dx,dy=rng.uniform(*config['translate_px'],2)
        candidate=np.array([[scale,0,(1-scale)*w/2+dx],[0,scale,(1-scale)*h/2+dy]])
        transformed={b:cv2.warpAffine(mask.astype(np.uint8),candidate,(w,h),flags=cv2.INTER_NEAREST)>0 for b,mask in masks.items()}
        if all(mask.sum()>=12 for mask in transformed.values()):
            matrix=candidate;selected=transformed;break
    image=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    ev=float(rng.uniform(*config['exposure_ev']));sat=float(rng.uniform(*config['saturation']))
    linear=hybrid.linear(image.astype(np.float32)/255)*2**ev
    luma=linear@np.array([.0722,.7152,.2126],np.float32)
    linear=luma[:,:,None]+sat*(linear-luma[:,:,None])
    image=hybrid.encoded(linear)
    blur=float(rng.uniform(*config['blur_sigma_px']))
    image=cv2.GaussianBlur(image,(0,0),blur)
    return image,selected,{'affine_matrix':matrix.tolist(),'exposure_ev':ev,'saturation':sat,'blur_sigma_px':blur,
                          'scope':'bounded whole-image sensor crop and tone; same transform applied to visible masks'}


def owned_outputs():
    owned=set()
    index=OUT/'workspace_sources.json'
    if index.exists():
        for row in full.read(index)['records']:
            for key in ('image','label'):
                path=(ROOT/row[key]).resolve()
                if not path.is_relative_to(OUT) or not path.is_file():raise ValueError('Prior export record is missing/outside task')
                owned.add(path.relative_to(OUT).as_posix())
    for path in [OUT/'data.yaml',OUT/'workspace_sources.json',OUT/'manifest.json',OUT/'audit.json',
                 *[OUT/s/'manifest.json' for s in ('train','val','test')]]:
        if path.is_file():owned.add(path.relative_to(OUT).as_posix())
    if (OUT/'manifest.json').exists():
        for name in full.read(OUT/'manifest.json').get('artifacts',[]):
            path=(OUT/name).resolve()
            if not path.is_relative_to(OUT) or not path.is_file():raise ValueError('Prior export artifact is missing/outside task')
            owned.add(name)
    return owned


def contact_sheet(stage,records,split):
    candidates=[r for r in records if r['split']==split and r['branch']=='blender_photo_composite']
    choices=[]
    for count,papers,yaw in [(1,0,'left'),(1,1,'left'),(2,0,'left'),(2,1,'left'),(2,2,'front'),
                              (3,0,'front'),(3,1,'left'),(3,2,'left'),(3,3,'left')]:
        match=next((r for r in candidates if len(r['buildings'])==count and len(r['instances'])==papers
                    and ('left' in r['layer_id'])==(yaw=='left')),None)
        if match:choices.append(match)
    font=ImageFont.truetype(str(source_path(full.TASK,'font')),20)
    sheet=Image.new('RGB',(1440,392*((len(choices)+2)//3)),'#20252b')
    for i,row in enumerate(choices):
        image=Image.open(stage/row['image']).convert('RGB');w,h=image.size;image.thumbnail((476,356))
        draw=ImageDraw.Draw(image)
        for obj in row['instances']:
            x0,y0,x1,y1=obj['xyxy'];draw.rectangle((x0*image.width/w,y0*image.height/h,x1*image.width/w,y1*image.height/h),outline='#20ed74',width=2)
        x=i%3*480;y=i//3*392;sheet.paste(image,(x,y+31))
        camera=row['camera'];title=f"{split} {row['buildings']} 纸{len(row['instances'])} 左偏{abs(camera['yaw_degrees']):g}° 高{camera['height_m']:.2f}m"
        ImageDraw.Draw(sheet).text((x+5,y+3),title,font=font,fill='white')
    path=f'{split}_contact_sheet.jpg';sheet.save(stage/path,quality=92)
    return path


def audit(stage,records,mask_stage,source_records):
    summary={};rgb={};file_hashes={};parents=defaultdict(set);structures={}
    for split in ('train','val','test'):
        subset=[r for r in records if r['split']==split]
        structures[split]=sorted({(r['buildings'],r['papers']) for r in subset if 'buildings' in r})
        summary[split]={'images':len(subset),'paper_boxes':sum(len(r['instances']) for r in subset),
            'empty_images':sum(not r['instances'] for r in subset),'branches':dict(Counter(r['branch'] for r in subset)),
            'building_paper_states':len(structures[split]),
            'partial_paper_multi_building_images':sum('buildings' in r and 0<len(r['papers'])<len(r['buildings']) for r in subset),
            'buildings_without_paper_images':sum('buildings' in r and not r['papers'] for r in subset),
            'camera_heights_m':dict(Counter(str(r['camera']['height_m']) for r in subset if 'camera' in r)),
            'views':dict(Counter('left_large_yaw' if r['camera']['yaw_degrees'] < -30 else 'front_or_level'
                                for r in subset if 'camera' in r)),
            'input_640_paper_short_side_px':dict.fromkeys(['0_8','8_16','16_32','over_32'],0)}
    for row in records:
        image=read_image(stage/row['image']);h,w=image.shape[:2]
        text=(stage/row['label']).read_text(encoding='utf8');check_yolo(text,1,0)
        expected=[yolo_row(obj['xyxy'],w,h) for obj in row['instances']]
        if text.splitlines()!=expected:raise ValueError('Paired label differs from final objects: '+row['id'])
        for obj in row['instances']:
            a,b,c,d=obj['xyxy'];size=min(c-a,d-b)*640/max(w,h)
            key='0_8' if size<=8 else '8_16' if size<=16 else '16_32' if size<=32 else 'over_32'
            summary[row['split']]['input_640_paper_short_side_px'][key]+=1
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        if digest in rgb:raise ValueError('Identical decoded images: '+rgb[digest]['id']+' / '+row['id'])
        rgb[digest]=row
        if row.get('visible_mask'):
            mask=read_image(mask_stage/row['visible_mask'],cv2.IMREAD_GRAYSCALE)
            for obj in row['instances']:
                selected=mask==ord(obj['building_id'])-64
                if int(selected.sum())!=obj['visible_pixels'] or mask_box(selected)!=obj['xyxy']:
                    raise ValueError('Final paper mask differs from final label')
        if row['branch']=='blender_photo_composite':parents[row['source_group']].add(row['split'])
    overlap={key:sorted(value) for key,value in parents.items() if len(value)>1}
    if overlap:raise ValueError('Real-photo background parent crosses splits: '+str(overlap))
    for split in summary:
        if summary[split]['images']!=full.split_plan(split)['budget']:raise ValueError('Actual split budget differs')
        if len(structures[split])!=26:raise ValueError('Missing building/paper structural controls in '+split)
    return {'status':'passed','summary':summary,'unique_decoded_images':len(rgb),'real_background_parent_overlap':overlap,
            'real_background_parents':{key:sorted(value) for key,value in parents.items()},'structures':structures,
            'scope':'Complete paper detection only. Shared Blender meshes and derived views are not independent captures. Val/test are diagnostic with prior development exposure retained.'}


def source_identity(row):
    """A composite is a new image; its background remains separate lineage."""
    return {'workspace_image':row['source_image'] if row['branch']=='direct_real' else relative(OUT/row['image']),
            'background_image':row.get('source_photo') or (row.get('source_image') if row['branch']!='direct_real' else None)}


def training_fields(row):
    """Freeze actual paper boxes and known generation axes for portable reports."""
    camera=row.get('camera') or {}
    metadata={'schema_version':1,'image_size':row['image_size'],
        'instances':[{'xyxy':p['xyxy'],'building_id':p.get('building_id','unknown')} for p in row['instances']],
        'scene':{'branch':row['branch'],'buildings':row.get('buildings'),'papers':row.get('papers'),
                 'yaw_degrees':camera.get('yaw_degrees'),'camera_height_m':camera.get('height_m'),
                 'view_basis':'render_control_with_estimated_camera' if camera else 'unknown'}}
    result={'paper':metadata}
    if row['branch'] in ('blender_render','blender_photo_composite'):
        result['physical_assets']=['datasets/assets/blender/aiaec_dataset.blend#Buildings']
    return result


def declared_reuse(records,reuse):
    assets=defaultdict(set)
    for row in records:
        for asset in row.get('physical_assets',[]):assets[asset].add(row['split'])
    return reuse+[{'relation':'physical_assets','source':asset,'splits':sorted(splits),
                   'evaluation_scope':'same_source_diagnostic'} for asset,splits in assets.items() if len(splits)>1]


def refresh_source_index():
    index=OUT/'workspace_sources.json';manifest=OUT/'manifest.json'
    batch=full.read(manifest);before=full.read(index);by_image={relative(OUT/r['image']):r for r in batch['records']}
    media=[OUT/r['image'] for r in batch['records']]
    configs=[manifest,index,Path(__file__).parent/'config/source.yaml',Path(__file__).parent/'config/dataset.yaml']
    with publisher.sample_lock(OUT):
        with generation_guard('building_name.refresh_source_index',media,[index],{'revision':'paper_training_metadata_v2'},
            root=ROOT,code_paths=[Path(__file__),Path(package.__file__)],procedural_inputs=configs,inplace_paths=[index]) as run:
            if run.reused:return
            records=[{**r,**source_identity(by_image[r['image']]),**training_fields(by_image[r['image']])} for r in before['records']]
            reuse=package.source_policy(records,{relative(p):p for p in media},load_task_config(full.TASK)[1],read_source_history(ROOT))
            updated={**before,'records':records,'same_source_reuse':reuse,'source_reuse':declared_reuse(records,reuse),
                     'image_identity':'workspace_image is the actual derived image or exact direct capture; background_image is lineage'}
            temporary=index.with_suffix('.json.new');write(temporary,updated);temporary.replace(index)
            run.set_result({'records':len(records),'images_and_labels_changed':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replace',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--refresh-source-index',action='store_true');args=parser.parse_args()
    if args.refresh_source_index:
        refresh_source_index();return
    if args.audit_only:
        result=audit(OUT,full.read(OUT/'manifest.json')['records'],MASKS,full.read(OUT/'workspace_sources.json')['records'])
        print(json.dumps(result,ensure_ascii=False));return
    composer=Composer();plans={s:full.split_plan(s) for s in ('train','val','test')}
    states=[dict(s,split=split) for split,plan in plans.items() for s in plan['strata']]
    if any(s['parameters'].get('template_id') not in composer.records for s in states if 'template_id' in s['parameters']):
        raise ValueError('Compile final plans against the completed accepted-template manifest first')
    declaration=deepcopy(plans['train']);declaration.update(budget=len(states),strata=states,
        purpose='Full current paper dataset, 3000 train plus 300 val and 300 test; supervision excludes OCR')
    media={ROOT/path for state in states for path in state['sources'] if (ROOT/path).suffix.lower()!='.blend'}
    media.update(p for path in composer.manifests for p in path.parent.rglob('*') if p.is_file() and p.suffix.lower()=='.png')
    procedural={*composer.manifests,source_path(full.TASK,'blender_scene'),source_path(full.TASK,'font'),
        *[source_path(full.TASK,k) for k in ('full_recipe','full_plates','full_render_plan','full_train_plan','full_val_plan','full_test_plan')],
        Path(__file__).parent/'config/source.yaml',Path(__file__).parent/'config/dataset.yaml'}
    code=[Path(__file__),Path(full.__file__),Path(hybrid.__file__),Path(rooftop.__file__),Path(package.__file__),
        ROOT/'datasets/scripts/common/recoverable_publish.py',ROOT/'datasets/scripts/common/annotations.py',
        ROOT/'datasets/scripts/common/images.py',ROOT/'datasets/scripts/common/paper.py',ROOT/'datasets/scripts/common/perspective.py',
        ROOT/'datasets/scripts/common/frame_environment.py',ROOT/'datasets/scripts/common/config.py']
    operation='building_name.full_dataset';committed=lambda token:receipt_completed(ROOT,operation,token)
    with publisher.sample_lock(OUT),publisher.sample_lock(MASKS):
        publisher.recover(OUT,committed);publisher.recover(MASKS,committed)
        owned=owned_outputs();mask_owned=set()
        if (MASKS/'manifest.json').exists():mask_owned=set(full.read(MASKS/'manifest.json')['files'])|{'manifest.json'}
        preserved=[relative(p) for folder,names in [(OUT,owned),(MASKS,mask_owned)] for p in folder.rglob('*')
                   if p.is_file() and p.relative_to(folder).as_posix() not in names]
        try:
            with generation_guard(operation,sorted(media),[OUT,MASKS],{'budget':{s:p['budget'] for s,p in plans.items()}},
                root=ROOT,production_plan=declaration,code_paths=code,procedural_inputs=sorted(procedural),
                replace_outputs=args.replace,replace_verifier=(lambda *_:{'preserved_paths':preserved}) if args.replace else None,
                sample_output_paths=[OUT/s/'images' for s in plans],auxiliary_output_paths=[MASKS,*[OUT/f'{s}_contact_sheet.jpg' for s in plans]]) as run:
                if run.reused:return
                stage=publisher.prepare(OUT,run.token);mask_stage=publisher.prepare(MASKS,run.token)
                publisher.copy_unowned(OUT,stage,owned);publisher.copy_unowned(MASKS,mask_stage,mask_owned)
                for split in plans:
                    for sub in ('images','labels'):(stage/split/sub).mkdir(parents=True,exist_ok=True)
                    (mask_stage/split).mkdir(exist_ok=True)
                records=[];sources=[];images={};mask_files=[]
                # Group templates to reuse native pixel layers without changing image quotas.
                ordered=sorted(states,key=lambda s:(s['parameters'].get('template_id',''),s['id']))
                for number,state in enumerate(ordered):
                    stem=state['id'];split=state['split'];branch=state['branch'];extra={};source_group=None
                    if branch=='direct_real':
                        item=state['parameters']['direct'];source=ROOT/item['source_image'];pixels=read_image(source)
                        text=item['label_text'];h,w=pixels.shape[:2];objects=[]
                        for line in text.splitlines():
                            _,x,y,bw,bh=map(float,line.split());objects.append({'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
                        metadata=item['metadata'];source_group=metadata.get('source_group') or metadata.get('source_id')
                        extra={'source_image':item['source_image'],'source_review':metadata.get('frame_environment') or metadata.get('source_review')}
                        extension=source.suffix.lower()
                    elif branch=='real_background_composite':
                        base=state['parameters']['replacement_state'];candidate=base['background']['candidates'][0]
                        pixels,papers=rooftop.compose_one(read_image(ROOT/candidate['image']),candidate,base['target']['papers'],font_path=source_path(full.TASK,'font'))
                        import random
                        pixels,aug=rooftop.augment_scene(pixels,papers,random.Random(state['parameters']['seed']))
                        objects=[{'xyxy':p['box_xyxy'],'building_id':p['building_id']} for p in papers]
                        source_group=candidate['parent_source'];extra={'source_image':candidate['image'],'source_review':candidate['source_review'],'augmentation':aug};extension='.png'
                    else:
                        pixels,objects,mask,extra=composer.compose(state)
                        source_group=extra['source_group'];extension='.png'
                        mask_name=f'{split}/{stem}.png'
                        if (mask_stage/mask_name).exists():raise ValueError('Final mask would overwrite unowned content')
                        write_image(mask_stage/mask_name,mask);mask_files.append(mask_name)
                        extra.update(visible_mask=mask_name,buildings=state['target']['buildings'],papers=state['target']['papers'])
                    h,w=pixels.shape[:2];image=f'{split}/images/{stem}{extension}';label=f'{split}/labels/{stem}.txt'
                    if any((stage/p).exists() for p in (image,label)):raise ValueError('Dataset would overwrite unowned content')
                    if branch=='direct_real':shutil.copyfile(source,stage/image)
                    else:write_image(stage/image,pixels)
                    text=''.join(yolo_row(obj['xyxy'],w,h)+'\n' for obj in objects)
                    (stage/label).write_text(text,encoding='utf8')
                    row={'id':stem,'state_id':stem,'split':split,'branch':branch,'image':image,'label':label,
                         'instances':objects,'image_size':[w,h],'source_group':source_group,**extra}
                    records.append(row)
                    output_image=relative(OUT/image);output_label=relative(OUT/label)
                    origin={**source_identity(row),**training_fields(row),
                            'source_review':extra.get('source_review'),'source_kind':'camera' if branch=='direct_real' else branch}
                    if branch=='blender_render':origin.update(mother_scene='Buildings',camera_reference_photo=extra['source_photo'])
                    else:origin.update(parent_video=source_group)
                    source_row=package.reviewed_record(OUT/image,OUT/label,split,len(objects),
                        'Buildings' if branch=='blender_render' else source_group,OUT/split/'manifest.json',
                        generation_method=branch,reviewed_by='machine',**origin)
                    sources.append(source_row);images[output_image]=stage/image
                    if (number+1)%100==0:print('COMPOSED',number+1,len(states),flush=True)
                reuse=package.source_policy(sources,images,load_task_config(full.TASK)[1],read_source_history(ROOT))
                results=audit(stage,records,mask_stage,sources)
                artifacts=[contact_sheet(stage,records,s) for s in plans]
                for split in plans:
                    write(stage/split/'manifest.json',{'task':full.TASK,'split':split,'status':'complete_detection_records',
                        'label_scope':'paper_detection','plan':relative(source_path(full.TASK,'full_'+split+'_plan')),
                        'generation_plan':plans[split],'summary':results['summary'][split],
                        'records':[r for r in records if r['split']==split]})
                write(stage/'data.yaml',{'train':'train/images','val':'val/images','test':'test/images','names':['paper']})
                write(stage/'workspace_sources.json',{'model_key':full.TASK,'selection_mode':'manifest_only',
                    'records':sources,'same_source_reuse':reuse,'source_reuse':declared_reuse(sources,reuse),'scope':results['scope']})
                write(stage/'audit.json',results)
                write(stage/'manifest.json',{'task':full.TASK,'status':'complete_full_dataset','label_scope':'paper_detection',
                    'summary':results['summary'],'artifacts':artifacts,'plans':{s:relative(source_path(full.TASK,'full_'+s+'_plan')) for s in plans},
                    'layer_manifests':[relative(p) for p in composer.manifests],
                    'visible_mask_root':relative(MASKS),'records':records})
                write(mask_stage/'manifest.json',{'task':full.TASK,'files':mask_files})
                list(full.plates())
                publisher.publish(MASKS,mask_stage,run.token,keep_root=True);publisher.publish(OUT,stage,run.token,keep_root=True)
                by_id={s['id']:s for s in run.plan['strata']}
                for row in records:
                    state=by_id[row['state_id']]
                    run.record_sample(state['id'],output=OUT/row['image'],sources=state['sources'],parameters=state['parameters'],actual=row)
                run.set_result(results)
            publisher.recover(OUT,committed);publisher.recover(MASKS,committed)
        except BaseException:
            publisher.recover(OUT,committed);publisher.recover(MASKS,committed);raise
    print(json.dumps(results['summary'],ensure_ascii=False))


if __name__=='__main__':main()
