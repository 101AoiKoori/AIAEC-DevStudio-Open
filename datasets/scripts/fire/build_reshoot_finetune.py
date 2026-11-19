"""Build the fire fine-tuning dataset inside its existing portable package; never train."""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.fire import compose_real_facades as facade
from datasets.scripts.fire.build_dataset import paper_artwork

PLAN = ROOT / 'datasets/scripts/fire/config/reshoot_finetune_plan.json'
SESSION = 'bins_reshoot_20261118'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def relative(path, root=ROOT):
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


@lru_cache(maxsize=48)
def rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'))


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('Empty mask')
    return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]


def negative_variant(source, index, rng):
    image = rgb(str(ROOT/source['image'])).copy()
    h, w = image.shape[:2]
    a, b, c, d = source['bins'][0]['bbox_xyxy']
    bw, bh = c-a, d-b
    # Whole context, whole bin, and the confusing rim/upper-body region all occur.
    if index == 0:
        crop = [0, 0, w, h]
    elif index % 3 == 2:
        margin = rng.uniform(.08, .25)
        crop = [a-margin*bw, b-.13*bh, c+margin*bw, b+rng.uniform(.42, .72)*bh]
    else:
        margin = rng.uniform(.15, .65)
        crop = [a-margin*bw, b-margin*bh, c+margin*bw, d+margin*bh]
    x0,y0,x1,y1 = [int(v) for v in crop]
    crop = [max(0,x0), max(0,y0), min(w,x1), min(h,y1)]
    x0,y0,x1,y1 = crop
    image = image[y0:y1,x0:x1]
    long_side = (960, 720, 640, 960, 720, 480)[index % 6]
    scale = long_side/max(image.shape[:2])
    image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    gain = 1.0 if index == 0 else rng.uniform(.78,1.09)
    image = np.uint8(np.clip(image.astype(np.float32)*gain,0,255))
    if index % 6 == 5:
        image = cv2.GaussianBlur(image,(3,3),.45)
    return image, {'crop_xyxy':crop,'gain':gain,'output_size':list(image.shape[1::-1]), 'variant':index}


def compose_wall(row, source, paper, count, seed, target_px, *, full_frame=False, tone='normal'):
    local_row = deepcopy(row)
    if full_frame:
        local_row['geometry']['supervision_crop_xyxy'] = [0,0,*row['image_size']]
    building = row['geometry']['building']
    use = {'row':local_row, 'branch':'real_background_composite', 'count':count, 'tone':tone,
           'target_building':building, 'visible_buildings':[building]}
    recipe = deepcopy(source['sample_recipe'])
    recipe['max_image_side'] = 960
    # Jitter the rectified wall anchor; the existing projector enforces the wall boundary.
    recipe['sticker_size']['anchor_jitter_fraction'] = .028
    fraction = .08
    for _ in range(14):
        recipe['sticker_size']['height_fraction'][building] = [fraction*.99, fraction*1.01]
        try:
            image, instances, transform = facade.compose_camera(use,paper,recipe,seed)
        except ValueError as error:
            if 'discernibility size' not in str(error) and 'non-overlapping scaled sticker layout' not in str(error):
                raise
            fraction *= 1.2 if 'discernibility size' in str(error) else .78
            continue
        sizes = [min(i['xyxy'][2]-i['xyxy'][0], i['xyxy'][3]-i['xyxy'][1])*640/max(image.shape[:2]) for i in instances]
        low,high = (8,16) if target_px < 16 else ((16,32) if target_px < 32 else (24,64))
        if all(low < size <= high for size in sizes):
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),instances,transform,sizes
        fraction *= target_px/float(np.median(sizes))
    raise ValueError(f'Could not realize {target_px}px targets on {row["frame_id"]}')


def add_bin(image, source, instances, gain, rng):
    """Retain the observed bin perspective; place on visually reviewed foreground floor."""
    h,w = image.shape[:2]
    with Image.open(ROOT/source['cutout']) as im:
        cutout = np.asarray(im.convert('RGBA'))
    for _ in range(100):
        height = round(h*rng.uniform(.14,.25))
        width = round(cutout.shape[1]*height/cutout.shape[0])
        base_y = round(h*rng.uniform(.90,.965))
        left = rng.randint(round(.03*w),max(round(.03*w),round(.97*w)-width))
        top = base_y-height
        if top < 0 or left+width >= w:
            continue
        patch = cv2.resize(cutout,(width,height),interpolation=cv2.INTER_AREA)
        alpha = patch[:,:,3].astype(np.float32)/255
        full_mask = np.zeros((h,w),np.uint8)
        full_mask[top:base_y,left:left+width] = np.uint8(alpha*255)
        if any(np.any(full_mask[int(b):int(d),int(a):int(c)] > 0) for a,b,c,d in (i['xyxy'] for i in instances)):
            continue
        # These three full-frame parent photos have visible floor throughout the bottom 16%.
        floor = np.float32([[.02*w,.84*h],[.98*w,.84*h],[.98*w,.99*h],[.02*w,.99*h]])
        contact_x = left+width*.5
        if cv2.pointPolygonTest(floor,(contact_x,float(base_y)),False) < 0:
            continue
        shadow = np.zeros((h,w),np.float32)
        cv2.ellipse(shadow,(round(contact_x),base_y),(max(3,round(width*.52)),max(2,round(height*.028))),0,0,360,.20,-1)
        shadow = cv2.GaussianBlur(shadow,(0,0),max(1,width*.045))
        result = image.astype(np.float32)*(1-shadow[:,:,None])
        # Blend in linear light, keeping the original cutout silhouette and color.
        region = result[top:base_y,left:left+width]/255
        foreground = np.clip(patch[:,:,:3].astype(np.float32)*gain,0,255)/255
        to_linear = lambda v: np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
        mixed = to_linear(region)*(1-alpha[:,:,None])+to_linear(foreground)*alpha[:,:,None]
        result[top:base_y,left:left+width] = np.where(mixed<=.0031308,mixed*12.92,1.055*np.maximum(mixed,0)**(1/2.4)-.055)*255
        return np.uint8(np.clip(result,0,255)), full_mask, {'xyxy':mask_box(full_mask>0),
            'kind':'real_bin','color':source['color'],'state':source['state'],
            'hard_negative_type':'red_orange_bin' if source['color']=='red' else 'other_color_bin',
            'source_record_id':source['id'],'floor_polygon_xy':floor.tolist(),
            'floor_contact_xy':[contact_x,base_y],'foreground_gain':gain,
            'perspective':'source photo perspective retained; floor placement approximate, no camera calibration',
            'shadow':'soft contact ellipse on reviewed floor; approximate ambient illumination',
            'target_occlusion_pixels':0}
    raise ValueError('No floor placement retaining every sticker')


class Builder:
    def __init__(self, plan, *, verify_only=False):
        self.plan = plan
        self.package = ROOT/plan['base_package']
        self.out = ROOT/plan['output']
        self.review = ROOT/plan['review_output']
        if self.out.exists() and not verify_only:
            raise FileExistsError('Fine-tune output exists; inspect its manifest before replacing: '+str(self.out))
        self.rows = []
        self.raw_sources = []
        self.review.mkdir(parents=True,exist_ok=True)
        self.original = json.loads((self.package/'dataset/workspace_sources.json').read_text(encoding='utf8'))
        self.original_sha = digest(self.package/'dataset/workspace_sources.json')
        if plan.get('base_source_manifest_sha256',self.original_sha)!=self.original_sha:
            raise ValueError('The pinned baseline source manifest changed')
        self.annotations = json.loads((ROOT/plan['annotation_manifest']).read_text(encoding='utf8'))['records']
        self.source = load_task_config('fire')[1]
        self.paper = paper_artwork(self.source)
        self.generated = 0

    def replay(self):
        for source in self.original['records']:
            row = deepcopy(source)
            for key in ('image','label'):
                old = self.package/source[key]
                new = self.out/Path(source[key]).relative_to('dataset')
                new.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(old,new)
                row[key] = relative(new,self.package)
            row['fine_tune_group'] = 'replay' if row['split']=='train' else 'original_regression'
            row['replay_origin'] = {'image':source['image'],'label':source['label'],
                                    'image_sha256':digest(self.package/source['image']),
                                    'label_sha256':digest(self.package/source['label'])}
            self.rows.append(row)
        print('REPLAY copied 3600 original pairs',flush=True)

    def sources(self):
        if len(self.annotations)!=36 or Counter(r['color'] for r in self.annotations)!={'red':8,'blue':8,'green':8,'gray':12}:
            raise ValueError('Reshoot inventory changed')
        for row in self.annotations:
            if row['capture_session']!=SESSION or row['split']!='train' or digest(ROOT/row['source_image'])!=row['source_sha256']:
                raise ValueError('Reshoot parent/hash changed')
            if digest(ROOT/row['mask'])!=row['mask_sha256']:
                raise ValueError('Mask bytes changed')
            proof = deepcopy(row)
            for key in ('image','mask','cutout'):
                old = ROOT/row[key];new = self.out/'source_records/reshoot'/key/old.name
                new.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,new)
                proof[key] = relative(new,self.package)
            proof['fire_review'] = {'task':'fire','presence':'absent','completeness':'complete',
                'scope':'full_original_photo','origin':'machine','human_confirmed':False,
                'method':'Codex visual review of all 36 source photographs; fixed fire artwork absent',
                'physical_parent_group':SESSION}
            self.raw_sources.append(proof)
        ids = {r.get('frame_id') for r in self.original['records'] if r['split']=='train' and r['branch']=='real_background_composite'}
        self.walls = [r for r in query_frames(task='fire',role='composite_background',split='train') if r['frame_id'] in ids]
        self.by_building = defaultdict(list)
        for row in self.walls:
            if row['label'].get('instances') or row['label'].get('completeness')!='complete':
                raise ValueError('Wall lacks complete source absence labels')
            if digest(ROOT/row['image'])!=row['content_sha256']:
                raise ValueError('Wall source changed')
            self.by_building[row['geometry']['building']].append(row)
            target = self.out/'source_records/walls'/f'{row["frame_id"]}{Path(row["image"]).suffix}'
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/row['image'],target)
        write_json(self.out/'source_records/reshoot_review.json',{'records':self.raw_sources,'physical_parent_group':SESSION,'scope':'fire-specific complete absence review; other-task sticker labels are not fire targets'})
        wall_proofs=deepcopy(self.walls)
        for row in wall_proofs:
            row['package_image']=relative(self.out/'source_records/walls'/f'{row["frame_id"]}{Path(row["image"]).suffix}',self.package)
        write_json(self.out/'source_records/wall_geometry.json',{'records':wall_proofs,'floor_review':{
            'frame_prefixes':self.plan['paired_wall_frame_prefixes'], 'method':'Codex visual full-frame floor review',
            'polygon_normalized':[[.02,.84],[.98,.84],[.98,.99],[.02,.99]],'camera_calibration':'unknown; placement approximate'}})
        shutil.copy2(ROOT/self.source['target_artwork'],self.out/'source_records/artwork.png')

    def add(self, group, image, instances, parents, **fields):
        self.generated += 1
        stem=f'finetune_{group}_{self.generated:06d}'
        ip=self.out/'images/train'/f'{stem}.jpg';lp=self.out/'labels/train'/f'{stem}.txt'
        Image.fromarray(image).save(ip,quality=95,subsampling=0)
        h,w=image.shape[:2]
        lp.write_text(''.join(facade.yolo_row(i['xyxy'],w,h)+'\n' for i in instances),encoding='utf8')
        row={'image':relative(ip,self.package),'label':relative(lp,self.package), 'split':'train','source_split':'train',
            'source_group':parents[0], 'parents':parents,'parent_samples':parents,
            'instances':instances,'building_counts':facade.counts_for(sorted({i['building'] for i in instances}),instances,'visible_stickers_on_registered_wall'),
            'fine_tune_group':group,'production_state_id':stem,'image_size':[w,h],
            'supervision_scope':'entire_output_frame','label_status':'reviewed',
            'review_state':'complete_positive' if instances else 'complete_negative',
            'review_origin':'machine','human_confirmed':False,
            'review_evidence':'dataset_finetune/verification.json',
            'development_contact':['fire_finetune_generation_and_visual_review'],**fields}
        self.rows.append(row)
        if self.generated%60==0: print('GENERATED',self.generated,group,flush=True)
        return row

    def negatives(self):
        for index,source in enumerate(self.annotations):
            for variant in range(12 if source['color']=='red' else 6):
                seed=self.plan['seed']+index*101+variant
                image,transform=negative_variant(source,variant,random.Random(seed))
                self.add('real_bin_negative',image,[],[SESSION],branch='direct_real',generation_method='camera_crop_gain_variant',
                    source_record_id=source['id'],source_sha256=source['source_sha256'],
                    source_image=next(r['image'] for r in self.raw_sources if r['id']==source['id']),
                    transform=transform,hard_negative=True,sample_seed=seed,
                    context={'distractor':'red_orange_bin' if source['color']=='red' else 'other_color_bin'},
                    distractor={'kind':'real_bin','color':source['color'],'state':source['state']})

    def positives(self):
        eligible = [s for s in self.annotations if s['composite_eligible'] and not s['truncated']]
        colors = {c:[s for s in eligible if s['color']==c] for c in ('red','blue','green','gray')}
        for bi,building in enumerate('ABC'):
            wall = next(r for r in self.by_building[building] if r['frame_id'].startswith(self.plan['paired_wall_frame_prefixes'][building]))
            for index in range(60):
                count = 1 if index%2==0 else 3
                color = 'red' if index<30 else ('blue','green','gray')[(index-30)%3]
                source = colors[color][(index//2+bi*3)%len(colors[color])]
                seed=self.plan['seed']+20000+bi*1000+index
                target_px=24 if count==3 or building=='C' else 36
                image,instances,transform,sizes=compose_wall(wall,self.source,self.paper,count,seed,target_px,full_frame=True,
                    tone='dim' if index%5==0 else 'normal')
                image,mask,prop=add_bin(image,source,instances,transform['brightness_gain'],random.Random(seed))
                row=self.add('positive_with_real_bin',image,instances,[wall['parent_group'],SESSION],
                    branch='real_background_composite',generation_method='registered_wall_fire_and_real_bin_composite',
                    frame_id=wall['frame_id'],transform=transform,sample_seed=seed,contrast_positive=True,
                    context={'distractor':prop['hard_negative_type']},distractor=prop,
                    size_control={'actual_input_short_sides_px':sizes},source_record_id=source['id'])
                mp=self.out/'masks/train'/f'{Path(row["image"]).stem}_bin.png';mp.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(mask).save(mp);row['distractor']['mask']=relative(mp,self.package)

    def small(self):
        for bi,building in enumerate('ABC'):
            walls=sorted(self.by_building[building],key=lambda r:r['frame_id'])
            for index in range(60):
                wall=walls[(index//2)%len(walls)]
                seed=self.plan['seed']+40000+bi*1000+index
                target=12 if index<40 else 24
                image,instances,transform,sizes=compose_wall(wall,self.source,self.paper,1 if index%2==0 else 3,seed,target,
                    tone='dim' if index%4==0 else 'normal')
                self.add('small_sticker',image,instances,[wall['parent_group']],branch='real_background_composite',
                    generation_method='registered_wall_small_fire_composite',frame_id=wall['frame_id'],transform=transform,
                    sample_seed=seed,size_control={'target_input_short_side_px':target,'actual_input_short_sides_px':sizes})

    def compile_plan(self, plan_path=PLAN):
        groups=Counter()
        for r in self.rows:
            if r['split']!='train':continue
            counts=tuple(sorted(Counter(i['building'] for i in r['instances']).items()))
            context=r.get('context',{}).get('distractor','unknown')
            groups[(r['fine_tune_group'],r['branch'],len(r['instances']),counts,context)]+=1
        strata=[]
        for index,((group,branch,count,counts,context),number) in enumerate(sorted(groups.items())):
            strata.append({'id':f'{group}_{index:04d}','min_count':number,'max_count':number,
                'failure_event':'fire_sticker_miss_or_color_confusion',
                'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,
                'target':{'count':count, **({'fire_counts':dict(counts)} if counts else {})},
                'context':{'distractor':context},'parameters':{'generation_method':branch},
                'sources':[group],'conditions':[]})
        self.plan['strata']=strata
        self.plan['base_source_manifest_sha256']=self.original_sha
        self.plan['initial_weights_sha256']=digest(ROOT/self.plan['initial_weights'])
        write_json(plan_path,self.plan)
        write_json(self.out/'generation_plan.json',self.plan)

    def verify(self):
        counts=Counter(); groups=Counter(); parents=defaultdict(set); image_hashes=defaultdict(set)
        building=defaultdict(Counter);extra_boxes=Counter(); sizes=defaultdict(Counter)
        group_buildings=defaultdict(Counter);negative_counts=Counter();new_hashes=[]
        failures=[]; unchanged=Counter()
        for row in self.rows:
            split=row['split'];counts[split]+=1;groups[(split,row['fine_tune_group'])]+=1
            ip=self.package/row['image'];lp=self.package/row['label']
            with Image.open(ip) as im:
                im.load();w,h=im.size
            values=[list(map(float,l.split())) for l in lp.read_text().splitlines() if l.strip()]
            negative_counts[split]+=not bool(values)
            if len(values)!=len(row['instances']):failures.append('label_count:'+row['image'])
            for fields,item in zip(values,row['instances']):
                if len(fields)!=5 or fields[0]!=0 or item['building'] not in 'ABC':failures.append('class_or_building:'+row['image']);continue
                _,x,y,bw,bh=fields;box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
                if not (bw>0 and bh>0 and min(box)>=-.02 and box[2]<=w+.02 and box[3]<=h+.02):failures.append('bounds:'+row['image'])
                if max(abs(a-b) for a,b in zip(box,item['xyxy']))>.03:failures.append('label_geometry:'+row['image'])
                building[split][item['building']]+=1
                group_buildings[row['fine_tune_group']][item['building']]+=1
                px=min(bw*w,bh*h)*640/max(w,h);bucket='8_16' if 8<px<=16 else '16_32' if 16<px<=32 else 'other'
                sizes[row['fine_tune_group']][bucket]+=1
                if row['fine_tune_group'] not in ('replay','original_regression'):
                    extra_boxes[row['fine_tune_group']]+=1
                    quad=np.float32(item['quad_source_px']);boundary=np.float32(item['wall_boundary_px'])
                    if any(cv2.pointPolygonTest(boundary,tuple(map(float,p)),True)<2.8 for p in quad):failures.append('off_wall:'+row['image'])
                    if next(r for r in self.walls if r['frame_id']==row['frame_id'])['geometry']['building']!=item['building']:failures.append('wall_building:'+row['image'])
            for name,meta in row.get('building_counts',{}).items():
                if meta['visible_count']!=sum(i['building']==name for i in row['instances']):failures.append('building_count:'+row['image'])
            for parent in row.get('parents',row.get('parent_samples',[])):parents[parent].add(split)
            ih=digest(ip);image_hashes[ih].add(split)
            if row.get('image_sha256',ih)!=ih or row.get('label_sha256',digest(lp))!=digest(lp):failures.append('declared_hash_changed:'+row['image'])
            if row['fine_tune_group'] not in ('replay','original_regression'):new_hashes.append(ih)
            row['image_sha256']=ih;row['label_sha256']=digest(lp)
            if 'replay_origin' in row:
                if ih!=row['replay_origin']['image_sha256'] or digest(lp)!=row['replay_origin']['label_sha256']:failures.append('changed_replay:'+row['image'])
                unchanged[split]+=1
            if row['fine_tune_group']=='positive_with_real_bin':
                mask=np.array(Image.open(self.package/row['distractor']['mask']))
                for a,b,c,d in (i['xyxy'] for i in row['instances']):
                    if np.any(mask[int(b):int(d),int(a):int(c)]):failures.append('occlusion:'+row['image'])
        overlap={p:sorted(v) for p,v in parents.items() if len(v)>1}
        hash_overlap=[k for k,v in image_hashes.items() if len(v)>1]
        if overlap or hash_overlap:failures.append('cross_split_overlap')
        if counts!={'train':self.plan['budget'],'val':300,'test':300}:failures.append('split_counts')
        expected={key:self.plan['train'][source] for key,source in
                  [('replay','replay'),('real_bin_negative','real_bin_negatives'),
                   ('positive_with_real_bin','positive_with_real_bin'),('small_sticker','small_sticker')]}
        if {g:n for (s,g),n in groups.items() if s=='train'}!=expected:failures.append('train_groups')
        if extra_boxes!={'positive_with_real_bin':360,'small_sticker':360}:failures.append('extra_boxes')
        for group in ('positive_with_real_bin','small_sticker'):
            if group_buildings[group]!={'A':120,'B':120,'C':120}:failures.append('new_building_balance:'+group)
        if sizes['small_sticker']!={'8_16':240,'16_32':120}:failures.append('small_target_size_bins')
        if len(new_hashes)!=len(set(new_hashes)):failures.append('duplicate_generated_image')
        for split in ('train','val','test'):
            for key,folder in (('image','images'),('label','labels')):
                actual={relative(p,self.package) for p in (self.out/folder/split).iterdir() if p.is_file()}
                expected={r[key] for r in self.rows if r['split']==split}
                if actual!=expected:failures.append('unpaired_files:'+split+'/'+folder)
        report={'status':'passed' if not failures else 'failed','errors':failures,'splits':dict(counts),
            'groups':{f'{s}/{g}':n for (s,g),n in groups.items()},'original_pairs_byte_identical':dict(unchanged),
            'building_instances':{s:dict(v) for s,v in building.items()},'extra_instances':dict(extra_boxes),
            'building_instances_by_group':{s:dict(v) for s,v in group_buildings.items()},
            'negative_images':dict(negative_counts),'generated_exact_duplicates':len(new_hashes)-len(set(new_hashes)),
            'input_short_side_bins':{s:dict(v) for s,v in sizes.items()},'parent_groups':len(parents),
            'cross_split_parents':overlap,'cross_split_image_sha256':hash_overlap,
            'reshoot_parent_splits':sorted(parents[SESSION]),'source_photos':len(self.annotations),
            'training_started':False,'model_inference_run':False,
            'scope':'All pairs decoded and compared to metadata; new wall quadrilaterals/attribution and real-bin occlusion checked. Whole-building counts remain unknown. Visual montage review recorded separately.'}
        if (self.out/'visual_review.json').is_file():
            visual=json.loads((self.out/'visual_review.json').read_text(encoding='utf8'))
            report.update(visual_review=relative(self.out/'visual_review.json',self.package),
                          visual_reviewed_generated_images=visual['generated_images_reviewed'])
        write_json(self.out/'verification.json',report);write_json(self.review/'verification.json',report)
        if failures:raise ValueError(str(failures[:10]))
        return report

    def previews(self):
        selected=[]
        for group in ('real_bin_negative','positive_with_real_bin','small_sticker'):
            pool=[r for r in self.rows if r['fine_tune_group']==group]
            if group=='real_bin_negative':chosen=[pool[i] for i in range(0,len(pool),max(1,len(pool)//12))][:12]
            else:
                chosen=[]
                for b in 'ABC':
                    rs=[r for r in pool if r['instances'][0]['building']==b]
                    chosen.extend(rs[i] for i in (0,1,40,41))
            selected.extend(chosen)
        for k in range(0,len(selected),12):
            sheet=Image.new('RGB',(1440,1280),'#ddd');draw=ImageDraw.Draw(sheet)
            for j,row in enumerate(selected[k:k+12]):
                im=Image.open(self.package/row['image']).convert('RGB');d=ImageDraw.Draw(im)
                for item in row['instances']:
                    d.polygon([tuple(p) for p in item['quad_xy']],outline='lime',width=2)
                    d.text(tuple(item['xyxy'][:2]),item['building'],fill='lime')
                im.thumbnail((350,285));x=j%4*360;y=j//4*420;sheet.paste(im,(x+(360-im.width)//2,y))
                draw.text((x+3,y+292),Path(row['image']).stem[-42:],fill='black')
                draw.text((x+3,y+308),str(Counter(i['building'] for i in row['instances'])),fill='black')
            sheet.save(self.out/f'contact_sheet_{k//12+1}.jpg',quality=92)

    def finish(self, report):
        write_json(self.out/'workspace_sources.json',{'task':'fire','model_key':'fire_sticker','records':self.rows,
            'plan':relative(PLAN),'generation_plan':'dataset_finetune/generation_plan.json',
            'source_notes':[self.plan['independence_basis']],'training_started':False})
        write_json(self.out/'manifest.json',{'task':'fire','status':'generated_pending_visual_review',
            'plan':relative(PLAN),'generation_plan':'dataset_finetune/generation_plan.json',
            'workspace_sources':'dataset_finetune/workspace_sources.json','verification':report,
            'base_manifest_sha256':self.original_sha,'initial_weights_sha256':self.plan['initial_weights_sha256']})
        (self.out/'data.yaml').write_text(yaml.safe_dump({'names':['fire_sticker'],'train':'images/train','val':'images/val','test':'images/test'},sort_keys=False),encoding='utf8')
        initial=self.package/'weights/finetune_initial.pt'
        if initial.exists() and digest(initial)!=self.plan['initial_weights_sha256']:
            raise ValueError('Existing fine-tune initial weight differs; preserve it before choosing a new run')
        shutil.copy2(ROOT/self.plan['initial_weights'],initial)
        # Rebuilding the preserved pool must not replace a newer selected training profile.
        if self.source.get('fine_tune_plan',relative(PLAN))!=relative(PLAN):
            print(json.dumps({'status':'generated_pending_visual_review','verification':report},ensure_ascii=False),flush=True)
            return
        config=yaml.safe_load((self.package/'config/train.yaml').read_text(encoding='utf8'))
        config.update(data='dataset_finetune/data.yaml',model='weights/finetune_initial.pt',
            description='Real-bin and small-fire-sticker fine-tuning; prepared only, user has not authorized training.',
            epochs=40,batch=4,lr0=.00004,warmup_epochs=1,warmup_bias_lr=0.0,evaluate_after_training=True,
            training_policy={'mode':'early_stopping','min_epochs':10,'patience':8,'min_delta':.001,'save_period':5},
            source_notes=[self.plan['independence_basis'],'Fresh optimizer from remote best.pt; never resume a stripped checkpoint.'])
        (self.package/'config/finetune.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
        (self.package/'config/finetune_launcher.yaml').write_text(yaml.safe_dump({'train_config':'config/finetune.yaml','wall_seconds':7200,'tensorboard':False,'open_browser':False,'port':0,'keep_tensorboard_open':False},sort_keys=False),encoding='utf8')
        workspace_config=deepcopy(config)
        for key in ('data','model'):workspace_config[key]=relative(self.package/config[key])
        (ROOT/'lab/features/fire/training/config/fire_sticker_reshoot_finetune_v4.yaml').write_text(yaml.safe_dump(workspace_config,allow_unicode=True,sort_keys=False),encoding='utf8')
        print(json.dumps({'status':'generated_pending_visual_review','verification':report},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true',help='Recheck existing pairs and provenance without regeneration or training')
    args=parser.parse_args()
    plan=json.loads(PLAN.read_text(encoding='utf8'))
    builder=Builder(plan,verify_only=args.verify)
    if args.verify:
        builder.rows=json.loads((builder.out/'workspace_sources.json').read_text(encoding='utf8'))['records']
        builder.walls=json.loads((builder.out/'source_records/wall_geometry.json').read_text(encoding='utf8'))['records']
        print(json.dumps(builder.verify(),ensure_ascii=False));return
    builder.replay();builder.sources();builder.negatives();builder.positives();builder.small()
    builder.compile_plan();report=builder.verify();builder.previews();builder.finish(report)


if __name__=='__main__':
    main()
