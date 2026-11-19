"""Mine qualified Train backgrounds and build paired waste-sticker fine-tuning data."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
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
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.images import read_image, write_image
from datasets.scripts.waste_sticker.assemble_full_dataset import labels, write_json
from datasets.scripts.waste_sticker.compose_sticker_real_plate import image8, linear
from datasets.scripts.waste_sticker.produce_surface_dataset import paper_on_face
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog

SOURCE_CONFIG = ROOT / 'datasets/scripts/waste_sticker/config/source.yaml'
PLAN = ROOT / yaml.safe_load(SOURCE_CONFIG.read_text(encoding='utf-8'))['finetune_plan']
VERIFIED = set()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def existing_path(record):
    path = next(ROOT / p for p in record['paths'] if (ROOT / p).is_file())
    key = (record['frame_id'], str(path))
    if key not in VERIFIED:
        if sha256(path.read_bytes()).hexdigest() != record['content_sha256']:
            raise ValueError('Source bytes differ from frame evidence: ' + str(path))
        VERIFIED.add(key)
    return path


def complete_empty(record):
    label = record.get('label', {})
    if label.get('presence') != 'absent' or label.get('completeness') != 'complete' or label.get('instances'):
        raise ValueError('Source lacks complete sticker absence evidence: ' + record['frame_id'])


def pixel_box(box, width, height):
    a,b,d,e=box
    if not (-.01<=a<d<=width+.01 and -.01<=b<e<=height+.01):
        raise ValueError('Box exceeds image beyond YOLO decimal rounding tolerance')
    return [max(0,a),max(0,b),min(width,d),min(height,e)]


def ground_stripes(image, polygon, seed, *, cards=False):
    """Project synthetic non-target paper/tape onto a declared ground patch."""
    rng = random.Random(seed)
    h, w = image.shape[:2]
    texture = np.zeros((512, 512), np.uint8)
    if cards:
        for _ in range(rng.randint(1, 4)):
            x, y = rng.randint(20, 380), rng.randint(20, 380)
            cv2.rectangle(texture, (x, y), (x+rng.randint(50, 110), y+rng.randint(45, 115)), 255, -1)
    else:
        spacing = rng.randint(65, 140)
        for x in range(rng.randint(-70, 20), 512, spacing):
            top = rng.randint(0, 150)
            cv2.rectangle(texture, (x, top), (x+rng.randint(22, max(23, spacing-20)), 511), 255, -1)
        if rng.random() < .45:
            cv2.rectangle(texture, (0, rng.randint(25, 140)), (511, rng.randint(145, 185)), 255, -1)
    quad = np.asarray(polygon, np.float32)
    transform = cv2.getPerspectiveTransform(np.float32([[0,0],[511,0],[511,511],[0,511]]), quad)
    mask = cv2.warpPerspective(texture, transform, (w, h)).astype(np.float32)/255
    mask = cv2.GaussianBlur(mask, (3,3), .4) * rng.uniform(.88, 1)
    # Preserve camera texture within the pale tape instead of laying opaque flat white.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    white = np.clip(rng.uniform(225, 249) + .09*(gray-gray.mean()), 180, 254)
    result = image.astype(np.float32)*(1-mask[:,:,None]) + white[:,:,None]*mask[:,:,None]
    return np.clip(result, 0, 255).astype(np.uint8)


def paper_on_image(image, face, quad, artwork=None):
    """Apply a white card or the unchanged fixed artwork with the same paper footprint."""
    h, w = image.shape[:2]
    paper = np.full((128,128,3), 240, np.uint8)
    if artwork is not None:
        icon = read_image(artwork, cv2.IMREAD_UNCHANGED)
        if icon.ndim != 3 or icon.shape[2] != 4:
            raise ValueError('Fixed artwork must have alpha')
        scale = min(102/icon.shape[1], 102/icon.shape[0])
        iw, ih = max(1, round(icon.shape[1]*scale)), max(1, round(icon.shape[0]*scale))
        icon = cv2.resize(icon, (iw,ih), interpolation=cv2.INTER_AREA)
        alpha = icon[:,:,3:4].astype(np.float32)/255
        x, y = (128-iw)//2, (128-ih)//2
        paper[y:y+ih,x:x+iw] = np.rint(icon[:,:,:3]*alpha+paper[y:y+ih,x:x+iw]*(1-alpha)).astype(np.uint8)
    transform = cv2.getPerspectiveTransform(np.float32([[0,0],[127,0],[127,127],[0,127]]), np.float32(quad))
    warped = cv2.warpPerspective(paper, transform, (w,h))
    mask = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(quad,np.int32), 255)
    for polygon in face['occluder_polygons']:
        cv2.fillPoly(mask, [np.asarray(polygon,np.int32)], 0)
    y,x = np.nonzero(mask)
    if not len(x): raise ValueError('Invisible paper')
    image[mask>0] = warped[mask>0]
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]


def mine(plan, out):
    """Use the baseline only on eligible Train negatives; predictions never create truth."""
    target = out/'mining.json'
    if target.exists():
        result = json.loads(target.read_text(encoding='utf-8'))
        if result['weights_sha256'] != plan['initial_weights_sha256']: raise ValueError('Different mining checkpoint')
        return result
    from lab.training_common.engine.runtime import run_context, prepare_ultralytics
    import torch
    torch.set_num_threads(2)
    records = query_frames(task='waste_sticker', role='empty_negative', split='train')
    for row in records: complete_empty(row)
    output = []
    with run_context(ROOT, {'phase':'negative_mining'}, out/'mining_runtime', 'mining'):
        model = prepare_ultralytics(out/'mining_runtime')(str(ROOT/plan['initial_weights']))
        device = '0' if torch.cuda.is_available() else 'cpu'
        for i,row in enumerate(records):
            image = existing_path(row)
            result = model.predict(str(image), imgsz=960, conf=.10, iou=.7, max_det=32,
                                   device=device, verbose=False, save=False)[0]
            predictions = [{'class_id':int(c),'confidence':float(p),'xyxy':b}
                           for c,p,b in zip(result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist(),
                                            result.boxes.xyxy.cpu().tolist())]
            output.append({'frame_id':row['frame_id'],'record_id':row['record_id'],
                           'source_image':relative(image),'parent_group':row['parent_group'],
                           'source_kind':row.get('source_kind'),'predictions':predictions,
                           'false_at_025':sum(p['confidence']>=.25 for p in predictions),
                           'max_confidence':max((p['confidence'] for p in predictions),default=0)})
            if i%50 == 0: print('MINED',i+1,'/',len(records),flush=True)
    result = {'weights_sha256':plan['initial_weights_sha256'],'split':'train',
              'truth_basis':'existing complete absence evidence, not model predictions','records':output}
    write_json(target,result)
    return result


class Builder:
    def __init__(self, plan, out, plan_path=PLAN):
        self.plan,self.out=plan,out
        self.plan_path=plan_path
        self.rng=random.Random(plan['seed'])
        self.rows=[]
        self.hashes={}
        self.names=yaml.safe_load((ROOT/plan['base_dataset']/'data.yaml').read_text(encoding='utf-8'))['names']
        self.art=artwork_catalog()
        self.base=json.loads((ROOT/plan['base_dataset']/'workspace_sources.json').read_text(encoding='utf-8'))
        self.base_by_image={r['image']:r for r in self.base['records']}

    def add(self, split, group, image, boxes, parent, *, sources, pair_id=None, extra=None, original=None):
        stem=f'{group}_{sum(r["split"]==split for r in self.rows):05d}'
        ip=self.out/split/'images'/(stem+'.png'); lp=self.out/split/'labels'/(stem+'.txt')
        ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
        h,w=image.shape[:2]
        digest=sha256(image.tobytes()).hexdigest()
        if digest in self.hashes: raise ValueError('Duplicate RGB: '+stem+' / '+self.hashes[digest])
        self.hashes[digest]=stem
        lines=[]
        for item in boxes:
            c=item['class_id'];a,b,d,e=pixel_box(item['bbox_xyxy'],w,h)
            if not (0<=c<24 and 0<=a<d<=w and 0<=b<e<=h): raise ValueError('Invalid composed box')
            item['bbox_xyxy']=[a,b,d,e]
            lines.append(f'{c} {(a+d)/(2*w):.8f} {(b+e)/(2*h):.8f} {(d-a)/w:.8f} {(e-b)/h:.8f}')
        write_image(ip,image);lp.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        record=deepcopy(original) if original else {}
        record.update(image=relative(ip),label=relative(lp),split=split,source_group=parent,source_id=parent,
                      workspace_image=relative(ip),workspace_label=relative(lp),
                      label_status='reviewed',review_state='complete_positive' if boxes else 'complete_negative',
                      review_evidence=relative(self.out/'manifest.json'),source_review_evidence=relative(self.out/'manifest.json'),
                      review_method='inherited_complete_source_labels_and_geometric_generation_audit',
                      human_review_claimed=False,group=group,pair_id=pair_id,sources=sources,
                      image_size=[w,h],instances=boxes,decoded_rgb_sha256=digest,
                      source_assets=list(set(record.get('source_assets',[])+['fixed_24_item_artworks'])),
                      evaluation_scope='controlled_known_parent_diagnostic')
        if not original:
            record.update(source_kind='derived_image',generation_method='composite',
                          image_processing={'background_origin':'direct_camera' if parent.lower().endswith('.mp4') else 'blender_render',
                                            'origin_review':'confirmed'})
        if extra: record.update(extra)
        self.rows.append(record)
        return record

    def old_records(self, split):
        return [r for r in self.base['records'] if r['split']==split]

    def copy_old(self, split, rows, group):
        for old in rows:
            image=read_image(ROOT/old['image']);h,w=image.shape[:2]
            boxes=labels(ROOT/old['label'],(w,h))
            self.add(split,group,image,boxes,old['source_group'],sources=[old['image'],old['label']],original=old)

    def replay(self, amount):
        pool=defaultdict(list)
        for r in self.old_records('train'):
            n=len((ROOT/r['label']).read_text(encoding='utf-8').splitlines())
            if n: pool[n].append(r)
        chosen=[];counts=Counter()
        for n in range(1,5):
            self.rng.shuffle(pool[n])
            candidates=[(r,[int(line.split()[0]) for line in (ROOT/r['label']).read_text().splitlines()]) for r in pool[n]]
            for _ in range(amount//4):
                k=min(range(len(candidates)),key=lambda j:sum(counts[c] for c in candidates[j][1])/n)
                row,ids=candidates.pop(k);chosen.append(row);counts.update(ids)
        self.copy_old('train',chosen,'replay_positive')

    def hybrid_pairs(self, split, amount):
        folder=ROOT/'datasets/annotated/waste_bin/waste_sticker_full/hybrid'/('test' if split=='test' else 'train')
        generated=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))['records']
        layers=folder/'layers'; lm=json.loads((layers/'manifest.json').read_text(encoding='utf-8'))
        templates={r['id']:r for r in lm['records']}
        by_count=defaultdict(list)
        for row in generated: by_count[row['visible_sticker_count']].append(row)
        selected=[]
        counts=sorted(by_count)
        for n in counts:
            self.rng.shuffle(by_count[n]);selected.extend(by_count[n][:amount//len(counts)])
        for index,row in enumerate(selected):
            spec=templates[row['template_id']];w,h=row['image_size']
            photo=cv2.resize(read_image(ROOT/row['source_photo']),(w,h),interpolation=cv2.INTER_AREA)
            quad=np.float32(row['ground_polygon'])*[w/row['source_size'][0],h/row['source_size'][1]]
            seed=self.plan['seed']+index+(20000 if split=='test' else 0)
            plate=ground_stripes(photo,quad,seed)
            transform=np.float32(row['variant']['affine'])
            rgba=cv2.warpAffine(read_image(layers/spec['foreground'],cv2.IMREAD_UNCHANGED),transform,(w,h))
            shadow=cv2.warpAffine(read_image(layers/spec['shadow_attenuation'],cv2.IMREAD_GRAYSCALE).astype(np.float32)/255,
                                  transform,(w,h),borderMode=cv2.BORDER_CONSTANT,borderValue=1)
            alpha=rgba[:,:,3:4].astype(np.float32)/255
            subject=rgba[:,:,:3].astype(np.float32)/255
            luma=subject @ np.array([.0722,.7152,.2126],np.float32)
            subject=.82*subject+.18*luma[:,:,None]
            composed=image8(linear(plate.astype(np.float32)/255)*shadow[:,:,None]*(1-alpha)+.96*linear(subject)*alpha)
            pair=f'{split}_hybrid_{index:04d}'
            common=dict(parent=row['source_parent'],sources=[row['source_photo'],relative(layers/'manifest.json')],pair_id=pair,
                        extra={'template_id':row['template_id'],'source_record_id':row['source_decision'],
                               'stripe_seed':seed,'stripe_quad':quad.tolist(),'mother_scene':row['source_blend'],
                               'white_distractor':('projected_white_road_markings' if np.ptp(quad,axis=0)[1] >= h*.1
                                                   else 'local_white_graphic_stripes')})
            self.add(split,'hybrid_pair_negative',plate,[],**common)
            boxes=labels(folder/row['label'],(w,h))
            self.add(split,'hybrid_pair_positive',composed,boxes,**common)

    def surface_pairs(self, split, amount):
        sources=[r for r in query_frames(task='waste_sticker',role='composite_background',split=split)
                 if r.get('geometry',{}).get('bins')]
        sizes=self.plan.get('surface_sizes_at_960',{}).get(split)
        if sizes:
            sources=[r for r in sources if r.get('source_kind')!='blender_render']
            if not sources or amount != 24*len(sizes):
                raise ValueError('Scale-balanced pairs require real surfaces and one pair per class/size')
        for index in range(amount):
            row=sources[index%len(sources)];complete_empty(row)
            photo=existing_path(row);image=read_image(photo);h,w=image.shape[:2]
            count=1 if sizes or split=='train' else 1+index%4
            faces=self.rng.sample(row['geometry']['bins'],count)
            positive=image.copy();negative=image.copy();boxes=[];placements=[]
            for j,face in enumerate(faces):
                cls=(index*count+j)%24 if split!='train' else index%24
                placement=paper_on_face(face,cls,(w,h),[0,0,w,h],self.rng)
                if sizes:
                    corners=np.float32(placement['paper_quad'])
                    target=sizes[index//24]*max(w,h)/960
                    factor=min(1.,target/np.ptp(corners,axis=0).min())
                    corners=(corners-corners.mean(axis=0))*factor+corners.mean(axis=0)
                    placement['paper_quad']=corners.tolist()
                    placement['requested_short_side_at_960']=sizes[index//24]
                quad=placement['paper_quad'];placements.append(placement)
                box=paper_on_image(positive,face,quad,ROOT/self.art[cls])
                paper_on_image(negative,face,quad)
                boxes.append({'class_id':cls,'bbox_xyxy':box})
            pair=f'{split}_surface_{index:04d}'
            common=dict(parent=row['parent_group'],sources=[relative(photo)],pair_id=pair,
                        extra={'source_record_id':row['record_id'],'source_frame_id':row['frame_id'],
                               'paper_placements':placements,'white_distractor':'blank_paper_card'})
            if row.get('source_kind')=='blender_render': common['extra']['mother_scene']=row['parent_group']
            self.add(split,'blank_pair_negative',negative,[],**common)
            self.add(split,'surface_pair_positive',positive,boxes,**common)

    def reshoot_negatives(self):
        """Use agent-reviewed complete empty frames, never whole-bin state labels."""
        for source in self.plan.get('reshoot_negative_sources',[]):
            photo=ROOT/source['image']
            if sha256(photo.read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError('Reshoot source bytes changed: '+str(photo))
            if source['presence']!='absent' or source['review_method']!='agent_visual_full_frame':
                raise ValueError('Reshoot needs task-specific full-frame absence evidence')
            image=read_image(photo);h,w=image.shape[:2]
            for fraction in (1., .52, .30):
                crop=image[:round(h*fraction)].copy()
                ch,cw=crop.shape[:2];scale=min(1.,960/max(ch,cw))
                crop=cv2.resize(crop,(round(cw*scale),round(ch*scale)),interpolation=cv2.INTER_AREA)
                self.add('train','reshoot_real_negative',crop,[],source['parent_group'],
                    sources=[source['image']],extra={'source_kind':'direct_camera',
                        'generation_method':'resize' if fraction==1 else 'crop_resize',
                        'image_processing':{'background_origin':'direct_camera','origin_review':'confirmed'},
                        'review_method':'agent_visual_full_frame_and_crop_inherits_absence',
                        'source_sha256':source['sha256'],'source_crop_xyxy':[0,0,w,round(h*fraction)],
                        'white_distractor':'real_chairs_lamps_white_surfaces_and_native_bin_marks'})

    def remaining_negatives(self, mining, amount):
        candidates=sorted(mining['records'],key=lambda r:(r['false_at_025'],r['max_confidence']),reverse=True)
        # Preserve genuine camera negatives even when the current baseline already rejects them.
        real=[r for r in candidates if r.get('source_kind')!='blender_render']
        chosen=real[:amount]+[r for r in candidates if r not in real][:max(0,amount-len(real))]
        qualified={r['frame_id']:r for r in query_frames(task='waste_sticker',role='empty_negative',split='train')}
        for row in chosen:
            source=qualified[row['frame_id']];complete_empty(source)
            image=read_image(existing_path(source))
            self.add('train','mined_negative',image,[],row['parent_group'],sources=[row['source_image']],
                     extra={'source_record_id':row['record_id'],'source_frame_id':row['frame_id'],
                            'baseline_false_at_025':row['false_at_025'],'baseline_max_confidence':row['max_confidence']})

    def extra_negatives(self, split, amount, *, cards=False):
        rows=query_frames(task='waste_sticker',role='composite_background',split=split)
        if split=='val': rows=[r for r in rows if r.get('geometry',{}).get('bins')]
        else: rows=[r for r in rows if not r.get('geometry',{}).get('bins')]
        for index in range(amount):
            row=rows[index%len(rows)];complete_empty(row)
            photo=existing_path(row);image=read_image(photo);h,w=image.shape[:2]
            bins=row.get('geometry',{}).get('bins',[])
            if bins:
                top=min(h-25,max(y for face in bins for _,y in face['allowed_polygon'])+12)
                quad=[[0,top],[w-1,top],[w-1,h-1],[0,h-1]]
            else: quad=row['geometry']['ground_polygon']
            seed=self.plan['seed']+30000+(split=='test')*10000+index+cards*500
            image=ground_stripes(image,quad,seed,cards=cards)
            self.add(split,'blank_ground_negative' if cards else 'stripe_negative',image,[],row['parent_group'],
                     sources=[relative(photo)],extra={'source_record_id':row['record_id'],
                       'source_frame_id':row['frame_id'],'stripe_quad':quad,'stripe_seed':seed,
                       'white_distractor':('blank_ground_card' if cards else
                           'projected_white_road_markings' if np.ptp(quad,axis=0)[1] >= h*.1 else 'local_white_graphic_stripes')})

    def finish(self):
        expected={'train':self.plan['budget'],'val':sum(self.plan['validation'].values()),
                  'test':sum(self.plan['test'].values())}
        actual=Counter(r['split'] for r in self.rows)
        if dict(actual)!=expected: raise ValueError(f'Incomplete counts: {actual}')
        pairs=defaultdict(set);parents=defaultdict(set);reuse=defaultdict(set)
        for row in self.rows:
            if row['pair_id']: pairs[row['pair_id']].add(row['split'])
            if row['source_group'].lower().endswith('.mp4'): parents[row['source_group']].add(row['split'])
            for relation in ['source_group','mother_scene','source_assets']:
                value=row.get(relation,[])
                for source in value if isinstance(value,list) else [value]:
                    if source: reuse[relation,source].add(row['split'])
        if any(len(v)>1 for v in pairs.values()) or any(len(v)>1 for v in parents.values()):
            raise ValueError('Pair or real parent crosses split')
        summary={}
        for split in expected:
            rows=[r for r in self.rows if r['split']==split]
            summary[split]={'images':len(rows),'negative_images':sum(not r['instances'] for r in rows),
                            'boxes':sum(len(r['instances']) for r in rows),
                            'groups':dict(Counter(r['group'] for r in rows)),
                            'classes':dict(sorted(Counter(i['class_id'] for r in rows for i in r['instances']).items()))}
            if len(summary[split]['classes'])!=24: raise ValueError('Missing class')
        # Retain the generation recipe, and resolve the registered training strata
        # to actual per-image class/count declarations for the existing statistics reader.
        resolved=deepcopy(self.plan)
        resolved.setdefault('recipe_strata',deepcopy(self.plan['strata']))
        resolved['strata']=[{'id':Path(row['image']).stem,'min_count':1,'max_count':1,
            'prior':{'alpha':2,'beta':8},'exposure_weight':1,'failure_cost':1,
            'failure_event':'white_background_false_positive_or_sticker_miss',
            'conditions':[row['group']],'background':{'image':row['sources'][0]},
            'target':{'kind':'explicit_bin_stickers','placements':[
                {'class_id':b['class_id'],'visible_bbox_xyxy':b['bbox_xyxy']} for b in row['instances']]},
            'parameters':{'finetune_group':row['group']}}
            for row in self.rows if row['split']=='train']
        write_json(self.plan_path,resolved)
        write_json(self.out/'manifest.json',{'status':'complete','task':'waste_sticker','plan':relative(self.plan_path),
                   'generation_plan':deepcopy(self.plan),'coverage':summary,'records':self.rows,
                   'label_review':'Machine source evidence and geometric audit; no invented human confirmation'})
        write_json(self.out/'workspace_sources.json',{'model_key':'waste_sticker','selection_mode':'manifest_only',
                   'records':self.rows,'source_reuse':[{'relation':r,'source':s,'splits':sorted(v)}
                          for (r,s),v in reuse.items() if len(v)>1],
                   'exposure_warnings':['Original Test already inspected to diagnose the failure. New challenge uses the same held-out real parent; controlled regression only.']})
        (self.out/'data.yaml').write_text(yaml.safe_dump({'names':self.names,**{s:s+'/images' for s in expected}},
                                                allow_unicode=True,sort_keys=False),encoding='utf-8')
        write_json(self.out/'audit.json',{'status':'passed','coverage':summary,'decoded_rgb_unique':len(self.hashes),
                   'paired_counterexamples':len(pairs),'real_parent_split_isolated':True,
                   'initial_weights_sha256':self.plan['initial_weights_sha256'],
                   'shared_assets':['fixed_24_item_artworks','datasets/assets/blender/aiaec_dataset.blend']})
        for split in expected:
            selected=[]
            for group in sorted({r['group'] for r in self.rows if r['split']==split}):
                selected.extend([r for r in self.rows if r['split']==split and r['group']==group][:3])
            sheet=Image.new('RGB',(960,210*((len(selected)+2)//3)),(25,25,25));draw=ImageDraw.Draw(sheet)
            for j,row in enumerate(selected):
                thumb=Image.open(ROOT/row['image']).convert('RGB')
                d=ImageDraw.Draw(thumb)
                for item in row['instances']: d.rectangle(item['bbox_xyxy'],outline='lime',width=3)
                thumb=ImageOps.contain(thumb,(310,175))
                x,y=(j%3)*320+5,(j//3)*210+5;sheet.paste(thumb,(x,y));draw.text((x,y+177),row['group']+' '+str(len(row['instances'])),fill='white')
            sheet.save(self.out/(split+'_contact_sheet.jpg'),quality=90)
        print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['mine','build'])
    parser.add_argument('--plan',type=Path,default=PLAN);args=parser.parse_args()
    plan_path=args.plan.resolve()
    plan=json.loads(plan_path.read_text(encoding='utf-8'));out=ROOT/plan['output']
    if sha256((ROOT/plan['initial_weights']).read_bytes()).hexdigest()!=plan['initial_weights_sha256']:
        raise ValueError('Baseline checkpoint differs from returned model')
    out.mkdir(parents=True,exist_ok=True)
    mined=mine(plan,out)
    if args.command=='mine': return
    if (out/'train').exists(): raise FileExistsError('Fine-tune dataset output exists; inspect it before rebuilding')
    groups={row['id']:row['min_count'] for row in plan.get('recipe_strata',plan['strata'])}
    if (groups['hybrid_pair_negative']!=groups['hybrid_pair_positive'] or
            groups['blank_pair_negative']!=groups['surface_pair_positive'] or
            plan['validation']['paired_positive']!=plan['validation']['paired_blank_negative'] or
            plan['test']['paired_positive']!=plan['test']['paired_negative']):
        raise ValueError('Paired counterexamples require equal positive and negative counts')
    if groups.get('replay_positive',0)%4 or groups['hybrid_pair_positive']%4:
        raise ValueError('Positive replay and train hybrid counts must divide into four sticker-count groups')
    builder=Builder(plan,out,plan_path)
    builder.remaining_negatives(mined,groups['mined_negative'])
    builder.hybrid_pairs('train',groups['hybrid_pair_positive'])
    builder.surface_pairs('train',groups['surface_pair_positive'])
    if plan.get('replay_all_original_train'):
        builder.copy_old('train',builder.old_records('train'),'original_train_replay')
        builder.reshoot_negatives()
    else:
        builder.replay(groups['replay_positive'])
    if (len(builder.old_records('val'))!=plan['validation']['original'] or
            len(builder.old_records('test'))!=plan['test']['original_regression']):
        raise ValueError('Original regression counts differ from the declared complete source splits')
    builder.copy_old('val',builder.old_records('val'),'original_val')
    builder.surface_pairs('val',plan['validation']['paired_positive'])
    builder.extra_negatives('val',plan['validation']['stripe_negative'])
    if plan.get('reuse_test_manifest'):
        previous=json.loads((ROOT/plan['reuse_test_manifest']).read_text(encoding='utf-8'))
        for row in previous['records']:
            if row['split']!='test': continue
            image=read_image(ROOT/row['image']);h,w=image.shape[:2]
            builder.add('test',row['group'],image,labels(ROOT/row['label'],(w,h)),row['source_group'],
                sources=[row['image'],row['label']],original=row,pair_id=row.get('pair_id'))
    else:
        builder.copy_old('test',builder.old_records('test'),'original_test_regression')
        builder.hybrid_pairs('test',plan['test']['paired_positive'])
        builder.extra_negatives('test',plan['test']['blank_negative'],cards=True)
        builder.extra_negatives('test',plan['test']['stripe_negative'])
    builder.finish()


if __name__=='__main__': main()
