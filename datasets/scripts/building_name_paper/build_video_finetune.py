"""Build new-video-only paper fine-tuning data in the user-selected training package."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.images import write_image,box as mask_box
from datasets.scripts.common import recoverable_publish as publisher
from lab.training_common.production_guard import generation_guard,receipt_completed

CONFIG=Path(__file__).parent/'config/video_finetune_review.json'
PLAN=Path(__file__).parent/'config/video_finetune_plan.json'
PACKAGE=ROOT/'training_packages/building_name_paper'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def decode(source,frames=None,*,gray=False):
    cap=cv2.VideoCapture(str((ROOT/source['video']).resolve()));result={};index=0
    maximum=max(frames) if frames is not None else None
    while maximum is None or index<=maximum:
        ok,frame=cap.read()
        if not ok:break
        if frames is None or index in frames:
            frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
            result[index]=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if gray else frame
        index+=1
    cap.release()
    if frames is not None and set(frames)-result.keys():raise ValueError('Video frames unavailable: '+source['video'])
    return result


def track_paper(gray,track):
    """Track the real paper plane; no predictions from the weak detector become labels.

    The older facade tracker assumes 640px facades larger than 3000px and cannot
    track small paper planes. This local fit uses forward/backward LK and RANSAC.
    """
    anchor=track['anchor'];quad=np.float32(track['quad']);mask=np.zeros_like(gray[anchor])
    cv2.fillConvexPoly(mask,np.rint(quad).astype(np.int32),255)
    mask=cv2.erode(mask,np.ones((3,3),np.uint8))
    initial=cv2.goodFeaturesToTrack(gray[anchor],maxCorners=180,qualityLevel=.005,minDistance=3,mask=mask)
    if initial is None or len(initial)<12:raise ValueError('Not enough real-paper anchor features')
    result={anchor:(quad.tolist(),{'anchor':anchor,'inliers':len(initial),'inlier_fraction':1.0})}
    for direction in (-1,1):
        end=min(track['frames']) if direction<0 else max(track['frames'])
        points=initial.copy();original=initial.copy();previous=anchor
        for number in range(anchor+direction,end+direction,direction):
            updated,status,_=cv2.calcOpticalFlowPyrLK(gray[previous],gray[number],points,None,winSize=(25,25),maxLevel=3)
            if updated is None:raise ValueError('Paper optical flow failed')
            backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray[number],gray[previous],updated,None,winSize=(25,25),maxLevel=3)
            keep=(status[:,0]==1)&(back_status[:,0]==1)&(np.linalg.norm(backward-points,axis=2)[:,0]<1.8)
            points=updated[keep];original=original[keep];previous=number
            if len(points)<10:raise ValueError(f'Paper optical flow lost features: {track["building_id"]} anchor {anchor} frame {number}, points={len(points)}')
            matrix,inliers=cv2.findHomography(original,points,cv2.RANSAC,2.0)
            if matrix is None or inliers is None or inliers.sum()<8 or inliers.mean()<.45:
                raise ValueError(f'Paper homography unreliable: {track["building_id"]} anchor {anchor} frame {number}, points={len(points)}, inliers={None if inliers is None else int(inliers.sum())}')
            fraction=float(inliers.mean());keep=inliers[:,0]==1;original=original[keep];points=points[keep]
            if number not in track['frames']:continue
            polygon=cv2.perspectiveTransform(quad.reshape(1,4,2),matrix)[0]
            if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):raise ValueError('Invalid tracked paper plane')
            result[number]=(polygon.tolist(),{'anchor':anchor,'inliers':int(inliers.sum()),'inlier_fraction':fraction,'matrix':matrix.tolist()})
    return {frame:result[frame] for frame in track['frames']}


def visible_box(quad,size=(1280,720)):
    w,h=size;mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.rint(quad).astype(np.int32),1)
    if mask.sum()<24:raise ValueError('Tracked paper is outside the image or too small')
    return mask_box(mask)


def compile_plan(config):
    bases=[]
    for source in config['sources']:
        gray=decode(source,gray=True);objects=defaultdict(list)
        for track in source['tracks']:
            for frame,(quad,evidence) in track_paper(gray,track).items():
                objects[frame].append({'building_id':track['building_id'],'quad':quad,'xyxy':visible_box(quad),'tracking':evidence})
        for frame in sorted(set(objects)|set(source['negative_frames'])):
            if frame in source['negative_frames'] and frame in objects:raise ValueError('Positive frame declared negative')
            bases.append({'id':Path(source['video']).stem+f'_f{frame:06d}','split':source['split'],
                'video':source['video'],'video_sha256':digest(ROOT/source['video']),'frame':frame,'objects':objects[frame]})
        del gray
        print('TRACKED',source['video'],sum(bool(v) for v in objects.values()),'positive frames',flush=True)
    states=[];by_id={r['id']:r for r in bases}
    def add(stem,split,branch,parents,parameters):
        count=0 if branch=='negative_crop' else sum(len(by_id[p]['objects']) for p in parents)
        states.append({'id':stem,'min_count':1,'max_count':1,'prior':{'alpha':1,'beta':1},
            'failure_event':'paper_miss_duplicate_or_false_positive','exposure_weight':1,'failure_cost':1,
            'conditions':[branch,split],'target':{'paper_count':count},'background':{'origin':'provided_new_video'},
            'parameters':{'branch':branch,'parents':parents,**parameters},
            'sources':sorted({by_id[p]['video'] for p in parents}),'labels':{'class_id':0,'scope':'complete_paper_detection'},'augmentation':{'mode':branch}})
    for row in bases:
        add(row['id'],row['split'],'video_frame',[row['id']],{})
        if row['split']=='train' and row['objects']:
            for i in range(config['augmentation']['positive_crops']):add(row['id']+f'_crop{i}','train','positive_crop',[row['id']],{'variant':i})
            for i in range(config['augmentation']['negative_crops']):add(row['id']+f'_negative{i}','train','negative_crop',[row['id']],{'variant':i})
    positive=defaultdict(list)
    for row in bases:
        if row['split']=='train' and len(row['objects'])==1:positive[row['objects'][0]['building_id']].append(row['id'])
    for i in range(config['augmentation']['multiview_montages']):
        ids='ABC' if i%2 else ('AB','AC','BC')[i//2%3]
        parents=[positive[b][(i*3+j)%len(positive[b])] for j,b in enumerate(ids)]
        add(f'train_multiview_{i:03d}','train','multiview_montage',parents,{'variant':i})
    plan={'schema_version':1,'model_key':'building_name_paper','target_split':'train','budget':len(states),
        'constraints':{'supervision':'complete_paper_detection','sources':'only_user_new_videos','original_training_images':0},
        'uncertainty_weight':0.0,'purpose':'New user videos only, no original training replay','original_training_images':0,
        'independence_basis':'Video-file split; all three videos share the same scene/session, so validation is a same-scene diagnostic.',
        'generation_review':config,'base_frames':bases,'strata':states,'scope':config['scope']}
    write(PLAN,plan);return plan


def crop_box(image,objects,variant,negative=False):
    h,w=image.shape[:2]
    if negative:
        roi=[0,250,640,610] if variant==0 else [450,260,1090,620]
        if any(max(roi[0],o['xyxy'][0])<min(roi[2],o['xyxy'][2]) and max(roi[1],o['xyxy'][1])<min(roi[3],o['xyxy'][3]) for o in objects):
            raise ValueError('Negative crop intersects a real paper')
        selected=[]
    else:
        a,b,c,d=objects[0]['xyxy'];width=min(w,max(480,round((c-a)*(5 if variant==0 else 8))))
        height=min(h,round(width*9/16));x=max(0,min(w-width,round((a+c)/2-width*(.35 if variant==0 else .65))))
        y=max(0,min(h-height,round((b+d)/2-height*.35)));roi=[x,y,x+width,y+height]
        selected=deepcopy(objects)
        for obj in selected:
            aa,bb,cc,dd=obj['xyxy']
            if not roi[0]<=aa<cc<=roi[2] or not roi[1]<=bb<dd<=roi[3]:raise ValueError('Positive crop would truncate the reviewed paper')
            obj['xyxy']=[(aa-roi[0])*1280/(roi[2]-roi[0]),(bb-roi[1])*720/(roi[3]-roi[1]),(cc-roi[0])*1280/(roi[2]-roi[0]),(dd-roi[1])*720/(roi[3]-roi[1])]
            obj['source_quad']=obj['quad'];obj['quad']=[[(x-roi[0])*1280/(roi[2]-roi[0]),(y-roi[1])*720/(roi[3]-roi[1])] for x,y in obj['quad']]
    return cv2.resize(image[roi[1]:roi[3],roi[0]:roi[2]],(1280,720)),selected,roi


def render_state(state,bases,images):
    pars=state['parameters'];parents=[bases[i] for i in pars['parents']];branch=pars['branch']
    first=parents[0];image=images[first['id']].copy();objects=deepcopy(first['objects']);extra={}
    if branch in ('positive_crop','negative_crop'):
        image,objects,roi=crop_box(image,objects,pars['variant'],branch=='negative_crop');extra['source_crop']=roi
    elif branch=='multiview_montage':
        canvas=np.full((720,1280,3),35,np.uint8);objects=[];count=len(parents)
        for i,row in enumerate(parents):
            tile,items,_=crop_box(images[row['id']],row['objects'],0)
            x0=round(i*1280/count);x1=round((i+1)*1280/count)
            canvas[:,x0:x1]=cv2.resize(tile,(x1-x0,720))
            for obj in items:
                a,b,c,d=obj['xyxy'];obj['xyxy']=[x0+a*(x1-x0)/1280,b,x0+c*(x1-x0)/1280,d]
                obj['quad']=[[x0+x*(x1-x0)/1280,y] for x,y in obj['quad']];objects.append(obj)
        image=canvas
    if branch!='video_frame':
        gain=.94+(int(hashlib.sha256(state['id'].encode()).hexdigest()[:4],16)%120)/1000
        image=np.clip(image.astype(np.float32)*gain,0,255).astype(np.uint8);extra['pixel_gain']=gain
    return image,objects,extra


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    config=read(CONFIG);plan=read(PLAN) if args.generate else compile_plan(config)
    bases={r['id']:r for r in plan['base_frames']};images={}
    for source in config['sources']:
        selected={r['frame'] for r in bases.values() if r['video']==source['video']}
        decoded=decode(source,selected)
        images.update({r['id']:decoded[r['frame']] for r in bases.values() if r['video']==source['video']})
    if not args.generate:
        folder=ROOT/'tmp/building_name_full/finetune_annotation';folder.mkdir(parents=True,exist_ok=True)
        positives=[r for r in bases.values() if r['objects']]
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
        for page in range(math.ceil(len(positives)/20)):
            sheet=Image.new('RGB',(1600,5*145),(25,28,32))
            for i,row in enumerate(positives[page*20:(page+1)*20]):
                image=images[row['id']].copy();obj=row['objects'][0];a,b,c,d=obj['xyxy']
                cv2.polylines(image,[np.rint(obj['quad']).astype(np.int32)],True,(0,255,70),1)
                crop=image[max(0,b-10):min(720,d+10),max(0,a-15):min(1280,c+15)]
                tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));tile.thumbnail((395,115));x=i%4*400;y=i//4*145
                sheet.paste(tile,(x,y+25));ImageDraw.Draw(sheet).text((x+3,y+3),row['id'],font=font,fill='white')
            sheet.save(folder/f'tracked_papers_{page}.jpg',quality=94)
        negatives=[r for r in bases.values() if not r['objects']]
        for page in range(math.ceil(len(negatives)/10)):
            sheet=Image.new('RGB',(1280,190*10),(25,28,32))
            for i,row in enumerate(negatives[page*10:(page+1)*10]):
                tile=Image.fromarray(cv2.cvtColor(images[row['id']][:160],cv2.COLOR_BGR2RGB));sheet.paste(tile,(0,i*190+25))
                ImageDraw.Draw(sheet).text((3,i*190+3),row['id']+' / reviewed no visible name paper',font=font,fill='white')
            sheet.save(folder/f'negative_tops_{page}.jpg',quality=94)
        print('PLAN',plan['budget'],'BASES',len(bases),flush=True);return
    output=(ROOT/config['output']).resolve();output.parent.mkdir(exist_ok=True,parents=True)
    if output.exists():raise ValueError('New fine-tune dataset output already exists: '+str(output))
    operation='building_name.new_video_finetune';completed=lambda token:receipt_completed(ROOT,operation,token)
    with publisher.sample_lock(output):
        publisher.recover(output,completed)
        with generation_guard(operation,[ROOT/s['video'] for s in config['sources']],[output],{'original_training_images':0},
            root=ROOT,code_paths=[Path(__file__),ROOT/'datasets/scripts/common/images.py',ROOT/'datasets/scripts/common/annotations.py'],
            procedural_inputs=[CONFIG,PLAN,Path(__file__).parent/'config/source.yaml',Path(__file__).parent/'config/dataset.yaml'],
            production_plan=plan,sample_output_paths=[output/'images/train',output/'images/val'],
            auxiliary_output_paths=[output/'contact_sheet.jpg']) as run:
            if run.reused:return
            stage=publisher.prepare(output,run.token)
            for split in ('train','val'):
                for sub in ('images','labels'):(stage/sub/split).mkdir(parents=True,exist_ok=True)
            (stage/'source_records').mkdir();write(stage/'source_records/review.json',config)
            records=[];source_rows=[];hashes=set()
            for state in run.plan['strata']:
                image,objects,extra=render_state(state,bases,images);split=state['parameters'].get('split') or next(bases[i]['split'] for i in state['parameters']['parents'])
                stem=state['id'];im=f'images/{split}/{stem}.jpg';label=f'labels/{split}/{stem}.txt'
                if len(objects)!=state['target']['paper_count']:raise ValueError('Actual paper count differs from state')
                fingerprint=hashlib.sha256(image.tobytes()).hexdigest()
                if fingerprint in hashes:raise ValueError('Duplicate fine-tune image')
                hashes.add(fingerprint);write_image(stage/im,image)
                text=''.join(yolo_row(o['xyxy'],1280,720)+'\n' for o in objects);(stage/label).write_text(text,encoding='utf8')
                row={'id':stem,'split':split,'image':im,'label':label,'image_size':[1280,720],'objects':objects,**state['parameters'],**extra};records.append(row)
                primary=bases[state['parameters']['parents'][0]];branch=state['parameters']['branch']
                prefix='finetune_dataset/'
                source_rows.append({'image':prefix+im,'label':prefix+label,'workspace_image':prefix+im,'split':split,'instances':len(objects),
                    'source_id':config['session_group'],'parent_video':str(Path(primary['video']).relative_to('training_packages/building_name_paper')),
                    'parent_samples':[bases[i]['video']+'#'+str(bases[i]['frame']) for i in state['parameters']['parents']],
                    'label_status':'reviewed','review_state':'complete_positive' if objects else 'complete_negative',
                    'review_evidence':prefix+'manifest.json','source_review_evidence':prefix+'source_records/review.json',
                    'generation_method':'provided_video_frame' if branch=='video_frame' else 'augmentation' if branch!='multiview_montage' else 'composite',
                    'source_kind':'video_frame' if branch=='video_frame' else 'derived_image',
                    'image_processing':{'origin_review':'confirmed','parent_origin':'direct_camera','background_origin':'direct_camera',
                        'resolution_review':'confirmed','source_size':[3840,2160],'output_size':[1280,720],'resampling':'resampled','clarity':'modified'},
                    'paper':{'schema_version':1,'image_size':[1280,720],'instances':[{'xyxy':o['xyxy'],'building_id':o['building_id']} for o in objects],
                        'scene':{'branch':branch,'buildings':None,'papers':None,'view_basis':'unknown'}},
                    'scope':config['scope']})
            summary={split:{'images':sum(r['split']==split for r in records),'paper_boxes':sum(len(r['objects']) for r in records if r['split']==split),
                    'negative_images':sum(not r['objects'] for r in records if r['split']==split)} for split in ('train','val')}
            write(stage/'manifest.json',{'model_key':'building_name_paper','status':'complete_new_video_finetune','original_training_images':0,
                'plan':PLAN.relative_to(ROOT).as_posix(),'generation_plan':plan,'summary':summary,'records':records,'scope':config['scope']})
            write(stage/'workspace_sources.json',{'model_key':'building_name_paper','selection_mode':'manifest_only','records':source_rows,
                'source_reuse':[{'relation':'source_group','source':config['session_group'],'splits':['train','val'],'evaluation_scope':'same_source_diagnostic'}],
                'scope':config['scope']})
            write(stage/'data.yaml',{'train':'images/train','val':'images/val','names':['paper']})
            choices=[]
            for split,branch in [('train','video_frame'),('train','positive_crop'),('train','negative_crop'),('train','multiview_montage'),('val','video_frame')]:
                selected=[r for r in records if r['split']==split and r['branch']==branch]
                choices.extend(selected[::max(1,len(selected)//3)][:3])
            sheet=Image.new('RGB',(1440,300*math.ceil(len(choices)/3)),(25,28,32));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
            for i,row in enumerate(choices):
                tile=Image.open(stage/row['image']).convert('RGB').resize((478,269));draw=ImageDraw.Draw(tile)
                for o in row['objects']:
                    a,b,c,d=o['xyxy'];draw.rectangle((a*478/1280,b*269/720,c*478/1280,d*269/720),outline='#20ed74',width=2)
                x=i%3*480;y=i//3*300;sheet.paste(tile,(x,y+29));ImageDraw.Draw(sheet).text((x+3,y+3),row['split']+' '+row['branch']+' paper '+str(len(row['objects'])),font=font,fill='white')
            sheet.save(stage/'contact_sheet.jpg',quality=93)
            publisher.publish(output,stage,run.token,keep_root=True)
            for state,row in zip(run.plan['strata'],records):run.record_sample(state['id'],output=output/row['image'],sources=state['sources'],parameters=state['parameters'],actual=row)
            run.set_result({'summary':summary,'original_training_images':0})
        publisher.recover(output,completed)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
