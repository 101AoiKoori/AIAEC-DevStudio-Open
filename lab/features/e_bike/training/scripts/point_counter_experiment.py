"""P2PNet-inspired independent point-set probe; not the upstream VGG16 implementation.

Uses project-trained frozen YOLO features, a new dense point/class head, and
one-to-one Hungarian supervision. No upstream P2PNet source is incorporated.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.training_common.engine.runtime import prepare_ultralytics,run_context
from lab.training_common.engine.config import resolve
from lab.training_common.engine.bounded import run_bounded
from lab.features.e_bike.training.scripts.report_pose_counts import pose_counts
from lab.features.e_bike.training.scripts.two_stage_experiment import truths
from datasets.scripts.common.images import read_image,write_image

POSES=('upright','fallen')
STAGES=(4,6,9)


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def letterbox(frame,size=640):
    h,w=frame.shape[:2];ratio=size/max(h,w)
    rw,rh=round(w*ratio),round(h*ratio)
    left,top=(size-rw)//2,(size-rh)//2
    canvas=np.full((size,size,3),114,np.uint8)
    canvas[top:top+rh,left:left+rw]=cv2.resize(frame,(rw,rh))
    tensor=torch.from_numpy(np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))).float()/255
    return tensor,{'scale_x':rw/w,'scale_y':rh/h,'left':left,'top':top,'width':w,'height':h}


def point_targets(record,transform,size=640):
    gt=truths(record);points=[];labels=[]
    for obj in gt:
        x1,y1,x2,y2=obj['xyxy'];x,y=(x1+x2)/2,(y1+y2)/2
        points.append([(x*transform['scale_x']+transform['left'])/size,
                       (y*transform['scale_y']+transform['top'])/size])
        labels.append(obj['class_id']+1)
    return torch.tensor(points,dtype=torch.float32).reshape(-1,2),torch.tensor(labels,dtype=torch.long)


def backbone_features(backbone,tensor):
    features=[]
    for i,layer in enumerate(backbone):
        tensor=layer(tensor)
        if i in STAGES:features.append(tensor)
    return features


class PointHead(nn.Module):
    def __init__(self,channels,hidden=96):
        super().__init__()
        self.fusion=nn.Sequential(nn.Conv2d(sum(channels),hidden,1),nn.GroupNorm(8,hidden),nn.ReLU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.ReLU(),
            nn.Conv2d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.ReLU())
        self.classifier=nn.Conv2d(hidden,3,1)
        self.offset=nn.Conv2d(hidden,2,1)
        nn.init.zeros_(self.classifier.weight)
        with torch.no_grad():self.classifier.bias.copy_(torch.tensor([math.log(.98),math.log(.01),math.log(.01)]))
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)

    def forward(self,features):
        size=features[0].shape[-2:]
        x=self.fusion(torch.cat([features[0],*[F.interpolate(f,size=size,mode='bilinear',align_corners=False) for f in features[1:]]],1))
        b,_,h,w=x.shape
        yy,xx=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
        anchors=torch.stack(((xx+.5)/w,(yy+.5)/h),-1).reshape(1,-1,2)
        offsets=self.offset(x).tanh().permute(0,2,3,1).reshape(b,-1,2)
        points=anchors+offsets*torch.tensor([2/w,2/h],device=x.device)
        return {'logits':self.classifier(x).permute(0,2,3,1).reshape(b,-1,3),'points':points}


def point_set_loss(outputs,targets):
    """Match one query to each GT; supervise every unmatched query as background."""
    classification=[];regression=[]
    for logits,points,target in zip(outputs['logits'],outputs['points'],targets):
        desired=torch.zeros(len(points),device=points.device,dtype=torch.long)
        n=len(target['points']);reg=points.sum()*0
        if n:
            with torch.no_grad():
                cost=.05*torch.cdist(points*640,target['points']*640)-logits.softmax(-1)[:,target['labels']]
                query,truth=linear_sum_assignment(cost.detach().cpu().numpy())
                query=torch.as_tensor(query,device=points.device);truth=torch.as_tensor(truth,device=points.device)
            desired[query]=target['labels'][truth]
            reg=F.smooth_l1_loss(points[query]*640,target['points'][truth]*640,beta=4.,reduction='sum')/n
        ce=F.cross_entropy(logits,desired,reduction='none')
        prob=logits.softmax(-1).gather(1,desired[:,None]).squeeze(1)
        weight=torch.where(desired==0,.05,1.)
        classification.append((ce*(1-prob).square()*weight).sum()/max(n,1))
        regression.append(reg)
    cls=torch.stack(classification).mean();reg=torch.stack(regression).mean()
    return cls+.05*reg,{'classification':float(cls.detach()),'regression_pixels':float(reg.detach())}


def match_points(gt,predictions):
    """Point metric only: class-aware one-to-one distance, separate from box IoU."""
    if not gt or not predictions:return [],list(range(len(gt))),list(range(len(predictions)))
    centers=np.array([[(g['xyxy'][0]+g['xyxy'][2])/2,(g['xyxy'][1]+g['xyxy'][3])/2] for g in gt])
    pts=np.array([p['point_xy'] for p in predictions])
    radii=np.array([np.clip(.15*math.hypot(g['xyxy'][2]-g['xyxy'][0],g['xyxy'][3]-g['xyxy'][1]),4,16) for g in gt])
    distance=np.linalg.norm(centers[:,None,:]-pts[None,:,:],axis=2)
    valid=(distance<=radii[:,None]) & (np.array([g['class_id'] for g in gt])[:,None]==np.array([p['class_id'] for p in predictions])[None,:])
    gi,pi=linear_sum_assignment(np.where(valid,distance/radii[:,None],1e6))
    matches=[(int(g),int(p),float(distance[g,p])) for g,p in zip(gi,pi) if valid[g,p]]
    usedg={g for g,_,_ in matches};usedp={p for _,p,_ in matches}
    return matches,[g for g in range(len(gt)) if g not in usedg],[p for p in range(len(predictions)) if p not in usedp]


def summarize(rows):
    counts=pose_counts(rows);counts.pop('records')
    counts['all_points_matched_images']=counts.pop('all_instances_matched_images')
    counts.update(point_tp=sum(len(r['matches']) for r in rows),extra_points=sum(len(r['false_prediction_indices']) for r in rows),
                  missed_points=sum(len(r['missed_gt_indices']) for r in rows))
    return counts


def load_cache(exp):
    metadata=read(exp/'point_manifest.json')
    return [(row,torch.load(exp/row['cache'],map_location='cpu',weights_only=True)) for row in metadata['records']]


@torch.no_grad()
def predict_rows(head,items,threshold):
    head.eval();rows=[]
    for row,item in items:
        outputs=head([f[None].cuda().float() for f in item['features']])
        scores,labels=outputs['logits'][0].softmax(-1).max(-1)
        chosen=(labels>0)&(scores>=threshold)
        points=outputs['points'][0][chosen].cpu().tolist();sc=scores[chosen].cpu().tolist();lab=labels[chosen].cpu().tolist()
        transform=row['transform'];found=[]
        for p,s,c in zip(points,sc,lab):
            x=(p[0]*640-transform['left'])/transform['scale_x'];y=(p[1]*640-transform['top'])/transform['scale_y']
            if 0<=x<transform['width'] and 0<=y<transform['height']:
                found.append({'class_id':c-1,'confidence':s,'point_xy':[x,y]})
        gt=row['ground_truth'];matches,missed,extra=match_points(gt,found)
        rows.append({'source_image':row['source_image'],'group':row['group'],'ground_truth':gt,'predictions':found,
                     'matches':matches,'missed_gt_indices':missed,'false_prediction_indices':extra})
    return rows


def prepare(exp):
    config={'output_root':'runs/e_bike','version':'point_set_probe','epochs':10,'batch':4,'lr':.001,'seed':42,
            'wall_seconds':600,'imgsz':640,'thresholds':[.25,.5,.75],'method':'P2PNet-inspired local point-set head, frozen project YOLO backbone; not upstream reproduction'}
    exp.mkdir(parents=True,exist_ok=False)
    for folder in ('metadata','logs','config','cache','temp','tensorboard'):(exp/folder).mkdir()
    with run_context(ROOT,config,exp,'prepare'):
        YOLO=prepare_ultralytics(exp)
        weight=ROOT/'runs/e_bike/keypoint_control_20261118/training/weights/best.pt'
        original=YOLO(str(weight)).model.model
        if any(layer.f!=-1 for layer in original[:10]):raise ValueError('Backbone requires non-sequential routing')
        backbone=nn.Sequential(*list(original[:10])).cuda().eval().requires_grad_(False)
        records=read(ROOT/'runs/e_bike/synthetic_short_inputs_20261118/manifest.json')['records']
        write(exp/'input_snapshot.json',{'records':records})
        (exp/'features').mkdir()
        metadata=[];channels=None;image_hashes={'train':set(),'val':set()}
        for index,record in enumerate(records):
            image=ROOT/record['image'];frame=read_image(image);tensor,transform=letterbox(frame)
            with torch.no_grad():features=[f[0].half().cpu() for f in backbone_features(backbone,tensor[None].cuda())]
            channels=[f.shape[0] for f in features]
            points,labels=point_targets(record,transform)
            cache=f'features/{index:04d}.pt';torch.save({'features':features,'points':points,'labels':labels},exp/cache)
            digest=hashlib.sha256(image.read_bytes()).hexdigest();image_hashes[record['split']].add(digest)
            metadata.append({'source_image':record['image'],'source_sha256':digest,'split':record['split'],'cache':cache,
                'transform':transform,'ground_truth':truths(record),'group':'synthetic_heldout' if record.get('condition') else 'original_real'})
        if image_hashes['train'] & image_hashes['val']:raise ValueError('Exact image content overlaps training and validation')
        torch.save({k:v.cpu() for k,v in backbone.state_dict().items()},exp/'backbone_state.pt')
        write(exp/'point_manifest.json',{'records':metadata,'target':'One point at visible-box center per vehicle, posture from original labels; independent of predicted boxes.',
              'scope':'Known-source diagnostic. Original image splits retained, shared sources and prior validation exposure.'})
        config.update(channels=channels,backbone_checkpoint=weight.relative_to(ROOT).as_posix(),backbone_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                      actual_images=dict(Counter(r['split'] for r in metadata)),classes=['background','upright','fallen'])
        write(exp/'config/point_counter.json',config)
        print(config,flush=True)


def train(exp):
    config=read(exp/'config/point_counter.json')
    with run_context(ROOT,config,exp,'point_train'):
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        items=load_cache(exp);training=[p for p in items if p[0]['split']=='train'];validation=[p for p in items if p[0]['split']=='val']
        head=PointHead(config['channels']).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10,eta_min=.0001)
        write(exp/'before.json',summarize(predict_rows(head,validation,.5)))
        history=[];best=None;updates=0
        backbone=torch.load(exp/'backbone_state.pt',map_location='cpu',weights_only=True)
        for epoch in range(1,11):
            head.train();order=torch.randperm(len(training)).tolist();losses=[]
            for offset in range(0,len(order),4):
                batch=[training[i][1] for i in order[offset:offset+4]]
                features=[torch.stack([b['features'][k] for b in batch]).cuda().float() for k in range(3)]
                targets=[{'points':b['points'].cuda(),'labels':b['labels'].cuda()} for b in batch]
                optimizer.zero_grad(set_to_none=True);loss,_=point_set_loss(head(features),targets)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite independent point loss')
                loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();updates+=1;losses.append(float(loss.detach()))
            summary=summarize(predict_rows(head,validation,.5));summary.update(epoch=epoch,mean_train_loss=float(np.mean(losses)),optimizer_updates=updates)
            history.append(summary);write(exp/'history.json',history);print(json.dumps(summary),flush=True)
            checkpoint={'head':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'backbone':backbone,
                        'config':config,'epoch':epoch,'optimizer_updates':updates,'validation':summary,'class_order':['background','upright','fallen']}
            torch.save(checkpoint,exp/'last.pt')
            score=(summary['mean_absolute_pose_count_error'],-summary['exact_images'])
            if best is None or score<best:
                best=score;torch.save(checkpoint,exp/'best.pt')
            scheduler.step()


def evaluate(exp):
    config=read(exp/'config/point_counter.json')
    with run_context(ROOT,config,exp,'point_evaluate'):
        items=[p for p in load_cache(exp) if p[0]['split']=='val']
        result={};cards=[]
        for checkpoint_name in ('last','best'):
            checkpoint=torch.load(exp/(checkpoint_name+'.pt'),map_location='cpu',weights_only=True)
            head=PointHead(config['channels']).cuda();head.load_state_dict(checkpoint['head'])
            for threshold in config['thresholds']:
                name=checkpoint_name+'_conf_'+str(threshold).replace('.','');folder=exp/'evaluation'/name;folder.mkdir(parents=True,exist_ok=False)
                rows=predict_rows(head,items,threshold)
                result[name]={'epoch':checkpoint['epoch'],'threshold':threshold,'overall':summarize(rows),
                    'groups':{g:summarize([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}}
                (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
                for row in rows:
                    frame=read_image(ROOT/row['source_image'])
                    for p in row['predictions']:cv2.circle(frame,tuple(round(v) for v in p['point_xy']),3,(40,220,40) if p['class_id']==0 else (0,160,255),-1)
                    g,p=[Counter(v['class_id'] for v in rr) for rr in (row['ground_truth'],row['predictions'])]
                    frame=cv2.copyMakeBorder(frame,30,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                    cv2.putText(frame,f'GT U={g[0]} F={g[1]} | Pred U={p[0]} F={p[1]}',(6,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    write_image(folder/(Path(row['source_image']).stem+'.jpg'),frame)
                print(json.dumps({name:result[name]}),flush=True)
        write(exp/'evaluation/summary.json',{'variants':result,'method':config['method'],'input_images':26,
            'matching':'Class-aware Hungarian point distance <=clip(0.15*bbox diagonal,4,16)px. Point matching is not comparable to box IoU matching.',
            'scope':'Counts comparable on same26 images; no predicted box or NMS used. Frozen project bike features, one fresh head-only10-epoch run, no independent real-scene claim.'})
        for row,_ in items:
            stem=Path(row['source_image']).stem
            cards.append('<h2>'+html.escape(stem)+'</h2><div class="grid">'+''.join(f'<figure><figcaption>{name}</figcaption><a href="{name}/{stem}.jpg"><img loading="lazy" src="{name}/{stem}.jpg"></a></figure>' for name in result)+'</div>')
        (exp/'evaluation/review.html').write_text('<!doctype html><meta charset="utf-8"><title>独立点计数小实验</title><style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:8px}figure{margin:0;background:white;padding:8px}img{width:100%}</style><h1>独立点计数，无框与NMS</h1><p>绿点直立，橙点倒伏。GT标注，Pred预测。P2PNet思路的项目轻量探针，非官方复现。</p>'+''.join(cards),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','train','train-child','evaluate'));p.add_argument('--experiment',required=True)
    args=p.parse_args();exp=resolve(ROOT,args.experiment)
    if exp.parent!=ROOT/'runs/e_bike':raise ValueError('Experiment must be under runs/e_bike')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':prepare(exp)
    elif args.phase=='train-child':train(exp)
    elif args.phase=='train':run_bounded([sys.executable,'-X','utf8','-B',str(Path(__file__)),'train-child','--experiment',args.experiment],time.monotonic()+600,exp/'logs/bounded_train.log',ROOT)
    else:evaluate(exp)


if __name__=='__main__':main()
