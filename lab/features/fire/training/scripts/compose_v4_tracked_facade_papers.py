"""Compose task-separated paper candidates on tracked real building facades.

The two input clips are distinct videos of a shared site and physical props.
Their actual frames, simulated paper artwork, camera motion and final labels are
recorded separately. The two output splits are provisional development and
synthetic stress sets, not independently collected blind-camera tests.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/.")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'lab/training_common'),str(Path(__file__).resolve().parent)]
from camera_tone import apply_tone,read_image,write_image  # noqa: E402
from fire_blender_labels import yolo_row  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402

SEQUENCES=ROOT/'datasets/annotated/fire/v4_facade_video_sequences_20261112'
EXPORTS=ROOT/'datasets/exports/fire'
FIRE_ICON=ROOT/'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'
FONT=ROOT/'datasets/raw/building_name_paper/targets/digital/typeface/NotoSansSC.ttf'
REVIEW=SEQUENCES/'facade_review.json'
SPECS=(
    {'id':'scene_001','split':'val','anchor':300,'targets':[220,260,300,340,380],
     'quad':[[307,19],[418,19],[408,248],[301,270]],'distractors':['red_bin','blue_bin','blue_vehicle'],
     'names':['春泽大厦','云和大楼','新雅公寓']},
    {'id':'scene_008','split':'test','anchor':210,'targets':[240,255,270,285,300],
     'quad':[[285,22],[371,25],[355,249],[269,265]],'distractors':['white_windows','road_markings','background_vehicle'],
     'names':['博宁大厦','明德中心','锦瑞商厦']},
)
STATES=('negative','fire_only','name_only','mixed_single','mixed_double')


def rel(path):return path.resolve().relative_to(ROOT).as_posix()


def track_quad(folder,anchor,quad,target):
    """Track reviewed facade features frame-by-frame and estimate its plane."""
    def gray(index):
        img=read_image(folder/f'frame_{index:06d}.png')
        if img.shape!=(360,640,3):raise ValueError('Unexpected frame size')
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    reference=gray(anchor);facade=np.zeros(reference.shape,np.uint8)
    cv2.fillConvexPoly(facade,np.asarray(quad,np.int32),255)
    original=cv2.goodFeaturesToTrack(reference,160,.01,4,mask=facade)
    if original is None or len(original)<20:raise ValueError('Too few facade features')
    points=original.copy();source=original.copy();previous=reference
    if target==anchor:return np.asarray(quad,np.float32),{'features':len(points),'inliers':len(points),'inlier_fraction':1.}
    step=1 if target>anchor else -1
    for number in range(anchor+step,target+step,step):
        current=gray(number);updated,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
        if updated is None or status is None:raise ValueError('Optical flow failed')
        keep=status.reshape(-1)==1;source=source[keep];points=updated[keep];previous=current
        if len(points)<20:raise ValueError('Too few tracked features')
    matrix,inliers=cv2.findHomography(source,points,cv2.RANSAC,2.)
    if matrix is None or inliers is None:raise ValueError('Homography fitting failed')
    count=int(inliers.sum());fraction=count/len(points)
    if count<20 or fraction<.8:raise ValueError('Facade tracking has insufficient inliers')
    tracked=cv2.perspectiveTransform(np.asarray(quad,np.float32).reshape(1,4,2),matrix)[0]
    if not np.isfinite(tracked).all() or not cv2.isContourConvex(tracked.astype(np.float32)):
        raise ValueError('Tracked facade polygon is invalid')
    if (np.any(tracked[:,0]<0) or np.any(tracked[:,0]>=640) or
            np.any(tracked[:,1]<0) or np.any(tracked[:,1]>=360) or abs(cv2.contourArea(tracked))<3000):
        raise ValueError('Tracked facade leaves the frame or is too small')
    return tracked,{'features':len(points),'inliers':count,'inlier_fraction':fraction,
                    'anchor_to_frame_homography':matrix.tolist()}


def paper_graphic(text=None):
    if text is None:
        raw=Image.open(FIRE_ICON).convert('RGB').resize((30,35),Image.Resampling.LANCZOS)
        return cv2.cvtColor(np.asarray(raw),cv2.COLOR_RGB2BGR)
    canvas=Image.new('RGB',(120,36),(239,239,235));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(str(FONT),21)
    bounds=draw.textbbox((0,0),text,font=font);width=bounds[2]-bounds[0];height=bounds[3]-bounds[1]
    draw.text(((120-width)/2-bounds[0],(36-height)/2-bounds[1]),text,font=font,fill=(25,25,25))
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def overlay_paper(image,facade,graphic,rect):
    """Warp one paper from facade coordinates; labels come from its raster mask."""
    plane=np.float32([[0,0],[200,0],[200,400],[0,400]])
    h=cv2.getPerspectiveTransform(plane,facade.astype(np.float32))
    x,y,w,height=rect;corners=np.float32([[x,y],[x+w,y],[x+w,y+height],[x,y+height]])
    dest=cv2.perspectiveTransform(corners.reshape(1,4,2),h)[0]
    if np.any(dest[:,0]<0) or np.any(dest[:,0]>=640) or np.any(dest[:,1]<0) or np.any(dest[:,1]>=360):
        return None
    gh,gw=graphic.shape[:2];source=np.float32([[0,0],[gw-1,0],[gw-1,gh-1],[0,gh-1]])
    warp=cv2.getPerspectiveTransform(source,dest.astype(np.float32))
    pixels=cv2.warpPerspective(graphic,warp,(640,360),flags=cv2.INTER_LINEAR)
    mask=cv2.warpPerspective(np.full((gh,gw),255,np.uint8),warp,(640,360),flags=cv2.INTER_LINEAR)
    ys,xs=np.nonzero(mask>=128)
    if not len(xs):return None
    box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
    alpha=mask.astype(np.float32)[:,:,None]/255
    result=np.clip(pixels.astype(np.float32)*alpha+image.astype(np.float32)*(1-alpha),0,255).astype(np.uint8)
    return result,box,dest.tolist()


def _generate(args):
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(EXPORTS.resolve()) or out.exists():raise ValueError('Use a new fire export directory')
    data=json.loads((SEQUENCES/'manifest.json').read_text(encoding='utf8'))
    if data.get('status')!='complete':raise ValueError('Source sequences are incomplete')
    videos={row['id']:row for row in data['videos']};times={}
    for spec in SPECS:
        rows=json.loads((ROOT/videos[spec['id']]['timestamp_index']).read_text(encoding='utf8'))
        times[spec['id']]={r['frame_number']:r['time_seconds'] for r in rows}
    review={'status':'AI_visual_candidate_not_human_accepted','physical_assets':'same building props, session and site may recur across videos',
        'scope':'Selected video frames reviewed at 640x360; no existing flame stickers or building-name papers discerned. '
                'Crop and augmented label apply only to visible task scope; unreadable target states are withheld.',
        'contact_sheet':'runs/fire_sticker/v4_facade_tracked_geometry.png','clips':[]}
    records=[];counts=Counter();rng=random.Random(20261121)
    try:
        for spec in SPECS:
            folder=SEQUENCES/spec['id'];numbers=[spec['anchor']] if args.pilot else spec['targets']
            for number in numbers:
                image_path=folder/f'frame_{number:06d}.png';image=read_image(image_path)
                quad,tracking=track_quad(folder,spec['anchor'],spec['quad'],number)
                review['clips'].append({'id':spec['id'],'frame':number,'image':rel(image_path),'time_seconds':times[spec['id']][number],
                    'source_video':videos[spec['id']]['source_video'],'quad_xy':quad.tolist(),'tracking':tracking,
                    'observed_fire':'candidate_complete_negative','observed_name':'candidate_complete_negative'})
                for scenario in STATES:
                    for profile,value,sat in (('normal',1.,1.),('dark_low_saturation',.7,.6)):
                        frame=image.copy();objects=[];partial=set();text=None
                        if scenario in ('name_only','mixed_single','mixed_double'):
                            text=spec['names'][rng.randrange(len(spec['names']))]
                            paper=overlay_paper(frame,quad,paper_graphic(text),(40,10,120,36))
                            if paper is None:partial.add('name')
                            else:
                                frame,box,corners=paper
                                if box[2]-box[0]<20 or box[3]-box[1]<7:partial.add('name')
                                objects.append({'kind':'name_sign','bbox_xyxy':box,'paper_quad_xy':corners,
                                    'text_requested':text,'ocr_truth_status':'not_verified_for_ocr'})
                        if scenario in ('fire_only','mixed_single','mixed_double'):
                            slots=[(118,150,30,35)] if scenario!='mixed_double' else [(35,180,30,35),(122,280,30,35)]
                            for slot in slots:
                                paper=overlay_paper(frame,quad,paper_graphic(),slot)
                                if paper is None:partial.add('fire');continue
                                frame,box,corners=paper
                                if box[2]-box[0]<14 or box[3]-box[1]<20:partial.add('fire')
                                objects.append({'kind':'fire_sticker','bbox_xyxy':box,'paper_quad_xy':corners,
                                    'estimated_paper_cm':[3.,3.5],'source_artwork':rel(FIRE_ICON)})
                        if profile!='normal':frame=apply_tone(frame,value_gain=value,saturation_gain=sat)
                        blur=.4 if scenario in ('mixed_single','mixed_double') else 0.
                        if blur:frame=cv2.GaussianBlur(frame,(0,0),blur)
                        stem=f"{spec['id']}_{number:06d}_{scenario}_{profile}";split=spec['split']
                        dest=out/'images'/split/(stem+'.png');dest.parent.mkdir(parents=True,exist_ok=True);write_image(dest,frame)
                        labels={}
                        for task,kind in (('fire','fire_sticker'),('name','name_sign')):
                            if task in partial:labels[task]=None;continue
                            label=out/'labels'/task/split/(stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
                            rows=[yolo_row(x['bbox_xyxy'],640,360) for x in objects if x['kind']==kind]
                            label.write_text(('\n'.join(rows)+'\n') if rows else '',encoding='utf8');labels[task]=rel(label)
                        records.append({'image':rel(dest),'labels':labels,'split':split,'scenario':scenario,'tone':profile,
                            'image_size':[640,360],'background_parent':rel(image_path),'background_video':videos[spec['id']]['source_video'],
                            'background_time_seconds':times[spec['id']][number],'background_track_group':spec['id']+'_facade_track',
                            'source_frame':number,'facade_quad_xy':quad.tolist(),'tracking':tracking,'objects':objects,'partial_tasks':sorted(partial),
                            'existing_targets_review':rel(REVIEW),'distractors':spec['distractors'],'shared_artwork':rel(FIRE_ICON),
                            'generation_method':'real_scene_composite','value_gain':value,'saturation_gain':sat,'blur_sigma_px':blur,
                            'status':'candidate_not_frozen'})
                        counts[split+'|'+scenario+'|'+profile]+=1
        (SEQUENCES/'facade_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        manifest={'status':'draft_not_deliverable','pilot':args.pilot,'seed':20261121,'counts':dict(counts),
            'scope':'Two separately recorded videos of the same physical site and building props; original frames contain no reviewed visible task targets. '
                    'Labels apply to generated paper on real backgrounds. Tone variants share a frame. Test is synthetic stress candidate; no independent real capture or final model result.',
            'source_sequence_manifest':rel(SEQUENCES/'manifest.json'),'records':records}
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'output':rel(out),'counts':dict(counts),'partial':sum(bool(r['partial_tasks']) for r in records)},ensure_ascii=False))
    except BaseException:
        out.mkdir(parents=True,exist_ok=True);(out/'FAILED.txt').write_text('Generation failed; inspect before reuse.\n',encoding='utf8');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    sequence_manifest=SEQUENCES/'manifest.json'
    data=json.loads(sequence_manifest.read_text(encoding='utf8'))
    videos={row['id']:row for row in data['videos']}
    metadata=[sequence_manifest,FONT]
    media=[FIRE_ICON]
    for spec in SPECS:
        metadata.append(ROOT/videos[spec['id']]['timestamp_index'])
        media.append(ROOT/videos[spec['id']]['source_video'])
        numbers=[spec['anchor']] if args.pilot else spec['targets']
        for number in numbers:
            lo,hi=sorted((spec['anchor'],number))
            media.extend(SEQUENCES/spec['id']/f'frame_{i:06d}.png' for i in range(lo,hi+1))
    code=[Path(__file__),ROOT/'lab/features/fire/training/scripts/fire_blender_labels.py',ROOT/'lab/training_common/camera_tone.py',ROOT/'datasets/scripts/common/annotations.py']
    with generation_guard('fire.compose_tracked_facade:'+str(out.relative_to(ROOT)),
                          list(dict.fromkeys([*metadata,*media])),[out,REVIEW],{'pilot':args.pilot,'specs':SPECS},
                          code_paths=code,procedural_inputs=list(dict.fromkeys(metadata)),root=ROOT) as run:
        if not run.reused:
            _generate(args)
            manifest=out/'manifest.json'
            result=json.loads(manifest.read_text(encoding='utf8'))
            result['batch_source_refs']=run.source_refs
            manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
