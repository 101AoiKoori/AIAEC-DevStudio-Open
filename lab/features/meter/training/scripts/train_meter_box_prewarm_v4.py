"""Bounded box-only prewarm from V4 synthetic meter masks; no pose supervision.

This research entry never changes the normal V4 engine or its label gate. It
encodes eight unknown points only inside an owned runs directory because the
installed pose loader requires 8x3 values. The point and presence losses are
forced to zero, the point head is frozen, and no validation or test is run.
The output is an intermediate box-prewarm checkpoint, not a trained pose model.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.bounded import BudgetExhausted, run_bounded  # noqa: E402
from engine.runtime import setup_paths, write_json  # noqa: E402

SOURCE = ROOT / 'datasets/exports/meter/v4_station_mask_train_candidates_20261112'
WEIGHTS = ROOT / 'runs/meter/runs/meter/v3_20261111_145506_148167/training/weights/best.pt'
RUNS = ROOT / 'runs/meter'
POINTS = ('temperature_pivot','temperature_tip','humidity_pivot','humidity_tip',
          'temperature_min','temperature_max','humidity_min','humidity_max')
POLICY = 'unknown_pose_points_encoded_as_zeros_for_loader_only_no_point_or_visibility_supervision'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def encode_box(box):
    """Encode one valid 640x480 box followed by eight all-unknown triplets."""
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in box)):
        raise ValueError('box must contain four finite pixel coordinates')
    x1,y1,x2,y2=box
    if not (0<=x1<x2<=640 and 0<=y1<y2<=480):
        raise ValueError('box must fit the 640x480 source image')
    return (f'0 {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f} '
            + ' '.join(['0 0 0']*8) + '\n')


def assert_unknown_points(row):
    """Reject sources that claim non-unknown keypoint supervision."""
    if row.get('keypoint_visibility') != dict.fromkeys(POINTS, 'unknown'):
        raise ValueError('all source keypoints must explicitly be unknown')
    evidence=row.get('dial_raster_evidence',{})
    if evidence.get('bbox_status')!='candidate_from_isolated_raster_and_visible_fraction':
        raise ValueError('source dial box lacks the expected raster evidence')
    return encode_box(evidence.get('bbox_xyxy_pixels'))


def checked_file(base, path):
    result=(base/path).resolve()
    if not result.is_relative_to(base.resolve()) or not result.is_file():
        raise ValueError('source file missing or outside candidate directory: '+str(path))
    return result


def read_mask(path):
    import cv2
    import numpy as np
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None or image.shape!=(480,640,4):raise ValueError('mask must be 640x480 RGBA: '+str(path))
    return (image[:,:,:3].max(axis=2)>63) & (image[:,:,3]>127)


def mask_bbox(mask):
    import numpy as np
    y,x=np.nonzero(mask)
    if not len(x):raise ValueError('empty target mask')
    return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]


def check_unknown_loss_semantics(run):
    """Verify the installed Ultralytics loss masks unknown points and zero kobj."""
    import torch
    from ultralytics.utils.loss import KeypointLoss, v8PoseLoss
    criterion=object.__new__(v8PoseLoss)
    criterion.keypoint_loss=KeypointLoss(torch.ones(8)/8)
    criterion.bce_pose=torch.nn.BCEWithLogitsLoss()
    foreground=torch.tensor([[True]]);assignment=torch.tensor([[0]]);batch_index=torch.tensor([0])
    stride=torch.ones((1,1));boxes=torch.tensor([[[0.,0.,10.,10.]]])
    truth=torch.zeros((1,8,3));prediction=torch.zeros((1,1,8,3),requires_grad=True)
    location,presence=criterion.calculate_keypoints_loss(foreground,assignment,truth,batch_index,stride,boxes,prediction)
    (location*0.+presence*0.).backward()
    gradient=prediction.grad.detach()
    if not torch.isfinite(gradient).all() or torch.count_nonzero(gradient).item()!=0 or float(location.detach())!=0.:
        raise RuntimeError('installed pose loss does not mask all-unknown points with zero gains')
    truth[0,0]=torch.tensor([3.,4.,2.]);control=torch.zeros((1,1,8,3),requires_grad=True)
    control_location,control_presence=criterion.calculate_keypoints_loss(foreground,assignment,truth,batch_index,stride,boxes,control)
    (control_location+control_presence*0.).backward();control_gradient=control.grad.detach()
    if (float(control_location.detach())<=0 or torch.count_nonzero(control_gradient[0,0,0,:2]).item()==0 or
            torch.count_nonzero(control_gradient[0,0,1:,:2]).item()!=0 or
            torch.count_nonzero(control_gradient[0,0,:,2]).item()!=0):
        raise RuntimeError('known-point control or masked-gradient check failed')
    result={'status':'passed','unknown_location_loss':float(location.detach()),
        'unscaled_unknown_presence_loss':float(presence.detach()),
        'unknown_total_gradient_max':float(gradient.abs().max()),
        'known_control_location_loss':float(control_location.detach()),
        'unknown_control_xy_gradient_max':float(control_gradient[0,0,1:,:2].abs().max()),
        'control_presence_gradient_max_at_zero_gain':float(control_gradient[0,0,:,2].abs().max()),
        'scope':'Installed keypoint loss function only; whole trainer enforces zero gains and frozen point head separately.'}
    write_json(run/'metadata/loss_semantics.json',result)
    return result


def prepare(run):
    """Recheck both masks and stage all forty image/box pairs under this run."""
    import cv2
    import numpy as np
    import yaml
    if not WEIGHTS.is_file():raise FileNotFoundError('V3 pose initialization is missing')
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
    mask_audit=json.loads((SOURCE/'mask_audit.json').read_text(encoding='utf8'))
    pixel_audit=json.loads((SOURCE/'mask_pixel_check.json').read_text(encoding='utf8'))
    rows=[json.loads(line) for line in (SOURCE/'samples.jsonl').read_text(encoding='utf8').splitlines()]
    if (manifest.get('status')!='complete_candidates' or manifest.get('completed_count')!=40 or len(rows)!=40 or
            mask_audit.get('status')!='pass' or mask_audit.get('samples')!=40 or mask_audit.get('errors') or
            pixel_audit.get('checked')!=40 or pixel_audit.get('errors')):
        raise ValueError('the 40-frame raster candidate and prior audits must be complete')
    image_dir=run/'data/images/train';label_dir=run/'data/labels/train'
    image_dir.mkdir(parents=True);label_dir.mkdir(parents=True)
    selected=[];ids=set()
    for row in rows:
        sid=row.get('id')
        if not isinstance(sid,str) or not re.fullmatch(r'station_\d{4}',sid) or sid in ids:
            raise ValueError('unexpected or repeated source identifier')
        ids.add(sid);label=assert_unknown_points(row)
        if row.get('split')!='unassigned' or row.get('image_size')!=[640,480]:
            raise ValueError('unexpected source split or image size')
        annotation=checked_file(SOURCE,'annotations/'+sid+'.json')
        if json.loads(annotation.read_text(encoding='utf8'))!=row:
            raise ValueError('individual annotation disagrees with source record')
        image=checked_file(SOURCE,row['image']);decoded=cv2.imdecode(np.fromfile(str(image),np.uint8),cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape!=(480,640,3):raise ValueError('invalid source image')
        evidence=row['dial_raster_evidence'];masks={}
        for kind in ('visible','isolated'):
            item=evidence[kind];path=checked_file(SOURCE,item['path']);mask=read_mask(path);box=mask_bbox(mask)
            if max(abs(a-b) for a,b in zip(box,item['bbox_xyxy_pixels']))>2:
                raise ValueError('mask pixels disagree with recorded box: '+sid)
            masks[kind]=mask
        fraction=float(np.logical_and(masks['visible'],masks['isolated']).sum()/masks['isolated'].sum())
        if fraction<.95 or abs(fraction-evidence['visible_fraction_candidate'])>.02:
            raise ValueError('mask overlap invalid: '+sid)
        vb,ib=mask_bbox(masks['visible']),mask_bbox(masks['isolated'])
        union=[min(vb[0],ib[0]),min(vb[1],ib[1]),max(vb[2],ib[2]),max(vb[3],ib[3])]
        if max(abs(a-b) for a,b in zip(union,evidence['bbox_xyxy_pixels']))>2:
            raise ValueError('mask union disagrees with candidate box: '+sid)
        dst=image_dir/(sid+'.png');shutil.copy2(image,dst);(label_dir/(sid+'.txt')).write_text(label,encoding='utf8')
        selected.append({'id':sid,'source_image':relative(image),'source_annotation':relative(annotation),
            'source_visible_mask':relative(SOURCE/evidence['visible']['path']),
            'source_isolated_mask':relative(SOURCE/evidence['isolated']['path']),
            'bbox_xyxy_pixels':evidence['bbox_xyxy_pixels'],'visible_fraction_candidate':fraction,
            'source_state':row['state'],'keypoint_visibility':dict.fromkeys(POINTS,'unknown'),
            'run_image':relative(dst),'run_label':relative(label_dir/(sid+'.txt'))})
    # Ultralytics insists on a val path even when the dedicated trainer skips it.
    # The alias is not an evaluation set; no validation iterator is consumed.
    spec={'path':str((run/'data').resolve()),'train':'images/train','val':'images/train',
          'kpt_shape':[8,3],'names':['meter']}
    (run/'data/data.yaml').write_text(yaml.safe_dump(spec,allow_unicode=True,sort_keys=False),encoding='utf8')
    evidence={'scope':'40 synthetic images from one Blender scene; train only, not independent collection',
              'source_manifest':relative(SOURCE/'manifest.json'),'source_records':relative(SOURCE/'samples.jsonl'),
              'source_mask_audit':relative(SOURCE/'mask_audit.json'),'source_pixel_audit':relative(SOURCE/'mask_pixel_check.json'),
              'policy':POLICY,'validation':'not run; loader-only path aliases train','test':'not run',
              'count':len(selected),'records':selected}
    write_json(run/'metadata/source_manifest.json',evidence)
    return evidence


def make_box_only_trainer(PoseTrainer, torch, ultralytics_version, trainer_state):
    """Construct the restricted trainer while keeping its safeguards testable."""
    from copy import deepcopy
    class BoxOnlyTrainer(PoseTrainer):
        def _setup_train(self, world_size):
            if world_size!=1:raise RuntimeError('box prewarm supports one process only')
            super()._setup_train(world_size)
            if float(self.args.pose)!=0. or float(self.args.kobj)!=0.:
                raise RuntimeError('keypoint and presence loss weights must both be zero')
            head=self.model.model[-1]
            if not hasattr(head,'cv4') or list(head.kpt_shape)!=[8,3]:
                raise RuntimeError('unexpected pose-head structure')
            self.frozen_point_head=head.cv4
            for parameter in self.frozen_point_head.parameters():parameter.requires_grad_(False)
            self.frozen_point_head.eval()
            self.initial_point_state={k:v.detach().cpu().clone() for k,v in self.frozen_point_head.state_dict().items()}

        def preprocess_batch(self,batch):
            if float(self.args.pose)!=0. or float(self.args.kobj)!=0.:
                raise RuntimeError('keypoint loss weights changed during training')
            self.frozen_point_head.eval()
            if any(p.requires_grad for p in self.frozen_point_head.parameters()):
                raise RuntimeError('point head unexpectedly trainable')
            batch=super().preprocess_batch(batch)
            if batch['keypoints'].shape[-2:]!=(8,3) or torch.any(batch['keypoints'][...,2]!=0):
                raise RuntimeError('loader keypoint visibility is not wholly unknown')
            return batch

        def validate(self):
            # Base trainer requests validation at the final epoch even for val=False.
            trainer_state['validation_calls_skipped']+=1
            return {}, 0.0

        def final_eval(self):
            trainer_state['final_eval_calls_skipped']+=1

        def save_model(self):
            # No best.pt or fitness: this checkpoint has never been validated.
            checkpoint={'epoch':self.epoch,'best_fitness':None,'model':None,
                'ema':deepcopy(self.ema.ema).half(),'updates':self.ema.updates,'optimizer':None,
                'train_args':vars(self.args),'train_metrics':{},
                'date':datetime.now(timezone.utc).isoformat(),'version':ultralytics_version,
                'box_prewarm_scope':'no pose supervision; no validation or model selection; not a final model'}
            torch.save(checkpoint,self.last)

    return BoxOnlyTrainer


def train_child(run, device, epochs):
    """Run box-only optimization with point head and its BN state frozen."""
    setup_paths(run)  # Must precede importing torch/ultralytics.
    import torch
    import ultralytics
    from torch.utils.tensorboard import SummaryWriter
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.train import PoseTrainer

    torch.set_num_threads(4)
    if device=='0' and not torch.cuda.is_available():raise RuntimeError('requested CUDA device is unavailable')
    check_unknown_loss_semantics(run)
    model=YOLO(str(WEIGHTS))
    if model.task!='pose' or list(model.model.kpt_shape)!=[8,3] or list(model.names.values())!=['meter']:
        raise ValueError('V3 initialization must be the one-class, eight-point pose model')
    trainer_state={'validation_calls_skipped':0,'final_eval_calls_skipped':0,'checked_batches':0}
    writer=SummaryWriter(log_dir=str(run/'tensorboard'))

    BoxOnlyTrainer=make_box_only_trainer(PoseTrainer,torch,ultralytics.__version__,trainer_state)

    def batch_check(trainer):
        losses=trainer.loss_items.detach().float().cpu().tolist()
        if len(losses)!=5 or any(not math.isfinite(x) for x in losses) or losses[1]!=0. or losses[2]!=0.:
            raise RuntimeError('pose or keypoint-presence loss was not exactly zero')
        if any(p.grad is not None for p in trainer.frozen_point_head.parameters()):
            raise RuntimeError('frozen point head unexpectedly received a gradient')
        trainer_state['checked_batches']+=1

    def epoch_log(trainer):
        for name,value in zip(('box','pose','kobj','class','dfl'),trainer.tloss.detach().cpu().flatten().tolist()):
            writer.add_scalar('train/'+name,float(value),trainer.epoch+1)
        for key,value in trainer.lr.items():writer.add_scalar(key,float(value),trainer.epoch+1)
        writer.flush()

    model.add_callback('on_train_batch_end',batch_check)
    model.add_callback('on_fit_epoch_end',epoch_log)
    started=time.monotonic()
    try:
        model.train(trainer=BoxOnlyTrainer,data=str(run/'data/data.yaml'),project=str(run/'training'),name='box_only',
                    exist_ok=False,epochs=epochs,imgsz=640,batch=4,nbs=4,device=device,workers=0,
                    optimizer='AdamW',lr0=.0002,lrf=.1,momentum=.9,weight_decay=.0005,warmup_epochs=0.,
                    pose=0.,kobj=0.,val=False,plots=False,amp=False,patience=0,seed=42,
                    degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,
                    fliplr=0.,flipud=0.,mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,verbose=False)
        trainer=model.trainer;point_state=trainer.frozen_point_head.state_dict()
        if any(not torch.equal(value.detach().cpu(),trainer.initial_point_state[key]) for key,value in point_state.items()):
            raise RuntimeError('point-head parameters or BN buffers changed during box prewarm')
        if (not trainer_state['checked_batches'] or not trainer.ema.updates or not trainer_state['validation_calls_skipped'] or
                not trainer_state['final_eval_calls_skipped'] or trainer.best.exists() or not trainer.last.is_file()):
            raise RuntimeError('optimization or no-validation/no-best invariants failed')
        info={'status':'completed','optimizer_updates':int(trainer.ema.updates),'epochs':int(epochs),
            'checked_batches':trainer_state['checked_batches'],'pose_loss_weight':float(trainer.args.pose),
            'presence_loss_weight':float(trainer.args.kobj),'point_head_parameters_and_buffers':'unchanged',
            'validation':'not_run','test':'not_run','best_selection':'not_performed',
            'checkpoint':relative(trainer.last),'device':str(trainer.device),
            'gpu_name':torch.cuda.get_device_name(0) if trainer.device.type=='cuda' else None,
            'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
            'elapsed_seconds':time.monotonic()-started,'counters':trainer_state,
            'limitation':'Shared backbone updates may change pose predictions; checkpoint is box-prewarm-only, not a trained or validated pose model.'}
        write_json(run/'metadata/result.json',info)
    finally:
        writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name');parser.add_argument('--device',choices=('cpu','0'),default='0')
    parser.add_argument('--epochs',type=int,default=1);parser.add_argument('--wall-seconds',type=float,default=150.)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--child',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.epochs not in (1,2,3) or not math.isfinite(args.wall_seconds) or not 0<args.wall_seconds<=600:
        parser.error('epochs must be 1..3 and wall-seconds must be 0 < seconds <= 600')
    if args.child:
        run=(ROOT/args.child).resolve()
        if not run.is_relative_to(RUNS.resolve()) or not (run/'metadata/source_manifest.json').is_file():
            parser.error('child requires a prepared run under runs/meter')
        try:train_child(run,args.device,args.epochs)
        except BaseException as exc:
            write_json(run/'metadata/child_failure.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
            raise
        return
    name=args.run_name or ('box_prewarm_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):parser.error('run-name must use letters, digits, underscore or hyphen')
    run=RUNS/name;run.mkdir(parents=True,exist_ok=False);(run/'metadata').mkdir();setup_paths(run)
    status={'status':'preparing','scope':'box prewarm only; unknown keypoints are not supervised; not a final Pose model',
            'validation':'not_run','test':'not_run','best_selection':'not_performed'}
    write_json(run/'metadata/status.json',status);started=time.monotonic()
    try:
        evidence=prepare(run)
        config={'source':relative(SOURCE),'initial_weights':relative(WEIGHTS),'samples':evidence['count'],
            'epochs':args.epochs,'device_requested':args.device,'image_size':640,'batch':4,'nominal_batch':4,
            'optimizer':'AdamW','lr0':.0002,'weight_decay':.0005,'warmup_epochs':0.,'pose':0.,'kobj':0.,
            'point_head':'frozen including BN buffers','validation':'not_run; required loader path aliases train',
            'test':'not_run','wall_seconds':args.wall_seconds,'seed':42,'policy':POLICY}
        write_json(run/'metadata/config.json',config)
        if args.check_only:
            check_unknown_loss_semantics(run)
            status.update(status='prepared_only',samples=evidence['count'])
        else:
            bootstrap=os.environ.get('AEC_PORTABLE_BOOTSTRAP')
            prefix=([sys.executable,'-I','-S','-B',bootstrap,'--script',str(Path(__file__).resolve()),'--'] if bootstrap
                    else [sys.executable,'-B',str(Path(__file__).resolve())])
            command=prefix+['--child',relative(run),'--device',args.device,'--epochs',str(args.epochs),'--wall-seconds',str(args.wall_seconds)]
            status.update(status='running',samples=evidence['count']);write_json(run/'metadata/status.json',status)
            run_bounded(command,started+args.wall_seconds,run/'logs/training.log',ROOT)
            result=json.loads((run/'metadata/result.json').read_text(encoding='utf8'))
            if result.get('status')!='completed':raise RuntimeError('child result is not complete')
            status.update(status='completed',optimizer_updates=result['optimizer_updates'],checkpoint=result['checkpoint'])
    except BaseException as exc:
        status.update(status='budget_exhausted' if isinstance(exc,BudgetExhausted) else 'failed',error=str(exc),
                      traceback=traceback.format_exc())
        write_json(run/'metadata/status.json',status);print('ERROR:',str(exc),file=sys.stderr);raise SystemExit(1)
    status['elapsed_seconds']=time.monotonic()-started;write_json(run/'metadata/status.json',status)
    print(json.dumps({'run':relative(run),**status},ensure_ascii=False))


if __name__=='__main__':main()
