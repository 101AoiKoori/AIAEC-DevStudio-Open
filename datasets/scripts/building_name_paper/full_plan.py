"""Read the executable full-data plans and exact per-plate source evidence."""
from copy import deepcopy
import json
from pathlib import Path

from datasets.scripts.common.config import source_path
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.paths import WORKSPACE, building_name_full_work_dir

TASK = 'building_name_paper'
WORK = building_name_full_work_dir()
LAYERS = WORK / 'layers'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def config():
    return read(source_path(TASK,'full_recipe'))


def plates(verify=True):
    rows = read(source_path(TASK,'full_plates'))['plates']
    for plate in rows:
        if verify:
            stored = query_frames(task=TASK,role='composite_background',split=plate['split'],
                                  record_id=plate['record_id'])
            if (len(stored)!=1 or not stored[0]['current'] or stored[0]['status']!='eligible'
                    or stored[0]['content_sha256']!=plate['content_sha256']
                    or stored[0]['geometry'].get('full_plate_id')!=plate['id']
                    or image_digest(WORKSPACE/plate['image'])!=plate['content_sha256']):
                raise ValueError('Stale full-dataset plate: '+plate['id'])
        yield plate


def render_plate(plate):
    result=deepcopy(plate)
    w,h=plate['image_size']; rw,rh=plate['render_size']; sx,sy=rw/w,rh/h
    result['source_size']=result['image_size']
    result['image_size']=plate['render_size']
    for key in ('ground_polygon','anchor_candidates_px'):
        result[key]=[[x*sx,y*sy] for x,y in plate[key]]
    result['forbidden_footprint_xyxy']=[[a*sx,b*sy,c*sx,d*sy] for a,b,c,d in plate['forbidden_footprint_xyxy']]
    result['perspective']={'horizon_y_px':plate['horizon_y_px']*sy,'basis':plate['perspective_basis']}
    result['focal_px_assumed']=plate['focal_px_assumed']*sx
    return result


def templates():
    return read(source_path(TASK,'full_render_plan'))


def split_plan(split):
    return read(source_path(TASK,'full_'+split+'_plan'))
