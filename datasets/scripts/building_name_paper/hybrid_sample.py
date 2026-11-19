"""Photo-ground evidence, planning and linear-light composition for paper samples."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import source_path
from datasets.scripts.common.frame_environment import edit_database, image_digest, query_frames
from datasets.scripts.common.paths import building_name_sample_work_dir

TASK = 'building_name_paper'
BASE = building_name_sample_work_dir() / 'hybrid'
LAYERS = BASE / 'layers'
WORK = BASE / 'composites'
PALETTE = {'A': (1, 0, 0), 'B': (0, 1, 0), 'C': (0, 0, 1)}


def recipe():
    return json.loads(source_path(TASK, 'hybrid_plan').read_text(encoding='utf8'))


def road_horizon(plate):
    lines = [np.cross((*a, 1), (*b, 1)) for a, b in plate['road_parallel_segments_px']]
    point = np.cross(*lines)
    if abs(point[2]) < 1e-8:
        raise ValueError('Road edges provide no finite vanishing point')
    point = point / point[2]
    if not np.isfinite(point).all() or not 0 < point[1] < plate['image_size'][1]:
        raise ValueError('Road vanishing point is outside the supported horizon range')
    return {'vanishing_point_px': point[:2].tolist(), 'horizon_y_px': float(point[1]),
            'segments_px': plate['road_parallel_segments_px'], 'basis': plate['review']['horizon_basis']}


def register_backgrounds():
    """Append only this task's reviewed evidence; preserve other tasks' observations."""
    config = recipe()
    evidence_path = source_path(TASK, 'hybrid_plan').relative_to(ROOT).as_posix()
    with edit_database() as writer:
        for plate in config['plates']:
            if image_digest(ROOT / plate['image']) != plate['content_sha256']:
                raise ValueError('Reviewed background bytes changed')
            current=query_frames(task=TASK,role='composite_background',split=plate['source_split'],frame_id=plate['frame_id'])
            if any(row['label']['presence']=='absent' and row['label']['completeness']=='complete'
                   and row['geometry'].get('ground_polygon')==plate['ground_polygon']
                   and row['geometry'].get('forbidden_footprint_xyxy')==plate['forbidden_footprint_xyxy'] for row in current):
                continue
            frame_id = writer.register(plate['image'], image_size=plate['image_size'],
                                       parent_group=plate['parent_group'])
            if frame_id != plate['frame_id']:
                raise ValueError('Reviewed background identity changed')
            evidence = {**plate['review'], 'manifest': evidence_path, 'plate_id': plate['id'],
                        'content_sha256': plate['content_sha256']}
            label = {'presence': 'absent', 'completeness': 'complete', 'scope': {'kind': 'frame'},
                     'instances': [], 'split': plate['source_split'], 'evidence': evidence}
            geometry = {'status': 'candidate', 'coordinate_size': plate['image_size'],
                        'coordinate_space': 'source_image_pixels', 'ground_polygon': plate['ground_polygon'],
                        'forbidden_footprint_xyxy': plate['forbidden_footprint_xyxy'],
                        'perspective': road_horizon(plate), 'evidence': evidence,
                        'reason_codes': ['reviewed_empty_plate_and_floor_footprint']}
            ids = {}
            for kind, scope, payload in [('label', 'frame', label), ('geometry', 'placement', geometry)]:
                ids[kind] = writer.observe(frame_id, task=TASK, kind=kind, scope=scope,
                    evidence_key='building_hybrid_plate', reference=evidence, payload=payload,
                    algorithm_version=plate['review']['algorithm_version'])
            writer.decide(frame_id, task=TASK, role='composite_background', split=plate['source_split'],
                status='eligible', evidence_ids=list(ids.values()), details={
                    'image': plate['image'], 'source_kind': 'camera', 'source_group': 'car_real_capture_20261112',
                    'parent_group': plate['parent_group'], 'capture_identity': frame_id,
                    'label_evidence_id': ids['label'], 'geometry_evidence_id': ids['geometry'],
                    'reason_codes': ['reviewed_empty_plate_and_floor_footprint']})


def qualified_plates(config):
    rows = []
    for plate in config['plates']:
        matches = query_frames(task=TASK, role='composite_background', split=plate['source_split'],
                               frame_id=plate['frame_id'])
        matches = [row for row in matches if row['geometry'].get('ground_polygon') == plate['ground_polygon']
                   and row['geometry'].get('forbidden_footprint_xyxy') == plate['forbidden_footprint_xyxy']
                   and row['label']['presence'] == 'absent' and row['label']['completeness'] == 'complete']
        if len(matches) != 1 or image_digest(ROOT / plate['image']) != plate['content_sha256']:
            raise ValueError('Photo plate lacks current matching task evidence: ' + plate['id'])
        rows.append({**deepcopy(plate), 'source_review': matches[0], 'perspective': road_horizon(plate)})
    return rows


def declaration(config, plates):
    states = []
    for plate in plates:
        for active in config['combinations']:
            for view in config['views']:
                yaw = view['single_yaw_degrees' if len(active) == 1 else 'multi_yaw_degrees']
                states.append({'id': f'hybrid_{plate["id"]}_{active}_{view["id"]}',
                    'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                    'failure_event': 'perspective_shadow_or_paper_visibility_failure',
                    'exposure_weight': 1, 'failure_cost': 1,
                    'conditions': [f'buildings:{active}', f'view:{view["id"]}'],
                    'background': {'plate_id': plate['id'], 'frame_id': plate['frame_id'],
                                   'record_id': plate['source_review']['record_id']},
                    'target': {'buildings': active, 'papers': active},
                    'labels': {'class_id': 0, 'scope': 'complete_paper_detection'},
                    'augmentation': {'mode': 'physical_foreground_and_receiver_ratio'},
                    'parameters': {'yaw_degrees': yaw, 'camera_height_m': view['camera_height_m'],
                                   'seed': config['seed'] + len(states)},
                    'sources': [plate['image'], 'datasets/assets/blender/aiaec_dataset.blend']})
    if len(states) != config['budget']:
        raise ValueError('Hybrid budget differs from executable combinations')
    return {'schema_version': 1, 'model_key': TASK, 'target_split': config['target_split'],
            'purpose': config['purpose'], 'budget': len(states), 'strata': states,
            'uncertainty_weight': 0.0,
            'independence_basis': 'Two frames from the same previously used vehicle-capture source; derived renders are dependent views.',
            'constraints': {'supervision': 'complete_paper_detection'}}


def same_render_recipe(before,after):
    """A source split reassignment changes provenance but not rendered pixels."""
    values=[]
    for config in (before,after):
        copied=deepcopy(config)
        for plate in copied['plates']:plate.pop('source_split',None)
        values.append(copied)
    return values[0]==values[1]


def linear(rgb):
    return np.where(rgb < .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def encoded(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.rint(np.where(rgb < .0031308, rgb * 12.92, 1.055 * rgb ** (1/2.4) - .055)*255).astype(np.uint8)


def compose_layers(photo, foreground, shadowed, unshadowed, *, strength, gain, blur_sigma, shadow_support=None):
    """Apply physical floor attenuation and straight-alpha foreground in linear light."""
    import cv2
    if foreground.shape != (*photo.shape[:2], 4) or shadowed.shape != photo.shape or unshadowed.shape != photo.shape:
        raise ValueError('Hybrid layers must share exact image coordinates')
    values = [linear(image.astype(np.float32)/255) for image in (shadowed, unshadowed)]
    ratio = np.clip(values[0].mean(axis=2)/np.maximum(values[1].mean(axis=2), .006), 0, 1)
    attenuation = 1 - np.maximum(0, 1-ratio-.015)*strength
    if shadow_support is not None:
        attenuation = 1 - (1-attenuation)*shadow_support
    alpha = foreground[:, :, 3:4].astype(np.float32)/255
    premultiplied = linear(foreground[:, :, :3].astype(np.float32)/255)*np.asarray(gain)*alpha
    if blur_sigma:
        premultiplied = cv2.GaussianBlur(premultiplied, (0, 0), blur_sigma)
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)[:, :, None]
    result = encoded(linear(photo.astype(np.float32)/255)*attenuation[:, :, None]*(1-alpha)+premultiplied)
    return result, attenuation, alpha[:, :, 0]


def compose_branch(*, replace=False):
    """Produce paired hybrids from persistent foreground, receiver and mask evidence."""
    import cv2
    from datasets.scripts.common.annotations import yolo_row
    from datasets.scripts.common.images import read_image, write_image
    from datasets.scripts.common import recoverable_publish as publisher
    from lab.training_common.production_guard import generation_guard, receipt_completed

    config = recipe()
    plates = qualified_plates(config)
    layer_manifest = LAYERS / 'manifest.json'
    data = json.loads(layer_manifest.read_text(encoding='utf8'))
    decisions = [plate['source_review']['record_id'] for plate in plates]
    if not same_render_recipe(data['source_plan'],config):
        raise ValueError('Re-render hybrid layers after changing their physical source recipe')
    if {row['state_id'] for row in data['records']} != {s['id'] for s in declaration(config, plates)['strata']}:
        raise ValueError('Hybrid render states do not match the declared schedule')
    media = {ROOT / plate['image'] for plate in plates}
    for row in data['records']:
        media.update(LAYERS / row[key] for key in ('foreground','visible_mask','shadowed_receiver','lit_receiver'))
        media.update(LAYERS / obj['visibility']['isolated_mask'] for obj in row['instances'])
    operation = 'building_name.hybrid_composites'
    committed = lambda token: receipt_completed(ROOT, operation, token)
    configs = [Path(__file__).parent/'config'/name for name in ('source.yaml','dataset.yaml','hybrid_plan.json')]
    with publisher.sample_lock(WORK):
        publisher.recover(WORK, committed)
        owned = set()
        if (WORK/'manifest.json').exists():
            previous = json.loads((WORK/'manifest.json').read_text(encoding='utf8'))
            if previous['task'] != TASK:
                raise ValueError('Hybrid output belongs to another task')
            owned.add('manifest.json')
            for row in previous['records']:
                for key in ('image','label','shadow_attenuation'):
                    path = (WORK / row[key]).resolve()
                    if not path.is_relative_to(WORK) or not path.is_file():
                        raise ValueError('Prior hybrid pair is missing or outside the task')
                    owned.add(path.relative_to(WORK).as_posix())
        preserved = [(p.relative_to(ROOT).as_posix()) for p in WORK.rglob('*')
                     if p.is_file() and p.relative_to(WORK).as_posix() not in owned]
        try:
            with generation_guard(operation, sorted(media), [WORK], {'recipe': config}, root=ROOT,
                code_paths=[Path(__file__), ROOT/'datasets/scripts/common/images.py',
                            ROOT/'datasets/scripts/common/annotations.py',ROOT/'datasets/scripts/common/recoverable_publish.py',
                            ROOT/'datasets/scripts/common/frame_environment.py',ROOT/'datasets/scripts/common/config.py'],
                procedural_inputs=[layer_manifest,*configs], production_plan=declaration(config,plates),
                replace_outputs=replace,
                replace_verifier=(lambda *_: {'preserved_paths': preserved}) if replace else None,
                sample_output_paths=[WORK/'images'], auxiliary_output_paths=[WORK/'shadows']) as run:
                if run.reused:
                    return
                stage = publisher.prepare(WORK, run.token)
                publisher.copy_unowned(WORK, stage, owned)
                for sub in ('images','labels','shadows'):
                    (stage/sub).mkdir(exist_ok=True)
                records = []
                for row in deepcopy(data['records']):
                    stem = row['id']
                    plate = next(p for p in plates if p['image'] == row['source_photo'])
                    photo = read_image(ROOT/row['source_photo'])
                    foreground = read_image(LAYERS/row['foreground'], cv2.IMREAD_UNCHANGED)
                    palette = cv2.cvtColor(read_image(LAYERS/row['visible_mask'], cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA)
                    union = np.zeros(photo.shape[:2], bool)
                    for obj in row['instances']:
                        selected = palette[:,:,3] >= 192
                        for channel, bit in enumerate(PALETTE[obj['building_id']]):
                            selected &= palette[:,:,channel] >= 192 if bit else palette[:,:,channel] <= 63
                        yy, xx = np.nonzero(selected)
                        if not len(xx) or len(xx) != obj['visibility']['visible_pixels']:
                            raise ValueError('Paper mask pixel count changed')
                        if obj['xyxy'] != [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]:
                            raise ValueError('Paper label differs from its visible mask')
                        if float((foreground[yy,xx,3] > 127).mean()) < .98:
                            raise ValueError('Visible paper does not coincide with the rendered foreground')
                        union |= selected
                    x0,y0,x1,y1 = plate['white_reference_xyxy']
                    reference = np.median(linear(photo[y0:y1,x0:x1].astype(np.float32)/255),axis=(0,1))
                    rendered_white = np.percentile(linear(foreground[:,:,:3][union].astype(np.float32)/255),85,axis=0)
                    gain = np.clip(reference/np.maximum(rendered_white,.01),.7,1.35)
                    support = np.zeros(photo.shape[:2],np.uint8)
                    cv2.fillPoly(support,[np.array(plate['ground_polygon'],np.int32)],1)
                    for a,b,c,d in plate['forbidden_footprint_xyxy']:
                        support[b:d,a:c] = 0
                    result, attenuation, alpha = compose_layers(photo,foreground,
                        read_image(LAYERS/row['shadowed_receiver']),read_image(LAYERS/row['lit_receiver']),
                        strength=config['lighting']['shadow_strength'],gain=gain,
                        blur_sigma=config['lighting']['foreground_blur_sigma_px'],shadow_support=support)
                    image, label, shadow = f'images/{stem}.png',f'labels/{stem}.txt',f'shadows/{stem}.png'
                    if any((stage/name).exists() for name in (image,label,shadow)):
                        raise ValueError('Hybrid output would overwrite unowned content')
                    text = ''.join(yolo_row(obj['xyxy'],*plate['image_size'])+'\n' for obj in row['instances'])
                    write_image(stage/image,result)
                    write_image(stage/shadow,np.rint(attenuation*255).astype(np.uint8))
                    (stage/label).write_text(text,encoding='utf8')
                    shadow_pixels = int(((attenuation < .95) & (alpha < .1)).sum())
                    if not shadow_pixels:
                        raise ValueError('Hybrid has no visible floor shadow: '+stem)
                    row.update(image=image,label=label,shadow_attenuation=shadow,
                        generation_frame_environment=row['frame_environment'],frame_environment=plate['source_review'],
                        source_split=plate['source_review']['split'],
                        lighting_match={'linear_bgr_gain':gain.tolist(),'white_reference_xyxy':plate['white_reference_xyxy'],
                                        'reference_basis':'painted white road marking, bounded approximate exposure and colour match'},
                        shadow_stats={'pixels_below_95pct_outside_foreground':shadow_pixels,
                                      'minimum_attenuation':float(attenuation.min())},
                        layer_evidence_root=LAYERS.relative_to(ROOT).as_posix())
                    records.append(row)
                qualified_plates(config)
                output = {**data,'generation_source_plan':data['source_plan'],
                    'generation_source_decisions':data['source_decisions'],'source_plan':config,'source_decisions':decisions,
                    'records':records,'status':'complete_hybrid_sample','label_scope':'paper_detection'}
                (stage/'manifest.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
                publisher.publish(WORK,stage,run.token,keep_root=True)
                for row in records:
                    state = next(s for s in run.plan['strata'] if s['id'] == row['state_id'])
                    run.record_sample(state['id'],output=WORK/row['image'],sources=state['sources'],
                                      parameters=state['parameters'],actual=row)
                run.set_result({'images':len(records),'paper_boxes':sum(len(r['instances']) for r in records)})
            publisher.recover(WORK,committed)
        except BaseException:
            publisher.recover(WORK,committed)
            raise


def loaded_records():
    """Validate source revisions and expose completed hybrid pairs to the current sample."""
    config = recipe()
    plates = qualified_plates(config)
    manifest = WORK / 'manifest.json'
    data = json.loads(manifest.read_text(encoding='utf8'))
    if data['source_plan'] != config or data['source_decisions'] != [p['source_review']['record_id'] for p in plates]:
        raise ValueError('Hybrid sample must be rendered again after source or recipe changes')
    if len(data['records']) != config['budget']:
        raise ValueError('Hybrid branch is incomplete')
    records, media, metadata = [], set(), {manifest, source_path(TASK, 'hybrid_plan')}
    for item in data['records']:
        image, label = WORK / item['image'], WORK / item['label']
        media.add(image)
        metadata.add(label)
        for key in ('foreground', 'visible_mask', 'shadowed_receiver', 'lit_receiver'):
            media.add(LAYERS / item[key])
        media.add(WORK / item['shadow_attenuation'])
        media.update(LAYERS / obj['visibility']['isolated_mask'] for obj in item['instances'])
        row = deepcopy(item)
        row.update(branch='blender_photo_composite', source_manifest=manifest.relative_to(ROOT).as_posix(),
                   source_image=image.relative_to(ROOT).as_posix())
        records.append((image, label.read_text(encoding='utf8'), row))
    media.update(ROOT / plate['image'] for plate in plates)
    return records, media, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-backgrounds', action='store_true', required=True)
    parser.parse_args()
    register_backgrounds()
    print(json.dumps([{'plate': p['id'], 'decision': p['source_review']['record_id'],
                       'perspective': p['perspective']} for p in qualified_plates(recipe())], ensure_ascii=False))
