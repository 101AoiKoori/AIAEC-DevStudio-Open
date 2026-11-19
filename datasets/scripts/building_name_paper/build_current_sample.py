"""Build the current paper-detection sample from camera, real composite and Blender branches."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.building_name_paper import compose_rooftop_papers as rooftop
from datasets.scripts.building_name_paper import hybrid_sample as hybrid
from datasets.scripts.building_name_paper import render_blender_names as renderer
from datasets.scripts.common import recoverable_publish as publisher
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.images import read_image
from datasets.scripts.common.package_direct_real import check_yolo, prepare, TARGETS
from datasets.scripts.common.paths import building_name_sample_work_dir, sample_dir
from datasets.scripts.common.working_inputs import WorkingInputs
from lab.training_common.production_guard import generation_guard, receipt_completed

TASK = 'building_name_paper'
HERE = Path(__file__).resolve().parent
WORK = building_name_sample_work_dir()
OUTPUT = sample_dir(TASK)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def digest(path):
    with Path(path).open('rb') as stream:
        from hashlib import file_digest
        return file_digest(stream, 'sha256').hexdigest()


def configs():
    return [HERE / 'config/source.yaml', HERE / 'config/dataset.yaml']


def import_blender():
    """Retain the accepted render pixels and masks before replacing the combined sample."""
    destination, plan_path = WORK / 'blender', WORK / 'blender_plan.json'
    if (destination / 'manifest.json').is_file() and plan_path.is_file():
        return
    original = OUTPUT / 'blender'
    manifest_path = original / 'manifest.json'
    data = read_json(manifest_path)
    if data.get('task') != TASK or data.get('label_scope') != 'paper_detection':
        raise ValueError('The accepted Blender paper-detection branch is missing')
    plan = renderer.build_plan({'single': 1, 'multiple': 0, 'negative': 0})
    template = plan['strata'][0]
    states = []
    media = []
    labels = []
    for row in data['records']:
        image, label = ROOT / row['image'], ROOT / row['label']
        if not image.resolve().is_relative_to(original) or not label.resolve().is_relative_to(original):
            raise ValueError('Accepted render paths escape the sample branch')
        validate_pair(image, label.read_text(encoding='utf8'), [
            [x * 640, y * 480, right * 640, bottom * 480]
            for x, y, right, bottom in (obj['bbox_xyxy_normalized'] for obj in row['objects'])])
        state = deepcopy(template)
        state.update(id=row['state_id'], background={'scene': row['source_scene'], 'buildings': row['buildings']},
                     target={'texts': {obj['building']: obj['text'] for obj in row['objects']}},
                     parameters=row['parameters'], augmentation={'exposure_ev': row['appearance']['exposure_ev']})
        states.append(state)
        media.append(image)
        labels.append(label)
        if row.get('visible_mask'):
            media.append(original / row['visible_mask'])
        media.extend(original / item['visibility']['isolated_mask'] for item in row['objects'])
    plan.update(budget=len(states), strata=states, purpose='Accepted low camera and left-oblique comparison sample')
    renderer.validate_plan(plan)
    inputs = [manifest_path, *labels, *configs(), source_path(TASK, 'blender_scene'), source_path(TASK, 'font')]
    with generation_guard('building_name.import_accepted_blender', media, [destination, plan_path],
                          {'source_manifest_sha256': digest(manifest_path)}, root=ROOT,
                          procedural_inputs=inputs,
                          code_paths=[Path(__file__), Path(renderer.__file__)]) as run:
        if run.reused:
            return
        if destination.exists() or plan_path.exists():
            raise ValueError('Partial Blender import requires recovery before replacement')
        WORK.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(original, destination)
            for row in data['records']:
                row['image'] = relative(destination / Path(row['image']).relative_to(original.relative_to(ROOT)))
                row['label'] = relative(destination / Path(row['label']).relative_to(original.relative_to(ROOT)))
            data.update(imported_from_manifest=relative(manifest_path), production_plan=relative(plan_path))
            write_json(destination / 'manifest.json', data)
            write_json(plan_path, plan)
            run.set_result({'images': len(states), 'source_blend_sha256': data['source_blend_sha256']})
        except BaseException:
            if destination.exists():
                shutil.rmtree(destination)
            plan_path.unlink(missing_ok=True)
            raise


def blender_records():
    import_blender()
    folder, plan = WORK / 'blender', WORK / 'blender_plan.json'
    manifest = folder / 'manifest.json'
    data = read_json(manifest)
    renderer.validate_plan(read_json(plan))
    records, media = [], set()
    for row in data['records']:
        image, label = ROOT / row['image'], ROOT / row['label']
        boxes = [[obj['bbox_xyxy_normalized'][i] * (640 if i % 2 == 0 else 480) for i in range(4)]
                 for obj in row['objects']]
        text = label.read_text(encoding='utf8')
        validate_pair(image, text, boxes)
        media.add(image)
        if row.get('visible_mask'):
            media.add(folder / row['visible_mask'])
        media.update(folder / obj['visibility']['isolated_mask'] for obj in row['objects'])
        metadata = {
            'branch': 'blender_render', 'source_split': data['split'], 'source_image': row['image'],
            'source_manifest': relative(manifest), 'source_group': row['source_scene'],
            'source_blend_sha256': data['source_blend_sha256'], 'source_scene': data['source_scene'],
            'camera': row['camera'], 'camera_height_m': row['camera_height_m'],
            'camera_matrix_world': row['camera_matrix_world'], 'parameters': row['parameters'],
            'buildings': row['buildings'], 'appearance': row['appearance'],
            'instances': [{'building_id': obj['building'], 'xyxy': box} for obj, box in zip(row['objects'], boxes)],
            'visible_mask': relative(folder / row['visible_mask']) if row.get('visible_mask') else None,
            'instance_evidence': row['objects'],
        }
        records.append((image, text, metadata))
    metadata = {manifest, plan, source_path(TASK, 'blender_scene'), *(ROOT / row['label'] for row in data['records'])}
    return records, media, metadata


def direct_records(recipe):
    rows, _ = prepare(TASK, split=recipe['direct_split'])
    extra, _ = (prepare(TASK, frame_record_ids=recipe['additional_direct_records'])
                if recipe['additional_direct_records'] else ([], None))
    records, media, metadata, seen = [], set(), {TARGETS}, set()
    exposure = {}
    index = ROOT / 'datasets/exports/building_name_paper/workspace_sources.json'
    if index.is_file():
        metadata.add(index)
        for item in read_json(index)['records']:
            exposure.setdefault(item.get('workspace_image'), set()).add(item['split'])
            frame = (item.get('source_review') or {}).get('frame_id')
            if frame:
                exposure.setdefault(frame, set()).add(item['split'])
    for manifest, row, truth, image, parent, label, text in rows + extra:
        key = row['frame_environment']['frame_id']
        if key in seen:
            continue
        seen.add(key)
        media.add(image)
        if parent:
            media.add(parent)
        metadata.add(manifest)
        if label:
            metadata.add(label)
        boxes = [item.get('xyxy') or item['box_xyxy'] for item in truth['instances']]
        validate_pair(image, text, boxes)
        records.append((image, text, {
            'branch': 'direct_real', 'source_split': truth['split'], 'source_image': relative(image),
            'source_manifest': relative(manifest), 'source_record_id': row['id'],
            'source_group': row.get('source_group') or row.get('parent_video') or relative(image),
            'parent_video': row.get('parent_video'), 'frame_environment': row['frame_environment'],
            'known_export_splits': sorted(exposure.get(relative(image), set()) | exposure.get(key, set())),
            'instances': [{'xyxy': box} for box in boxes],
        }))
    return records, media, metadata


def composite_plan(recipe):
    declaration = read_json(source_path(TASK, 'production_plan'))
    rooftop.validate_request(declaration)
    selected, counts = [], Counter()
    for state in declaration['strata']:
        if not state['target']['papers']:
            continue
        candidate = state['background']['candidates'][0]
        key = candidate['source_review']['frame_id']
        if counts[key] >= recipe['composites_per_background']:
            continue
        for variant in range(counts[key], recipe['composites_per_background']):
            copied = deepcopy(state)
            copied.update(id=state['id'] + f'_sample_{variant:02d}', min_count=1, max_count=1)
            copied['augmentation'] = {'mode': 'scene_bounded_v1' if variant % 2 else 'none'}
            copied['parameters']['seed'] = recipe['seed'] + len(selected)
            for paper in copied['target']['papers']:
                paper['text'] = rooftop.make_name(recipe['seed'] + len(selected))
            selected.append(copied)
            counts[key] += 1
    if not selected:
        raise ValueError('No eligible real-background paper composition states')
    declaration.update(budget=len(selected), strata=selected, purpose='Real photo paper replacement sample')
    rooftop.validate_request(declaration)
    rooftop.current_environment_sources(declaration)
    return declaration


def validate_pair(image, text, boxes):
    with Image.open(image) as pixels:
        pixels.load()
        width, height = pixels.size
    check_yolo(text, 1, 0)
    lines = text.splitlines()
    if len(lines) != len(boxes):
        raise ValueError('Paper count differs from the complete annotation')
    for line, box in zip(lines, boxes):
        _, x, y, w, h = map(float, line.split())
        measured = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
        if max(abs(a-b) for a,b in zip(measured, box)) > 1:
            raise ValueError('Paper box differs from the recorded detection')
    return [width, height]


def owned_files():
    """Identify prior generated paths and preserve every other file at publication."""
    owned = set()
    manifest = OUTPUT / 'manifest.json'
    if manifest.is_file():
        data = read_json(manifest)
        if data.get('task') != TASK:
            raise ValueError('Current sample manifest belongs to another task')
        owned.add('manifest.json')
        for row in data['records']:
            for key in ('image', 'label'):
                value = Path(row[key])
                path = ROOT / value if value.parts[:1] == ('datasets',) else OUTPUT / value
                if not path.resolve().is_relative_to(OUTPUT) or not path.is_file():
                    raise ValueError('Prior generated sample path is missing or outside the task')
                owned.add(path.relative_to(OUTPUT).as_posix())
        for name in data.get('artifacts', []):
            path = OUTPUT / name
            if not path.resolve().is_relative_to(OUTPUT) or not path.is_file():
                raise ValueError('Prior sample artifact is missing or outside the task')
            owned.add(name)
    legacy = OUTPUT / 'blender'
    if (legacy / 'manifest.json').is_file():
        imported = WORK / 'blender'
        old = read_json(legacy / 'manifest.json')
        paths = {legacy / 'manifest.json'}
        for row in old['records']:
            paths.update(ROOT / row[key] for key in ('image', 'label'))
            if row.get('visible_mask'):
                paths.add(legacy / row['visible_mask'])
            paths.update(legacy / item['visibility']['isolated_mask'] for item in row['objects'])
        for path in paths:
            if not path.resolve().is_relative_to(legacy):
                raise ValueError('Prior Blender evidence escapes its branch')
            counterpart = imported / path.relative_to(legacy)
            if path.name != 'manifest.json' and (not counterpart.is_file() or digest(path) != digest(counterpart)):
                raise ValueError('Prior Blender branch differs from the retained evidence')
            owned.add(path.relative_to(OUTPUT).as_posix())
    return owned


def contact_sheet(stage, records, filename='contact_sheet.jpg', columns=4, tile=(320,270)):
    if (stage / filename).exists():
        raise ValueError('Contact sheet would overwrite an unowned file')
    font = ImageFont.truetype(str(source_path(TASK, 'font')), 20)
    sheet = Image.new('RGB', (columns*tile[0], ((len(records)+columns-1)//columns)*tile[1]), '#20252b')
    for i, row in enumerate(records):
        image = Image.open(stage / row['image']).convert('RGB')
        width, height = image.size
        image.thumbnail((tile[0]-4, tile[1]-32))
        draw = ImageDraw.Draw(image)
        for obj in row['instances']:
            x0, y0, x1, y1 = obj['xyxy']
            draw.rectangle((x0*image.width/width, y0*image.height/height,
                            x1*image.width/width, y1*image.height/height), outline='#20ed74', width=2)
        x, y = i % columns * tile[0], i // columns * tile[1]
        sheet.paste(image, (x, y+29))
        title = {'direct_real': '实拍', 'real_background_composite': '实拍底图替字',
                 'blender_render': 'Blender', 'blender_photo_composite': '楼体＋实拍底图'}[row['branch']]
        if filename == 'hybrid_contact_sheet.jpg':
            title = ('低机位 左前' + str(abs(row['parameters']['yaw_degrees'])) + '°'
                     if row['parameters']['yaw_degrees'] else '平视 正前')
            title = Path(row['source_photo']).stem.split('_')[0] + ' ' + row['buildings'] + ' ' + title
        ImageDraw.Draw(sheet).text((x+5, y+3), f'{i+1} {title} 纸条{len(row["instances"])}', font=font, fill='white')
    sheet.save(stage / filename, quality=92)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args(argv)
    _, source = load_task_config(TASK)
    recipe = source['sample_recipe']
    hybrid.compose_branch(replace=args.replace)
    hybrid_records, hybrid_media, hybrid_metadata = hybrid.loaded_records()
    rendered, render_media, render_metadata = blender_records()
    direct, direct_media, direct_metadata = direct_records(recipe)
    direct_snapshot = deepcopy(direct)
    composite = composite_plan(recipe)
    working = WorkingInputs(TASK, backgrounds=[c['image'] for s in composite['strata'] for c in s['background']['candidates']],
                            targets=[rooftop.FONT], split=composite['target_split'])
    background_media = {ROOT / c[key] for s in composite['strata'] for c in s['background']['candidates']
                        for key in ('image', 'parent_source')}
    procedural = {*configs(), *render_metadata, *direct_metadata, *hybrid_metadata, *working.evidence,
                  rooftop.FONT, working.path(rooftop.FONT), source_path(TASK, 'production_plan')}
    media = render_media | direct_media | background_media | hybrid_media | working.media
    declaration = deepcopy(composite)
    states = list(declaration['strata'])
    for number, (image, _, record) in enumerate(direct + rendered + hybrid_records):
        state = {'id': f'{record["branch"]}_{number:04d}', 'min_count': 1, 'max_count': 1,
                 'prior': {'alpha': 1, 'beta': 1}, 'failure_event': 'paper_detection_failure',
                 'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
                 'background': {'source_group': record['source_group']}, 'target': {'paper_count': len(record['instances'])},
                 'labels': {'class_id': 0}, 'augmentation': {'mode': 'none'}, 'sources': [relative(image)],
                 'parameters': {'source_image': relative(image)}}
        states.append(state)
        record['state_id'] = state['id']
    declaration.update(budget=len(states), strata=states, purpose='Current four-branch building paper detection sample',
                       constraints={'supervision': 'complete_paper_detection'})
    operation = 'building_name.current_sample'
    committed = lambda token: receipt_completed(ROOT, operation, token)
    code = [Path(__file__), Path(rooftop.__file__), Path(hybrid.__file__), ROOT / 'datasets/scripts/common/package_direct_real.py',
            ROOT / 'datasets/scripts/common/working_inputs.py', ROOT / 'datasets/scripts/common/frame_environment.py',
            ROOT / 'datasets/scripts/common/recoverable_publish.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/paper.py',
            ROOT / 'datasets/scripts/common/perspective.py', ROOT / 'datasets/scripts/common/names.py',
            ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/source_history.py',
            ROOT / 'lab/scripts/data/export_site_adaptation.py', ROOT / 'lab/training_common/bayesian_allocation.py']
    with publisher.sample_lock(OUTPUT):
        publisher.recover(OUTPUT, committed)
        owned = owned_files()
        preserved = [relative(p) for p in OUTPUT.rglob('*') if p.is_file() and p.relative_to(OUTPUT).as_posix() not in owned]
        try:
            with generation_guard(operation, sorted(media), [OUTPUT], {'recipe': recipe}, root=ROOT,
                                  procedural_inputs=sorted(procedural), code_paths=code,
                                  production_plan=declaration, sample_output_paths=[OUTPUT/'images'],
                                  auxiliary_output_paths=[OUTPUT/'contact_sheet.jpg',OUTPUT/'hybrid_contact_sheet.jpg'], replace_outputs=args.replace,
                                  replace_verifier=(lambda *_: {'preserved_paths': preserved}) if args.replace else None) as run:
                if run.reused:
                    print(json.dumps(read_json(OUTPUT / 'manifest.json')['summary'], ensure_ascii=False))
                    return
                if OUTPUT.exists() and any(OUTPUT.iterdir()) and not args.replace:
                    raise ValueError('Use --replace to rebuild the current sample')
                stage = publisher.prepare(OUTPUT, run.token)
                publisher.copy_unowned(OUTPUT, stage, owned)
                for name in ('blender/images', 'blender/labels', 'blender/masks', 'blender'):
                    legacy = stage / name
                    if legacy.is_dir() and not any(legacy.iterdir()):
                        legacy.rmdir()
                (stage/'images').mkdir(exist_ok=True)
                (stage/'labels').mkdir(exist_ok=True)
                records = []
                for number, (image, text, record) in enumerate(direct + rendered + hybrid_records):
                    stem = f'{record["branch"]}_{number:04d}'
                    row = dict(record, image=f'images/{stem}{image.suffix.lower()}', label=f'labels/{stem}.txt')
                    row['image_size'] = validate_pair(image, text, [obj['xyxy'] for obj in row['instances']])
                    if any((stage / row[key]).exists() for key in ('image', 'label')):
                        raise ValueError('Generated sample would overwrite an unowned image or label')
                    shutil.copyfile(image, stage/row['image'])
                    (stage/row['label']).write_text(text, encoding='utf8')
                    records.append(row)
                for state in plan_production(composite, ROOT)['strata']:
                    candidate = state['background']['candidates'][0]
                    frame = working.path(candidate['image'], role='background')
                    image, papers = rooftop.compose_one(read_image(frame), candidate, state['target']['papers'],
                                                         font_path=working.path(rooftop.FONT))
                    augmentation = {'mode': 'none'}
                    if state['augmentation']['mode'] == 'scene_bounded_v1':
                        import random
                        image, augmentation = rooftop.augment_scene(image, papers, random.Random(state['parameters']['seed']))
                    height, width = image.shape[:2]
                    stem = 'composite_' + state['id']
                    text = ''.join(yolo_row(paper['box_xyxy'], width, height) + '\n' for paper in papers)
                    row = {'image': f'images/{stem}.png', 'label': f'labels/{stem}.txt', 'state_id': state['id'],
                           'branch': 'real_background_composite', 'source_split': 'train',
                           'source_image': relative(frame), 'source_group': candidate['parent_source'],
                           'frame_environment': candidate['source_review'], 'source_candidate': candidate,
                           'augmentation': augmentation, 'image_size': [width, height],
                           'instances': [{'xyxy': paper['box_xyxy'], 'building_id': paper['building_id']} for paper in papers]}
                    if any((stage / row[key]).exists() for key in ('image', 'label')):
                        raise ValueError('Composite would overwrite an unowned image or label')
                    cv2.imencode('.png', image)[1].tofile(str(stage/row['image']))
                    (stage/row['label']).write_text(text, encoding='utf8')
                    validate_pair(stage/row['image'], text, [obj['xyxy'] for obj in row['instances']])
                    records.append(row)
                summary = {'images': len(records), 'paper_boxes': sum(len(row['instances']) for row in records),
                           'by_branch': dict(Counter(row['branch'] for row in records)),
                           'boxes_by_branch': {branch: sum(len(row['instances']) for row in records if row['branch'] == branch)
                                               for branch in {row['branch'] for row in records}},
                           'by_source_split': dict(Counter(row['source_split'] for row in records)),
                           'empty_images': sum(not row['instances'] for row in records)}
                contact_sheet(stage, records)
                selected = [row for plate in hybrid.recipe()['plates'] for view in ('left_low','front_level')
                            for combo in ('A','AB','ABC') for row in records
                            if row['branch'] == 'blender_photo_composite' and row['buildings'] == combo
                            and row['source_photo'] == plate['image'] and row['id'].endswith(view)]
                contact_sheet(stage,selected,'hybrid_contact_sheet.jpg',columns=3,tile=(480,392))
                if direct_records(recipe)[0] != direct_snapshot:
                    raise ValueError('Direct source eligibility changed while preparing the sample')
                rooftop.current_environment_sources(composite)
                hybrid.qualified_plates(hybrid.recipe())
                write_json(stage/'manifest.json', {
                    'task': TASK, 'status': 'complete_four_branch_sample', 'label_scope': 'paper_detection',
                    'summary': summary, 'production_plan': run.plan, 'records': records,
                    'artifacts': ['contact_sheet.jpg','hybrid_contact_sheet.jpg'], 'blender_plan': relative(WORK/'blender_plan.json'),
                    'plan': relative(source_path(TASK,'hybrid_plan')),
                    'plan_scope': 'Hybrid branch recipe; production_plan is the full combined generation snapshot',
                    'source_scope': 'Development sample; retain original splits, parent reuse and known evaluation contact.'})
                publisher.publish(OUTPUT, stage, run.token, keep_root=True)
                by_id = {state['id']: state for state in run.plan['strata']}
                for row in records:
                    state = by_id[row['state_id']]
                    run.record_sample(state['id'], output=OUTPUT/row['image'], sources=state['sources'],
                                      parameters=state['parameters'], actual=row)
                run.set_result(summary)
            publisher.recover(OUTPUT, committed)
        except BaseException:
            publisher.recover(OUTPUT, committed)
            raise
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
