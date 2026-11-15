"""Write planned, source-linked person annotation candidates for camera images."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import sample_dir, split_dir
from lab.training_common.production_guard import generation_guard  # noqa: E402
from lab.scripts.data.export_site_adaptation import label_lines  # noqa: E402


def workspace_file(value, parent):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('a workspace-relative source path is required')
    path = (ROOT / value).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise ValueError('source is missing or outside its declared area')
    return path


def prepare(declaration):
    if (not isinstance(declaration, dict) or declaration.get('model_key') != 'person'
            or declaration.get('target_split') not in ('train', 'val', 'test')):
        raise ValueError('a person training candidate declaration is required')
    if declaration.get('constraints') != {'scope': 'near_frontal', 'white_back': 'no_person_boxes',
                                          'review': 'ai_candidate_not_human_accepted'}:
        raise ValueError('unsupported annotation constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('annotation states are required')
    allowed = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight',
               'failure_cost', 'conditions', 'background', 'target', 'labels', 'augmentation',
               'parameters', 'sources'}
    result, seen = {}, set()
    for state in states:
        if not isinstance(state, dict) or set(state) != allowed:
            raise ValueError('annotation state has missing or unsupported fields')
        sid = state['id']
        if not isinstance(sid, str) or not sid.strip() or sid in result:
            raise ValueError('state IDs must be unique nonempty strings')
        if state['min_count'] != 1 or state['max_count'] != 1 or state['conditions'] != []:
            raise ValueError('each explicitly annotated source can be used once')
        background, target = state['background'], state['target']
        if not isinstance(background, dict) or set(background) != {'source_image'}:
            raise ValueError('one camera image must be specified')
        image = workspace_file(background['source_image'], source_path('person', 'real_capture_root'))
        if image in seen:
            raise ValueError('duplicate camera image')
        seen.add(image)
        if not isinstance(target, dict) or set(target) != {'instances'} or not isinstance(target['instances'], list):
            raise ValueError('explicit person instances are required')
        if (state['labels'] != {'person': 'yolo_visible_boxes', 'white_back': 'no_boxes'}
                or state['augmentation'] != {'kind': 'none'}
                or state['parameters'] != {'image_size': [640, 480], 'image_operation': 'byte_copy',
                                            'annotation': 'ai_visual_candidate'}):
            raise ValueError('unsupported annotation, image or augmentation parameters')
        references = []
        boxes = []
        for item in target['instances']:
            if not isinstance(item, dict) or set(item) != {'class_id', 'xyxy', 'reference_asset'}:
                raise ValueError('each candidate requires class, visible box and artwork reference')
            cls = item['class_id']
            if type(cls) is not int or cls not in (0, 1):
                raise ValueError('unknown person class')
            folder = '社区人员' if cls == 0 else '非社区人员'
            ref = workspace_file(item['reference_asset'], source_path('person', 'person_artwork_root')/folder)
            references.append(ref)
            boxes.append({'class_id': cls, 'xyxy': item['xyxy']})
        if not boxes:
            raise ValueError('this entry supports positive candidates only')
        expected = sorted({image.relative_to(ROOT).as_posix(),
                           *(p.relative_to(ROOT).as_posix() for p in references)})
        if state['sources'] != expected:
            raise ValueError('declared sources differ from image and artwork references')
        with Image.open(image) as source:
            if source.size != (640, 480):
                raise ValueError('camera image size differs from the annotation')
        result[sid] = {'image': image, 'references': references,
                       'label': label_lines(boxes, 640, 480, 2, 0)}
    return result


def generate(plan_path, output):
    plan_path = workspace_file(plan_path, ROOT)
    output = (ROOT / output).resolve()
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    if output not in {sample_dir('person').resolve(), split_dir('person', declaration['target_split']).resolve()}:
        raise ValueError('output must be the person sample or declared split')
    inputs = prepare(declaration)
    config_inputs = [ROOT/'datasets/scripts/person/config/source.yaml', ROOT/'datasets/scripts/person/config/dataset.yaml']
    media = {item['image'] for item in inputs.values()}
    media.update(ref for item in inputs.values() for ref in item['references'])
    with generation_guard('person.annotate_real:' + output.relative_to(ROOT).as_posix(),
                          [*media, plan_path], [output], {'version': 1},
                          code_paths=[Path(__file__), ROOT/'lab/scripts/data/export_site_adaptation.py', ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py'],
                          procedural_inputs=[plan_path, *config_inputs], root=ROOT, production_plan=declaration,
                          sample_output_paths=[output/'images']) as run:
        if run.reused:
            return json.loads((output/'manifest.json').read_text(encoding='utf8'))
        if output.exists() and any(output.iterdir()):
            raise ValueError('output exists; refusing replacement')
        (output/'images').mkdir(parents=True)
        (output/'labels').mkdir()
        records = []
        for state in run.plan['strata']:
            item = inputs[state['id']]
            image = output/'images'/item['image'].name
            label = output/'labels'/(image.stem+'.txt')
            shutil.copyfile(item['image'], image)
            label.write_text(item['label'], encoding='utf8')
            actual = {'source_image': item['image'].relative_to(ROOT).as_posix(),
                      'reference_assets': [p.relative_to(ROOT).as_posix() for p in item['references']],
                      'instances': state['target']['instances'],
                      'label': label.relative_to(ROOT).as_posix(),
                      'annotation_basis': 'AI visual observation and comparison to supplied artwork',
                      'label_status': 'ai_candidate_not_human_accepted'}
            run.record_sample(state['id'], output=image, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            records.append({'state_id': state['id'], 'image': image.relative_to(ROOT).as_posix(), **actual})
        manifest = {'status': 'ai_candidate_not_human_accepted', 'source_kind': 'direct_camera_capture',
                    'image_operation': 'byte_copy_not_new_capture', 'records': records}
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = generate(args.production_plan, args.output)
    print(json.dumps({'status': result['status'], 'images': len(result['records'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
