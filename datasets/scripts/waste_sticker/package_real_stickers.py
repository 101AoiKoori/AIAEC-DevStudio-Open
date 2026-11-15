"""Import designated real sticker frames through their recorded labels and split uses."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir
from datasets.scripts.common import images
from datasets.scripts.waste_sticker.source_inputs import real_source_catalog, select_direct_source
from lab.training_common.camera_tone import apply_tone
from lab.training_common.production_guard import generation_guard


def prepare_inputs(plan, catalog, formal=False):
    if plan.get('model_key') != 'waste_sticker' or plan.get('target_split') not in ('train', 'val', 'test'):
        raise ValueError('a waste_sticker plan and explicit target split are required')
    if plan.get('constraints') != {'labels': 'recorded_full_frame_stickers', 'resize': 'none'}:
        raise ValueError('unsupported direct sticker constraints')
    result = {}
    identities = set()
    for state in plan['strata']:
        image = state.get('background', {}).get('image')
        row = select_direct_source(catalog, image, plan['target_split'], formal=formal)
        if state['background'] != {'image': image} or state['target'] != {'kind': 'existing_stickers'}:
            raise ValueError('direct sticker plan must retain the existing image targets')
        expected = list(dict.fromkeys([image, row['parent']]))
        if state['sources'] != expected:
            raise ValueError('direct sticker plan sources must retain image and parent')
        if state['parameters'] != {'annotation_record_index': row['record_index']}:
            raise ValueError('direct sticker plan must bind its actual annotation record')
        if state['labels'] != {'format': 'yolo_detection', 'annotation_manifest': row['annotation_manifest']}:
            raise ValueError('direct sticker plan must bind its actual annotation manifest')
        aug = state['augmentation']
        if set(aug) != {'value_gain', 'saturation_gain'} or any(
                type(v) not in (int, float) or not .25 <= v <= 1.5 for v in aug.values()):
            raise ValueError('unsupported direct sticker tone transform')
        identity = (image, aug['value_gain'], aug['saturation_gain'])
        if identity in identities or state.get('max_count') != 1 or state.get('conditions'):
            raise ValueError('each exact direct sticker transform must occur once')
        identities.add(identity)
        if state['id'] in result:
            raise ValueError('duplicate direct sticker state id')
        result[state['id']] = row
    return result


def _write(output, run, selected, formal):
    if output.exists() and any(output.iterdir()):
        raise ValueError('sticker output must be empty')
    label_dir = 'labels' if formal else 'labels_candidate'
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / label_dir).mkdir()
    records = []
    for state in run.plan['strata']:
        if not state['planned_count']:
            continue
        row = selected[state['id']]
        annotation = row['annotation']
        image = images.read_image(ROOT / row['image'])
        width, height = annotation['image_size']
        if image.shape[:2] != (height, width):
            raise ValueError('real sticker dimensions differ from annotation')
        image = apply_tone(image, **state['augmentation'])
        stem = f'real_{len(records):06d}'
        image_out, label_out = output/'images'/f'{stem}.png', output/label_dir/f'{stem}.txt'
        images.write_image(image_out, image)
        lines = []
        for box in annotation['boxes']:
            x1, y1, x2, y2 = box['xyxy']
            lines.append(f"{box['class_id']} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
        label_out.write_text('\n'.join(lines)+'\n', encoding='utf8')
        actual = {'image': row['image'], 'parent': row['parent'], 'stationary': row['stationary'],
                  'annotation_manifest': row['annotation_manifest'], 'record_index': row['record_index'],
                  'label_status': annotation['annotation_level'], 'source_role': annotation.get('source_role'),
                  'augmentation': state['augmentation']}
        run.record_sample(state['id'], output=image_out, sources=state['sources'],
                          parameters=state['parameters'], actual=actual)
        records.append({'image': image_out.relative_to(output).as_posix(),
                        'label': label_out.relative_to(output).as_posix(), 'actual': actual})
    (output/'manifest.json').write_text(json.dumps({'status': 'complete_candidates',
        'split': run.plan['target_split'], 'records': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--inspect-sources', action='store_true', help='print source and annotation status without writing')
    args = parser.parse_args()
    _, config = load_task_config('waste_sticker')
    catalog = real_source_catalog(config)
    if args.inspect_sources:
        print(json.dumps(catalog, ensure_ascii=False, indent=2))
        return
    if args.plan is None or args.out is None:
        parser.error('--plan and --out are required for the guarded import')
    plan_path, output = (ROOT / args.plan).resolve(), (ROOT / args.out).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('plan must be an existing workspace JSON')
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    formal = output != sample_dir('waste_sticker').resolve()
    if formal and output != split_dir('waste_sticker', plan['target_split']).resolve():
        raise ValueError('output must be the current sticker sample or declared split')
    selected = prepare_inputs(plan, catalog, formal)
    media = sorted({ROOT / p for state in plan['strata'] for p in state['sources']})
    configs = [ROOT/'datasets/scripts/waste_sticker/config'/name for name in ('source.yaml', 'dataset.yaml')]
    with generation_guard('waste_sticker.direct_real:' + output.relative_to(ROOT).as_posix(), media, [output],
                          {'formal': formal}, code_paths=[Path(__file__),
                          ROOT/'datasets/scripts/waste_sticker/source_inputs.py',
                          ROOT/'datasets/scripts/common/images.py', ROOT/'lab/training_common/camera_tone.py'],
                          procedural_inputs=[plan_path, ROOT/config['real_annotations'], *configs], root=ROOT,
                          production_plan=plan, sample_output_paths=[output/'images']) as run:
        if not run.reused:
            prepare_inputs(run.plan, catalog, formal)
            _write(output, run, selected, formal)


if __name__ == '__main__':
    main()
