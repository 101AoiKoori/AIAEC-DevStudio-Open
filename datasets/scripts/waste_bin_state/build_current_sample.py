"""Package the current representative waste-bin-state images and labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.package_direct_real import TARGETS, check_yolo, prepare, workspace_file
from datasets.scripts.common.paths import SPLITS, sample_dir
from lab.training_common.production_guard import generation_guard, receipt_completed
from datasets.scripts.waste_bin_state import sample_publish

TASK = 'waste_bin_state'
ANNOTATED = ROOT / 'datasets/annotated/waste_bin_state'
DIRECT_IDS = {
    'train': (
        'archive-3409--images-0002',
        'archive-3410--images-0005',
        'db_073c67d6d9716c95_empty_negative_train',
        'db_5b5317d8c817dacb_empty_negative_train',
        'db_ae6c5cfe299c87b9_empty_negative_train',
    ),
    'val': ('archive-3412--images-bins_004_001', 'archive-3412--images-bins_004_020'),
    'test': (
        'db_018029e2e97f7bf8_empty_negative_test',
        'db_0c5ebe66f6808c74_empty_negative_test',
        'db_6c39b8604ec443ce_direct_target_test',
    ),
}
CLASS_NAMES = ('open', 'closed')
COMPOSITE_KINDS = ('real_background_composite', 'blender_route_composite')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rel(path):
    return path.resolve().relative_to(ROOT).as_posix()


def direct_inputs(by_split):
    media, procedural, rows = set(), {TARGETS}, []
    for split, ids in by_split.items():
        selected, _ = prepare(TASK, split=split, record_ids=ids)
        for manifest, row, truth, image, parent, label, text in selected:
            media.add(image)
            if parent is not None:
                media.add(parent)
            for path in (row.get('source_image'), row.get('canonical_manifest')):
                if path:
                    source_path = workspace_file(path)
                    if source_path.suffix.lower() == '.json':
                        procedural.add(source_path)
                    else:
                        media.add(source_path)
            procedural.add(manifest)
            if label is not None:
                procedural.add(label)
            rows.append((image, text, {
                'source_kind': 'direct_real' if text else 'direct_empty_negative',
                'split': split,
                'source_image': rel(image),
                'source_manifest': rel(manifest),
                'source_record_id': row['id'],
                'parent_group': row.get('source_group') or row.get('parent_video') or rel(image),
                'frame_environment': row['frame_environment'],
                'bins': [{'color': instance.get('color') or instance.get('bin_color') or 'unknown',
                          'state': CLASS_NAMES[instance['class_id']], 'bbox_xyxy': instance['xyxy']}
                         for instance in truth['instances'] if instance.get('disposition', 'target') == 'target'],
            }))
    return rows, media, procedural


def composite_inputs(manifest, plan_path, allocation_path):
    report, plan, allocation = map(read_json, (manifest, plan_path, allocation_path))
    if report.get('model_key') != TASK or report.get('status') != 'complete_candidates':
        raise ValueError('Expected a complete waste-bin composition receipt')
    split = report['split']
    if split not in SPLITS or split != plan['target_split'] or split != allocation['target_split']:
        raise ValueError('Composition and allocation split differ')
    by_id = {row['id']: row for row in plan['strata']}
    chosen = {row['id'] for row in allocation['strata'] if row['planned_count']}
    if {row['state_id'] for row in report['records']} != chosen or len(chosen) != plan['budget']:
        raise ValueError('Actual composition differs from the selected states')
    media, procedural, rows = set(), {manifest, plan_path, allocation_path}, []
    for row in report['records']:
        axes = by_id[row['state_id']]['axes']
        actual = row['actual']
        background = actual['background']
        if background['split'] != split:
            raise ValueError('Composition contains an incorrect source split')
        kind = ('blender_route_composite' if 'rendered' in Path(background['image']).parts
                else 'real_background_composite')
        if axes['source_kind'] != kind or kind not in COMPOSITE_KINDS:
            raise ValueError('Composition axes and actual background origin disagree')
        image = workspace_file(row['image'])
        label = workspace_file(row['label'])
        media.add(image)
        procedural.add(label)
        boxes = [*actual['bins'], *actual['retained_background_bins']]
        rows.append((image, label.read_text(encoding='utf-8'), {
            'source_kind': kind,
            'split': split,
            'source_image': rel(image),
            'source_manifest': rel(manifest),
            'source_record_id': row['state_id'],
            'parent_group': background.get('parent_group') or background.get('parent_image'),
            'frame_environment': background['frame_environment'],
            'background_image': background['image'],
            'render_image': actual['render_image'],
            'render_manifest': actual['render_manifest'],
            'scale_basis': actual['scale_basis'],
            'augmentation': actual['augmentation'],
            'axes': axes,
            'bins': [{'color': item.get('bin') or item.get('color') or item.get('bin_color') or 'unknown',
                      'state': item['state'], 'bbox_xyxy': item['bbox_xyxy']} for item in boxes],
        }))
    return rows, media, procedural


def verify_pair(image, text, record):
    with Image.open(image) as pixels:
        width, height = pixels.size
        pixels.load()
    check_yolo(text, len(CLASS_NAMES), 0)
    labels = []
    for line in text.splitlines():
        cls, x, y, w, h = map(float, line.split())
        box = ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
        labels.append((CLASS_NAMES[int(cls)], box))
    remaining = list(record['bins'])
    for state, box in labels:
        match = next((i for i, item in enumerate(remaining) if item['state'] == state and
                      max(abs(a-b) for a,b in zip(box,item['bbox_xyxy'])) <= 1), None)
        if match is None:
            raise ValueError('Label does not match a documented whole-bin box: ' + rel(image))
        remaining.pop(match)
    if remaining:
        raise ValueError('Missing whole-bin label: ' + rel(image))
    return [width, height]


def summarize(records):
    source_mix = Counter(row['source_kind'] for row in records)
    quantity = Counter(str(len(row['bins'])) for row in records)
    state_count = Counter(item['state'] for row in records for item in row['bins'])
    color_state = {}
    for row in records:
        for item in row['bins']:
            color_state.setdefault(item['color'], Counter())[item['state']] += 1
    return {
        'images': len(records), 'by_source': dict(source_mix), 'by_quantity': dict(quantity),
        'by_split': dict(Counter(row['split'] for row in records)),
        'state_instances': dict(state_count),
        'color_state_instances': {color: dict(counts) for color, counts in color_state.items()},
    }


def package(composite_manifest, plan, allocation, direct_selectors=DIRECT_IDS, *, replace=False):
    output = sample_dir(TASK)
    if output.is_symlink():
        raise ValueError('Sample root must not be a symlink')
    output = output.resolve()
    direct, direct_media, direct_metadata = direct_inputs(direct_selectors)
    composites, generated, synthesis_metadata = composite_inputs(composite_manifest, plan, allocation)
    data, source = load_task_config(TASK)
    if (ROOT / data['sample_output']).resolve() != output:
        raise ValueError('Sample root differs from dataset.yaml')
    configs = [ROOT / 'datasets/scripts' / TASK / 'config' / name for name in ('source.yaml', 'dataset.yaml')]
    direct_database = ROOT / source['databases']['frame_environment']
    metadata = direct_metadata | synthesis_metadata | {*configs, direct_database}
    code = [Path(__file__), Path(sample_publish.__file__), ROOT / 'datasets/scripts/common/recoverable_publish.py',
            ROOT / 'datasets/scripts/common/package_direct_real.py',
            ROOT / 'datasets/scripts/common/frame_environment.py',
            ROOT / 'datasets/scripts/common/source_history.py']
    params = {'direct_record_ids': direct_selectors, 'allocation': rel(allocation),
              'production_plan': rel(plan), 'composite_manifest': rel(composite_manifest)}
    operation = f'{TASK}.build_current_sample'
    committed = lambda token: receipt_completed(ROOT, operation, token)
    with sample_publish.sample_lock(output):
        sample_publish.recover(output, committed)
        # An otherwise empty scaffold may contain .gitkeep. Preserve it on first run.
        scaffold = (output.is_dir() and {item.name for item in output.iterdir()} <= {'.gitkeep'})
        replacing = replace or scaffold
        owned = set()

        def verify_existing(workspace, output_state, previous):
            owned.update(sample_publish.owned_files(output, previous, output_state, workspace))
            return True

        stage = None
        try:
            with generation_guard(operation, sorted(direct_media | generated), [output], params,
                                  code_paths=code, procedural_inputs=sorted(metadata), root=ROOT,
                                  replace_outputs=replacing,
                                  replace_verifier=verify_existing if replacing else None) as run:
                if run.reused:
                    return read_json(output / 'manifest.json')
                if output.exists() and any(output.iterdir()) and not replacing:
                    raise ValueError('Current sample directory requires --replace before packaging')
                if direct_inputs(direct_selectors)[0] != direct:
                    raise ValueError('Direct source eligibility changed before packaging')
                stage = sample_publish.prepare(output, run.token)
                sample_publish.copy_unowned(output, stage, owned)
                (stage / 'images').mkdir(exist_ok=True)
                (stage / 'labels').mkdir(exist_ok=True)
                records = []
                for n, (image, label, record) in enumerate(direct + composites, 1):
                    size = verify_pair(image, label, record)
                    stem = f'{record["source_kind"]}_{n:04d}'
                    record = dict(record, image=f'images/{stem}{image.suffix.lower()}',
                                  label=f'labels/{stem}.txt', image_size=size)
                    if (stage / record['image']).exists() or (stage / record['label']).exists():
                        raise ValueError('Generated sample would overwrite an unowned file')
                    shutil.copyfile(image, stage / record['image'])
                    (stage / record['label']).write_text(label, encoding='utf-8')
                    records.append(record)
                summary = summarize(records)
                manifest = {'task': TASK, 'status': 'complete_representative_sample',
                            'class_names': list(CLASS_NAMES),
                            'purpose': 'Representative subset of available data production branches',
                            'split_policy': 'retain existing source split; composition sampled from train',
                            'bayesian_composition_plan': rel(plan), 'bayesian_allocation': rel(allocation),
                            'composition_manifest': rel(composite_manifest), 'summary': summary, 'records': records}
                if (stage / 'manifest.json').exists():
                    raise ValueError('Generated manifest would overwrite an unowned file')
                (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',
                                                    encoding='utf-8')
                sample_publish.publish(output, stage, run.token)
                run.set_result(summary)
            # Guard exit has committed the receipt; remove the previous version.
            sample_publish.recover(output, committed)
        except BaseException:
            # Do not discard a backup if the receipt store cannot be read.
            if sample_publish._metadata(output)[1].exists():
                sample_publish.recover(output, committed)
            elif stage is not None and stage.exists():
                shutil.rmtree(stage)
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composites', type=Path, default=ANNOTATED / 'sample_composites/manifest.json')
    parser.add_argument('--plan', type=Path, default=ANNOTATED / 'sample_composite_plan.json')
    parser.add_argument('--allocation', type=Path, default=ANNOTATED / 'sample_composite_allocation.json')
    parser.add_argument('--replace', action='store_true',
                        help='Replace only generated files verified against the previous sample receipt.')
    args = parser.parse_args()
    manifest = package(args.composites.resolve(), args.plan.resolve(), args.allocation.resolve(), replace=args.replace)
    print(json.dumps(manifest['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
