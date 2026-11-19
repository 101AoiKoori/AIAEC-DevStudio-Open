"""Render an immutable, train-only hard-negative layer for incremental exports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.fire.render_hard_negative_preview import build_scene, set_variant, write_json
from datasets.scripts.fire.render_blender_fire_dataset import material_override, render_id_pass
from datasets.scripts.fire.fire_blender_labels import visible_bbox
from lab.training_common.production_guard import generation_guard

CONFIG = ROOT/'datasets/scripts/fire/config/negative_replacement.json'
KINDS = {'red_signal_lamp': ('signal_lamp', 0), 'red_orange_bin': ('bin', 1), 'other_red_orange_object': ('cone', 2)}


def render():
    cfg = json.loads(CONFIG.read_text(encoding='utf8'))
    recipe = json.loads((ROOT/cfg['preview_config']).read_text(encoding='utf8'))
    snapshot = ROOT/cfg['source_snapshot']; output = ROOT/cfg['render_output']
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != snapshot.resolve():
        raise ValueError('Open the immutable hard-negative source snapshot in background Blender')
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if (output/'manifest.json').is_file():
        raise ValueError('Render stage already exists; the replacement entry verifies and reuses it')
    recipe['scene_name'] = 'Fire Hard Negatives Batch'; recipe['samples'] = cfg['samples']
    states = []
    for category, count in cfg['train_types'].items():
        for index in range(count):
            color = 'red' if category == 'red_signal_lamp' else ('orange' if category == 'other_red_orange_object' else 'red')
            if category == 'red_orange_bin' and index % 3 == 2:
                color = 'orange'
            params = {'category': category, 'index': index, 'color': color, 'seed': cfg['seed']+len(states)*1009}
            states.append({'id': f'hard_negative_{len(states)+1:06d}', 'failure_event': 'color_shortcut',
                'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
                'failure_cost': 1, 'conditions': [category, 'zero_designated_stickers'],
                'background': {'scene': recipe['scene_name']}, 'target': {'count': 0, 'building': 'A', 'distractor': category},
                'labels': {'class_id': 0, 'scope': 'entire_output'}, 'augmentation': recipe['appearance'],
                'sources': [cfg['source_snapshot']], 'parameters': params})
    plan = {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train', 'purpose': cfg['purpose'],
        'budget': len(states), 'uncertainty_weight': .5, 'independence_basis': 'One Blender mother, train only',
        'constraints': {'supervision': 'complete_empty'}, 'strata': states}
    with generation_guard('fire.hard_negative_render', [snapshot], [output], {'config': cfg},
            procedural_inputs=[CONFIG, ROOT/cfg['preview_config'], CONFIG.with_name('source.yaml'), CONFIG.with_name('dataset.yaml')],
            code_paths=[Path(__file__), ROOT/'datasets/scripts/fire/render_hard_negative_preview.py',
                ROOT/'datasets/scripts/fire/render_blender_fire_dataset.py', ROOT/'datasets/scripts/fire/fire_blender_labels.py'],
            production_plan=plan, sample_output_paths=[output/'images'], auxiliary_output_paths=[output/'instance_pass']) as run:
        if run.reused:
            return
        rig = build_scene(recipe); scene = rig['scene']; mask = material_override()
        for part in ('images', 'labels', 'instance_pass'):
            (output/part).mkdir(parents=True, exist_ok=True)
        records = []
        for state in states:
            p = state['parameters']; kind, kind_index = KINDS[p['category']]
            current = deepcopy(recipe); current['seed'] = p['seed']
            actual = set_variant(rig, current, kind, p['color'], kind_index)
            rig['paper'].hide_render = True; bpy.context.view_layer.update()
            image = output/'images'/(state['id']+'.png'); label = output/'labels'/(state['id']+'.txt')
            scene.render.filepath = str(image); bpy.ops.render.render(write_still=True)
            targets = [{'object': obj, 'id': i+2} for i,obj in enumerate(rig['props'][kind])]
            ids = render_id_pass(scene, targets, output/'instance_pass'/(state['id']+'.png'), mask)
            bbox, area = visible_bbox(ids>=2)
            if bbox is None or area < 100:
                raise ValueError('Hard-negative object is not visible')
            label.write_text('', encoding='utf8')
            row = {'image': f'images/{image.name}', 'label': f'labels/{label.name}',
                'instance_pass': f'instance_pass/{image.name}', 'branch': 'blender_render', 'generation_method': 'blender_render',
                'split': 'train', 'source_split': 'train', 'source_group': cfg['source_scene'], 'parents': [cfg['source_scene']],
                'source_scene': cfg['source_scene']+'#'+recipe['scene_name'], 'source_blend_sha256': digest,
                'scene_snapshot': cfg['source_snapshot'], 'instances': [],
                'building_counts': {'A': {'visible_count': 0, 'whole_building_count': None, 'scope': 'visible_output'}},
                'supervision_scope': 'entire_output_frame', 'production_state_id': state['id'],
                'distractor': {'kind': kind, 'color': p['color'], 'hard_negative_type': p['category'],
                    'object_ids': [t['id'] for t in targets], 'xyxy': bbox, 'visible_pixels': area},
                'hard_negative': True, 'context': {'distractor': p['category']}, 'actual': actual, 'sample_seed': p['seed']}
            records.append(row)
            run.record_sample(state['id'], output=image, sources=state['sources'], parameters=p, actual=row)
            write_json(output/'progress.json', {'completed': len(records), 'total': len(states)})
        write_json(output/'manifest.json', {'task': 'fire', 'status': 'complete', 'generation_plan': run.plan,
            'source_blend_sha256': digest, 'records': records, 'images': len(records)})
        run.set_result({'images': len(records), 'fire_instances': 0, 'split': 'train'})


if __name__ == '__main__':
    render()
