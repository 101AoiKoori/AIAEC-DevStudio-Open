"""Use an existing labeled Blender render in the regular person sample plan.

The render is not regenerated: only its value and saturation are changed.
"""
import json
import math
from pathlib import Path

from lab.training_common.camera_tone import apply_tone, read_image

ROOT = Path(__file__).resolve().parents[3]
BASE = 'datasets/annotated/common/review/person/toolchain_20261115/blender_raw_preview'
IMAGE = BASE + '.png'
LABEL = BASE + '.txt'
EVIDENCE = BASE + '_evidence.json'
MASK_MANIFEST = BASE + '_mask_manifest.json'


def _evidence():
    return json.loads((ROOT / EVIDENCE).read_text(encoding='utf-8'))


def _objects(evidence):
    return [dict(item, kind='person', category=item['class'], source=item['texture'],
                 angles_degrees={'yaw': round(math.degrees(item['yaw_radians']), 3),
                                 'pitch': 0, 'roll': 0})
            for item in evidence['objects']]


def candidate_states(lighting):
    """Return feasible 640x480 states, including one guaranteed Blender state."""
    evidence = _evidence()
    scene = evidence['scene']['path']
    actors = _objects(evidence)
    states = []
    for name in lighting['priority']:
        low, high = lighting['image_value_gain'][name]
        for saturation in (0.7, 1.0):
            state = {
                'id': f'blender_existing_{name}_{int(saturation * 100)}',
                'failure_event': 'person_blender_existing',
                'min_count': 1 if not states else 0, 'max_count': 1,
                'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
                'failure_cost': 1, 'conditions': [],
                'background': {'role': 'blender_existing', 'image': IMAGE, 'manifest': EVIDENCE},
                'axes': {'person_count': len(actors), 'yaw': 0, 'lighting': name,
                         'background': 'blender:' + scene, 'vehicle_occlusion': 'none',
                         'source_resolution': '640x480', 'class_composition': 'both'},
                'target': {'actors': actors},
                'labels': {'person': 'existing_blender_visible_yolo', 'white_back': 'not_labeled'},
                'augmentation': {'value_gain': round((low + high) / 2, 3),
                                 'saturation_gain': saturation},
                'parameters': {'resolution': [640, 480], 'source_scene': scene,
                               'source_label': LABEL, 'mask_manifest': MASK_MANIFEST,
                               'source_kind': 'existing_blender_render'},
                'sources': [IMAGE],
            }
            states.append(state)
    return states


def scene_sources(declaration):
    """Return media and accompanying procedural records for production guard."""
    return [IMAGE], [LABEL, EVIDENCE, MASK_MANIFEST, _evidence()['scene']['path']]


def prepare_scene(declaration):
    evidence = _evidence()
    return {'row': {'human_accepted': False, 'timestamp_seconds': None},
            'image': ROOT / IMAGE, 'existing': [], 'protected': [],
            'evidence': evidence}


def produce_scene(prepared, state):
    image = read_image(prepared['image'])
    tone = state['augmentation']
    image = apply_tone(image, value_gain=tone['value_gain'],
                       saturation_gain=tone['saturation_gain'])
    labels = (ROOT / LABEL).read_text(encoding='utf-8').splitlines()
    return {'image': image, 'labels': labels, 'objects': _objects(prepared['evidence'])}
