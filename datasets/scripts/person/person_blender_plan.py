"""Pure person Blender state and source selection contracts."""

def validate_production_plan(declaration, args):
    if (not isinstance(declaration, dict) or declaration.get('model_key') != 'person'
            or declaration.get('target_split') not in ('train', 'val', 'test')):
        raise ValueError('a person training production declaration is required')
    if declaration.get('constraints') != {'white_back': 'no_person_boxes',
                                          'frontal_visibility': 'minimum_0.8',
                                          'geometry': 'visually_fitted'}:
        raise ValueError('unsupported person render constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('person render requires explicit states')
    fields = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight',
              'failure_cost', 'conditions', 'background', 'target', 'labels', 'augmentation',
              'parameters', 'sources'}
    ids = set()
    for state in states:
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError('person render state has missing or unsupported fields')
        state_id = state['id']
        if not isinstance(state_id, str) or not state_id.strip() or state_id in ids:
            raise ValueError('render state IDs must be unique nonempty strings')
        ids.add(state_id)
        if state['background'] != {'scene': args.scene, 'camera_preset': args.camera_preset}:
            raise ValueError('state scene or camera differs from the loaded scene')
        if args.scene == 'original' and args.camera_preset != 'person-near':
            raise ValueError('original scene supports only the person-near camera')
        target = state['target']
        base_target = {'front_count', 'noncommunity_front', 'white_back_class', 'occlusion'}
        explicit_vehicle = {'vehicle_class', 'occluded_person_class'}
        if not isinstance(target, dict) or set(target) not in (base_target, base_target | explicit_vehicle):
            raise ValueError('unsupported person render targets')
        count, non = target['front_count'], target['noncommunity_front']
        back, occlusion = target['white_back_class'], target['occlusion']
        if (type(count) is not int or count not in (3, 4, 5, 6) or type(non) is not int
                or non not in (1, 2) or back not in ('community', 'noncommunity')
                or occlusion not in ('none', 'person', 'vehicle')
                or (back == 'noncommunity' and non != 1)
                or (occlusion == 'vehicle' and count == 6)):
            raise ValueError('requested person render combination is unsupported')
        if explicit_vehicle <= set(target) and (occlusion != 'vehicle' or
                target['vehicle_class'] not in ('upright', 'fallen') or
                target['occluded_person_class'] not in ('community', 'noncommunity')):
            raise ValueError('vehicle pose and occluded person class require vehicle occlusion')
        if state['conditions'] != [] or state['labels'] != {'person': 'visible_yolo', 'white_back': 'negative_no_boxes'}:
            raise ValueError('unsupported person render conditions or labels')
        aug = state['augmentation']
        if (not isinstance(aug, dict) or set(aug) != {'exposure_ev'}
                or type(aug['exposure_ev']) not in (int, float) or aug['exposure_ev'] not in (0, -.25, -.55)):
            raise ValueError('unsupported render exposure')
        if state['parameters'] != {'resolution': [640, 480], 'camera_jitter_m': args.camera_jitter,
                                   'position_jitter_m': args.position_jitter, 'render_device': args.render_device}:
            raise ValueError('render parameters differ from the invocation')
        sources = ['person.whiteborder_textures', 'arena.' + args.scene]
        if occlusion == 'vehicle':
            sources.append('person.vehicle_occluders')
        if state['sources'] != sources:
            raise ValueError('render source groups differ from the requested state')

def select_occlusion_sources(objects, vehicles, target, rng, index):
    """Select the requested foreground class and vehicle pose for one scene."""
    people = [item for item in objects if item['front'] and
              (not target.get('occluded_person_class') or item['class'] == target['occluded_person_class'])]
    options = [item for item in vehicles if not target.get('vehicle_class') or
               item['class'] == target['vehicle_class']]
    if not people or not options:
        raise ValueError('requested person class or vehicle pose has no source')
    person = rng.choice(people)
    vehicle = rng.choice(options) if target.get('vehicle_class') else vehicles[index % len(vehicles)]
    return person, vehicle
