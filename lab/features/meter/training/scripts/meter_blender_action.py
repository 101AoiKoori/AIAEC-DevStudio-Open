"""Access an object's own animation curves across Blender Action versions."""


def object_action_curves(obj):
    """Return the active object's curves, selecting its slot in layered Actions."""
    animation = obj.animation_data
    if animation is None or animation.action is None:
        raise RuntimeError('Object has no active animation action: %s' % obj.name)
    action = animation.action
    if hasattr(action, 'fcurves'):
        return list(action.fcurves)
    slot = getattr(animation, 'action_slot', None)
    if slot is None:
        raise RuntimeError('Layered action has no active object slot: %s' % obj.name)
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', ()):
                if bag.slot == slot:
                    curves.extend(bag.fcurves)
    if not curves:
        raise RuntimeError('Active action slot has no animation curves: %s' % obj.name)
    return curves


def find_object_curve(curves, data_path, index):
    """Find one component without accidentally selecting a different slot."""
    matches = [curve for curve in curves
               if curve.data_path == data_path and curve.array_index == index]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one %s[%d] curve, got %d' %
                           (data_path, index, len(matches)))
    return matches[0]
