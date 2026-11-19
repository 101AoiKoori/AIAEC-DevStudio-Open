"""Meter-specific input declarations for the shared production receipt guard."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/meter/.")

from contextlib import contextmanager
from array import array
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.blender_workspace import WORKSPACE_BLEND


def _config_references(path, raw):
    """Find explicitly named workspace files in a JSON configuration."""
    value = json.loads(raw.decode('utf-8-sig'))
    found = []
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key != 'output':
                    visit(value)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)
        elif isinstance(obj, str) and obj.startswith(('datasets/', 'lab/', 'config/')):
            workspace_ref = WORKSPACE_BLEND.relative_to(ROOT).as_posix()
            target = ROOT / (workspace_ref if obj.startswith(workspace_ref + '#') else obj)
            if not target.is_file():
                raise ValueError('referenced workspace input missing: ' + obj)
            found.append(target)
    visit(value)
    return found


def blender_media(bpy, face_source=None):
    """Declare file-backed images actually loaded in the current Blender file."""
    result = []
    for image in bpy.data.images:
        if image.source != 'FILE':
            continue
        if face_source is not None and image.name == 'face.png' and image.packed_file:
            path = Path(face_source).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or bytes(image.packed_file.data) != path.read_bytes():
                raise ValueError('pending station face binding differs from its configured workspace source')
            result.append(path)
            continue
        if not image.filepath:
            if (image.name == 'Synthetic station room environment' and image.packed_file
                    and tuple(image.size) == (192, 96) and not image.is_dirty):
                from datasets.scripts.meter.power_meter_room import room_environment_pixels
                expected = room_environment_pixels()
                actual = array('f', [0.0]) * len(expected)
                image.pixels.foreach_get(actual)
                if actual != expected:
                    raise ValueError('packed procedural environment differs from its generator')
                result.append(Path(__file__).with_name('power_meter_room.py').resolve())
                continue
            raise ValueError('Blender FILE media has no source path: ' + image.name +
                             '; identify and register its workspace source before production')
        path = Path(bpy.path.abspath(image.filepath)).resolve()
        if path.is_file():
            if not path.is_relative_to(ROOT):
                raise ValueError('external Blender media is not allowed: ' + str(path))
            if image.packed_file and bytes(image.packed_file.data) != path.read_bytes():
                raise ValueError('packed Blender media differs from workspace source: ' + str(path))
            if getattr(image, 'is_dirty', False):
                raise ValueError('unsaved Blender image changes are not an indexed source: ' + image.name)
            result.append(path)
        else:
            raise ValueError('Blender media source is missing: ' + str(path) +
                             '; packed media also requires a resolvable registered workspace source')
    return result


def reload_unpacked_media(bpy):
    """Refresh file-backed Blender caches only after the guard has entered."""
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            image.reload()


def rebind_station_face(bpy, face_path):
    """Resolve the appended packed face to its matching current workspace file."""
    face_path = Path(face_path).resolve()
    if not face_path.is_relative_to(ROOT) or not face_path.is_file():
        raise ValueError('current station face is missing or outside the workspace')
    images = [image for image in bpy.data.images if image.source == 'FILE' and image.name == 'face.png']
    if len(images) != 1 or not images[0].packed_file:
        raise ValueError('expected one packed station face')
    if bytes(images[0].packed_file.data) != face_path.read_bytes():
        raise ValueError('packed station face differs from its current workspace source')
    if Path(bpy.path.abspath(images[0].filepath)).resolve() != face_path:
        images[0].filepath = str(face_path)


def ensure_registered_blender_media(bpy, run):
    """Check FILE media appended by a guarded Blender library load."""
    allowed = {ROOT / p for p in run.source_refs}
    allowed.update(ROOT / p for p in run.staged_inputs)
    current = blender_media(bpy)
    missing = [p for p in current if p not in allowed]
    if missing:
        raise ValueError('appended Blender media was not declared before the guard: ' + str(missing[0]))
    reload_unpacked_media(bpy)
    if blender_media(bpy) != current:
        raise ValueError('appended Blender media changed while refreshing file-backed inputs')


@contextmanager
def guarded(kind, args, script, source_blend, extra_inputs=(), code_dependencies=(), inplace=False,
            outputless=False, media_probe=None, media_reload=None, extra_outputs=(),
            production_plan=None, sample_output_paths=(), auxiliary_output_paths=()):
    """Check the shared database before any scene mutation or output creation."""
    output = Path(args.output).resolve()
    source = Path(source_blend).resolve()
    configs = [Path(v).resolve() for v in extra_inputs]
    media_before = list(media_probe()) if media_probe else []
    inputs = [source] + configs + media_before
    snapshots = {}
    pending = [p for p in configs if p.suffix.lower() == '.json']
    while pending:
        config = pending.pop()
        if config in snapshots:
            continue
        raw = config.read_bytes()
        snapshots[config] = raw
        refs = _config_references(config, raw)
        inputs.extend(refs)
        pending.extend(p for p in refs if p.suffix.lower() == '.json' and p not in snapshots)
    # A JSON can refer back to the loaded blend or to another listed input.
    inputs = list(dict.fromkeys(inputs))
    procedural = [p for p in inputs if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.npz'}]
    code = list(dict.fromkeys([Path(script).resolve(), Path(__file__).resolve()] +
                              [Path(v).resolve() for v in code_dependencies]))
    outputs = [] if outputless else [output, *(Path(v).resolve() for v in extra_outputs)] + (
        [source] if inplace and not source.is_relative_to(output) else [])
    params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    operation = 'meter.' + kind + ':' + output.relative_to(ROOT).as_posix()
    planning = ({'production_plan': production_plan, 'sample_output_paths': sample_output_paths,
                 'auxiliary_output_paths': auxiliary_output_paths} if production_plan is not None else {})
    with generation_guard(operation, inputs, outputs, params, code_paths=code,
                          procedural_inputs=procedural, inplace_paths=[source] if inplace else [],
                          **planning) as run:
        if any(path.read_bytes() != raw for path, raw in snapshots.items()):
            raise ValueError('JSON source changed while entering production guard; retry after index sync')
        if media_probe and list(media_probe()) != media_before:
            raise ValueError('Blender media declarations changed while entering production guard')
        if not run.reused and media_reload:
            media_reload()
            if media_probe and list(media_probe()) != media_before:
                raise ValueError('Blender media declarations changed while refreshing file-backed inputs')
        yield run
        if not run.reused and not outputless:
            # Preserve the index references in the existing batch record, if present.
            for name in ('manifest.json', 'prototype_manifest.json', 'endpoints.json'):
                path = output / name
                if path.is_file():
                    data = json.loads(path.read_text(encoding='utf-8-sig'))
                    data['production_source_refs'] = run.source_refs
                    if run.staged_inputs:
                        data['production_staged_inputs'] = run.staged_inputs
                    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
                    break
