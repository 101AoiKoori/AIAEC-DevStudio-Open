"""Dependencies consulted by the plate source-gate preparation scripts."""
from pathlib import Path
import csv
import json


def source_gate_dependencies(root, known):
    from lab.scripts.data import v4_source_gate as gate
    root = Path(root)
    files = {root / value for value in known['inputs']}
    media_suffixes = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.mp4', '.mov', '.avi', '.mkv'}
    def referenced(value):
        if isinstance(value, dict):
            for item in value.values(): referenced(item)
        elif isinstance(value, list):
            for item in value: referenced(item)
        elif isinstance(value, str):
            normalized = value.replace('\\', '/')
            if normalized.startswith(('datasets/', 'Archive/', 'ws/', 'runs/')) and Path(normalized).suffix.lower() in media_suffixes:
                files.add(root / normalized)
    for path in list(files):
        if path.suffix.lower() == '.json':
            referenced(json.loads(path.read_text(encoding='utf-8-sig')))
        elif path.suffix.lower() == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                referenced(list(csv.DictReader(stream)))
    files.add(root / 'config/perception_tasks.json')
    for name in gate.MODELS:
        config = root / 'lab/features' / gate.OWNERS.get(name, name) / 'training/config' / (name + '_v3.yaml')
        files.add(config)
        files.add(root / gate.read_yaml(config)['data'])
    code = {Path(__file__), root / 'lab/scripts/data/v4_source_gate.py',
            root / 'lab/training_common/engine/data.py', root / 'lab/training_common/engine/config.py',
            root / 'lab/training_common/train.py',
            root / 'lab/features/meter/training/scripts/generate_meter_synthetic.py'}
    return files, code


def blender_external_dependencies(bpy):
    """Read external scene assets without altering the scene; reject opaque dependencies."""
    if bpy.data.libraries:
        raise ValueError('Linked Blender libraries require an explicit dependency inventory')
    paths = set()
    for image in bpy.data.images:
        if getattr(image, 'packed_file', None) or image.source in {'GENERATED', 'VIEWER'}:
            continue
        if image.source not in {'FILE', 'MOVIE'}:
            raise ValueError('Unsupported external Blender image source: ' + image.name)
        if not image.filepath:
            raise ValueError('External Blender image has no path: ' + image.name)
        paths.add(Path(bpy.path.abspath(image.filepath)).resolve())
    for collection in (bpy.data.movieclips, bpy.data.sounds, bpy.data.fonts):
        for item in collection:
            if getattr(item, 'packed_file', None) or not getattr(item, 'filepath', ''):
                continue
            paths.add(Path(bpy.path.abspath(item.filepath)).resolve())
    return paths
