"""Keep source identities and working-copy names stable after physical moves."""

import json


RELOCATIONS = 'datasets/raw_source_relocations.json'


def read_source_history(root):
    manifest = root / RELOCATIONS
    if not manifest.is_file():
        return ()
    data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    mappings = []
    for field in ('moves', 'aliases'):
        entries = [(row['current_path'].rstrip('/'), row['original_path'].rstrip('/'))
                   for row in data.get(field, [])]
        mappings.extend(sorted(entries, key=lambda item: -len(item[0])))
    return tuple(mappings)


def historical_source(path, history):
    """Recover a former source name for identity, never to open a file."""
    for current, original in history:
        if path == current or path.startswith(current + '/'):
            return original + path[len(current):]
    return path


def capture_identity(row, history):
    identity = row.get('capture_identity')
    if identity:
        return identity
    return historical_source(row.get('duplicate_of') or row.get('source_image') or row['image'], history)
