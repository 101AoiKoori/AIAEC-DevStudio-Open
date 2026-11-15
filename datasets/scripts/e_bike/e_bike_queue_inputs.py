"""List the workspace inputs consulted by the V4 source queue.

This only discovers inputs; the production guard performs the index lookup.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/e_bike/. Production remains paused.")

import json
from pathlib import Path

from lab.scripts.data import v4_source_gate as gate
from lab.scripts.media.media_index import IMAGES, VIDEOS


_MEDIA_SUFFIXES = IMAGES | VIDEOS | {'.npz'}


def queue_input_report(root, queue, checked_sources=()):
    root = Path(root).resolve()
    queue_media, queue_procedural, _ = gate.queue_inputs(root)
    paths = set(queue_media) | set(queue_procedural)
    paths.update(root / value for value in queue['inputs'])
    # source_role compares each existing exposed image by content.
    paths.update(root / value for value in queue['exposed_images'])
    sources = [Path(value).resolve().relative_to(root).as_posix() for value in checked_sources]
    if any(value.startswith(gate.SITE_SEQUENCE + '/') for value in sources):
        paths.add(root / gate.SITE_SEQUENCE / 'ffmpeg.log')
    if any(value.startswith(gate.PERSON_SEQUENCE + '/') for value in sources):
        paths.add(root / gate.PERSON_SEQUENCE / 'frames_probe.json')
    ledger = gate.read_json(root, gate.DEVELOPMENT_EXPOSURE)
    for item in ledger['items']:
        paths.add(root / item['image'])
        if item.get('source_time_seconds') is not None:
            paths.add(root / item['parent_video'])
        paths.update(root / value for value in item['evidence'])
    for pair in ledger.get('near_duplicate_pairs', []):
        paths.update(root / pair[key] for key in ('a', 'b', 'evidence'))
    # Excluded historical references are not current media. Explicitly selected
    # sources remain guarded even if the historical ledger also names them.
    selected = {(root / value).resolve() for value in sources}
    unavailable = {(root / item['path']).resolve() for item in queue.get('unavailable_exposure_paths', [])}
    paths = {path for path in paths if path.resolve() not in unavailable or path.resolve() in selected}
    existing = {path for path in paths if path.is_file()}
    missing = sorted(path.relative_to(root).as_posix() for path in paths - existing)
    media = sorted(path for path in existing if path.suffix.lower() in _MEDIA_SUFFIXES)
    procedural = sorted(existing - set(media))
    return media, procedural, missing


def queue_inputs(root, queue, checked_sources=()):
    media, procedural, _ = queue_input_report(root, queue, checked_sources)
    return media, procedural
