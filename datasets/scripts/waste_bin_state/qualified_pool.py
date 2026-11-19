"""Task-local adapter for eligible raw background frames.

The source ledger owns eligibility.  This adapter only supplies the legacy
background record shape needed by the bin-state compositor.
"""

from pathlib import Path

from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.paths import WORKSPACE


def qualified_backgrounds(source, split, *, frame_ids=(), database=None):
    """Return current eligible frames, optionally restricted to explicit IDs."""
    pool = source['current_selection']['qualified_pool']
    database_key = pool['database']
    database = Path(database or WORKSPACE / source['databases'][database_key])
    selected = set(frame_ids)
    rows = query_frames(task='waste_bin_state', role=pool['role'],
                        split=split, database=database)
    result = []
    for row in rows:
        frame_id = row['frame_id']
        if selected and frame_id not in selected:
            continue
        paths = sorted((path for path in row['paths']
                        if path.startswith('datasets/raw/') and (WORKSPACE / path).is_file()),
                       key=lambda path: (not path.startswith('datasets/raw/shared/backgrounds/'), path))
        if not paths:
            if frame_id in selected:
                raise ValueError('selected eligible frame has no raw background image: ' + frame_id)
            continue
        image = paths[0]
        result.append({
            'id': 'pool_' + frame_id,
            'image': image,
            'parent_image': row.get('parent_image') or image,
            'video': row.get('parent_video'),
            'parent_group': row['parent_group'],
            'split': row['split'],
            'frame_id': frame_id,
            'record_id': row['record_id'],
            'manifest': source['databases'][database_key],
        })
    missing = selected - {row['frame_id'] for row in result}
    if missing:
        raise ValueError('selected frame lacks current eligible pool background: ' + ', '.join(sorted(missing)))
    return result
