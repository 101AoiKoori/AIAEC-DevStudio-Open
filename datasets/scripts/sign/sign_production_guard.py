"""Task-local input discovery for the shared production guard."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/sign/.")

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402

def guarded(operation, output, media, procedural, parameters, code, **kwargs):
    """Declare concrete inputs and output before the caller writes anything."""
    media = sorted({Path(p).resolve() for p in media})
    procedural = sorted({Path(p).resolve() for p in procedural})
    code = sorted({Path(p).resolve() for p in code} | {Path(__file__).resolve(),
                  ROOT / 'lab/training_common/production_guard.py', ROOT / 'lab/scripts/media/media_index.py',
                  ROOT / 'lab/scripts/media/media_index_sqlite.py'})
    relative = Path(output).resolve().relative_to(ROOT).as_posix()
    return generation_guard('sign:' + operation + ':' + relative,
                            list(dict.fromkeys(media + procedural)), [output], parameters,
                            code_paths=code, procedural_inputs=procedural, root=ROOT, **kwargs)


def existing_result(output):
    output = Path(output)
    for name in ('manifest.json', 'summary.json'):
        path = output / name
        if path.is_file():
            return json.loads(path.read_text(encoding='utf8'))
    raise ValueError('reused output lacks its existing result manifest')
