"""Run the shared training CLI with the person V4 configuration."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2] if HERE.parent.name == 'task' else HERE.parents[1]
PACKAGED = (PACKAGE_ROOT / 'scripts/common/engine').is_dir()
ROOT = PACKAGE_ROOT if PACKAGED else HERE.parents[5]
COMMON = ROOT / 'scripts/common' if PACKAGED else ROOT / 'lab/training_common'
DEFAULT_CONFIG = 'config/train.yaml' if PACKAGED else 'lab/features/person/training/config/person_v4.yaml'
sys.path.insert(0, str(COMMON))
from engine.cli import main


if __name__ == '__main__':
    raise SystemExit(main(default_root=ROOT, default_config=DEFAULT_CONFIG))
