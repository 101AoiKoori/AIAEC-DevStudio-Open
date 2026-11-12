"""Independent waste_bin_state training, evaluation and tuning entry point."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab' / 'training_common'))
from engine.cli import main

if __name__ == '__main__':
    raise SystemExit(main(default_root=ROOT,
                         default_config='lab/features/waste_bin/training/config/waste_bin_state_v3.yaml'))
