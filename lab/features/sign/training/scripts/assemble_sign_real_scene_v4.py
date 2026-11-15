"""Select the existing measured real-scene composites into a V4 candidate."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/sign/. Production remains paused.")

import json
import shutil
from pathlib import Path

import compose_sign_scene as base
from sign_production_guard import guarded, existing_result

ROOT = base.ROOT
INPUT = ROOT / 'datasets/exports/sign/v4_measured_multibackground_train_20261112'
BACK = ROOT / 'datasets/annotated/sign/v4_training_backgrounds_20261112'
OUTPUT = ROOT / 'runs/sign/v4_real_scene_selection_v4_20261113'


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts task entry and its source.yaml")


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
