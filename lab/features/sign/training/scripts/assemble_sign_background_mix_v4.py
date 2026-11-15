"""Combine indexed sign candidates with explicit original-background evidence."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/sign/. Production remains paused.")

import json
import shutil
from pathlib import Path

from sign_production_guard import guarded, existing_result

ROOT = Path(__file__).resolve().parents[5]
REAL = ROOT / 'datasets/exports/sign/v4_real_scene_selection_v4_20261113'
BLEND = ROOT / 'datasets/exports/sign/v4_mixed_background_batch_20261113'
SECOND = ROOT / 'datasets/exports/sign/v4_second_background_60_20261113'
OUT = ROOT / 'runs/sign/v4_background_mix_v2_20261113'
REAL_VIDEO = 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
SECOND_VIDEO = 'datasets/raw/e_bike/archive_assets_legacy/视频/人物立牌正背面_电动车直立_倒伏和遮挡_场地扫描_012.mp4'
SCENE = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts task entry and its source.yaml")


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
