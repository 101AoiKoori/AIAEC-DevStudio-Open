"""Create a small source-grouped real-video sign evaluation draft.

Test is only proposed and must not be evaluated or used for tuning until the
dataset and model-selection scheme are frozen.  No independent capture claim.
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import source_role  # noqa: E402
from lab.scripts.data.v4_source_gate import SITE_SEQUENCE, SITE_VIDEO, build_queue, site_sequence_times  # noqa: E402

OUT = ROOT / 'runs/sign/v4_real_valtest_draft_20261112'
TONES = ROOT / 'datasets/exports/mixed_scene/v4_dark_sign_valtest_draft_20261112'
REVIEW = ROOT / 'datasets/annotated/site_capture_20261112/sign_plate_v4_review'
SOURCES = {433: ('val', 'positive', 'side_top_clipped'), 733: ('val', 'positive', 'front_top_clipped'),
           1261: ('val', 'negative', 'building_and_person'),
           2833: ('test', 'positive', 'back_top_clipped_crop'), 11095: ('test', 'negative', 'e_bike')}


def _main():
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


def main():
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


if __name__ == '__main__':
    main()
