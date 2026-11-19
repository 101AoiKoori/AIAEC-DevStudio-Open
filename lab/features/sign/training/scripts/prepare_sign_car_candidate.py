"""Convert the recorded complete-scope P05 sign candidate to YOLO format."""
import json
from pathlib import Path

from sign_production_guard import guarded, existing_result

ROOT = Path(__file__).resolve().parents[5]
IMAGE = ROOT / 'Archive/Assets/captures/car_real_capture_20261112/P05_NOENTRY.png'
ANNOTATIONS = ROOT / 'datasets/annotated/car_real_capture_20261112/field_eval_v1/annotations_root.json'
OBSERVATIONS = ROOT / 'datasets/annotated/car_real_capture_20261112/target_instances_manifest.jsonl'
OUTPUT = ROOT / 'datasets/annotated/sign/car_real_candidate_20261114'


def main():
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
