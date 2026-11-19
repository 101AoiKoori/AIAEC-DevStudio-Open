"""Discover person backgrounds in the media database and analyze candidate frames."""

import argparse
import json
import hashlib
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from datasets.scripts.common.review import cli as review
from lab.training_common.bayesian_allocation import allocate

DATABASE = ROOT / "datasets/annotated/media_index/query_cache.sqlite3"
SOURCE_PREFIX = "datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113/images/"
SELECTION = ROOT / "datasets/scripts/person/config/review_selection.json"
REVIEW_PLAN = ROOT / "datasets/scripts/person/config/review_plan.json"
HIGH_RES_DIRECTORY = ROOT / "datasets/annotated/person/background_720p"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path):
    return path.relative_to(ROOT).as_posix()


def discover(limit_per_parent, seed=17):
    """Select indexed frames across source videos and timeline using shared allocation."""
    records = _read(SELECTION)["records"]
    existing = {row.get("frame", {}).get("path") for row in records}
    existing.update((row.get("evidence", {}).get("source_record") or {}).get("derived_from") for row in records)
    uri = DATABASE.resolve().as_uri() + "?mode=ro"
    groups = {}
    with sqlite3.connect(uri, uri=True) as connection:
        prefix = SOURCE_PREFIX.casefold()
        rows = connection.execute(
            "SELECT item_json FROM media WHERE path_key >= ? AND path_key < ? ORDER BY path_key, position",
            (prefix, prefix + "\uffff"),
        )
        seen = set()
        for (raw,) in rows:
            item = json.loads(raw)
            path = item.get("path")
            if not path or path in existing or path in seen or not (ROOT / path).is_file():
                continue
            seen.add(path)
            evidence = next((e for e in item.get("evidence", [])
                             if e.get("claims", {}).get("source_video")), None)
            if evidence is None:
                continue
            claims = evidence["claims"]
            parent = claims["source_video"]
            if not (ROOT / parent).is_file():
                continue
            frame = {
                "path": path,
                "parent_video": parent,
                "source_group": claims.get("source_group") or parent,
                "timestamp_seconds": claims.get("source_time_seconds"),
                "record_index": evidence.get("record_index"),
                "parent_link_verified": False,
                "role": "background",
            }
            source = {"manifest": evidence.get("manifest"), "record_index": evidence.get("record_index")}
            groups.setdefault(parent, []).append((frame, source))
    candidates = []
    strata = []
    for parent, frames in sorted(groups.items()):
        frames.sort(key=lambda item: (item[0].get('timestamp_seconds') or 0, item[0]['path']))
        for index, item in enumerate(frames):
            frame = item[0]
            candidates.append(item)
            strata.append({
                'id': frame['path'], 'failure_event': 'background_sampling',
                'min_count': 0, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
                'axes': {'source_group': parent, 'timeline_third': min(2, index * 3 // len(frames))},
            })
    if not strata:
        return [], None
    budget = sum(min(limit_per_parent, len(frames)) for frames in groups.values())
    declaration = {
        'schema_version': 1, 'model_key': 'person', 'target_split': 'train',
        'purpose': 'Select indexed backgrounds for automatic geometry and target observation.',
        'budget': budget, 'uncertainty_weight': 0,
        'independence_basis': 'Recorded parent video and relative temporal position.',
        'conditional_balance': {'axes': ['source_group', 'timeline_third'],
                                'stratify_by': 'source_group', 'prior': 1, 'seed': seed},
        'strata': strata,
    }
    allocation = allocate(declaration, root=ROOT)
    selected = [item for item, row in zip(candidates, allocation['strata']) if row['planned_count']]
    return selected, allocation['conditional_balance']


def _extract_720p(selected):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required to extract 720p source frames')
    HIGH_RES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    extracted = []
    errors = []
    for frame, source in selected:
        timestamp = frame.get('timestamp_seconds')
        if timestamp is None:
            errors.append({'image': frame['path'], 'reason': 'missing source timestamp'})
            continue
        key = hashlib.sha256(frame['path'].encode('utf-8')).hexdigest()[:12]
        target = HIGH_RES_DIRECTORY / (Path(frame['path']).stem + '_' + key + '.png')
        if not target.is_file():
            command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-ss', str(timestamp),
                       '-i', str(ROOT / frame['parent_video']), '-frames:v', '1',
                       '-vf', 'scale=1280:720', '-y', str(target)]
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode or not target.is_file():
                errors.append({'image': frame['path'], 'reason': result.stderr[-500:]})
                continue
        source = dict(source, derived_from=frame['path'], extraction_time_seconds=timestamp,
                      processing='video frame decoded at requested timestamp and resized to 1280x720')
        frame = dict(frame, path=_relative(target))
        extracted.append((frame, source))
    if errors:
        print(json.dumps({'extraction_failures': errors}, ensure_ascii=False))
    manifest_path = HIGH_RES_DIRECTORY / 'manifest.json'
    existing = _read(manifest_path) if manifest_path.exists() else {'task': 'person', 'records': []}
    known = {item['image'] for item in existing['records']}
    existing['records'].extend({'image': frame['path'], 'source_video': frame['parent_video'],
                                'source_time_seconds': frame['timestamp_seconds'],
                                'source_record': source, 'generation_method': 'video_frame_resize_720p'}
                               for frame, source in extracted if frame['path'] not in known)
    manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return extracted


def run(limit_per_parent, seed=17, resolution='original'):
    selected, allocation = discover(limit_per_parent, seed)
    if not selected:
        print("数据库没有新的可用背景帧")
        return 0
    if resolution == '720p':
        selected = _extract_720p(selected)
        if not selected:
            print('没有可提取的 720p 背景帧')
            return 1
    batch = "person_database_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
    directory = ROOT / "datasets/tmp/person_database_review" / batch
    directory.mkdir(parents=True)
    plan = _read(REVIEW_PLAN)
    plan["batch"] = batch
    plan["frames"] = [frame for frame, _ in selected]
    plan["database_selection"] = {"database": _relative(DATABASE), "frames": len(selected),
                                  "parents": len({f["parent_video"] for f, _ in selected}),
                                  "allocation": allocation}
    plan["outputs"] = {"metrics_dir": _relative(directory), "selection": _relative(directory / "selection.json")}
    plan_path = directory / "review_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for command in ("plan", "analyze", "apply"):
        status = review.main([command, "--task", "person", "--plan", _relative(plan_path)])
        if status and (command != "analyze" or not (directory / "metrics.jsonl").exists()):
            raise SystemExit(status)
    sources = {frame["path"]: source for frame, source in selected}
    results = _read(directory / "selection.json")["records"]
    for record in results:
        path = record["frame"]["path"]
        evidence = record.setdefault("evidence", {})
        evidence.update({"source_batch": batch, "source_record": sources[path]})
    current = _read(SELECTION)
    existing = {row.get("frame", {}).get("path") for row in current["records"]}
    additions = [record for record in results if record["frame"]["path"] not in existing]
    current["records"].extend(additions)
    temporary = SELECTION.with_name(SELECTION.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(SELECTION)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"database_frames": len(selected), "new_records": len(additions),
                      "parent_videos": len({f["parent_video"] for f, _ in selected}),
                      "batch": batch, "report": _relative(directory)}, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-parent", type=int, default=3, help="frames to sample per source video")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resolution", choices=('original', '720p'), default='original')
    args = parser.parse_args()
    if args.per_parent < 1:
        parser.error("--per-parent must be positive")
    return run(args.per_parent, args.seed, args.resolution)


if __name__ == "__main__":
    raise SystemExit(main())
