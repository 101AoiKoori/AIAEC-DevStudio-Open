"""Assemble and audit the isolated person thick-plastic fine-tuning dataset."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

FORMAL = ROOT / "datasets/exports/person"
OUTPUT = FORMAL / "white_plastic_finetune"
SOURCE_INDEX = FORMAL / "workspace_sources.json"
TEXTURE_INDEX = ROOT / "datasets/annotated/person/v4_whiteborder_textures_20261113/manifest.json"
BLENDER_SOURCE = ROOT / "datasets/assets/blender/aiaec_dataset.blend"
RENDER_SCRIPT = Path(__file__).with_name("render_person_plastic_pairs.py")
CLASSES = ("community", "noncommunity")
QUOTAS = {"train": {"close": 100, "other_positive": 56, "mixed": 38, "back_only": 26},
          "val": {"close": 22, "other_positive": 14, "mixed": 8, "back_only": 4}}
RENDER_COUNTS = {"train": 120, "val": 36}


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def contours(output):
    art = {split: {name: [] for name in CLASSES} for split in QUOTAS}
    held_out = {"community_3", "community_7", "community_12", "community_15", "noncommunity_F2"}
    for row in read(TEXTURE_INDEX)["records"]:
        source = ROOT / row["image"]
        data = np.fromfile(source, dtype=np.uint8)
        rgba = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        height, width, channels = rgba.shape
        if channels != 4:
            raise ValueError(f"Texture has no alpha channel: {source}")
        found, tree = cv2.findContours((rgba[..., 3] >= 128).astype("uint8"),
                                       cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        parents = tree[0, :, 3]
        largest = max(np.flatnonzero(parents == -1), key=lambda i: cv2.contourArea(found[i]))
        boundaries = [cv2.approxPolyDP(border, .85, True).reshape(-1, 2).tolist()
                      for i, border in enumerate(found)
                      if i == largest or (parents[i] == largest and cv2.contourArea(border) >= 15)]
        split = "val" if source.stem in held_out else "train"
        art[split][row["class"]].append({"image": row["image"], "artwork": row["artwork"],
                                         "size": [width, height], "contours": boundaries,
                                         "sha256": hashlib.sha256(data.tobytes()).hexdigest()})
    write(output, art)


def annotations(label):
    return [tuple(float(value) for value in line.split())
            for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay(split, sources, original):
    groups = defaultdict(lambda: defaultdict(list))
    keep = []
    for row in sources:
        if row["split"] != split:
            continue
        image = row["image"]
        manifest_row = original[image]
        objects = manifest_row.get("objects") or []
        listed = annotations(ROOT / row["label"])
        row = {**row, "source_assets": sorted(set(row.get("source_assets", [])) |
                                               {obj["source"] for obj in objects if obj.get("source")})}
        if "/direct_real/" in image:
            keep.append(("direct_real", row))
            continue
        if row.get("image_processing", {}).get("background_origin") != "direct_camera":
            continue
        white_back = bool(manifest_row.get("background_white_backs")) or any(
            obj.get("kind") == "person" and obj.get("view") == "back" for obj in objects)
        if white_back:
            bucket = "mixed" if listed else "back_only"
        elif not listed:
            continue
        else:
            if any(h >= .28 for _, _, _, _, h in listed):
                groups["close"][row["source_group"]].append(row)
            bucket = "other_positive"
        groups[bucket][row["source_group"]].append(row)
    rng = random.Random(111678 if split == "train" else 115678)
    chosen_paths = set()
    for bucket, wanted in QUOTAS[split].items():
        buckets = list(groups[bucket].values())
        for rows in buckets:
            rows[:] = [row for row in rows if row["image"] not in chosen_paths]
            rng.shuffle(rows)
            # Favor both categories in the replay whose noncommunity class is scarcer.
            if bucket in {"close", "other_positive"}:
                rows.sort(key=lambda row: not any(cls == 1 for cls, *_ in annotations(ROOT / row["label"])))
        chosen = []
        used_backgrounds = Counter()
        while len(chosen) < wanted:
            available = [rows for rows in buckets if rows]
            if not available:
                raise ValueError(f"{split} replay cannot fill {bucket} ({len(chosen)}/{wanted})")
            for candidates in available:
                rank = min(range(len(candidates)), key=lambda i: used_backgrounds[
                    original[candidates[i]["image"]].get("working_background") or candidates[i]["source_group"]])
                row = candidates.pop(rank)
                chosen_paths.add(row["image"])
                used_backgrounds[original[row["image"]].get("working_background") or row["source_group"]] += 1
                chosen.append((bucket, row))
                if len(chosen) == wanted:
                    break
        keep.extend(chosen)
    return keep


def transfer(row, split, bucket, origin, destination):
    source_image, source_label = ROOT / row["image"], ROOT / row["label"]
    image = destination / split / "images" / origin / source_image.name
    label = destination / split / "labels" / origin / source_label.name
    for source, target in ((source_image, image), (source_label, label)):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"Duplicate package filename: {target}")
        shutil.copy2(source, target)
    return {**row, "image": rel(image), "label": rel(label), "bucket": bucket,
            "workspace_image": row.get("workspace_image", row["image"]) if origin != "blender" else rel(image),
            "workspace_label": row.get("workspace_label", row["label"]) if origin != "blender" else rel(label),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()}


def draw_boxes(image, objects, resize):
    draw = ImageDraw.Draw(image)
    for obj in objects:
        coords = [v * resize for v in obj["box_xyxy"]]
        color = ("#55ed8c" if obj["class"] == "community" else "#76ceff") if obj["positive"] else "#ffc770"
        draw.rectangle(coords, outline=color, width=2)


def render_audit(batch, rows, evidence):
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    panels = []
    summaries = []
    taken = Counter()
    for pair, (soft, highlight) in sorted(by_pair.items()):
        if (soft["light"], highlight["light"]) != ("soft", "highlight") or soft["objects"] != highlight["objects"]:
            raise ValueError(f"Invalid light pairing: {pair}")
        if (ROOT / soft["label"]).read_bytes() != (ROOT / highlight["label"]).read_bytes():
            raise ValueError(f"Light variants have different labels: {pair}")
        negative = any(not obj["positive"] for obj in soft["objects"])
        frames = [Image.open(ROOT / row["image"]).convert("RGB") for row in (soft, highlight)]
        delta = np.abs(np.asarray(frames[0], dtype=np.int16) - np.asarray(frames[1], dtype=np.int16)).max(axis=2)
        if not delta.any():
            raise ValueError(f"Pair has identical pixels: {pair}")
        box = next((obj["box_xyxy"] for obj in soft["objects"] if not obj["positive"]), soft["objects"][0]["box_xyxy"])
        x1, y1, x2, y2 = (int(v) for v in box)
        roi = delta[y1:y2+1, x1:x2+1]
        summaries.append({"pair": pair, "layout": soft["layout"], "targets": len(soft["objects"]),
                          "negative_present": negative, "pixels_in_target_changed_gt_8": int((roi > 8).sum()),
                          "pixels_full_changed_gt_8": int((delta > 8).sum()), "target_delta_p99": float(np.quantile(roi, .99))})
        key = soft["layout"]
        if taken[key] < 2:
            taken[key] += 1
            panels.append((pair, key, frames, soft["objects"]))
    tile_w, tile_h = 320, 240
    canvas = Image.new("RGB", (4 * tile_w, ((len(panels) + 1) // 2) * (tile_h + 27)), "#151b25")
    for i, (pair, layout, frames, objects) in enumerate(panels):
        x, y = (i % 2) * 2 * tile_w, (i // 2) * (tile_h + 27)
        for j, (frame, light) in enumerate(zip(frames, ("soft", "highlight"))):
            view = frame.resize((tile_w, tile_h))
            draw_boxes(view, objects, .5)
            canvas.paste(view, (x + j * tile_w, y + 26))
            ImageDraw.Draw(canvas).text((x + j * tile_w + 5, y + 5), f"{pair} | {layout} | {light}", fill="white")
    canvas.save(evidence / f"{batch}_contact.jpg", quality=91)
    write(evidence / f"{batch}_light_deltas.json", summaries)


def reuse_records(records):
    counts = defaultdict(set)
    for row in records:
        for field in ("source_group", "parent_video", "mother_scene", "source_assets", "near_duplicate_group", "fragment"):
            values = row.get(field)
            relation = {"fragment": "segment_id", "near_duplicate_group": "lineage_group"}.get(field, field)
            scope = row["source_group"] if field == "fragment" else None
            for value in (values if isinstance(values, list) else [values]):
                if value:
                    counts[relation, value, scope].add(row["split"])
    allowed = []
    for (relation, source, group), splits in counts.items():
        if len(splits) != 2:
            continue
        if relation == "source_assets" and ("whiteborder_textures" in source or "/digital/person/" in source or "/数字人物/" in source or "/targets/digital/" in source):
            reason = "Shared standee artwork in existing replay, separate scenes/backgrounds."
        elif relation == "source_assets" and ("occluder" in source or "/e_bike/" in source):
            reason = "Shared vehicle occluder asset in existing replay, separate backgrounds."
        else:
            raise ValueError(f"Unexplained train/val {relation} reuse: {source}")
        allowed.append({"relation": relation, "source": source, "source_group": group,
                        "splits": sorted(splits), "reason": reason})
    return allowed


def package(args):
    renders = {split: read(args.render_root / f"render_{split}" / "render_manifest.json") for split in QUOTAS}
    sources = read(SOURCE_INDEX)["records"]
    selected = {}
    formal_manifests = []
    for split in QUOTAS:
        manifest = FORMAL / split / "manifest.json"
        formal_manifests.append(manifest)
        selected[split] = replay(split, sources, {r["image"]: r for r in read(manifest)["records"]})
        if len(renders[split]["records"]) != RENDER_COUNTS[split]:
            raise ValueError(f"Incorrect {split} Blender quota")
    render_plan = args.render_root / "contours.json"
    for split, manifest in renders.items():
        if manifest["source_blend_sha256"] != hashlib.sha256(BLENDER_SOURCE.read_bytes()).hexdigest():
            raise ValueError(f"Blender source changed since rendering {split}")
        if manifest["contour_plan_sha256"] != hashlib.sha256(render_plan.read_bytes()).hexdigest():
            raise ValueError(f"Contour plan changed since rendering {split}")
    render_paths = [args.render_root / f"render_{split}" for split in QUOTAS]
    procedural = [SOURCE_INDEX, TEXTURE_INDEX, BLENDER_SOURCE, render_plan, *formal_manifests,
                  *(folder / name for folder in render_paths
                    for name in ("render_manifest.json", "paired_plastic_standees.blend"))]
    media = []
    for split in QUOTAS:
        rows = [row for _, row in selected[split]] + renders[split]["records"]
        media += [ROOT / row["image"] for row in rows]
        procedural += [ROOT / row["label"] for row in rows]
    with generation_guard("person.white_plastic_finetune", media, [OUTPUT],
                          {"render_counts": RENDER_COUNTS, "replay_quotas": QUOTAS},
                          code_paths=[Path(__file__), RENDER_SCRIPT,
                                      ROOT / "datasets/scripts/common/blender_standees.py",
                                      ROOT / "datasets/scripts/common/blender_workspace.py"],
                          procedural_inputs=procedural, root=ROOT) as guard:
        if guard.reused:
            print("Reused complete package receipt.")
            return
        result = assemble(renders, selected, render_paths, render_plan)
        guard.set_result(result)
        print(json.dumps(result, indent=2))


def assemble(renders, selected, render_paths, render_plan):
    evidence = OUTPUT / "evidence"
    records, counts = [], {}
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(render_plan, evidence / "contours.json")
    for split, folder in zip(QUOTAS, render_paths):
        manifest = renders[split]
        editable = evidence / f"paired_plastic_standees_{split}.blend"
        render_manifest = evidence / f"render_manifest_{split}.json"
        shutil.copy2(folder / "paired_plastic_standees.blend", editable)
        moved = []
        scene_ref = manifest["source_blend"] + "#" + manifest["source_scene"]
        for row in manifest["records"]:
            positive = any(obj["positive"] for obj in row["objects"])
            source_group = f"person_plastic_{manifest['source_scene']}"
            transferred = transfer({
                **row, "source_group": source_group, "source_id": source_group,
                "mother_scene": scene_ref, "near_duplicate_group": row["pair"],
                "generation_method": "blender_render", "source_kind": "blender_scene",
                "source_assets": sorted({obj["artwork"] for obj in row["objects"]}),
                "label_status": "reviewed",
                "review_state": "complete_positive" if positive else "complete_negative",
                "source_review_evidence": rel(render_manifest), "review_evidence": rel(render_manifest),
                "label_authorship": "procedural geometric annotation",
                "strata_review": "confirmed", "strata": {"conditions": [row["layout"], row["light"]]},
            }, split, row["layout"], "blender", OUTPUT)
            moved.append(transferred)
        write(render_manifest, {**manifest, "editable_scene": rel(editable), "records": moved})
        render_audit(split, moved, evidence)
        replay_rows = [transfer(row, split, bucket, "replay", OUTPUT) for bucket, row in selected[split]]
        packaged = moved + replay_rows
        objects = [annotations(ROOT / row["label"]) for row in packaged]
        counts[split] = {"images": len(packaged), "blender_images": len(moved),
                         "blender_paired_layouts": dict(Counter(row["layout"] for row in moved)),
                         "replay_images": len(replay_rows),
                         "replay_buckets": dict(Counter(row["bucket"] for row in replay_rows)),
                         "negative_images": sum(not obj for obj in objects),
                         "near_positive_images": sum(any(h >= .28 for _, _, _, _, h in obj) for obj in objects),
                         "boxes": {name: sum(cls == index for obj in objects for cls, *_ in obj)
                                   for index, name in enumerate(CLASSES)}}
        records.extend(packaged)
    shared = reuse_records(records)
    write(OUTPUT / "workspace_sources.json", {
        "model_key": "person", "records": records, "source_reuse": shared, "coverage": counts,
        "split_policy": "Original formal replay split; Arena vs Arena Detailed and disjoint artwork in new Blender renders. "
                        "Both Blender scenes approximate the same venue. No test images enter this dataset.",
    })
    write(OUTPUT / "data.yaml", {"names": list(CLASSES), **{split: f"{split}/images" for split in QUOTAS}})
    write(evidence / "coverage.json", counts)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--contours", type=Path)
    modes.add_argument("--render-root", type=Path)
    args = parser.parse_args()
    if args.contours is not None:
        contours(args.contours.resolve())
    else:
        args.render_root = args.render_root.resolve()
        package(args)
