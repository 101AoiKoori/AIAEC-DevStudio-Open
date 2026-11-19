"""Score the frozen 53-frame person regression at the original operating point."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "lab/training_common"))
from engine.reporting import box_iou, diagnostics, match_detections
from engine.runner import assert_model_contract
from engine.runtime import prepare_ultralytics

FORMAL = ROOT / "training_packages/person/runs/person/v4_formal_20261116"
BASELINE = FORMAL / "evaluation/test_direct_real_last_inference_20261116"
DATASET = ROOT / "datasets/exports/person/white_plastic_finetune"
COLORS = {"TP": "#44f49c", "FP": "#ff6459", "FN": "#ffda6c"}
CONFIG = {"version": "v4", "task": "detect", "imgsz": 640, "diagnostic_conf": .25,
          "match_iou": .50, "nms_iou": .70, "device": "0", "max_det": 300}
WEIGHT_HASH = "c51903d4db5b78ceb190518c3925a3785d100ce85818269bbcba0a6808b0f31b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_write(path, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labeled_inputs():
    spec = read(BASELINE / "inputs.json")
    original = (ROOT / spec["manifest"]).parent
    frozen = {row["run_image"]: row for row in read(original / "samples.json")}
    assert len(spec["records"]) == 53
    for row in spec["records"]:
        frozen_row = frozen[row["run_image"]]
        if any(row[key] != frozen_row[key] for key in ("image", "label", "run_label", "instances")):
            raise ValueError("Frozen labels or image identity changed.")
    return original, spec["records"]


def stage(destination):
    original, records = labeled_inputs()
    holdout = destination / "data"
    trainval = read(DATASET / "workspace_sources.json")["records"]
    disallowed = {row["image_sha256"] for row in trainval}
    hashes = []
    for row in records:
        image = original / Path(row["run_image"]).relative_to("data")
        label = original / Path(row["run_label"]).relative_to("data")
        fingerprints = {"image_sha256": digest(image), "label_sha256": digest(label)}
        if fingerprints["image_sha256"] in disallowed:
            raise ValueError(f"Frozen test image overlaps fine-tuning data: {image}")
        for path in (image, label):
            target = holdout / path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        hashes.append({**row, **fingerprints})
    write(holdout / "samples.json", hashes)
    write(holdout / "data.yaml", {"names": ["community", "noncommunity"], "test": "images/test"})
    return holdout


def evaluate(run):
    base_model = FORMAL / "training/weights/last_inference.pt"
    if digest(base_model) != WEIGHT_HASH:
        raise ValueError("Old inference weight differs from the frozen hash")
    artifacts = read(run / "metadata/artifacts.json")
    if artifacts["selection"]["split"] != "val":
        raise ValueError("Expected a validation-selected checkpoint")
    if read(run / "metadata/post_training.json")["status"] != "complete":
        raise ValueError("Training finalization is incomplete")
    best_model = run / artifacts["best"]
    destination = run / "evaluation/test_direct_real_white_plastic"
    destination.mkdir(parents=True, exist_ok=False)
    data = stage(destination)
    os.environ["YOLO_CONFIG_DIR"] = str(run / "cache/ultralytics")
    YOLO = prepare_ultralytics(run)
    write(destination / "inputs.json", {
        "config": CONFIG, "select_by": artifacts["selection"],
        "models": {mode: {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}
                   for mode, path in (("before", base_model), ("after", best_model))},
        "held_out": "Fixed 53 old test real images. Excluded from train, validation and checkpoint selection.",
    })
    loaded = {}
    for mode, checkpoint in (("before", base_model), ("after", best_model)):
        model = YOLO(str(checkpoint), task="detect")
        assert_model_contract(model, data / "data.yaml", CONFIG)
        folder = destination / mode
        diagnostics(model, data / "data.yaml", "test", CONFIG, folder)
        loaded[mode] = read_lines(folder / "predictions/predictions.jsonl")
    before = loaded["before"]
    original = read_lines(BASELINE / "predictions/predictions.jsonl")
    for a, b in zip(before, original, strict=True):
        keys = ("image", "ground_truth", "missed_gt_indices", "false_prediction_indices")
        if any(a[k] != b[k] for k in keys):
            raise ValueError(f"Baseline replay disagrees with original predictions: {a['image']}")
    make_comparison(destination, data, loaded)


def metrics(records):
    counts = Counter()
    by_class = defaultdict(Counter)
    for row in records:
        counts.update({"tp": len(row["matches"]), "fp": len(row["false_prediction_indices"]),
                       "fn": len(row["missed_gt_indices"])})
        agnostic_truth = [{**obj, "class_id": 0} for obj in row["ground_truth"]]
        agnostic_preds = [{**obj, "class_id": 0} for obj in row["predictions"]]
        agnostic, _, _ = match_detections(agnostic_truth, agnostic_preds, CONFIG["match_iou"])
        matched = {i for i, _, _ in row["matches"]}
        detected = {i for i, _, _ in agnostic} | matched
        for i, obj in enumerate(row["ground_truth"]):
            tally = by_class[obj["class_id"]]
            tally["gt"] += 1
            tally["class_matched"] += i in matched
            if (obj["xyxy"][3] - obj["xyxy"][1]) / row["shape_hw"][0] >= .28:
                tally["near_gt"] += 1
                tally["near_detected"] += i in detected
                tally["near_class_matched"] += i in matched
    return {**counts, "images": len(records), "recall": counts["tp"] / (counts["tp"] + counts["fn"]),
            "precision": counts["tp"] / (counts["tp"] + counts["fp"]),
            "by_class": {("community", "noncommunity")[cls]: dict(values) for cls, values in by_class.items()}}


def load_original_reviews():
    with (BASELINE / "review/false_positive_review.csv").open(encoding="utf-8-sig", newline="") as file:
        return {(int(row["frame_id"]), int(row["prediction_index"])): row["review"] for row in csv.DictReader(file)}


def fp_rows(loaded):
    prior = load_original_reviews()
    tables = {}
    for mode, records in loaded.items():
        rows = []
        for index, item in enumerate(records):
            for pred_id in item["false_prediction_indices"]:
                pred = item["predictions"][pred_id]
                old_index, old_iou = pred_id, 1.
                if mode == "after":
                    base = loaded["before"][index]
                    old_index, old_iou = max(
                        ((bp, box_iou(pred["xyxy"], base["predictions"][bp]["xyxy"]))
                         for bp in base["false_prediction_indices"]), key=lambda pair: pair[1], default=(-1, 0))
                rows.append({"id": f"{index:02d}-{pred_id:02d}", "frame": index, "prediction": pred_id,
                             "image": item["image"], "source": item["source_image"], "class_id": pred["class_id"],
                             "confidence": round(pred["confidence"], 6), "xyxy": pred["xyxy"],
                             "old_review": prior.get((index, old_index), "") if old_iou >= .5 else "",
                             "old_iou": round(old_iou, 4), "visual_review": ""})
        tables[mode] = rows
    return tables


def draw_result(frame, row):
    draw = ImageDraw.Draw(frame)
    for _, pred_idx, _ in row["matches"]:
        pred = row["predictions"][pred_idx]
        draw.rectangle(pred["xyxy"], outline=COLORS["TP"], width=2)
        x, y = pred["xyxy"][:2]
        draw.text((x, max(0, y - 11)), f'{("C","NC")[pred["class_id"]]} {pred["confidence"]:.2f}', fill=COLORS["TP"], stroke_width=1, stroke_fill="#151b25")
    for pred_idx in row["false_prediction_indices"]:
        pred = row["predictions"][pred_idx]
        draw.rectangle(pred["xyxy"], outline=COLORS["FP"], width=2)
        x, y = pred["xyxy"][:2]
        draw.text((x, max(0, y - 11)), f'{("C","NC")[pred["class_id"]]} {pred["confidence"]:.2f}', fill=COLORS["FP"], stroke_width=1, stroke_fill="#151b25")
    for gi in row["missed_gt_indices"]:
        draw.rectangle(row["ground_truth"][gi]["xyxy"], outline=COLORS["FN"], width=2)


def fp_contacts(folder, data, table):
    panels = []
    for row in table:
        image = Image.open(data.parent / row["image"]).convert("RGB")
        a, b, c, d = row["xyxy"]
        crop = [max(0, int(a-13)), max(0, int(b-9)), min(image.width, int(c+14)), min(image.height, int(d+12))]
        patch = image.crop(crop)
        ImageDraw.Draw(patch).rectangle((a-crop[0], b-crop[1], c-crop[0], d-crop[1]), outline="#ff5856", width=1)
        patch.thumbnail((103, 177), Image.Resampling.LANCZOS)
        panels.append((row["id"], patch))
    cols, rows = 10, 5
    for start in range(0, len(panels), cols * rows):
        canvas = Image.new("RGB", (cols * 112, rows * 201), "#181d27")
        draw = ImageDraw.Draw(canvas)
        for i, (identifier, patch) in enumerate(panels[start:start + cols * rows]):
            x, y = (i % cols) * 112, (i // cols) * 201
            draw.text((x + 5, y + 3), identifier, fill="white")
            canvas.paste(patch, (x+(112-patch.width)//2, y+18))
        canvas.save(folder / f"fp_crops_{start//(cols*rows)+1:02d}.jpg", quality=94)


def make_comparison(destination, data, loaded):
    summary = {mode: metrics(rows) for mode, rows in loaded.items()}
    for field in ("tp", "fp", "fn"):
        if summary["before"][field] != {"tp": 83, "fp": 127, "fn": 6}[field]:
            raise ValueError("Baseline totals disagree with handoff")
    tables = fp_rows(loaded)
    for mode, table in tables.items():
        if table:
            csv_write(destination / mode / "fp_review.csv", table)
            fp_contacts(destination / mode, data, table)
    view = destination / "same_frame"
    view.mkdir()
    for index, (before, after) in enumerate(zip(loaded["before"], loaded["after"], strict=True)):
        if before["image"] != after["image"] or before["ground_truth"] != after["ground_truth"]:
            raise ValueError("Comparison inputs disagree")
        orig = Image.open(data.parent / before["image"]).convert("RGB")
        width, height = orig.size
        fig = Image.new("RGB", (width * 2, height + 48), "#151b25")
        draw = ImageDraw.Draw(fig)
        for side, (phase, which) in enumerate((("Before | official EMA", before), ("After | val-selected", after))):
            pred = orig.copy()
            draw_result(pred, which)
            x = side * width
            fig.paste(pred, (x, 48))
            draw.text((x + 8, 6), phase + f" | frame {index:02d} | " + f"TP {len(which['matches'])} FP {len(which['false_prediction_indices'])} FN {len(which['missed_gt_indices'])}", fill="white")
            draw.text((x + 8, 27), "Green matched | Red false detection | Yellow missed truth", fill="#d5dfec")
        fig.save(view / f"{index:02d}_{Path(before['source_image']).stem}.jpg", quality=93)
    write(destination / "comparison.json", summary)


def probe_renders(run):
    """Use fixed settings with the old EMA to check dark/bright pairs after training is frozen."""
    weight = FORMAL / "training/weights/last_inference.pt"
    if digest(weight) != WEIGHT_HASH:
        raise ValueError("Old inference weight differs from the frozen hash")
    os.environ["YOLO_CONFIG_DIR"] = str(run / "cache/ultralytics")
    model = prepare_ultralytics(run)(str(weight), task="detect")
    options = {"imgsz": CONFIG["imgsz"], "conf": .001, "iou": CONFIG["nms_iou"],
               "device": CONFIG["device"], "max_det": CONFIG["max_det"], "verbose": False}
    images = [row for split in ("train", "val")
              for row in read(DATASET / f"evidence/render_manifest_{split}.json")["records"]]
    counts = Counter()
    pairs = defaultdict(dict)
    records = []
    for row in images:
        result = model.predict(str(ROOT / row["image"]), **options)[0]
        boxes = result.boxes
        predictions = [{"class_id": int(cls.item()), "confidence": float(conf.item()), "xyxy": box.tolist()}
                       for cls, conf, box in zip(boxes.cls.cpu(), boxes.conf.cpu(), boxes.xyxy.cpu())]
        accepted = [pred for pred in predictions if pred["confidence"] >= CONFIG["diagnostic_conf"]]
        truths = [{"class_id": ("community", "noncommunity").index(obj["class"]), "xyxy": obj["box_xyxy"]}
                  for obj in row["objects"] if obj["positive"]]
        _, _, false = match_detections(truths, accepted, CONFIG["match_iou"])
        board_boxes = [obj["box_xyxy"] for obj in row["objects"] if not obj["positive"]]
        fp_indices = [i for i in false if any(box_iou(accepted[i]["xyxy"], box) >= CONFIG["match_iou"]
                                              for box in board_boxes)]
        low_on_board = max((pred["confidence"] for pred in predictions
                            if any(box_iou(pred["xyxy"], box) >= CONFIG["match_iou"] for box in board_boxes)),
                           default=None)
        if board_boxes:
            pairs[row["pair"]][row["light"]] = len(fp_indices)
            counts[f'{row["light"]}_white_board_fp'] += len(fp_indices)
            counts[f'{row["light"]}_frames_with_white_board_fp'] += bool(fp_indices)
            counts[f'{row["light"]}_frames_with_white_board_candidate_ge_001'] += low_on_board is not None
            counts[f'{row["light"]}_any_fp_in_board_frames'] += len(false)
        records.append({"image": row["image"], "pair": row["pair"], "layout": row["layout"],
                        "light": row["light"], "unlabeled_side_back_boxes": board_boxes,
                        "white_board_fp_count": len(fp_indices), "total_fp_count": len(false),
                        "max_white_board_candidate_confidence_ge_001": low_on_board,
                        "predictions_ge_001": predictions})
    transitions = Counter()
    for pair in pairs.values():
        transitions[f'soft_{bool(pair["soft"])}_highlight_{bool(pair["highlight"])}'] += 1
    output = {"weight": weight.relative_to(ROOT).as_posix(), "sha256": WEIGHT_HASH,
              "settings": CONFIG, "recorded_prediction_floor": .001,
              "render_images": len(images), "pairs_containing_side_or_back": len(pairs),
              "counts": dict(counts), "transitions": dict(transitions), "records": records}
    write(run / "evaluation/blender_white_plastic_old_ema.json", output)
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--render-probe", action="store_true", help="Check same-geometry material pairs with the old EMA")
    args = parser.parse_args()
    if args.render_probe:
        probe_renders(args.run.resolve())
    else:
        evaluate(args.run.resolve())
