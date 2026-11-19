"""Install the audited narrow-edge dataset alongside the existing portable person training package."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / "lab/training_common")]
from engine.data import _lineage
from lab.training_common.production_guard import generation_guard
from package_person_plastic_finetune import FORMAL, SOURCE_INDEX, annotations, read, replay, write
from evaluate_person_plastic_regression import labeled_inputs

PACKAGE = ROOT / "training_packages/person"
DATA_NAME = "narrow_edge_finetune_20261116"
DATASET = PACKAGE / "dataset" / DATA_NAME
CLASSES = ("community", "noncommunity")
REPLAY = {"close": 70, "other_positive": 45, "mixed": 20, "back_only": 12, "direct_real": 8}
# The persistent false alarm clip is 000710–000732. Four spaced stills represent it.
# The other records contribute different camera positions, edge crops, or positive context.
REAL = {1:"mixed_task_point", 3:"other_mixed_task_point", 7:"motion_cropped_white", 9:"moving_oblique_side",
        11:"moving_wider_white", 13:"distant_and_near_front_context", 14:"right_edge_white", 18:"left_edge_sliver",
        19:"oblique_left_edge", 20:"recurring_bright_side_start", 27:"recurring_bright_side_early_middle",
        34:"recurring_bright_side_late_middle", 41:"recurring_bright_side_end", 42:"white_sides_oblique_motion",
        44:"white_sides_clipped_motion", 47:"route_background", 52:"mixed_front_white_second_view",
        53:"different_area_mixed_close_front"}
SOURCE_SPLITS = ("train", "val")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def pkg(path):
    return path.relative_to(PACKAGE).as_posix()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_real(sources):
    frozen, old_records = labeled_inputs()
    records = {int(Path(row["image"]).stem.rsplit("_",1)[1]): row for row in sources
               if row["split"] == "test" and "/direct_real/" in row["image"]}
    if len(records) != 53 or len(old_records) != 53:
        raise ValueError("The old direct test pool has changed")
    source_parents = {row["parent_video"] for row in records.values()}
    if None in source_parents or len(source_parents) != 1:
        raise ValueError("The source pool must match the audited vehicle route")
    chosen = []
    for index, role in REAL.items():
        row = records[index]
        previous = old_records[index-1]
        for field in ("image","label"):
            source = ROOT / row[field]
            original = frozen / Path(previous[f"run_{field}"]).relative_to("data")
            if digest(source) != digest(original):
                raise ValueError(f"Current test and frozen source differ: {source}")
        chosen.append({**row,"split":"train","bucket":"former_test_real","selection_reason":role,
                       "historical_split":"test","evaluation_scope":"retired_from_blind_holdout",
                       "test_pool_index":index})
    return chosen, source_parents


def pick(scratch):
    sources = read(SOURCE_INDEX)["records"]
    real, parents = choose_real(sources)
    selected = []
    for split in SOURCE_SPLITS:
        original_rows = {row["image"]:row for row in read(FORMAL/split/"manifest.json")["records"]}
        taken = Counter()
        for bucket,row in replay(split,sources,original_rows):
            if split == "train" and taken[bucket] >= REPLAY[bucket]:
                continue
            taken[bucket] += 1
            selected.append({**row,"bucket":f"replay_{bucket}"})
        if split == "train" and dict(taken) != REPLAY:
            raise ValueError(f"Replay quota mismatch: {taken}")
    if any(row.get("parent_video") in parents for row in selected):
        raise ValueError("Retired test parent overlaps the existing replay")
    manifest = read(scratch/"manifest.json")
    raw = manifest["records"]
    if Counter(row["split"] for row in raw) != {"train":128, "val":36}:
        raise ValueError("Render counts differ from the declared design")
    for source_key,digest_key in (("source","source_sha256"),("contour_plan","contour_plan_sha256")):
        if digest(ROOT/manifest[source_key]) != manifest[digest_key]:
            raise ValueError(f"Render {source_key} changed")
    rendered = []
    for row in raw:
        assets = {row["artwork"], *(obj["artwork"] for obj in row["companions"])}
        bucket = "render_" + row["state"]
        rendered.append({**row,"bucket":bucket,"source_group":row["group"],"source_id":row["group"],
                         "label_status":"reviewed","review_state":"complete_positive" if row["boxes"] else "complete_negative",
                         "source_kind":"blender_scene","generation_method":"blender_render",
                         "mother_scene":manifest["source"]+"#"+manifest["scene"],"source_assets":sorted(assets),
                         "near_duplicate_group":row["pair"],"label_authorship":"procedural geometry",
                         "strata_review":"confirmed","strata":{"conditions":[row["state"],"width_paired"]}})
    return {**manifest,"records":rendered}, selected+real+rendered, sorted(parents)


def shared_assets(records):
    usage = defaultdict(set)
    for row in records:
        for relation,values in _lineage(row,True).items():
            scope = row["source_group"] if relation in ("episode_group","segment_id") else None
            for value in ([values] if isinstance(values,str) else values):
                usage[relation,value,scope].add(row["split"])
    shared = []
    for (relation,value,scope),splits in sorted(usage.items(),key=str):
        if splits != set(SOURCE_SPLITS):
            continue
        if relation == "mother_scene" and value.endswith("#Arena Indoor Soft Light"):
            reason = "Same lighting scene; split by camera-placement family and newly rendered artwork."
        elif relation == "source_assets" and any(key in value for key in
                                                ("whiteborder_textures","/digital/person/","/targets/digital/","occluder","/e_bike/")):
            reason = "Shared artwork or occluder in existing replay, with disjoint original scene parent groups."
        else:
            raise ValueError(f"Unexpected train/val overlap: {relation} {value}")
        shared.append({"relation":relation,"source":value,"source_group":scope,"splits":sorted(splits),"reason":reason})
    return shared


def copy_records(rows):
    output = []
    for row in rows:
        split,bucket = row["split"],row["bucket"]
        render = bucket.startswith("render_")
        kind = "blender" if render else ("former_test" if bucket=="former_test_real" else "replay")
        image = DATASET/split/"images"/kind/Path(row["image"]).name
        label = DATASET/split/"labels"/kind/Path(row["label"]).name
        for field,target in (("image",image),("label",label)):
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise ValueError(f"Duplicate destination: {target}")
            shutil.copy2(ROOT/row[field],target)
        evidence = DATASET/"evidence"/("blender_manifest.json" if render else kind+"_source.json")
        output.append({**row,"image":pkg(image),"label":pkg(label),"archived_input_image":row["image"],
                       "archived_input_label":row["label"],"source_review_evidence":pkg(evidence),
                       "review_evidence":pkg(evidence),"workspace_image":row.get("workspace_image",pkg(image)),
                       "workspace_label":row.get("workspace_label",pkg(label)),
                       "image_sha256":digest(image),"label_sha256":digest(label)})
    return output


def contacts(rows,evidence,split):
    def frame(row,w,h):
        overlay = Image.open(PACKAGE/row["image"]).convert("RGB").resize((w,h))
        ink = ImageDraw.Draw(overlay)
        for cls,x,y,bw,bh in annotations(PACKAGE/row["label"]):
            ink.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),
                          outline=("#55fb98","#63d6ff")[int(cls)])
        if row["bucket"]=="render_white_back":
            a,b,c,d = row["target_box_xyxy"]
            ink.rectangle((a*w/640,b*h/480,c*w/640,d*h/480),outline="#ffbf60")
        return overlay

    selected = [r for r in rows if r["split"]==split and r["bucket"].startswith("render_")]
    count = len(selected)//2
    keys = [selected[2*(i*count//12)]["pair"] for i in range(12)]
    samples = [r for key in keys for r in selected if r["pair"]==key]
    w,h=240,180
    sheet = Image.new("RGB",(4*w, 6*(h+24)),"#141a24")
    ink = ImageDraw.Draw(sheet)
    for i,row in enumerate(samples):
        x,y=(i%4)*w,(i//4)*(h+24)
        sheet.paste(frame(row,w,h),(x,y+22))
        ink.text((x+3,y+3),f"{row['pair']} {row['state']} | {row['width_delta_px']:.1f}px",
                 fill="white")
    sheet.save(evidence/f"{split}_blender_pairs.jpg",quality=92)
    if split == "train":
        real = [r for r in rows if r["bucket"]=="former_test_real"]
        sheet = Image.new("RGB",(4*w, ((len(real)+3)//4)*(h+24)),"#141a24")
        ink = ImageDraw.Draw(sheet)
        for i,row in enumerate(real):
            x,y=(i%4)*w,(i//4)*(h+24)
            sheet.paste(frame(row,w,h),(x,y+22))
            ink.text((x+3,y+3),f"{row['test_pool_index']:02d} {row['selection_reason']}",fill="white")
        sheet.save(evidence/"former_test_selected.jpg",quality=92)


def build(scratch):
    manifest, rows, retired_parents = pick(scratch)
    shared = shared_assets(rows)
    evidence = DATASET/"evidence"
    provenance = [SOURCE_INDEX, *(FORMAL/split/"manifest.json" for split in ("train","val","test")),
                  ROOT/manifest["source"], ROOT/manifest["contour_plan"], scratch/"manifest.json"]
    with generation_guard("person.narrow_edge_finetune_20261116",
                          [ROOT/row["image"] for row in rows],[DATASET],
                          {"real_indices":sorted(REAL),"replay_quotas":REPLAY,"source_parent_retirement":retired_parents},
                          code_paths=[Path(__file__), Path(__file__).with_name("render_person_narrow_finetune_pairs.py"),
                                      Path(__file__).with_name("package_person_plastic_finetune.py"),
                                      Path(__file__).with_name("evaluate_person_plastic_regression.py"),
                                      Path(__file__).with_name("render_person_plastic_pairs.py")],
                          procedural_inputs=provenance+[ROOT/row["label"] for row in rows],root=ROOT) as guard:
        if guard.reused:
            print("Reused unchanged completed dataset receipt")
            return
        evidence.mkdir(parents=True,exist_ok=False)
        installed = copy_records(rows)
        for source_kind,filename in (("former_test_real","former_test_source.json"),("replay_","replay_source.json")):
            write(evidence/filename,{"records":[row for row in rows if row["bucket"].startswith(source_kind)],
                                     "original_index":rel(SOURCE_INDEX),"original_labels_preserved":True})
        write(evidence/"blender_manifest.json",{**manifest,"records":[r for r in installed if r["bucket"].startswith("render_")]})
        shutil.copy2(ROOT/manifest["contour_plan"],evidence/"contours.json")
        shutil.copy2(ROOT/manifest["source"],evidence/"lighting_scene_no_base.blend")
        coverage = {}
        for split in SOURCE_SPLITS:
            selected = [r for r in installed if r["split"]==split]
            labels = [annotations(PACKAGE/r["label"]) for r in selected]
            if len({r["image_sha256"] for r in selected}) != len(selected):
                raise ValueError(f"Duplicate image in {split}")
            coverage[split] = {"images":len(selected),"negative_images":sum(not rows for rows in labels),
                               "boxes_by_class":{name:sum(obj[0]==c for group in labels for obj in group)
                                                 for c,name in enumerate(CLASSES)},
                               "by_bucket":dict(Counter(r["bucket"] for r in selected)),
                               "max_width_delta_px":max(r["width_delta_px"] for r in selected if r["bucket"].startswith("render_"))}
            contacts(installed,evidence,split)
        status = {"retired_parent_videos":retired_parents,"former_test_pool_total":53,"selection_size":len(REAL),
                  "holdout_status":"Old 53-frame test is development-only; no new blind test collected."}
        write(evidence/"coverage.json",coverage)
        write(evidence/"holdout_status.json",status)
        write(DATASET/"workspace_sources.json",{"model_key":"person","records":installed,"source_reuse":shared,
                                                "coverage":coverage,"test_lineage":status})
        write(DATASET/"data.yaml",{"names":list(CLASSES),**{split:f"{split}/images" for split in SOURCE_SPLITS}})
        guard.set_result(coverage)
        print(json.dumps(coverage,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve())
