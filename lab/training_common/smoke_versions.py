"""Prepare, run and archive bounded V1/V2 experiments using frozen V3 code.

The model definitions and source splits come from the task's V3 configuration.
All generated experiment records and artifacts are written below ``runs``.
This tool does not infer that a short run has poor or good model performance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "person": "person", "e_bike": "e_bike", "fire_sticker": "fire",
    "building_name_paper": "fire", "sign": "sign",
    "waste_bin_state": "waste_bin", "waste_sticker": "waste_bin",
    "meter": "meter", "plate_detection": "plate",
}
PROFILES = {"v1": {"epochs": 1, "lr0": 0.01},
            "v2": {"epochs": 2, "lr0": 0.005}}

# This launcher imports the code snapshot in this bundle, including after archive.
# It derives only the runtime output location from its current location.
LAUNCHER = '''"""Run this recorded smoke experiment with its frozen engine."""
from pathlib import Path
import sys

bundle = Path(__file__).resolve().parent.parent
root = next((p for p in bundle.parents if (p / "config/perception_tasks.json").is_file()), None)
if root is None:
    raise SystemExit("This experiment requires its recorded workspace datasets and model weights.")
sys.dont_write_bytecode = True
sys.path.insert(0, str(bundle / "scripts/common"))
import yaml
from engine.cli import main

if __name__ == "__main__":
    config = yaml.safe_load((bundle / "config/train.yaml").read_text(encoding="utf-8"))
    config["output_root"] = (bundle / "runs").relative_to(root).as_posix()
    effective = bundle / "runs/effective_config.yaml"
    effective.parent.mkdir(parents=True, exist_ok=True)
    effective.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    raise SystemExit(main(default_root=root, default_config=effective.relative_to(root).as_posix()))
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inside(root, value):
    path = Path(value)
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    result.relative_to(root.resolve())
    return result


def relative(root, path):
    return path.resolve().relative_to(root.resolve()).as_posix()


def validate_suite(root, value):
    result = inside(root, value)
    result.relative_to((root / "runs/smoke_versions").resolve())
    if result == (root / "runs/smoke_versions").resolve():
        raise ValueError("A named suite directory below runs/smoke_versions is required")
    return result


def git_context(root):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"head": git("rev-parse", "HEAD"),
            "source_worktree_status": git("status", "--short", "--", "lab/training_common", "lab/features")}


def planned_config(root, model_key, version):
    task = MODELS[model_key]
    source = root / f"lab/features/{task}/training/config/{model_key}_v3.yaml"
    if not source.is_file():
        raise FileNotFoundError(source)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 3:
        raise ValueError(f"Expected a V3 training configuration: {source}")
    original = dict(config)
    config.update(version=version, imgsz=320, train_limit=50, eval_limit=None,
                  batch=8, workers=0, seed=42, optimizer="SGD", amp=False,
                  warmup_epochs=0.0, close_mosaic=0,
                  **PROFILES[version])
    return source, original, config


def frozen_sources(root):
    common = root / "lab/training_common"
    expected = [common / "engine/cli.py", common / "engine/runner.py", common / "train.py"]
    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(f"V3 engine not ready: {path}")
    sources = sorted((common / "engine").rglob("*.py"))
    sources.extend(p for p in (common / "train.py", common / "tensorboard_log.py") if p.is_file())
    return common, sources


def prepare(root, suite, models, versions):
    if suite.exists():
        raise FileExistsError(f"Refusing to replace suite: {suite}")
    common, sources = frozen_sources(root)
    plans = [(model, version, planned_config(root, model, version))
             for model in models for version in versions]
    context = git_context(root)
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    suite.mkdir(parents=True)
    manifest = {"schema_version": 1, "created_utc": now(), "status": "prepared",
                "purpose": "bounded 50-image 320px smoke experiments",
                "comparison": "V2 changes both training duration and initial learning rate; scores are measured, not predetermined.",
                "evaluation": "Full configured val/test splits; source reuse remains as documented by their original manifests.",
                "code": context, "experiments": []}
    write_json(suite / "suite.json", manifest)
    for model, version, (source, original, config) in plans:
        bundle = suite / f"{model}_{version}"
        (bundle / "config").mkdir(parents=True)
        (bundle / "runs").mkdir()
        (bundle / "scripts/common").mkdir(parents=True)
        mapping = []
        for item in sources:
            inside(root, item)
            destination = bundle / "scripts/common" / item.relative_to(common)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            if destination.read_bytes() != item.read_bytes():
                raise RuntimeError(f"Training source changed while freezing: {item}")
            mapping.append({"source": relative(root, item), "snapshot": destination.relative_to(bundle).as_posix()})
        (bundle / "scripts/run.py").write_text(LAUNCHER, encoding="utf-8")
        config["output_root"] = relative(root, bundle / "runs")
        (bundle / "config/train.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        changes = {key: {"v3": original.get(key), "experiment": value}
                   for key, value in config.items() if original.get(key) != value}
        provenance = {"created_utc": now(), "model_key": model, "task_id": MODELS[model],
                      "version": version, "source_config": relative(root, source), "code": context,
                      "source_files": mapping, "configuration_changes": changes,
                      "launcher": "Generated by lab/training_common/smoke_versions.py",
                      "original_bundle": relative(root, bundle), "status": "prepared"}
        write_json(bundle / "runs/provenance.json", provenance)
        write_json(bundle / "runs/preparation_environment.json",
                   {"python": sys.version, "packages": dict(sorted(packages.items()))})
        manifest["experiments"].append({"model_key": model, "task_id": MODELS[model], "version": version,
                                         "bundle": relative(root, bundle), "status": "prepared"})
        write_json(suite / "suite.json", manifest)
    return manifest


def invoke(root, bundle, command, arguments, timeout):
    output = bundle / "runs/launcher"
    output.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": now(), "command": command, "arguments": arguments,
              "python_version": sys.version, "status": "running", "timeout_seconds": timeout}
    record_path = output / f"{command}.json"
    if record_path.exists():
        raise FileExistsError(f"Command already attempted; preserve its evidence: {record_path}")
    bootstrap = os.environ.get("AEC_PORTABLE_BOOTSTRAP")
    if bootstrap:
        bootstrap_path = inside(root, bootstrap)
        if not bootstrap_path.is_file():
            raise FileNotFoundError(bootstrap_path)
        argv = [sys.executable, "-I", "-S", "-B", str(bootstrap_path), "--offline",
                "--script", str(bundle / "scripts/run.py"), "--", command, *arguments]
    else:
        argv = [sys.executable, "-B", str(bundle / "scripts/run.py"), command, *arguments]
    write_json(record_path, record)
    begin = time.monotonic()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    try:
        with (output / f"{command}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=environment)
            record["process_id"] = process.pid
            write_json(record_path, record)
            try:
                returncode = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name == "nt":
                    # Windows venv redirectors can own a second python process.
                    # Kill only the tree belonging to this live Popen instance.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False, timeout=15)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
                raise
        record.update(status="completed" if returncode == 0 else "failed", returncode=returncode)
    except subprocess.TimeoutExpired:
        record.update(status="timeout", returncode=None)
    except KeyboardInterrupt:
        record.update(status="interrupted", returncode=None)
        raise
    except Exception as error:
        record.update(status="failed", returncode=None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(finished_utc=now(), elapsed_seconds=round(time.monotonic() - begin, 3))
        write_json(record_path, record)
    return record


def _run_suite(root, suite, limit, timeout, evaluation_timeout):
    manifest_path = suite / "suite.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = 0
    for item in manifest["experiments"]:
        if item["status"] != "prepared":
            continue
        if limit is not None and completed >= limit:
            break
        bundle = inside(root, item["bundle"])
        item.update(status="running", started_utc=now())
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        print(f"START {item['model_key']} {item['version']}", flush=True)
        try:
            result = invoke(root, bundle, "train", ["--run-name", "smoke"], timeout)
            item["train"] = result
            if result["status"] != "completed":
                item["status"] = result["status"]
                manifest["status"] = "stopped_after_failure"
                break
            run_dir = bundle / "runs/smoke"
            weights = run_dir / "training/weights/best.pt"
            if not weights.is_file():
                raise FileNotFoundError(f"Training did not produce best.pt: {weights}")
            result = invoke(root, bundle, "evaluate", ["--weights", relative(root, weights), "--split", "test",
                            "--run-dir", relative(root, run_dir)], evaluation_timeout)
            item["test"] = result
            item["status"] = "completed" if result["status"] == "completed" else result["status"]
            item["run_dir"] = relative(root, run_dir)
            print(f"END {item['model_key']} {item['version']} {item['status']}", flush=True)
            if item["status"] != "completed":
                manifest["status"] = "stopped_after_failure"
                break
            completed += 1
        except KeyboardInterrupt:
            item["status"] = "interrupted"
            manifest["status"] = "interrupted"
            raise
        except Exception as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
            manifest["status"] = "stopped_after_failure"
            raise
        finally:
            item["finished_utc"] = now()
            write_json(manifest_path, manifest)
    else:
        manifest["status"] = "completed" if all(x["status"] in ("completed", "archived") for x in manifest["experiments"]) else "incomplete"
    if manifest["status"] == "running":
        manifest["status"] = "partially_completed"
    write_json(manifest_path, manifest)
    return manifest


def run_suite(root, suite, limit, timeout, evaluation_timeout):
    lock = suite / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f"Suite already has a run lock; inspect its recorded process before retrying: {lock}")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_utc": now()}, stream)
        return _run_suite(root, suite, limit, timeout, evaluation_timeout)
    finally:
        lock.unlink()


def audit_suite(root, suite):
    """Check completed outputs while recording native plots not produced."""
    manifest = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
    errors, rows, selections = [], [], {}
    reference_scripts = None
    for item in manifest["experiments"]:
        if item["status"] not in ("completed", "archived"):
            continue
        key = item["model_key"] + "/" + item["version"]
        bundle = inside(root, item["bundle"])
        run_dir = bundle / "runs/smoke"
        try:
            config = yaml.safe_load((bundle / "config/train.yaml").read_text(encoding="utf-8"))
            samples = json.loads((run_dir / "data/samples.json").read_text(encoding="utf-8"))
            header = yaml.safe_load((run_dir / "data/data.yaml").read_text(encoding="utf-8"))
            selected = {split: [r for r in samples if r["split"] == split] for split in ("train", "val", "test")}
            if len(selected["train"]) != 50 or config["imgsz"] != 320 or config["seed"] != 42:
                raise ValueError("Expected 50 training images, 320px and seed 42")
            classes = {c for record in selected["train"] for c in record["classes"]}
            if classes != set(range(len(header["names"]))):
                raise ValueError("The 50-image selection does not cover all declared classes")
            selections.setdefault(item["model_key"], {})[item["version"]] = [r["image"] for r in selected["train"]]
            code = {p.relative_to(bundle / "scripts").as_posix(): p.read_bytes()
                    for p in (bundle / "scripts").rglob("*.py")}
            if reference_scripts is None:
                reference_scripts = code
            if code != reference_scripts:
                raise ValueError("Snapshots differ within this same-engine configuration experiment")
            for record in samples:
                if not (run_dir / record["run_image"]).is_file() or not (run_dir / record["run_label"]).is_file():
                    raise ValueError("A staged sample is missing: " + record["run_image"])
            for name in ("best.pt", "last.pt"):
                if not (run_dir / "training/weights" / name).is_file():
                    raise FileNotFoundError(name)
            if not list((run_dir / "tensorboard").rglob("events.out.tfevents*")):
                raise ValueError("TensorBoard event files missing")
            for split in ("val", "test"):
                directory = run_dir / "evaluation" / split
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                required = ("diagnostics.json", "per_image.csv", "by_source.csv", "per_class_metrics.csv",
                            "negative_false_positives.csv", "confusion_matrix.png", "diagnostic_counts.png")
                missing = [name for name in required if not (directory / name).is_file()]
                if missing:
                    raise ValueError(f"{split} missing outputs: {missing}")
                with (directory / "per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
                    image_rows = list(csv.DictReader(stream))
                with (directory / "per_class_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                    class_rows = list(csv.DictReader(stream))
                if summary["images"] != len(selected[split]) or len(image_rows) != len(selected[split]):
                    raise ValueError(f"{split} image counts disagree")
                if summary["diagnostics"]["images"] != len(selected[split]) or len(class_rows) != len(header["names"]):
                    raise ValueError(f"{split} diagnostic/class counts disagree")
                if not list((directory / "predictions/previews").glob("*")):
                    raise ValueError(f"{split} prediction previews missing")
                prefixes = ("Box", "Pose") if config["task"] == "pose" else ("",)
                optional = [prefix + kind + "_curve.png" for prefix in prefixes for kind in ("PR", "F1", "P", "R")]
                optional += ["box_per_class.png"] + (["pose_per_class.png"] if config["task"] == "pose" else [])
                rows.append({"model_key": item["model_key"], "version": item["version"], "split": split,
                             "run_dir": relative(root, run_dir), "images": summary["images"],
                             "annotated_instances": summary["annotated_instances"], "metrics": summary["metrics"],
                             "diagnostics": summary["diagnostics"],
                             "native_plot_availability": {name: (directory / name).is_file() for name in optional}})
        except (OSError, ValueError, KeyError) as error:
            errors.append({"experiment": key, "error": f"{type(error).__name__}: {error}"})
    for model, versions in selections.items():
        if set(versions) == {"v1", "v2"} and versions["v1"] != versions["v2"]:
            errors.append({"experiment": model, "error": "V1 and V2 training selections differ"})
    report = {"created_utc": now(), "status": "failed" if errors else "passed", "errors": errors,
              "completed_experiments_checked": len([x for x in manifest["experiments"] if x["status"] in ("completed", "archived")]),
              "pending_experiments": [x["model_key"] + "/" + x["version"] for x in manifest["experiments"]
                                      if x["status"] not in ("completed", "archived")],
              "frozen_script_count": len(reference_scripts or {}), "evaluations": rows,
              "native_plots": "Missing native plots are recorded as unavailable; no curve values are inferred or fabricated."}
    write_json(suite / "artifact_audit.json", report)
    return report


def archive_suite(root, suite):
    path = suite / "suite.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (suite / "run.lock").exists():
        raise ValueError("Suite has a run lock; wait for its orchestrator before archive")
    if any(item["status"] == "running" for item in manifest["experiments"]):
        raise ValueError("An experiment is marked running; inspect its process before archive")
    audit = audit_suite(root, suite)
    if audit["errors"]:
        raise ValueError("Artifact audit failed; see suite artifact_audit.json")
    selected = [item for item in manifest["experiments"] if item["status"] == "completed"]
    moves = []
    archive_root = (root / "Archive/technical/revisions").resolve()
    for item in selected:
        source = inside(root, item["bundle"])
        source.relative_to(suite.resolve())
        target = archive_root / item["task_id"] / f"{item['model_key']}_yolo_{item['version']}"
        target.resolve().relative_to(archive_root)
        if target.exists():
            raise FileExistsError(f"Archive version already exists: {target}")
        if any(p.suffix.lower() in (".md", ".rst", ".docx", ".pdf") for p in (source / "scripts").rglob("*")):
            raise ValueError("Document snapshots must not enter the technical archive")
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Symbolic links need explicit review before moving this bundle")
        moves.append((item, source, target))
    for item, source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Both resolved paths were checked inside the workspace and intended areas.
        shutil.move(str(source), str(target))
        item.update(status="archived", original_bundle=item["bundle"], bundle=relative(root, target), archived_utc=now())
        item["run_dir"] = relative(root, target / "runs/smoke")
        write_json(path, manifest)
        print(f"ARCHIVED {relative(root, target)}", flush=True)
    manifest["status"] = "archived" if all(x["status"] == "archived" for x in manifest["experiments"]) else "partially_archived"
    write_json(path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "run", "audit", "archive"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--suite", required=True, help="Workspace-relative named directory under runs/smoke_versions")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--versions", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--limit", type=int, help="Run at most this many pending experiments, then return")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds for one training process")
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if len(args.models) != len(set(args.models)) or len(args.versions) != len(set(args.versions)):
        parser.error("Model and version lists must not contain duplicates")
    if args.timeout < 1 or args.evaluation_timeout < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Timeouts and limit must be positive")
    root = args.root.resolve()
    if not (root / "config/perception_tasks.json").is_file():
        parser.error("Expected the AIAEC workspace root")
    try:
        suite = validate_suite(root, args.suite)
        if args.command == "plan":
            result = [{"model": model, "version": version,
                       "source": relative(root, planned_config(root, model, version)[0]),
                       "epochs": PROFILES[version]["epochs"], "lr0": PROFILES[version]["lr0"],
                       "train_limit": 50, "imgsz": 320, "val_test": "full configured splits"}
                      for model in args.models for version in args.versions]
        elif args.command == "prepare":
            result = prepare(root, suite, args.models, args.versions)
        elif args.command == "run":
            result = run_suite(root, suite, args.limit, args.timeout, args.evaluation_timeout)
        elif args.command == "audit":
            result = audit_suite(root, suite)
        else:
            result = archive_suite(root, suite)
        display = result
        if isinstance(result, dict) and "experiments" in result:
            display = {"suite": relative(root, suite), "status": result["status"],
                       "experiments": [{key: item[key] for key in ("model_key", "version", "status", "bundle")}
                                       for item in result["experiments"]]}
        elif isinstance(result, dict) and "evaluations" in result:
            display = {key: value for key, value in result.items() if key != "evaluations"}
        print(json.dumps(display, ensure_ascii=False, indent=2))
        return 1 if isinstance(result, dict) and result.get("status") in ("stopped_after_failure", "interrupted", "incomplete", "failed") else 0
    except (OSError, ValueError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
