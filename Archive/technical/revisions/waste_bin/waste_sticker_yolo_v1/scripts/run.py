"""Run this recorded smoke experiment with its frozen engine."""
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
