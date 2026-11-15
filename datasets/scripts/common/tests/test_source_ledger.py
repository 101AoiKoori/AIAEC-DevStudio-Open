"""Contract checks for the cross-task background/source ledger.

Every fixture is built in a temporary workspace. The checks use only declared
configuration, manifests and files; they do not read real media, alter task
plans or claim label/semantic validity.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from datasets.scripts.common.source_ledger import (
    background_candidates,
    build_ledger,
    issue_counts,
    write_ledger,
)
from datasets.scripts.common.source_ledger.cli import main

POOL = "datasets/annotated/mixed_scene/v4_compositing_backgrounds_20261113"


def base_pairing(**overrides):
    pairing = {
        "schema_version": 1,
        "status": "draft",
        "composition_policy": {
            "target_source": "fixed_artwork",
            "blender_priority": "background_only",
            "target_components": ["artwork"],
            "background_components": ["real_backgrounds"],
            "blender_share_target": 0.30,
        },
        "policy": {
            "parent_classes": ["task_real", "background"],
            "split_unit": "parent_group",
            "train_parent_cap": 0.8,
            "val_test_mode": "reserve_only",
            "derived_inherits_parent_split": True,
            "cross_task_duplicates": "allowed_with_separate_labels",
            "same_source_splits": "record_and_report",
        },
        "task_real": [],
        "backgrounds": [],
        "data_sources": [],
        "blender": [],
        "captures": [],
        "plan_bindings": {"train": [], "val": [], "test": []},
    }
    pairing.update(overrides)
    return pairing


class SourceLedgerTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.review_manifest = self.write_json(
            "datasets/annotated/person/review.json", {"records": [{"image": "x"}]}
        )
        self.background_manifest = self.write_json(
            "datasets/annotated/person/backgrounds.json", {"records": [{"image": "x"}]}
        )

    def tearDown(self):
        self._temporary.cleanup()

    # ------------------------------------------------------------- helpers

    def write_text(self, relative: str, text: str = "") -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return relative

    def write_json(self, relative: str, data) -> str:
        return self.write_text(relative, json.dumps(data, ensure_ascii=False))

    def write_source(self, task: str, data) -> None:
        path = self.root / "datasets/scripts" / task / "config" / "source.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def evidence(self, manifest: str | None = None, record_index: int = 0) -> dict:
        return {"manifest": manifest or self.background_manifest, "record_index": record_index, "relation": "describes"}

    def background(
        self,
        group_id: str = "bg1",
        path: str = "datasets/raw/person/bg/a.png",
        *,
        state: str = "confirmed_empty",
        roles=("composite_background", "empty_negative"),
        primary: str = "train",
        same_source=None,
        labels: str = "complete",
    ) -> dict:
        split = {"primary": primary, "status": "proposed"}
        if same_source:
            split["same_source_allowed_in"] = list(same_source)
        return {
            "id": group_id,
            "class": "background",
            "kind": "parent_photo",
            "paths": [path],
            "evidence": [self.evidence()],
            "review": {"state": state, "method": "manual", "accepted_by": None},
            "labels": {"status": labels},
            "roles": list(roles),
            "split": split,
        }

    def task_real(
        self,
        group_id: str = "tr1",
        path: str = "datasets/raw/person/videos/a.mp4",
        *,
        primary: str = "train",
        roles=("direct_target",),
    ) -> dict:
        return {
            "id": group_id,
            "class": "task_real",
            "kind": "parent_video",
            "paths": [path],
            "evidence": [self.evidence(self.review_manifest)],
            "capture_session": "unknown",
            "labels": {"status": "partial"},
            "roles": list(roles),
            "split": {"primary": primary, "status": "proposed"},
        }

    def person_pairing(self, **overrides):
        self.write_text("datasets/raw/person/videos/a.mp4")
        self.write_text("datasets/raw/person/bg/a.png")
        self.write_text("datasets/raw/person/art/a.png")
        self.write_text("datasets/raw/person/art/b.png")
        self.write_text("datasets/assets/blender/shared/a.blend")
        pairing = base_pairing(
            task_real=[self.task_real()],
            backgrounds=[self.background(same_source=["val"])],
            data_sources=[
                {
                    "id": "ds1",
                    "kind": "fixed_artwork",
                    "paths": [
                        "datasets/raw/person/art/a.png",
                        "datasets/raw/person/art/b.png",
                    ],
                    "parent_group": None,
                    "roles": ["target_foreground"],
                    "split": {"primary": "shared", "status": "confirmed"},
                }
            ],
            blender=[
                {
                    "id": "bl1",
                    "kind": "blender_scene",
                    "path": "datasets/assets/blender/shared/a.blend",
                    "roles": ["background_scene"],
                    "split": {"primary": "shared", "status": "confirmed"},
                }
            ],
        )
        pairing.update(overrides)
        self.write_source("person", {"task": "person", "pairing": pairing})
        return pairing

    # --------------------------------------------------------------- tests

    def test_valid_pairing_builds_ledger_and_candidates(self):
        self.person_pairing()
        ledger = build_ledger(self.root, ["person"])
        entry = ledger["tasks"]["person"]
        self.assertEqual(entry["adoption"], "pairing")
        self.assertEqual(len(entry["groups"]), 3)
        self.assertEqual(len(entry["data_sources"]), 1)
        self.assertEqual(issue_counts(ledger)["error"], 0)

        train = background_candidates(ledger, "person", "composite_background", "train")
        self.assertEqual([item["id"] for item in train["eligible"]], ["bg1"])
        val = background_candidates(ledger, "person", "empty_negative", "val")
        self.assertEqual([item["id"] for item in val["eligible"]], ["bg1"])
        test = background_candidates(ledger, "person", "empty_negative", "test")
        self.assertTrue(test["excluded"])
        self.assertEqual(test["excluded"][0]["reason"], "split_not_assigned")

    def test_missing_pairing_falls_back_to_legacy_observation(self):
        self.write_source(
            "person",
            {
                "task": "person",
                "roots": ["datasets/raw/person"],
                "records": [],
                "databases": {},
            },
        )
        ledger = build_ledger(self.root, ["person"])
        entry = ledger["tasks"]["person"]
        self.assertEqual(entry["adoption"], "legacy")
        self.assertEqual(entry["groups"], [])
        issue = background_candidates(ledger, "person", "composite_background", "train")
        self.assertEqual(issue["reason"], "pairing_not_adopted")
        codes = [item["code"] for item in ledger["issues"]]
        self.assertIn("BSL018", codes)
        self.assertEqual(issue_counts(ledger)["error"], 0)

    def test_missing_background_path_is_an_error(self):
        self.person_pairing(backgrounds=[self.background(path="datasets/raw/person/bg/missing.png")])
        ledger = build_ledger(self.root, ["person"])
        errors = [item for item in ledger["issues"] if item["level"] == "error"]
        self.assertTrue(any(item["code"] == "BSL003" and item["group"] == "bg1" for item in errors))

    def test_empty_negative_requires_confirmed_empty(self):
        self.person_pairing(backgrounds=[self.background(state="candidate_empty")])
        ledger = build_ledger(self.root, ["person"])
        self.assertTrue(
            any(
                item["code"] == "BSL007" and item["group"] == "bg1"
                for item in ledger["issues"]
                if item["level"] == "error"
            )
        )

    def test_train_cap_is_enforced_per_role(self):
        paths = "datasets/raw/person/bg/a.png"
        self.write_text(paths)
        backgrounds = []
        for index in range(5):
            item = self.background(group_id=f"bg{index}", path=paths, roles=("composite_background",))
            item["evidence"] = [self.evidence()]
            backgrounds.append(item)
        self.person_pairing(task_real=[], backgrounds=backgrounds, data_sources=[], blender=[])
        ledger = build_ledger(self.root, ["person"])
        errors = [
            item
            for item in ledger["issues"]
            if item["level"] == "error" and item["code"] == "BSL011"
        ]
        self.assertTrue(errors)
        self.assertIn("background/composite_background", errors[0]["message"])

    def test_reserve_only_rejects_budget_fields(self):
        pairing = self.person_pairing()
        pairing["backgrounds"][0]["count"] = 10
        self.write_source("person", {"task": "person", "pairing": pairing})
        ledger = build_ledger(self.root, ["person"])
        self.assertTrue(
            any(
                item["code"] == "BSL014"
                for item in ledger["issues"]
                if item["level"] == "error"
            )
        )

    def test_derived_data_source_must_inherit_parent_split(self):
        self.person_pairing(
            data_sources=[
                {
                    "id": "ds1",
                    "kind": "fixed_artwork",
                    "path": "datasets/raw/person/art/a.png",
                    "parent_group": "tr1",
                    "roles": ["target_foreground"],
                    "split": {"primary": "shared", "status": "confirmed"},
                }
            ]
        )
        ledger = build_ledger(self.root, ["person"])
        self.assertTrue(
            any(
                item["code"] == "BSL009" and item["group"] == "ds1"
                for item in ledger["issues"]
                if item["level"] == "error"
            )
        )

    def test_capture_placeholder_and_capture_link(self):
        placeholder = "datasets/raw/sign/capture_test/"
        pairing = base_pairing(
            task_real=[
                {
                    "id": "tr_sign_capture",
                    "class": "task_real",
                    "kind": "parent_video",
                    "paths": [placeholder],
                    "evidence": [{"note": "待采；采后登记原视频/照片与真值清单"}],
                    "capture_session": "planned",
                    "capture_id": "cap1",
                    "labels": {"status": "unknown"},
                    "roles": ["direct_target"],
                    "split": {"primary": "val", "status": "proposed", "action": "to_capture"},
                }
            ],
            captures=[
                {
                    "id": "cap1",
                    "target_split": "val",
                    "status": "proposed",
                    "reason": "no_independent_parent",
                    "save_to": placeholder,
                }
            ],
        )
        self.write_source("sign", {"task": "sign", "pairing": pairing})
        ledger = build_ledger(self.root, ["sign"])
        self.assertFalse(any(item["code"] == "BSL013" for item in ledger["issues"]))

        pairing["captures"][0]["target_split"] = "train"
        self.write_source("sign", {"task": "sign", "pairing": pairing})
        ledger = build_ledger(self.root, ["sign"])
        self.assertTrue(
            any(item["code"] == "BSL013" for item in ledger["issues"] if item["level"] == "error")
        )

    def test_shared_pool_aggregates_parent_videos_and_tasks(self):
        records = [
            {"image": f"{POOL}/images/a_001.jpg", "source_group": "video_a"},
            {"image": f"{POOL}/images/a_002.jpg", "source_group": "video_a"},
            {"image": f"{POOL}/images/b_001.jpg", "source_group": "video_b"},
        ]
        self.write_json(f"{POOL}/manifest.json", {"records": records})
        for record in records:
            self.write_text(record["image"])
        self.write_source(
            "person",
            {
                "task": "person",
                "planning_background_manifest": f"{POOL}/manifest.json",
                "current_selection": {
                    "backgrounds": [records[0]["image"], records[2]["image"]]
                },
            },
        )
        ledger = build_ledger(self.root, ["person"])
        pool = ledger["shared"]["background_pool"]
        self.assertEqual(pool["records"], 3)
        self.assertEqual(len(pool["parent_groups"]), 2)
        by_parent = {item["source_video"]: item for item in pool["parent_groups"]}
        self.assertEqual(by_parent["video_a"]["tasks"][0]["selected"], 1)
        self.assertTrue(by_parent["video_a"]["tasks"][0]["declared"])
        self.assertFalse(any(item["code"] == "BSL023" for item in ledger["issues"]))

        self.write_source(
            "person",
            {
                "task": "person",
                "current_selection": {"backgrounds": [records[0]["image"]]},
            },
        )
        ledger = build_ledger(self.root, ["person"])
        self.assertTrue(
            any(item["code"] == "BSL023" for item in ledger["issues"] if item["level"] == "warning")
        )

    def test_blender_scene_usage_is_aggregated_across_tasks(self):
        self.write_text("datasets/assets/blender/shared/a.blend")
        for task in ("person", "e_bike"):
            self.write_source(
                task,
                {
                    "task": task,
                    "blender_master_scene": "datasets/assets/blender/shared/a.blend",
                },
            )
        ledger = build_ledger(self.root, ["person", "e_bike"])
        scenes = ledger["shared"]["blender_scenes"]
        self.assertEqual(len(scenes), 1)
        self.assertEqual({item["task"] for item in scenes[0]["tasks"]}, {"person", "e_bike"})
        self.assertTrue(
            any(item["code"] == "BSL025" and "Blender" in item["message"] for item in ledger["issues"])
        )

    def test_cli_write_outputs_ledger(self):
        self.person_pairing()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["write", "--root", str(self.root), "--task", "person"])
        self.assertEqual(code, 0)
        target = self.root / "datasets/annotated/common/sources/ledger.json"
        self.assertTrue(target.is_file())
        ledger = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(ledger["schema_version"], 1)
        self.assertEqual(ledger["tasks"]["person"]["adoption"], "pairing")

    def test_write_ledger_is_atomic_and_readable(self):
        ledger = build_ledger(self.root, ["person"])
        target = write_ledger(ledger, root=self.root)
        self.assertTrue(target.is_file())
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["schema_version"], 1)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
