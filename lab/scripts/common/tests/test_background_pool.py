from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lab.scripts.common import background_pool


class BackgroundPoolTest(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        (workspace / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace / "tmp")
        self.root = Path(self.temp.name)
        task_dir = self.root / "datasets/scripts/person/config"
        task_dir.mkdir(parents=True)
        (task_dir / "source.yaml").write_text(
            """
pairing:
  schema_version: 1
  status: draft
  backgrounds:
    - id: bg_scene
      class: background
      kind: parent_video
      paths: [datasets/raw/site/scene.mp4]
      review: {state: candidate_empty, method: contact_sheet, accepted_by: null}
      labels: {status: unknown}
      roles: [composite_background]
      split: {primary: val, status: proposed, same_source_allowed_in: [train]}
""",
            encoding="utf-8",
        )
        self.plan = self.root / "plan.json"
        self.selection = self.root / "selection.json"
        self.plan.write_text(
            json.dumps(
                {
                    "task": "person",
                    "batch": "test",
                    "frames": [
                        {
                            "path": "datasets/raw/site/frame_a.jpg",
                            "parent_video": "datasets/raw/site/scene.mp4",
                            "source_group": "scene",
                            "role": "background",
                        },
                        {
                            "path": "datasets/raw/site/frame_b.jpg",
                            "parent_video": "datasets/raw/site/scene.mp4",
                            "source_group": "scene",
                            "role": "background",
                        },
                        {
                            "path": "datasets/raw/site/frame_c.jpg",
                            "parent_video": "datasets/raw/site/scene.mp4",
                            "source_group": "scene",
                            "role": "background",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.selection.write_text(
            json.dumps(
                {
                    "task": "person",
                    "records": [
                        {
                            "frame": {"path": "datasets/raw/site/frame_a.jpg"},
                            "verdict": "ready",
                            "reason_codes": ["no_unlabelled_target"],
                            "accepted_by": "reviewer",
                            "existing_targets": [],
                        },
                        {
                            "frame": {"path": "datasets/raw/site/frame_b.jpg"},
                            "verdict": "ready",
                            "reason_codes": ["no_unlabelled_target"],
                            "accepted_by": None,
                            "existing_targets": [],
                            "near_duplicate_group": "dup_0",
                        },
                        {
                            "frame": {"path": "datasets/raw/site/frame_c.jpg"},
                            "verdict": "unknown",
                            "reason_codes": ["geometry_rejected"],
                            "accepted_by": None,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_states_are_conservative_and_split_is_declared(self):
        report = background_pool.summarize_review(
            task="person",
            batch="test",
            review_plan=self.plan,
            review_selection=self.selection,
            root=self.root,
        )
        states = [record["review_state"] for record in report["records"]]
        self.assertEqual(states, ["confirmed_empty", "candidate_empty", "unknown"])
        self.assertEqual(report["summary"]["human_confirmed_empty"], 1)
        self.assertEqual(report["summary"]["candidate_empty_not_negative"], 1)
        self.assertEqual(report["summary"]["unknown_not_negative"], 1)
        self.assertEqual(
            report["records"][0]["split_reservation"],
            {
                "status": "declared",
                "group_id": "bg_scene",
                "class": "background",
                "kind": "parent_video",
                "primary": "val",
                "same_source_allowed_in": ["train"],
                "reservation_status": "proposed",
                "review_state": "candidate_empty",
                "labels_status": "unknown",
            },
        )

    def test_excluded_target_is_contains_target_not_empty(self):
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["records"][0].update(
            {
                "verdict": "excluded",
                "reason_codes": ["unlabelled_target_suspected"],
                "accepted_by": "reviewer",
            }
        )
        self.selection.write_text(json.dumps(selection), encoding="utf-8")
        report = background_pool.summarize_review(
            task="person",
            batch="test",
            review_plan=self.plan,
            review_selection=self.selection,
            root=self.root,
        )
        self.assertEqual(report["records"][0]["review_state"], "contains_target")

    def test_missing_review_stays_unknown(self):
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["records"] = selection["records"][1:]
        self.selection.write_text(json.dumps(selection), encoding="utf-8")
        report = background_pool.summarize_review(
            task="person",
            batch="test",
            review_plan=self.plan,
            review_selection=self.selection,
            root=self.root,
        )
        self.assertEqual(report["records"][0]["review_state"], "unknown")
        self.assertIn("review_missing", report["records"][0]["review_reasons"])


if __name__ == "__main__":
    unittest.main()
