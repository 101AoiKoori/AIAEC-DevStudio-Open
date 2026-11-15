from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.scripts.person.human_review_state import backup_files, has_human_content, merge_cases


class HumanReviewStateTest(unittest.TestCase):
    def test_candidate_and_pending_do_not_count_as_human_content(self):
        review = {
            "cases": [
                {
                    "accepted_by": None,
                    "box_accepted_by": "Codex visual review",
                    "box_acceptance": "candidate",
                    "full_frame_checked": False,
                }
            ]
        }
        self.assertFalse(has_human_content(review))

    def test_merge_preserves_human_fields_by_stable_id(self):
        existing = {
            "cases": [
                {
                    "id": "P01",
                    "accepted_by": "aoi",
                    "human_decision": "add one target",
                    "human_box_corrections": [{"box": [1, 2, 3, 4]}],
                }
            ]
        }
        proposed = {
            "status": "pending",
            "cases": [{"id": "P01", "accepted_by": None, "overlay": "new.png"}],
        }
        merged, preserved = merge_cases(existing, proposed)
        self.assertEqual(preserved, ["P01"])
        self.assertEqual(merged["cases"][0]["overlay"], "new.png")
        self.assertEqual(merged["cases"][0]["accepted_by"], "aoi")
        self.assertTrue(merged["cases"][0]["preserved_from_existing_review"])

    def test_backup_keeps_nonempty_review_file(self):
        workspace = Path(__file__).resolve().parents[3]
        (workspace / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace / "tmp") as tmp:
            source = Path(tmp) / "review.json"
            source.write_text('{"accepted_by": "aoi"}\n', encoding="utf-8")
            backups = backup_files([source])
            self.assertEqual(len(backups), 1)
            self.assertTrue(backups[0].is_file())
            self.assertEqual(backups[0].read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
