"""Check that source-role dependencies are declared for guarded generation."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/features/e_bike/training/scripts')]
import e_bike_queue_inputs as subject


class QueueInputsTest(unittest.TestCase):
    def test_excluded_history_is_not_input_unless_actually_selected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            image = root / 'images/excluded.png'
            image.parent.mkdir()
            image.touch()
            queue = {'inputs': [], 'exposed_images': ['images/excluded.png'],
                     'unavailable_exposure_paths': [{'path': 'images/excluded.png',
                                                     'status': 'lifecycle_excluded_not_comparable'}]}
            with mock.patch.object(subject.gate, 'read_json', return_value={'items': []}), \
                 mock.patch.object(subject.gate, 'queue_inputs', return_value=([], [], [])):
                media, _, _ = subject.queue_input_report(root, queue)
                self.assertNotIn(image, media)
                media, _, _ = subject.queue_input_report(root, queue, [image])
                self.assertIn(image, media)

    def test_dynamic_evidence_and_source_role_inputs_are_included(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            exposed = root / 'images/exposed.png'
            exposed.parent.mkdir()
            exposed.touch()
            ledger = {'items': [{'image': 'images/seen.png', 'parent_video': 'videos/source.mp4',
                                 'source_time_seconds': 1.0, 'evidence': ['review/one.json']}],
                      'near_duplicate_pairs': [{'a': 'images/a.png', 'b': 'images/b.png',
                                                'evidence': 'review/two.json'}]}
            base = [root / 'config/perception_tasks.json',
                    root / 'lab/features/e_bike/training/config/e_bike_v3.yaml']
            for path in [*base, *(root / p for p in ('images/seen.png', 'images/a.png', 'videos/source.mp4',
                         'review/base.json', 'review/one.json', 'review/two.json',
                         subject.gate.SITE_SEQUENCE + '/ffmpeg.log'))]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with mock.patch.object(subject.gate, 'read_json', return_value=ledger), \
                 mock.patch.object(subject.gate, 'queue_inputs', return_value=([], base, [])):
                media, procedural, missing = subject.queue_input_report(
                    root, {'inputs': ['review/base.json'], 'exposed_images': ['images/exposed.png']},
                    [root / subject.gate.SITE_SEQUENCE / 'frame_000001.png'])
            media = {p.relative_to(root).as_posix() for p in media}
            procedural = {p.relative_to(root).as_posix() for p in procedural}
            self.assertTrue({'images/exposed.png', 'images/seen.png', 'images/a.png',
                             'videos/source.mp4'} <= media)
            self.assertNotIn('images/b.png', media)
            self.assertIn('images/b.png', missing)
            self.assertTrue({'review/base.json', 'review/one.json', 'review/two.json',
                             'config/perception_tasks.json',
                             subject.gate.SITE_SEQUENCE + '/ffmpeg.log',
                             'lab/features/e_bike/training/config/e_bike_v3.yaml'} <= procedural)
            self.assertNotIn('dataset/data.yaml', procedural)


if __name__ == '__main__':
    unittest.main()
