import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'lab/features/e_bike/training/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('build_occlusion_candidates', SCRIPTS / 'build_occlusion_candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OcclusionCandidatesTest(unittest.TestCase):
    def test_source_validation_grouping_and_visible_masks(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            review = root / 'review'
            (review / 'images').mkdir(parents=True)
            frame = review / 'images' / 'frame.jpg'
            frame.write_bytes(b'frame')
            video = root / 'video.mp4'
            video.write_bytes(b'video')
            rgba = np.full((24, 12, 4), 255, np.uint8)
            person = root / 'person.png'
            vehicle = root / 'vehicle.png'
            cv2.imencode('.png', rgba)[1].tofile(str(person))
            cv2.imencode('.png', rgba)[1].tofile(str(vehicle))
            review_manifest = review / 'manifest.json'
            review_manifest.write_text(json.dumps({'sources': [{'source': str(video),
                                                                 'frames': ['images/frame.jpg']}]}), encoding='utf-8')
            selection = root / 'selection.json'
            data = {'split': None, 'group': 'one_source', 'approved_source_videos': [str(video)],
                    'review_manifest': str(review_manifest),
                    'people': [{'class': 'community', 'path': str(person)}],
                    'vehicles': [{'class': 'upright', 'path': str(vehicle), 'source_video': str(video),
                                  'source_frame': str(frame)}]}
            selection.write_text(json.dumps(data), encoding='utf-8')
            out = root / 'candidates'
            samples = module._build(selection, out)
            self.assertEqual(len(samples), 2)
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete_candidates')
            self.assertIsNone(manifest['split'])
            self.assertTrue(all(sample['source_group'] == 'one_source' for sample in samples))
            self.assertTrue(all((out / sample['person_visible_mask']).is_file() for sample in samples))
            with self.assertRaises(ValueError):
                module._build(selection, out)
            data['approved_source_videos'] = []
            selection.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                module._build(selection, root / 'other')
            self.assertFalse((root / 'other').exists())


if __name__ == '__main__':
    unittest.main()
