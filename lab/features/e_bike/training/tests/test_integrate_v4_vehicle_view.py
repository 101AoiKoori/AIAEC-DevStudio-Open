import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/integrate_v4_vehicle_view.py'
spec = importlib.util.spec_from_file_location('vehicle_view_gate', SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class VehicleViewValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(gate, 'ROOT', self.root)
        self.patch.start()
        self.master = patch.object(gate, 'MASTER', 'scene.blend')
        self.master.start()
        self.file('scene.blend', b'blend')
        self.file('video1.mp4', b'video1')
        self.file('video2.mp4', b'video2')
        self.real = {'records': []}
        self.blender = {'records': []}
        for i in range(2):
            image = f'real{i}.png'
            self.image(image, i + 1)
            self.image(f'parent{i}.png', i + 1)
            self.file(f'real{i}.txt', b'')
            self.file(f'parent{i}.txt', b'')
            self.real['records'].append({'split': 'train', 'image': image, 'label': f'real{i}.txt',
                'source_image': f'parent{i}.png', 'source_label': f'parent{i}.txt',
                'parent_video': f'video{i+1}.mp4', 'label_state': 'complete_negative'})
        for i, split in enumerate(('train', 'val', 'test')):
            image, background, label, mask = f'out{i}.png', f'view{i}.png', f'out{i}.txt', f'mask{i}.png'
            self.image(image, 20 + i)
            self.image(background, 30 + i)
            self.image(mask, 40 + i)
            self.file(label, b'0 0.5 0.5 0.5 0.5\n')
            self.blender['records'].append({'split': split, 'image': image, 'label': label,
                'background_image': background, 'mother_scene': 'scene.blend', 'background_origin': 'blender',
                'background': {'kind': 'blender', 'source_group': 'scene.blend', 'view_group': Path(background).stem},
                'objects': [{'class': 'upright', 'bbox_xyxy': [2, 2, 6, 6], 'visible_mask': mask}]})

    def tearDown(self):
        self.master.stop()
        self.patch.stop()
        self.tmp.cleanup()

    def file(self, name, data):
        (self.root / name).write_bytes(data)

    def image(self, name, color):
        Image.new('RGB', (8, 8), (color, 0, 0)).save(self.root / name)

    def test_counts_and_actual_background_groups(self):
        stats = gate._validate_rows(self.real, self.blender, expected=False)
        self.assertEqual(stats['total']['images'], 5)
        self.assertEqual(stats['total']['blender_ratio'], .6)
        self.assertEqual(stats['total']['real_source_groups'], ['video1.mp4', 'video2.mp4'])

    def test_cross_split_duplicate_rejected(self):
        (self.root / 'out2.png').write_bytes((self.root / 'out1.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'across splits'):
            gate._validate_rows(self.real, self.blender, expected=False)

    def test_candidate_manifest_cannot_replace_review(self):
        with self.assertRaisesRegex(ValueError, 'QA must explicitly'):
            gate._plan({'samples': []}, {'additional_records': []}, [], self.real, self.blender,
                       {'status': 'candidate_QA_pending'}, self.root / 'qa.json', expected=False)

    def test_unknown_label_not_negative(self):
        self.real['records'][0]['label_state'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unreviewed'):
            gate._validate_rows(self.real, self.blender, expected=False)

    def test_blender_group_not_arbitrary(self):
        self.blender['records'][0]['background']['view_group'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            gate._validate_rows(self.real, self.blender, expected=False)


if __name__ == '__main__':
    unittest.main()
