"""Optional Blender integration test; set BLENDER_EXE to enable it."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'datasets/raw/meter/pdf/page15-005.jpg'
TICKS = ROOT / 'lab/features/meter/inference/config/pdf_ticks.json'
BUILDER = ROOT / 'lab/features/meter/training/scripts/build_blender_9011.py'
BLENDER = os.environ.get('BLENDER_EXE', '')


@unittest.skipUnless(BLENDER and Path(BLENDER).is_file() and SOURCE.is_file(),
                     'Set BLENDER_EXE and extract PDF photo to run Blender integration')
class BlenderMeterTests(unittest.TestCase):
    def test_model_renders_and_two_hands_rotate_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'dataset'
            command = [sys.executable, str(BUILDER), '--blender', BLENDER, '--source', str(SOURCE),
                       '--ticks', str(TICKS), '--output', str(output), '--count', '1', '--seed', '9012']
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            data = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['keypoint_order']), 8)
            self.assertEqual(len(data['samples']), 1)
            row = (output / 'labels/train/meter_00000.txt').read_text().split()
            self.assertEqual(len(row), 29)
            self.assertEqual(row[0], '0')
            for i in range(8):
                self.assertTrue(0 <= float(row[5 + i * 3]) <= 1)
                self.assertTrue(0 <= float(row[6 + i * 3]) <= 1)
                self.assertIn(int(row[7 + i * 3]), (0, 1, 2))
            self.assertTrue((output / 'images/train/meter_00000.png').is_file())
            self.assertTrue((output / 'previews/meter_00000.png').is_file())
            expression = ("import bpy; a=bpy.data.objects['temperature_needle_rotate_Z']; "
                          "b=bpy.data.objects['humidity_needle_rotate_Z']; z=b.rotation_euler.z; "
                          "assert a.type=='CURVE' and b.type=='CURVE'; "
                          "assert bpy.data.objects['metal_bezel_bezier'].type=='CURVE'; "
                          "assert any(m.type=='BEVEL' for m in bpy.data.objects['metal_shell'].modifiers); "
                          "a.rotation_euler.z=0.8; assert abs(b.rotation_euler.z-z)<1e-6; "
                          "z=a.rotation_euler.z; b.rotation_euler.z=-0.5; "
                          "assert abs(a.rotation_euler.z-z)<1e-6; "
                          "assert any(i.packed_file for i in bpy.data.images)")
            subprocess.run([BLENDER, '--background', '--factory-startup',
                            str(output / 'meter_9011_approx.blend'), '--python-exit-code', '1',
                            '--python-expr', expression], check=True, capture_output=True, text=True, timeout=60)
            # A second invocation must refuse to overwrite the generated model.
            again = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('refusing to overwrite', again.stderr)


if __name__ == '__main__':
    unittest.main()
