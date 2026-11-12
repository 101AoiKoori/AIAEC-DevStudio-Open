"""Check warm-start equivalence, trainability and inference checkpoint dispatch."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws/src/aec_vision/src'))
sys.path.insert(0, str(ROOT / 'lab/features/plate/training/scripts'))
import torch
from aec_vision.plate import CNNReader, create_cnn, load_charsets
from train_cnn_ocr import initialize_model


class TestSpatialCNN(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.charset_file = ROOT / 'ws/src/aec_vision/config/plate/charset.json'
        self.charsets = load_charsets(self.charset_file)
        self.counts = [len(c) for c in self.charsets]
        self.legacy = create_cnn(self.counts).eval()
        self.checkpoint = {'model': self.legacy.state_dict(), 'charsets': self.charsets, 'image_size': [224, 64]}

    def test_warm_start_preserves_logits(self):
        spatial = initialize_model(self.checkpoint, self.counts, 'spatial_v2').eval()
        with torch.inference_mode():
            image = torch.rand(2, 3, 64, 224)
            for old, new in zip(self.legacy(image), spatial(image)):
                torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)

    def test_new_parameters_receive_gradients(self):
        spatial = initialize_model(self.checkpoint, self.counts, 'spatial_v2').train()
        outputs = spatial(torch.rand(2, 3, 64, 224))
        loss = sum(torch.nn.functional.cross_entropy(out, torch.zeros(2, dtype=torch.long)) for out in outputs)
        loss.backward()
        self.assertGreater(spatial.refine[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(spatial.heads[0][0].weight.grad.abs().sum().item(), 0)

    def test_reader_dispatch_and_legacy_compatibility(self):
        # Temporary checkpoints stay in the repository's ignored temporary area.
        temp_root = ROOT / 'tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / 'model.pt'
            torch.save(self.checkpoint, path)
            self.assertEqual(CNNReader(path, self.charset_file).architecture, 'legacy')
            spatial = initialize_model(self.checkpoint, self.counts, 'spatial_v2')
            torch.save(dict(self.checkpoint, model=spatial.state_dict(), architecture='spatial_v2'), path)
            reader = CNNReader(path, self.charset_file)
            self.assertEqual(reader.architecture, 'spatial_v2')
            with self.assertRaises(ValueError):
                create_cnn(self.counts, 'unknown')


if __name__ == '__main__':
    unittest.main()
