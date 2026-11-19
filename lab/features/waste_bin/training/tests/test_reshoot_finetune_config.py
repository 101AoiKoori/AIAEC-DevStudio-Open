"""Guard the real-bin fine-tune bias warmup at the actual trainer option boundary."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.config import load_config
from engine.runner import _training_options


class ReshootFineTuneConfigurationTests(unittest.TestCase):
    def test_zero_bias_warmup_reaches_trainer_options(self):
        config = load_config(ROOT, 'lab/features/waste_bin/training/config/waste_bin_state_reshoot_finetune_v4.yaml')
        options = _training_options(config, ROOT / config['data'], ROOT / 'tmp/unexecuted_bin_training')
        # Omitting this override silently restores Ultralytics' 0.1 bias LR,
        # although the nominal fine-tuning LR remains 0.0001.
        self.assertEqual(options['warmup_bias_lr'], 0.0)
        self.assertGreater(options['warmup_epochs'], 0)
        self.assertLessEqual(options['warmup_bias_lr'], options['lr0'])
        self.assertEqual(options['optimizer'], 'AdamW')
        self.assertNotIn('resume', options)


if __name__ == '__main__':
    unittest.main()
