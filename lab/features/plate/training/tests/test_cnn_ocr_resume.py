"""Verify independent OCR resume metadata without training or inference."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/plate/training/scripts'))
import train_cnn_ocr as ocr


class OCRResumeTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'tmp'
        base.mkdir(exist_ok=True)
        temporary = TemporaryDirectory(dir=base)
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        (self.run / 'metadata').mkdir()
        self.best, self.last = self.run / 'best.pt', self.run / 'last.pt'
        self.config = {'epochs': 5, 'max_minutes': 10}

    def save_checkpoint(self):
        (self.run / 'run_metadata.json').write_text(json.dumps({'run_id': 'fixture', 'config': self.config}),
                                                   encoding='utf-8')
        checkpoint = {'charsets': [['A']] * 7, 'image_size': [32, 16], 'architecture': 'legacy',
                      'epoch': 1, 'metrics': {'source': {'plate_accuracy': 0.5}}}
        state = {'schema_version': 1, 'run_id': 'fixture', 'config': self.config, 'device': 'cpu',
                 'data_identity': {'train': 'signature', 'val': 'signature'}, 'best_name': 'best.pt',
                 'best_epoch': 1, 'best_score': 0.5, 'stale': 0, 'history': [{'epoch': 1, 'train_loss': 0.2}],
                 'elapsed_seconds': 30.5, 'stop_reason': None}
        model = SimpleNamespace(state_dict=lambda: {'weight': torch.tensor([1.5])})
        optimizer = SimpleNamespace(state_dict=lambda: {'state': {0: {'step': torch.tensor(2)}}})
        scheduler = SimpleNamespace(state_dict=lambda: {'last_epoch': 1})
        scaler = SimpleNamespace(state_dict=lambda: {})
        generator = torch.Generator().manual_seed(74)
        ocr._commit_epoch(self.best, checkpoint, state, model, torch.device('cpu'), optimizer,
                          scheduler, scaler, generator)
        return torch.load(self.last, weights_only=True)

    def test_resume_reuses_original_configuration_and_training_state(self):
        checkpoint = self.save_checkpoint()
        self.assertNotIn('ocr_resume', torch.load(self.best, weights_only=True))
        self.assertEqual(checkpoint['ocr_resume']['scheduler']['last_epoch'], 1)
        self.assertEqual(checkpoint['ocr_resume']['optimizer']['state'][0]['step'].item(), 2)
        with patch.object(ocr, '_run') as run:
            ocr.main(['--resume', str(self.last)])
        self.assertEqual(run.call_args.args[1], self.config)
        self.assertEqual(run.call_args.kwargs['checkpoint']['epoch'], 1)
        self.assertEqual(run.call_args.kwargs['resume_state']['elapsed_seconds'], 30.5)
        self.assertEqual(run.call_args.kwargs['output'], self.best)
        (self.run / 'run_metadata.json').write_text('{"run_id": "other"}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not belong to the recorded run'):
            ocr.main(['--resume', str(self.last)])

    def test_resume_recovers_companion_files_and_preserves_history_prefix(self):
        checkpoint = self.save_checkpoint()
        self.best.unlink()
        log = self.run / 'metrics.jsonl'
        first_record = log.read_text(encoding='utf-8')
        checkpoint['epoch'] = 2
        state = checkpoint['ocr_resume']
        state['best_epoch'] = 2
        state['history'].append({'epoch': 2, 'train_loss': 0.15})
        ocr._recover_outputs(self.best, checkpoint, state)
        self.assertEqual(torch.load(self.best, weights_only=True)['epoch'], 2)
        self.assertTrue(log.read_text(encoding='utf-8').startswith(first_record))
        self.assertEqual(len(log.read_text(encoding='utf-8').splitlines()), 2)
        log.write_text('{"epoch": 9}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'metrics log differs'):
            ocr._recover_outputs(self.best, checkpoint, state)

    def test_old_weights_and_terminal_attempt_are_not_resumed(self):
        checkpoint = self.save_checkpoint()
        with self.assertRaisesRegex(ValueError, 'lacks full training state'):
            ocr._resume_state(torch.load(self.best, weights_only=True), self.best)
        checkpoint['ocr_resume']['stop_reason'] = 'early_stopping'
        torch.save(checkpoint, self.last)
        with patch.object(ocr, '_run') as run:
            with self.assertRaisesRegex(ValueError, 'already stopped'):
                ocr.main(['--resume', str(self.last)])
        run.assert_not_called()

    def test_data_identity_covers_annotation_and_image_contents(self):
        annotation, image = self.run / 'samples.json', self.run / 'image.png'
        annotation.write_text(json.dumps({'annotations': ['image.png']}), encoding='utf-8')
        image.write_bytes(b'pixels')
        sources = {'source': str(annotation)}
        datasets = {'source': SimpleNamespace(root=self.run, rows=[{'image': 'image.png'}])}
        baseline = ocr._data_identity(sources, datasets)
        image.write_bytes(b'PIXELS')
        changed_image = ocr._data_identity(sources, datasets)
        self.assertNotEqual(changed_image['source']['images_sha256'], baseline['source']['images_sha256'])
        annotation.write_text('{}', encoding='utf-8')
        changed_annotation = ocr._data_identity(sources, datasets)
        self.assertNotEqual(changed_annotation['source']['annotations_sha256'], baseline['source']['annotations_sha256'])


if __name__ == '__main__':
    unittest.main()
