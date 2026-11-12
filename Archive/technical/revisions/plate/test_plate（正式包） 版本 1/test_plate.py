import sys
import json
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from aec_vision.plate import DualPlateOCR, PlateRecognizer, confirm_plate_samples, format_plate_text, load_charsets, normalize_text


class Reader:
    def __init__(self, text='', score=0.0, error=None):
        self.text, self.score, self.error, self.seen = text, score, error, []

    def read(self, image):
        self.seen.append(image)
        if self.error:
            raise RuntimeError(self.error)
        return self.text, self.score


class TestDualOCR(unittest.TestCase):
    def setUp(self):
        self.charsets = load_charsets(PACKAGE / 'config/plate/charset.json')

    def run_readers(self, cnn, paddle, policy='agreement_only'):
        crop = object()
        result = DualPlateOCR(cnn, paddle, self.charsets, policy=policy).read(crop)
        self.assertIs(cnn.seen[0], crop)
        self.assertIs(paddle.seen[0], crop)
        return result

    def test_agreement_preserves_both_readers(self):
        result = self.run_readers(Reader('苏A12345', .81), Reader('苏A·12345', .92))
        self.assertEqual((result.status, result.text, result.decision), ('ok', '苏A12345', 'agreement'))
        self.assertEqual((result.cnn_confidence, result.paddle_confidence), (.81, .92))

    def test_disagreement_uses_province_paddle_threshold(self):
        result = self.run_readers(Reader('苏A12345', .99), Reader('苏A12346', .75), policy='province_baseline')
        self.assertEqual((result.text, result.decision), ('苏A12346', 'paddle_override'))
        result = self.run_readers(Reader('苏A12345', .3), Reader('苏A12346', .69), policy='province_baseline')
        self.assertEqual((result.text, result.decision), ('苏A12345', 'paddle_rejected'))

    def test_default_disagreement_is_unknown(self):
        result = self.run_readers(Reader('苏A12345', .99), Reader('苏A12346', .99))
        self.assertEqual((result.status, result.text, result.decision), ('unknown', '', 'disagreement'))
        self.assertEqual((result.cnn_text, result.paddle_text), ('苏A12345', '苏A12346'))

    def test_config_defaults_to_single_frame_and_agreement(self):
        config = json.loads((PACKAGE / 'config/plate/inference.json').read_text(encoding='utf-8'))
        self.assertEqual(config['arbitration']['policy'], 'agreement_only')
        self.assertEqual((config['confirmation']['sample_window'], config['confirmation']['required_matches']), (1, 1))

    def test_error_cannot_be_success(self):
        result = self.run_readers(Reader('苏A12345', .9), Reader(error='unavailable'))
        self.assertEqual((result.status, result.text), ('error', ''))
        self.assertEqual(result.cnn_text, '苏A12345')
        self.assertIn('unavailable', result.error)

    def test_invalid_and_missing_second_reader(self):
        result = self.run_readers(Reader('bad', .9), Reader('bad', .9))
        self.assertEqual((result.status, result.decision), ('unknown', 'invalid_outputs'))
        with self.assertRaises(ValueError):
            DualPlateOCR(Reader(), None, self.charsets)

    def test_normalization(self):
        self.assertEqual(normalize_text('苏a·12345'), '苏A12345')
        self.assertEqual(format_plate_text('苏A12345'), '苏A·12345')

    def test_province_three_frame_vote_and_single_frame_option(self):
        def detection(text, score):
            return {'ocr': {'status': 'ok', 'text': text, 'decision': 'agreement', 'cnn_confidence': score}}
        samples = [[detection('苏A12345', .6)], [detection('苏A12345', .9)], [detection('苏A12346', .95)]]
        result = confirm_plate_samples(samples, sample_window=3, required_matches=2)
        self.assertEqual((result['status'], result['text'], result['selected_index']), ('ok', '苏A12345', 1))
        self.assertEqual(confirm_plate_samples(samples[:1], sample_window=3, required_matches=2)['status'], 'not_confirmed')
        self.assertEqual(confirm_plate_samples(samples[:1])['status'], 'ok')

    def test_multiple_plate_frames_do_not_vote(self):
        item = {'ocr': {'status': 'ok', 'text': '苏A12345', 'decision': 'agreement', 'cnn_confidence': .9}}
        self.assertEqual(confirm_plate_samples([[item, item], []])['status'], 'not_confirmed')

    def test_detector_crop_reaches_both_readers(self):
        import numpy as np

        class Scalar:
            def item(self):
                return .91

        class Box:
            xyxy = [np.array([10, 10, 30, 20])]
            conf = [Scalar()]

        class Detector:
            def predict(self, *args, **kwargs):
                return [type('Result', (), {'boxes': [Box()]})()]

        cnn, paddle = Reader('苏A12345', .8), Reader('苏A12345', .9)
        recognizer = PlateRecognizer(Detector(), DualPlateOCR(cnn, paddle, self.charsets), padding=0)
        results = recognizer.infer(np.zeros((40, 50, 3), dtype=np.uint8))
        self.assertEqual(results[0]['bbox'], (10, 10, 30, 20))
        self.assertEqual(results[0]['ocr']['decision'], 'agreement')
        self.assertEqual(cnn.seen[0].shape, (10, 20, 3))
        self.assertIs(cnn.seen[0], paddle.seen[0])


if __name__ == '__main__':
    unittest.main()
