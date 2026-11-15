import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import produce_integrated_fire as producer
import produce_facade_crop_candidates as inspection


class CandidateStateTests(unittest.TestCase):
    def test_ai_review_is_not_promoted_to_verified(self):
        entry = {'image': 'crop.png', 'quads': [[[1, 1], [21, 1], [21, 31], [1, 31]]],
                 'supported_counts': [0, 1], 'review_state': 'ai_reviewed_empty_candidate'}
        state = {'parameters': {'source_image': 'crop.png', 'quads': entry['quads'], 'crop_xyxy': None},
                 'target': {'kind': 'fire_sticker', 'count': 1}}
        with tempfile.TemporaryDirectory(dir=inspection.ROOT / 'tmp') as temporary:
            source = Path(temporary) / 'source.json'
            with patch.object(producer, '_generate'):
                producer.generate_for_state(state, {'crop.png': entry}, 431, source, Path('icon'), Path('out'), None)
            row = json.loads(source.read_text(encoding='utf8'))[0]
        self.assertEqual(row['review_state'], 'ai_reviewed_empty_candidate')
        self.assertIs(row['verified_empty'], False)

    def test_prior_and_cost_change_allocation(self):
        entries, _ = inspection.load_entries()
        declaration = inspection.make_plan(entries)
        weighted = inspection.plan_production(declaration, inspection.ROOT)
        for state in declaration['strata']:
            state['prior'] = {'alpha': 1, 'beta': 1}
            state['exposure_weight'] = state['failure_cost'] = 1
        equal = inspection.plan_production(declaration, inspection.ROOT)
        self.assertNotEqual([s['planned_count'] for s in weighted['strata']],
                            [s['planned_count'] for s in equal['strata']])
        self.assertTrue(all(s['min_count'] < s['max_count'] for s in weighted['strata']))


if __name__ == '__main__':
    unittest.main()
