import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import compose_v4_grounded_train as grounded
import compose_vehicle_view_arena as arena


class UnplannedEntryTest(unittest.TestCase):
    def test_grounded_entry_fails_before_sources_and_direct_call_fails(self):
        with mock.patch.object(sys, 'argv', ['entry', '--output', 'unused']), \
             mock.patch.object(grounded.bike, 'sources') as sources, mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                grounded.main()
            sources.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            grounded._produce()

    def test_arena_entry_fails_before_sources_and_direct_call_fails(self):
        with mock.patch.object(sys, 'argv', ['entry', '--output', 'datasets/exports/e_bike/new']), \
             mock.patch.object(arena, 'source_inputs') as sources:
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                arena.main()
            sources.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            arena.produce(None, 0, None, None, None, None)


if __name__ == '__main__':
    unittest.main()
