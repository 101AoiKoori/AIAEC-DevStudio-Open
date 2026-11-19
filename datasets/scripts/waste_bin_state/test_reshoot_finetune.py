"""Split and color-state-size allocation regressions for bin fine-tuning."""
from collections import Counter
import unittest

from datasets.scripts.waste_bin_state.build_reshoot_finetune import background_order, choose_replay, composition_schedule


class ReshootFineTuneTests(unittest.TestCase):
    def test_replay_keeps_real_photos_negative_quota_and_color_states(self):
        rows = []
        for i in range(200):
            empty = i < 24
            rows.append({'split': 'train', 'source_kind': 'direct_empty_negative' if empty else
                         'direct_real' if i < 28 else 'real_background_composite' if i % 2 else 'blender_route_composite',
                         'source_record_id': f'source_{i:03d}',
                         'bins': [] if empty else [{'color': ('gray', 'blue', 'red', 'green')[i % 4],
                                                   'state': 'open' if (i // 4) % 2 else 'closed'}]})
        selected = choose_replay(rows)
        self.assertEqual(len(selected), 128)
        self.assertEqual(sum(not r['bins'] for r in selected), 16)
        self.assertTrue(all(r in selected for r in rows[24:28]))
        self.assertEqual(len({r['source_record_id'] for r in selected}), 128)
        self.assertEqual({(b['color'], b['state']) for r in selected for b in r['bins']},
                         {(color, state) for color in ('gray', 'blue', 'red', 'green') for state in ('open', 'closed')})
        self.assertEqual(selected, choose_replay(rows))

    def test_non_train_replay_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only Train'):
            choose_replay([{'split': 'test'}])

    def test_replay_prioritizes_scarce_real_empty_scenes(self):
        rows = [{'split': 'train', 'source_kind': 'direct_empty_negative' if i < 4 else
                 'blender_empty_negative', 'source_record_id': f'empty_{i}', 'bins': []}
                for i in range(100)]
        selected = choose_replay(rows, total=64, negative_count=64)
        self.assertEqual(len(selected), 64)
        self.assertEqual(sum(r['source_kind'] == 'direct_empty_negative' for r in selected), 4)

    def test_truncated_photo_stays_out_of_cutouts_and_sizes_are_balanced(self):
        photos = [{'id': f'closed_{i}', 'color': 'gray', 'state': 'closed', 'composite_eligible': i != 3} for i in range(4)]
        photos += [{'id': f'open_{i}', 'color': 'gray', 'state': 'open', 'composite_eligible': True} for i in range(4)]
        heights = [60, 75, 90, 110, 160, 220]
        jobs = composition_schedule(photos, heights)
        self.assertEqual(len(jobs), 96)
        self.assertEqual(Counter((r['state'], height) for r, height, _ in jobs),
                         Counter({(state, height): 8 for state in ('open', 'closed') for height in heights}))
        counts = Counter(r['id'] for r, _, _ in jobs)
        self.assertNotIn('closed_3', counts)
        self.assertTrue(all(counts[f'closed_{i}'] == 16 for i in range(3)))
        self.assertTrue(all(counts[f'open_{i}'] == 12 for i in range(4)))

    def test_four_colors_remain_state_size_balanced_with_unequal_source_counts(self):
        photos = []
        for color in ('gray', 'red', 'blue', 'green'):
            for state in ('open', 'closed'):
                # Five eligible views exercises a quota that does not divide 48.
                photos.extend({'id': f'{color}_{state}_{i}', 'color': color, 'state': state}
                              for i in range(5 if color == 'red' else 4))
        heights = [60, 75, 90, 110, 160, 220]
        jobs = composition_schedule(photos, heights)
        self.assertEqual(len(jobs), 384)
        self.assertEqual(Counter((r['color'], r['state'], h) for r, h, _ in jobs),
                         Counter({(color, state, h): 8 for color in ('gray', 'red', 'blue', 'green')
                                  for state in ('open', 'closed') for h in heights}))
        for color in ('gray', 'red', 'blue', 'green'):
            for state in ('open', 'closed'):
                use = Counter(r['id'] for r, _, _ in jobs if r['color'] == color and r['state'] == state)
                self.assertLessEqual(max(use.values()) - min(use.values()), 1)
                for identity in use:
                    self.assertEqual({h for r, h, _ in jobs if r['id'] == identity}, set(heights))

    def test_uncovered_color_state_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'red:closed'):
            composition_schedule([{'color': 'red', 'state': 'open'}], [60, 90])

    def test_background_recipes_balance_domains_and_repeat_across_color_states(self):
        backgrounds = [({'source_kind': 'direct_empty_negative', 'image': f'camera_{i}'}, None, None)
                       for i in range(4)]
        backgrounds += [({'source_kind': 'blender_empty_negative', 'image': f'blender_{i}'}, None, None)
                        for i in range(60)]
        recipes = [background_order(backgrounds, i) for i in range(48)]
        self.assertEqual(Counter(kind for kind, _, _ in recipes), {'camera': 24, 'blender': 24})
        for height_index in range(6):
            self.assertEqual(Counter(recipes[i][0] for i in range(height_index, 48, 6)),
                             {'camera': 4, 'blender': 4})
        self.assertEqual(len({name for kind, name, _ in recipes if kind == 'camera'}), 4)
        for _ in range(8):
            self.assertEqual([(kind, name) for kind, name, _ in recipes],
                             [background_order(backgrounds, i)[:2] for i in range(48)])
        for requested, _, ordered in recipes:
            kinds = ['blender' if item[0]['source_kind'] == 'blender_empty_negative' else 'camera'
                     for item in ordered]
            first_alternative = kinds.index('blender' if requested == 'camera' else 'camera')
            self.assertTrue(all(kind == requested for kind in kinds[:first_alternative]))
            self.assertEqual(len(ordered), 64)

    def test_missing_background_domain_falls_back_without_inventing_sources(self):
        backgrounds = [({'source_kind': 'blender_empty_negative', 'image': 'only_blender'}, None, None)]
        requested, preferred, ordered = background_order(backgrounds, 0)
        self.assertEqual((requested, preferred, len(ordered)), ('camera', 'only_blender', 1))


if __name__ == '__main__':
    unittest.main()
