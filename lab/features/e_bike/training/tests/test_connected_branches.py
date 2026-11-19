"""No-output contracts for task YAML, clipping and Blender entry relocation."""
import ast
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
from datasets.scripts.e_bike import compose_v4_grounded_train as entry
from datasets.scripts.e_bike import plan_blender_batch as planner


class ConnectedBranchesTest(unittest.TestCase):
    def test_clip_matches_visible_pixels_and_rejects_invisible_actor(self):
        frame = np.zeros((40,40,3), np.uint8)
        sprite = np.full((20,20,4),255,np.uint8)
        mask = entry.place_actor(frame, sprite, 5,30,True)
        self.assertEqual(int(mask.sum()), 300)
        self.assertEqual(int(np.count_nonzero(frame[:,:,0])),300)
        with self.assertRaisesRegex(ValueError,'outside'):
            entry.place_actor(frame,sprite,5,30,False)
        with self.assertRaisesRegex(ValueError,'entirely'):
            entry.place_actor(frame,sprite,-30,30,True)

    def test_person_occluder_must_use_own_configured_root(self):
        spec = dict(foot_x=20,foot_y=30,size_px=20,yaw=0,pitch=0,roll=0,relative_exposure=1)
        root = entry.ROOT/'Archive/Assets/working_sources/digital/person/team_assets'
        with mock.patch('datasets.scripts.common.config.source_path', return_value=root) as resolve, \
             mock.patch.object(Path,'is_file',return_value=True):
            self.assertEqual(entry.validate_people([{'source':'Archive/Assets/working_sources/digital/person/team_assets/a.png'}],[spec]),
                             {'Archive/Assets/working_sources/digital/person/team_assets/a.png'})
            resolve.assert_called_with('e_bike','person_occluder_artwork_root')
            with self.assertRaisesRegex(ValueError,'outside'):
                entry.validate_people([{'source':'datasets/raw/fire/a.png'}],[spec])

    def test_blender_sources_and_mutations_are_inside_guarded_entry(self):
        self.assertTrue(planner.FOREGROUNDS.is_file())
        self.assertTrue(planner.USE.is_file())
        path = Path(entry.__file__).with_name('render_e_bike_blender_batch.py')
        tree = ast.parse(path.read_text(encoding='utf8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
        guard = next(n for n in main.body if isinstance(n, ast.With))
        assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and
                       any(ast.unparse(t).startswith(('scene.render.','scene.camera','scene.cycles.')) for t in n.targets)]
        self.assertTrue(assignments)
        self.assertTrue(all(guard.lineno < n.lineno <= guard.end_lineno for n in assignments))


if __name__ == '__main__':
    unittest.main()
