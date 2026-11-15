"""Selection of scene batches preserves provenance and enforces background ratios."""
import importlib.util
import json
from unittest import mock
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media/tests')]
spec = importlib.util.spec_from_file_location('integrate_sticker_scene_v4', ROOT / 'lab/features/waste_bin/training/scripts/integrate_sticker_scene_v4.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import test_media_index_sqlite as fixtures
from lab.training_common import production_guard


def original():
    rows = []
    for i in range(54):
        rows.append({'id': f'old-real-{i}', 'image': f'{m.PRIOR}/images/train/{i}.png', 'split': 'train',
                     'parent_video': m.PRIOR_VIDEO, 'tasks': {'waste_sticker': {'state': 'complete_positive'}}})
    rows.append({'id': 'old-flat', 'image': 'datasets/old.png', 'split': 'test'})
    return {'samples': rows}


def batch():
    rows, refs = [], {}
    for split in m.SPLITS:
        for i in range(30):
            image = f'datasets/exports/new/images/{split}/{i}.png'
            row = {'image': image, 'label': f'datasets/exports/new/labels/{split}/{i}.txt',
                   'split': split, 'class_id': i % 24, 'source_name': '纸箱',
                   'source_image': f'datasets/render/{split}/{i}.png', 'arena_master': 'datasets/arena.blend',
                   'source_camera': {'location': [i % 2, 0, 0], 'rotation': [0, 0, 0], 'lens_mm': 22},
                   'tone': {'normal_or_dark': 'dark' if i == 0 else 'normal'}}
            refs[image] = [{'manifest': 'datasets/new.json', 'record_index': len(rows), 'relation': 'describes'}]
            rows.append(row)
    image = 'datasets/exports/new/images/train/real.png'
    refs[image] = [{'manifest': 'datasets/new.json', 'record_index': len(rows), 'relation': 'describes'}]
    rows.append({'image': image, 'label': 'datasets/exports/new/labels/train/real.txt', 'split': 'train',
                 'class_id': 1, 'source_name': '纸箱', 'background_image': 'datasets/real.png',
                 'parent_video': 'datasets/real2.mp4', 'brightness': 'normal'})
    return [('datasets/new.json', rows)], refs


def assets(batches):
    return {row['image']: ['source.png', 'cache.npz'] for _, rows in batches for row in rows
            if row.get('class_id') is not None}


class TestSelection(unittest.TestCase):
    def test_reselects_only_prior_video_and_records_backgrounds(self):
        source = original()
        batches, refs = batch()
        selected, report = m.reset_selection(source, batches, refs, artwork_by_image=assets(batches))
        self.assertEqual(source['samples'][-1]['split'], 'test')
        self.assertEqual(next(x for x in selected['samples'] if x['id'] == 'old-flat')['split'], 'unassigned')
        self.assertEqual(report['train']['images'], 85)
        self.assertEqual(report['val']['images'], 30)
        self.assertEqual(report['test']['images'], 30)
        self.assertEqual(report['train']['non_blender_positive_videos'], {m.PRIOR_VIDEO: 54, 'datasets/real2.mp4': 1})
        sample = next(x for x in selected['samples'] if x['id'].startswith('sticker-scene:'))
        self.assertEqual(sample['generation_method'], 'composite')
        self.assertEqual(sample['image_processing']['background_origin'], 'synthetic')
        self.assertEqual(sample['indexed_source_references'][0]['record_index'], 0)
        self.assertEqual(sample['source_assets'], ['source.png', 'cache.npz'])
        self.assertNotIn('generation_record', sample)

    def test_real_positive_preserves_sources_negative_has_none(self):
        batches, refs = batch()
        entries = batches[0][1]
        positive = entries[-1]
        negative = dict(positive, image='datasets/exports/new/images/val/negative.png',
                        label='datasets/exports/new/labels/val/negative.txt', split='val',
                        class_id=None, source_name=None)
        refs[negative['image']] = [{'manifest': 'datasets/new.json', 'record_index': len(entries), 'relation': 'describes'}]
        entries.append(negative)
        source_map = assets(batches)
        source_map[positive['image']] = [m.ARTWORK_DIR + '/纸箱.png', m.CACHE_DIR + '/纸箱.npz']
        selected, _ = m.reset_selection(original(), batches, refs, artwork_by_image=source_map)
        by_image = {row['image']: row for row in selected['samples'] if row.get('image')}
        self.assertEqual(by_image[positive['image']]['source_assets'], source_map[positive['image']])
        self.assertEqual(by_image[negative['image']]['source_assets'], [])
        self.assertEqual(by_image[negative['image']]['image_processing']['background_origin'], 'direct_camera')

    def test_blender_only_with_two_actual_cameras(self):
        batches, refs = batch()
        batches[0][1].pop()
        selected, report = m.reset_selection(original(), batches, refs, retain_prior=False, artwork_by_image=assets(batches))
        self.assertTrue(all(x['background']['kind'] == 'blender' for x in selected['samples']
                            if x.get('split') in m.SPLITS))
        self.assertEqual(report['train']['blender_background_fraction'], 1.0)

    def test_rejects_one_camera(self):
        batches, refs = batch()
        for row in batches[0][1]:
            if 'source_camera' in row:
                row['source_camera']['location'] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'at least two camera view groups'):
            m.reset_selection(original(), batches, refs, artwork_by_image=assets(batches))

    def test_positive_artwork_and_cache_sources_and_negative(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temp:
            root = Path(temp)
            for directory in (root / m.ARTWORK_DIR, root / m.CACHE_DIR):
                directory.mkdir(parents=True)
            (root / m.ARTWORK_DIR / '纸箱.png').write_bytes(b'icon')
            (root / m.CACHE_DIR / '纸箱.npz').write_bytes(b'cache')
            row = {'class_id': 0, 'source_name': '纸箱', 'image': 'sample.png'}
            expected = [m.ARTWORK_DIR + '/纸箱.png', m.CACHE_DIR + '/纸箱.npz']
            self.assertEqual(m.artwork_paths(root, row, ['纸箱']), expected)
            self.assertEqual(m.artwork_paths(root, {'class_id': None}, ['纸箱']), [])
            with self.assertRaisesRegex(ValueError, 'class id and generator class name disagree'):
                m.artwork_paths(root, dict(row, source_name='苹果'), ['纸箱'])
            (root / m.CACHE_DIR / '纸箱.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'artwork/cache is missing'):
                m.artwork_paths(root, row, ['纸箱'])

    def test_dark_low_saturation_explicit_brightness(self):
        self.assertEqual(m.tone_conditions({'brightness': 'dark_low_saturation'}),
                         ['dark', 'low_light', 'low_saturation'])

    def test_real_guard_refusal_and_reuse(self):
        fix = fixtures.CacheTests('runTest')
        fix.setUp()
        self.addCleanup(fix.doCleanups)
        root = fix.root
        shard = root / fix.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        item['modified_ns'] = fix.image.stat().st_mtime_ns
        shard.write_text(json.dumps(item, ensure_ascii=False) + '\n', encoding='utf8')
        fix.write_catalog()
        source, config, distribution, manifest = (root / 'source.json', root / 'config.yaml',
                                                   root / 'distribution.json', root / 'batch.json')
        (root / 'label.txt').write_text('', encoding='utf8')
        (root / 'classes.yaml').write_text('items:\n' + ''.join(f'- name: class{i}\n' for i in range(24)), encoding='utf8')
        source.write_text(json.dumps({'samples': []}), encoding='utf8')
        config.write_text('data_overrides:\n  train: []\n', encoding='utf8')
        distribution.write_text(json.dumps({'additional_records': []}), encoding='utf8')
        image = fix.image.relative_to(root).as_posix()
        row = {'image': image, 'source_image': image, 'arena_master': 'arena.blend', 'label': 'label.txt',
               'source_camera': {'location':[0,0,1], 'rotation':[1,0,0], 'lens_mm':22}}
        manifest.write_text(json.dumps({'records': [row]}), encoding='utf8')
        script = root / 'script.py'; script.write_text('# fixture', encoding='utf8')
        gate_file = root / 'lab/scripts/data/v4_source_gate.py'; gate_file.parent.mkdir(exist_ok=True)
        gate_file.write_text('# fixture', encoding='utf8')
        patches = [mock.patch.object(m, 'ROOT', root), mock.patch.object(m, 'SOURCE', 'source.json'),
                   mock.patch.object(m, 'CONFIG', 'config.yaml'), mock.patch.object(m, 'DISTRIBUTION', 'distribution.json'),
                   mock.patch.object(m, 'CLASSES', 'classes.yaml'),
                   mock.patch.object(m, '__file__', str(script)), mock.patch.object(m.gate, '_selection_files', return_value=set()),
                   mock.patch.object(m.gate, 'audit', return_value={'status':'passed'}), mock.patch.object(m, 'inspect_records'),
                   mock.patch.object(m, 'reset_selection', return_value=({'samples':[{'id':'new'}]}, {})),
                   mock.patch.object(m, 'configured_copy', return_value=({'changed':True}, {'changed':True})),
                   mock.patch.object(production_guard, '_index', return_value=fixtures.cache)]
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)
        before = source.read_text(encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            m.main(['--manifest', 'batch.json'])
        self.assertEqual(source.read_text(encoding='utf8'), before)
        self.assertEqual(fix.call(['sync'])[0], 0)
        m.main(['--manifest', 'batch.json'])
        stamp = source.stat().st_mtime_ns
        m.main(['--manifest', 'batch.json'])
        self.assertEqual(source.stat().st_mtime_ns, stamp)

    def test_rejects_one_real_group(self):
        batches, refs = batch()
        batches[0][1].pop()
        with self.assertRaisesRegex(ValueError, 'at least two actual parent groups'):
            m.reset_selection(original(), batches, refs, artwork_by_image=assets(batches))

    def test_rejects_missing_reference(self):
        batches, refs = batch()
        refs.clear()
        with self.assertRaisesRegex(ValueError, 'indexed batch record missing'):
            m.reset_selection(original(), batches, refs, artwork_by_image=assets(batches))


if __name__ == '__main__':
    unittest.main()
