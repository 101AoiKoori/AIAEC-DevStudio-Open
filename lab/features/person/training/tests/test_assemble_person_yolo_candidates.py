import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import assemble_person_yolo_candidates as packer


class PersonPackageTest(unittest.TestCase):
    def test_layout_and_provenance_without_training(self):
        old = packer.ROOT, packer.EXPORTS, packer.EXPECTED
        with tempfile.TemporaryDirectory() as temp:
            try:
                root = Path(temp).resolve()
                packer.ROOT, packer.EXPORTS = root, root/'exports'
                packer.EXPECTED = {'train': 1, 'val': 1, 'test': 1}
                train, evaluation = packer.EXPORTS/'source_train', packer.EXPORTS/'source_eval'
                train.mkdir(parents=True)
                evaluation.mkdir()
                (train/'summary.json').write_text(json.dumps({'status':'complete_candidates',
                    'completed':1,'source_group':'train_video'}), encoding='utf8')
                (evaluation/'summary.json').write_text(json.dumps({'status':'complete_candidates',
                    'completed':{'val':1,'test':1}, 'source_groups':{'val':['v'], 'test':['t']}}), encoding='utf8')
                for split in ('train', 'val', 'test'):
                    base = train if split == 'train' else evaluation/split
                    base.mkdir(exist_ok=True)
                    (base/'image.png').write_bytes(b'png-source')
                    (base/'label.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf8')
                    row = {'image':'image.png', 'label':'label.txt'}
                    if split == 'train': row.update({'scenario':'single', 'split':'train_candidate'})
                    else: row.update({'scenario':{'kind':'single'}, 'split_candidate':split})
                    (base/'manifest.jsonl').write_text(json.dumps(row)+'\n', encoding='utf8')
                output = packer.EXPORTS/'package'
                summary = packer.package(train,evaluation,output)
                self.assertEqual(summary['completed'], {'train':1,'val':1,'test':1})
                self.assertEqual(summary['train_permission'],'not_authorized')
                config = yaml.safe_load((output/'data.yaml').read_text(encoding='utf8'))
                self.assertNotIn('path', config)
                self.assertEqual(config['names'], {0:'community',1:'noncommunity'})
                for split in ('train','val','test'):
                    self.assertTrue((output/config[split]).is_dir())
                    self.assertEqual(len(list((output/'labels'/split).glob('*.txt'))), 1)
                    self.assertTrue((output/('manifest_'+split+'.jsonl')).is_file())
            finally:
                packer.ROOT, packer.EXPORTS, packer.EXPECTED = old


if __name__ == '__main__':
    unittest.main()
