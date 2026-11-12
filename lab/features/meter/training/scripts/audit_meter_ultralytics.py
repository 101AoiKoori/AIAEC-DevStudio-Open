"""Validate meter YOLO Pose files using the installed Ultralytics verifier.

No weights, full training, dataset caches, network access, or model downloads.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import v8_transforms
from ultralytics.data.utils import verify_image_label
from ultralytics.utils.loss import v8PoseLoss


class _CaptureMask:
    def __init__(self):
        self.mask = None

    def __call__(self, prediction, target, mask, area):
        self.mask = mask.detach().cpu().tolist()
        return torch.tensor(0.0)


def visibility_behavior():
    loss = object.__new__(v8PoseLoss)
    capture = _CaptureMask()
    loss.keypoint_loss = capture
    loss.bce_pose = torch.nn.BCEWithLogitsLoss()
    keypoints = torch.tensor([[[1., 1., 0.], [1., 1., 1.], [1., 1., 2.]]])
    loss.calculate_keypoints_loss(torch.tensor([[True]]), torch.tensor([[0]]), keypoints,
        torch.tensor([0]), torch.tensor([1.]), torch.tensor([[[0., 0., 10., 10.]]]),
        torch.zeros((1, 1, 3, 3)))
    if capture.mask != [[False, True, True]]:
        raise RuntimeError('Visibility behavior changed; inspect installed Ultralytics')
    return {'0': False, '1': True, '2': True}


def flip_behavior(data):
    class Dataset:
        use_keypoints = True
        use_segments = False
        use_obb = False

    dataset = Dataset()
    dataset.data = data
    hyp = copy.deepcopy(DEFAULT_CFG)
    v8_transforms(dataset, imgsz=1024, hyp=hyp)
    if hyp.fliplr != 0 or hyp.flipud != 0:
        raise RuntimeError('Unexpected default image flips; set fliplr=0 and flipud=0')
    return {'fliplr': hyp.fliplr, 'flipud': hyp.flipud}


def validate_dataset_header(data, root):
    """An omitted path uses the directory containing the dataset YAML."""
    configured_root = Path(data.get('path', root)).resolve()
    if configured_root != root or data['kpt_shape'] != [8, 3] or data['names'] != {0: 'meter'}:
        raise RuntimeError('Dataset YAML root, keypoint shape, or class mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    data = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf8'))
    validate_dataset_header(data, root)
    count = {}
    for split in ('train', 'val', 'test'):
        if data[split] != 'images/' + split:
            raise RuntimeError('Unexpected split path in YAML')
        paths = sorted((root / 'images' / split).glob('*.png'))
        if not paths:
            raise RuntimeError('Empty split ' + split)
        for path in paths:
            label = root / 'labels' / split / (path.stem + '.txt')
            result = verify_image_label((str(path), str(label), '', True, 1, 8, 3))
            image, labels, shape, _, keypoints, missing, found, empty, corrupt, message = result
            if (image is None or missing or not found or empty or corrupt or message or
                    shape != (1024, 1024) or labels.shape != (1, 5) or
                    keypoints.shape != (1, 8, 3)):
                raise RuntimeError('Ultralytics rejected %s: %s' % (path, message))
        count[split] = len(paths)
    report = {'ultralytics_version': ultralytics.__version__, 'pair_counts': count,
              'keypoint_shape': [8, 3], 'position_loss_mask': visibility_behavior(),
              'default_flips_without_flip_idx': flip_behavior(data),
              'scope': 'installed pair verifier and loss-mask/augmentation unit checks; no model training'}
    (root / 'audit_ultralytics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print('ULTRALYTICS_COMPATIBILITY_OK', json.dumps(report))


if __name__ == '__main__':
    main()
