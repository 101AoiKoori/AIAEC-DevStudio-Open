#!/usr/bin/env python3
"""Compare shared CNN against the original provincial vision.py and checkpoint."""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PACKAGE = ROOT / 'ws/src/aec_vision'
sys.path.insert(0, str(PACKAGE / 'src'))
import torch
from aec_vision.plate import CNNReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--province-vision', required=True, help='Workspace-local adapted provincial vision.py')
    parser.add_argument('--province-weights', required=True, help='Trusted provincial CNN checkpoint')
    args = parser.parse_args()
    source = Path(args.province_vision).resolve()
    try:
        source.relative_to(ROOT.resolve())
    except ValueError:
        parser.error('province-vision must be an independently maintained script inside the workspace')
    spec = importlib.util.spec_from_file_location('province_vision', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = torch.load(args.province_weights, map_location='cpu', weights_only=False)
    original = module.PlateOCRBase.create([len(group) for group in checkpoint['charsets']])
    original.load_state_dict(checkpoint['model'], strict=True)
    original.eval()
    migrated = CNNReader(PACKAGE / 'models/plate_ocr_best.pt', PACKAGE / 'config/plate/charset.json')
    torch.manual_seed(42)
    image = torch.rand(1, 3, 64, 224)
    with torch.inference_mode():
        old_heads, new_heads = original(image), migrated.model(image)
    if len(old_heads) != 7 or not all(torch.equal(a, b) for a, b in zip(old_heads, new_heads)):
        raise SystemExit('CNN head outputs differ from provincial model')
    print('All seven CNN head outputs match the provincial model.')


if __name__ == '__main__':
    main()
