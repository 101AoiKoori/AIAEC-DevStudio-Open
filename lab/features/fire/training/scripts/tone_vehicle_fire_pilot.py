"""Disabled legacy tone variant of the P12 traffic-signal/building composite."""
import argparse


DISABLED_REASON = (
    'disabled: P12_TL2.png contains traffic-signal targets, while its parent fire pilot composites '
    'synthetic buildings into that frame; the resulting co-occurrence contradicts the confirmed route layout'
)


def generate(source=None, output=None):
    raise ValueError(DISABLED_REASON)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.parse_args()
    parser.error(DISABLED_REASON)
