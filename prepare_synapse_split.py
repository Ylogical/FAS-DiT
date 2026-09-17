"""
Write Synapse/split.json, the case-level split the Synapse loader reads.

Input (download and arrange it yourself, see the 'Data' section of README.md):
  Synapse/
    train_npz/caseXXXX_sliceYYY.npz   2D slices, image + label, 512x512, 18 cases
    test_vol_h5/caseXXXX.npy.h5       3D volumes, 12 cases

Splitting:
  * test : the 12 .h5 volumes, exactly as in TransUNet (case-level isolation)
  * train: the 18 training cases, split by case so that slices of one case never
           end up in two splits. --val_cases 0 (the default, and the protocol
           reported in the paper) keeps all 18 for training and leaves no
           validation set; --val_cases N holds N of them out for early stopping.

Usage:
    python prepare_synapse_split.py
    python prepare_synapse_split.py --val_cases 4 --seed 42
"""
import os
import json
import random
import argparse


SYNAPSE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Synapse')

CLASS_NAMES = ['background', 'aorta', 'gallbladder', 'kidney_left',
               'kidney_right', 'liver', 'pancreas', 'spleen', 'stomach']


def parse_args():
    p = argparse.ArgumentParser(description='Generate the Synapse train/val/test split.')
    p.add_argument('--root', type=str, default=SYNAPSE_ROOT,
                   help='Synapse root directory (holds train_npz / test_vol_h5)')
    p.add_argument('--val_cases', type=int, default=0,
                   help='How many of the 18 training cases to hold out for '
                        'validation (0 = the protocol used in the paper)')
    p.add_argument('--seed', type=int, default=42,
                   help='Seed of the case shuffle, so the split is reproducible')
    p.add_argument('--output', type=str, default=None,
                   help='Output path (default: <root>/split.json)')
    return p.parse_args()


def main():
    args = parse_args()

    train_npz_dir = os.path.join(args.root, 'train_npz')
    test_h5_dir   = os.path.join(args.root, 'test_vol_h5')
    for d in (train_npz_dir, test_h5_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    npz_files = [f for f in os.listdir(train_npz_dir) if f.endswith('.npz')]
    all_train_cases = sorted({f.split('_')[0] for f in npz_files})

    test_files = sorted(f for f in os.listdir(test_h5_dir) if f.endswith('.h5'))
    test_cases = [f.split('.')[0] for f in test_files]

    if args.val_cases >= len(all_train_cases):
        raise ValueError(f"--val_cases {args.val_cases} must be smaller than the "
                         f"{len(all_train_cases)} available training cases")

    shuffled = list(all_train_cases)
    random.Random(args.seed).shuffle(shuffled)
    val_cases   = sorted(shuffled[:args.val_cases])
    train_cases = sorted(shuffled[args.val_cases:])

    n_train_slices = sum(1 for f in npz_files if f.split('_')[0] in train_cases)
    n_val_slices   = sum(1 for f in npz_files if f.split('_')[0] in val_cases)

    split = {
        'description': 'Synapse multi-organ segmentation '
                       '(9 classes: background + aorta, gallbladder, left kidney, '
                       'right kidney, liver, pancreas, spleen, stomach)',
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'seed': args.seed,
        'val_cases_n': args.val_cases,
        'train': {'cases': train_cases, 'n_cases': len(train_cases),
                  'n_slices': n_train_slices, 'source': 'train_npz'},
        'val':   {'cases': val_cases, 'n_cases': len(val_cases),
                  'n_slices': n_val_slices, 'source': 'train_npz'},
        'test':  {'cases': test_cases, 'files': test_files,
                  'n_cases': len(test_cases), 'source': 'test_vol_h5 (3D volumes)'},
    }

    out_path = args.output or os.path.join(args.root, 'split.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2)

    print(f"Train cases ({len(train_cases)}): {train_cases}")
    print(f"Val   cases ({len(val_cases)}): {val_cases}")
    print(f"Test  cases ({len(test_cases)}): {test_cases}")
    print(f"Train slices (2D npz): {n_train_slices}")
    print(f"Val   slices (2D npz): {n_val_slices}")
    print(f"Test  volumes (3D h5): {len(test_cases)}")
    if not val_cases:
        print("No validation split: training runs for the full --epochs and "
              "final_model.pth is the model to evaluate.")
    print(f"\nSplit written to {out_path}")


if __name__ == '__main__':
    main()
