"""
split.py - Split the dataset into training and validation sets.
Usage: python split.py [--dataset data.csv] [--ratio 0.8] [--seed 42]
"""

import argparse
import numpy as np
import sys
import os


def load_csv(filepath):
    """Load CSV file and return header and data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line.split(','))
    return data


def save_csv(filepath, rows):
    """Save rows to CSV."""
    with open(filepath, 'w') as f:
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')


def split_dataset(filepath, ratio=0.8, seed=42):
    np.random.seed(seed)

    data = load_csv(filepath)
    if not data:
        print("Error: empty dataset", file=sys.stderr)
        sys.exit(1)

    # Detect if first row is a header (non-numeric first field)
    has_header = False
    try:
        float(data[0][0])
    except ValueError:
        has_header = True

    header = None
    rows = data
    if has_header:
        header = data[0]
        rows = data[1:]

    n = len(rows)
    indices = np.random.permutation(n)
    split = int(n * ratio)
    train_idx = indices[:split]
    valid_idx = indices[split:]

    train_rows = [rows[i] for i in train_idx]
    valid_rows = [rows[i] for i in valid_idx]

    if header:
        train_rows = [header] + train_rows
        valid_rows = [header] + valid_rows

    base = os.path.splitext(filepath)[0]
    train_path =  base + '_train.csv'
    valid_path =  base + '_valid.csv'

    save_csv(train_path, train_rows)
    save_csv(valid_path, valid_rows)

    print(f"Dataset split with ratio {ratio} (seed={seed})")
    print(f"  Training set:   {len(train_rows) - (1 if header else 0)} samples -> {train_path}")
    print(f"  Validation set: {len(valid_rows) - (1 if header else 0)} samples -> {valid_path}")
    return train_path, valid_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/validation sets')
    parser.add_argument('--dataset', type=str, default='./dataset/data.csv', help='Path to CSV dataset')
    parser.add_argument('--ratio', type=float, default=0.8, help='Training ratio (default 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    split_dataset(args.dataset, args.ratio, args.seed)