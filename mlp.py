"""
mlp.py - Single entry point combining split / train / predict phases.

Usage:
  python mlp.py split   --dataset data.csv [--ratio 0.8] [--seed 42]
  python mlp.py train   --dataset data_train.csv [--valid data_valid.csv]
                        [--layer 24 24] [--epochs 100] [--batch_size 8]
                        [--learning_rate 0.001] [--loss categoricalCrossentropy]
                        [--optimizer adam] [--early_stopping] [--patience 15]
                        [--seed 42] [--model saved_model.npy]
  python mlp.py predict --dataset data_valid.csv [--model saved_model.npy]
  python mlp.py explore --dataset data.csv           # bonus: data exploration plots

Quick start:
  python mlp.py split   --dataset data.csv
  python mlp.py train   --dataset data_train.csv --valid data_valid.csv
  python mlp.py predict --dataset data_valid.csv
"""

import argparse
import sys


def cmd_split(argv):
    import numpy as np
    from split import split_dataset

    parser = argparse.ArgumentParser(prog='mlp split')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    split_dataset(args.dataset, args.ratio, args.seed)


def cmd_train(argv):
    # delegate to train.py main
    import sys
    sys.argv = ['train.py'] + argv
    from train import main
    main()


def cmd_predict(argv):
    import sys
    sys.argv = ['predict.py'] + argv
    from predict import main
    main()


def cmd_explore(argv):
    import argparse
    import numpy as np
    from preprocess import load_and_preprocess
    from visualize import plot_data_exploration

    parser = argparse.ArgumentParser(prog='mlp explore')
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args(argv)

    X, y, scaler = load_and_preprocess(args.dataset, fit_scaler=True)
    print(f"Dataset shape: {X.shape}")
    print(f"Malignant: {int(y[:,1].sum())}  Benign: {int(y[:,0].sum())}")
    plot_data_exploration(X, y, 'data_exploration.png')


COMMANDS = {
    'split': cmd_split,
    'train': cmd_train,
    'predict': cmd_predict,
    'explore': cmd_explore,
}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    COMMANDS[cmd](sys.argv[2:])