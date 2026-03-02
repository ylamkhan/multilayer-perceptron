"""
train.py - Training program for the Multilayer Perceptron.

Usage:
  python train.py --dataset data_train.csv [--valid data_valid.csv]
                  [--layer 24 24] [--epochs 100] [--batch_size 8]
                  [--learning_rate 0.001] [--loss categoricalCrossentropy]
                  [--optimizer adam] [--early_stopping] [--patience 15]
                  [--seed 42] [--model saved_model.npy]

Examples:
  python train.py --dataset data_train.csv --valid data_valid.csv
  python train.py --dataset data_train.csv --valid data_valid.csv --layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
"""

import argparse
import sys
import numpy as np

from preprocess import load_and_preprocess, StandardScaler
from network import DenseLayer, MLP
from visualize import plot_learning_curves


def main():
    parser = argparse.ArgumentParser(description='Train a Multilayer Perceptron')
    parser.add_argument('--dataset', type=str, required=True, help='Training CSV file')
    parser.add_argument('--valid', type=str, default=None, help='Validation CSV file (optional)')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24],
                        help='Hidden layer sizes (default: 24 24)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='categoricalCrossentropy',
                        choices=['categoricalCrossentropy', 'binaryCrossentropy'])
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'rmsprop', 'sgd'])
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='saved_model.npy',
                        help='Output path for saved model')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable learning curve plots')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Load & preprocess training data ──────────────────────────────────────
    print(f"Loading training data from {args.dataset}...")
    X_train, y_train, scaler = load_and_preprocess(args.dataset, fit_scaler=True)
    print(f"x_train shape : {X_train.shape}")

    # Save scaler for use during prediction
    scaler.save('scaler.npy')

    # ── Load validation data ──────────────────────────────────────────────────
    if args.valid:
        X_valid, y_valid, _ = load_and_preprocess(args.valid, scaler=scaler, fit_scaler=False)
    else:
        # Auto-split 80/20
        n = len(X_train)
        split = int(n * 0.8)
        idx = np.random.permutation(n)
        X_valid = X_train[idx[split:]]
        y_valid = y_train[idx[split:]]
        X_train = X_train[idx[:split]]
        y_train = y_train[idx[:split]]
        print("No validation file provided – using auto 80/20 split from training data.")

    print(f"x_valid shape : {X_valid.shape}")
    print()

    # ── Build network ─────────────────────────────────────────────────────────
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]  # 2

    layers = [
        DenseLayer(input_dim, activation='sigmoid'),  # input layer (passthrough shape)
    ]
    for units in args.layer:
        layers.append(DenseLayer(units, activation='sigmoid', weights_initializer='heUniform'))
    layers.append(DenseLayer(output_dim, activation='softmax', weights_initializer='heUniform'))

    model = MLP(
        layers=layers,
        loss=args.loss,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        early_stopping=args.early_stopping,
        patience=args.patience,
    )

    print(f"Network architecture: {input_dim} -> {' -> '.join(str(u) for u in args.layer)} -> {output_dim}")
    print(f"Optimizer: {args.optimizer} | LR: {args.learning_rate} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    if args.early_stopping:
        print(f"Early stopping enabled (patience={args.patience})")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    model.fit(X_train, y_train, X_valid, y_valid,
              epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(args.model)

    # ── Plot learning curves ──────────────────────────────────────────────────
    if not args.no_plot:
        plot_learning_curves(model.history, save_path='learning_curves.png')
        print("Learning curves saved to learning_curves.png")


if __name__ == '__main__':
    main()