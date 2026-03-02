"""
predict.py - Prediction program for the Multilayer Perceptron.

Usage:
  python predict.py --dataset data_valid.csv [--model saved_model.npy] [--scaler scaler.npy]
"""

import argparse
import numpy as np
from preprocess import load_and_preprocess, StandardScaler
from network import MLP


def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy as specified in the subject."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # Use malignant probability (class index 1)
    p = y_pred[:, 1]
    y = y_true[:, 1]
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    parser = argparse.ArgumentParser(description='Run predictions with a trained MLP')
    parser.add_argument('--dataset', type=str, required=True, help='CSV file to predict on')
    parser.add_argument('--model', type=str, default='saved_model.npy', help='Path to saved model')
    parser.add_argument('--scaler', type=str, default='scaler.npy', help='Path to saved scaler')
    args = parser.parse_args()

    # ── Load scaler & preprocess ──────────────────────────────────────────────
    print(f"Loading scaler from {args.scaler}...")
    scaler = StandardScaler.load(args.scaler)

    print(f"Loading dataset from {args.dataset}...")
    X, y_true, _ = load_and_preprocess(args.dataset, scaler=scaler, fit_scaler=False)
    print(f"x shape : {X.shape}\n")

    # ── Load model & predict ──────────────────────────────────────────────────
    print(f"Loading model from {args.model}...")
    model = MLP.load(args.model)

    y_pred = model.predict(X)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)

    accuracy = np.mean(pred_labels == true_labels)

    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))

    precision = tp / (tp + fp + 1e-15)
    recall    = tp / (tp + fn + 1e-15)
    f1        = 2 * precision * recall / (precision + recall + 1e-15)
    bce       = binary_cross_entropy(y_true, y_pred)

    print(f"{'─'*40}")
    print(f"  Samples evaluated : {len(y_true)}")
    print(f"  Binary Cross-Entropy (E) : {bce:.6f}")
    print(f"  Accuracy                 : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision                : {precision:.4f}")
    print(f"  Recall                   : {recall:.4f}")
    print(f"  F1-Score                 : {f1:.4f}")
    print(f"{'─'*40}")
    print(f"  Confusion Matrix:")
    print(f"            Predicted B  Predicted M")
    print(f"  True B       {tn:4d}         {fp:4d}")
    print(f"  True M       {fn:4d}         {tp:4d}")
    print(f"{'─'*40}")

    # Per-sample output
    label_map = {0: 'B', 1: 'M'}
    print("\nSample predictions (first 10):")
    print(f"  {'Index':>6}  {'True':>5}  {'Pred':>5}  {'P(M)':>8}  {'Match':>5}")
    for i in range(min(10, len(y_true))):
        match = '✓' if pred_labels[i] == true_labels[i] else '✗'
        print(f"  {i:>6}  {label_map[true_labels[i]]:>5}  {label_map[pred_labels[i]]:>5}  {y_pred[i,1]:>8.4f}  {match:>5}")


if __name__ == '__main__':
    main()