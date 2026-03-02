"""
preprocess.py - Data loading and preprocessing utilities for the breast cancer dataset.
"""

import numpy as np


def load_dataset(filepath):
    """
    Load the Wisconsin breast cancer CSV.
    Columns: id, diagnosis (M/B), feature1..feature30
    Returns X (float array), y_labels (list of 'M'/'B'), feature_names
    """
    ids, labels, features = [], [], []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # First column: id, second: diagnosis, rest: features
            if parts[1] in ('M', 'B'):
                ids.append(parts[0])
                labels.append(parts[1])
                features.append([float(x) for x in parts[2:]])
            else:
                # Might be a header row – skip
                continue

    X = np.array(features, dtype=float)
    y = np.array(labels)
    return X, y


def encode_labels(y):
    """
    Encode ['M','B'] labels to one-hot vectors.
    M -> [0, 1]  (class 1 = malignant)
    B -> [1, 0]  (class 0 = benign)
    """
    one_hot = np.zeros((len(y), 2))
    for i, label in enumerate(y):
        one_hot[i, 1 if label == 'M' else 0] = 1
    return one_hot


class StandardScaler:
    """Z-score normalisation: (x - mean) / std"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1  # avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, filepath):
        np.save(filepath, {'mean': self.mean_, 'std': self.std_}, allow_pickle=True)

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=True).item()
        scaler = cls()
        scaler.mean_ = data['mean']
        scaler.std_ = data['std']
        return scaler


def load_and_preprocess(filepath, scaler=None, fit_scaler=True):
    """
    Full pipeline: load CSV -> encode labels -> standardise features.
    Returns X_scaled, y_onehot, scaler
    """
    X, y = load_dataset(filepath)
    y_oh = encode_labels(y)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y_oh, scaler