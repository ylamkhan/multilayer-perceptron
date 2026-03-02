"""
visualize.py - Plotting utilities for learning curves and data exploration.
"""

import numpy as np


def plot_learning_curves(history, save_path='learning_curves.png', compare_histories=None):
    """
    Plot loss and accuracy learning curves.
    compare_histories: list of (label, history_dict) for multi-model comparison.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Learning Curves', fontsize=14, fontweight='bold')

    def _plot_metric(ax, key, title, ylabel):
        if compare_histories:
            for label, h in compare_histories:
                ax.plot(h[key], label=f'{label} train')
                ax.plot(h['val_' + key], linestyle='--', label=f'{label} val')
        else:
            ax.plot(history[key], label='training ' + key)
            ax.plot(history['val_' + key], linestyle='--', label='validation ' + key)
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    _plot_metric(axes[0], 'loss', 'Loss', 'Loss')
    _plot_metric(axes[1], 'accuracy', 'Accuracy', 'Accuracy')
    _plot_metric(axes[2], 'recall', 'Recall', 'Recall')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_data_exploration(X, y, save_path='data_exploration.png'):
    """
    Plot histograms for the first 10 features, coloured by diagnosis.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping data exploration plot.")
        return

    # y here is one-hot: col 0 = B, col 1 = M
    true_labels = np.argmax(y, axis=1)
    X_M = X[true_labels == 1]
    X_B = X[true_labels == 0]

    n_features = min(10, X.shape[1])
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Feature Distribution by Diagnosis', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= n_features:
            ax.axis('off')
            continue
        ax.hist(X_B[:, i], bins=20, alpha=0.6, color='steelblue', label='Benign')
        ax.hist(X_M[:, i], bins=20, alpha=0.6, color='tomato', label='Malignant')
        ax.set_title(f'Feature {i+1}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Data exploration plot saved to {save_path}")


def plot_confusion_matrix(y_true_labels, y_pred_labels, save_path='confusion_matrix.png'):
    """Plot a simple confusion matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tp = int(np.sum((y_pred_labels == 1) & (y_true_labels == 1)))
    fp = int(np.sum((y_pred_labels == 1) & (y_true_labels == 0)))
    fn = int(np.sum((y_pred_labels == 0) & (y_true_labels == 1)))
    tn = int(np.sum((y_pred_labels == 0) & (y_true_labels == 0)))

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_yticklabels(['Benign', 'Malignant'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()