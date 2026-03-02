"""
network.py - Multilayer Perceptron implementation from scratch.
Includes: DenseLayer, activations, losses, optimizers, and the MLP model.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Dummy derivative (softmax + cross-entropy combined in backward pass)
def softmax_derivative(x):
    return np.ones_like(x)


ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'softmax': (softmax, softmax_derivative),
}


# ─────────────────────────────────────────────────────────────────────────────
# Weight initializers
# ─────────────────────────────────────────────────────────────────────────────

def he_uniform(fan_in, fan_out):
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def xavier_uniform(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def random_normal(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out) * 0.01

INITIALIZERS = {
    'heUniform': he_uniform,
    'xavierUniform': xavier_uniform,
    'randomNormal': random_normal,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dense Layer
# ─────────────────────────────────────────────────────────────────────────────

class DenseLayer:
    def __init__(self, units, activation='sigmoid', weights_initializer='xavierUniform'):
        self.units = units
        self.activation_name = activation
        self.activation_fn, self.activation_deriv = ACTIVATIONS[activation]
        self.initializer = INITIALIZERS.get(weights_initializer, xavier_uniform)

        self.W = None
        self.b = None

        # cache for backprop
        self.input = None
        self.z = None
        self.a = None

        # Adam / momentum states
        self.mW = None
        self.vW = None
        self.mb = None
        self.vb = None

    def initialize(self, input_dim):
        self.W = self.initializer(input_dim, self.units)
        self.b = np.zeros((1, self.units))
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.input = x
        self.z = x @ self.W + self.b
        self.a = self.activation_fn(self.z)
        return self.a

    def backward(self, delta):
        """
        delta: gradient flowing from the next layer (dL/da for this layer)
        Returns dL/d(input) to pass to previous layer.
        """
        if self.activation_name == 'softmax':
            # Combined softmax + cross-entropy: delta is already dL/dz
            dz = delta
        else:
            dz = delta * self.activation_deriv(self.z)

        m = self.input.shape[0]
        self.dW = self.input.T @ dz / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        return dz @ self.W.T


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def categorical_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_cross_entropy_grad(y_true, y_pred):
    # For softmax output layer
    return y_pred - y_true

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

LOSSES = {
    'categoricalCrossentropy': (categorical_cross_entropy, categorical_cross_entropy_grad),
    'binaryCrossentropy': (binary_cross_entropy, None),
}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, layers, loss='categoricalCrossentropy',
                 learning_rate=0.01, optimizer='adam',
                 beta1=0.9, beta2=0.999, epsilon=1e-8,
                 early_stopping=False, patience=10):
        self.layers = layers
        self.loss_name = loss
        self.loss_fn, self.loss_grad = LOSSES[loss]
        self.lr = learning_rate
        self.optimizer_name = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.early_stopping = early_stopping
        self.patience = patience
        self.t = 0  # Adam timestep

        # History
        self.history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'precision': [], 'val_precision': [],
            'recall': [], 'val_recall': [],
        }

    def _build(self, input_dim):
        """Initialize all layer weights."""
        dim = input_dim
        for layer in self.layers:
            layer.initialize(dim)
            dim = layer.units

    def _forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _backward(self, y_true, y_pred):
        delta = self.loss_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def _update_weights(self):
        self.t += 1
        for layer in self.layers:
            if self.optimizer_name == 'adam':
                # Adam update
                layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
                layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * layer.dW ** 2
                mW_hat = layer.mW / (1 - self.beta1 ** self.t)
                vW_hat = layer.vW / (1 - self.beta2 ** self.t)
                layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)

                layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
                layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * layer.db ** 2
                mb_hat = layer.mb / (1 - self.beta1 ** self.t)
                vb_hat = layer.vb / (1 - self.beta2 ** self.t)
                layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)

            elif self.optimizer_name == 'rmsprop':
                decay = 0.9
                layer.vW = decay * layer.vW + (1 - decay) * layer.dW ** 2
                layer.W -= self.lr * layer.dW / (np.sqrt(layer.vW) + self.epsilon)
                layer.vb = decay * layer.vb + (1 - decay) * layer.db ** 2
                layer.b -= self.lr * layer.db / (np.sqrt(layer.vb) + self.epsilon)

            else:  # SGD
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def _compute_metrics(self, y_true, y_pred):
        """Compute accuracy, precision, recall."""
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(pred_labels == true_labels)

        # Binary: class 1 = malignant
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))

        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        return accuracy, precision, recall

    def fit(self, X_train, y_train, X_valid, y_valid,
            epochs=100, batch_size=32, verbose=True):
        """Train the network."""
        self._build(X_train.shape[1])

        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            idx = np.random.permutation(len(X_train))
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            # Mini-batch training
            for start in range(0, len(X_train), batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred_batch = self._forward(X_batch)
                self._backward(y_batch, y_pred_batch)
                self._update_weights()

            # Epoch metrics
            y_pred_train = self._forward(X_train)
            y_pred_valid = self._forward(X_valid)

            loss = self.loss_fn(y_train, y_pred_train)
            val_loss = self.loss_fn(y_valid, y_pred_valid)
            acc, prec, rec = self._compute_metrics(y_train, y_pred_train)
            val_acc, val_prec, val_rec = self._compute_metrics(y_valid, y_pred_valid)

            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['precision'].append(prec)
            self.history['val_precision'].append(val_prec)
            self.history['recall'].append(rec)
            self.history['val_recall'].append(val_rec)

            if verbose:
                print(f"epoch {epoch:02d}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f} "
                      f"- acc: {acc:.4f} - val_acc: {val_acc:.4f}")

            # Early stopping
            if self.early_stopping:
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            print(f"\nEarly stopping triggered at epoch {epoch}")
                        break

    def predict(self, X):
        return self._forward(X)

    def save(self, filepath):
        """Save model topology and weights."""
        model_data = {
            'loss': self.loss_name,
            'optimizer': self.optimizer_name,
            'learning_rate': self.lr,
            'layers': []
        }
        for layer in self.layers:
            model_data['layers'].append({
                'units': layer.units,
                'activation': layer.activation_name,
                'W': layer.W,
                'b': layer.b,
            })
        np.save(filepath, model_data, allow_pickle=True)
        print(f"> saving model '{filepath}' to disk...")

    @classmethod
    def load(cls, filepath):
        """Load a saved model."""
        model_data = np.load(filepath, allow_pickle=True).item()
        layers = []
        for ld in model_data['layers']:
            layer = DenseLayer(ld['units'], activation=ld['activation'])
            layer.W = ld['W']
            layer.b = ld['b']
            layer.mW = np.zeros_like(layer.W)
            layer.vW = np.zeros_like(layer.W)
            layer.mb = np.zeros_like(layer.b)
            layer.vb = np.zeros_like(layer.b)
            layers.append(layer)
        model = cls(
            layers,
            loss=model_data.get('loss', 'categoricalCrossentropy'),
            optimizer=model_data.get('optimizer', 'adam'),
            learning_rate=model_data.get('learning_rate', 0.01),
        )
        return model