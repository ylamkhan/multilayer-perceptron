# Multilayer Perceptron – Breast Cancer Diagnosis

A complete from-scratch implementation of a multilayer perceptron (MLP) for binary classification of breast cancer diagnoses (Malignant / Benign).

---

## Project Structure

```
mlp/
├── mlp.py          # Single entry-point (split / train / predict / explore)
├── split.py        # Dataset splitting utility
├── train.py        # Training program
├── predict.py      # Prediction & evaluation program
├── network.py      # MLP core: layers, activations, optimizers
├── preprocess.py   # Data loading, encoding, normalisation
├── visualize.py    # Learning curves & data exploration plots
└── README.md
```

---

## Quick Start

### 1 – Split the dataset
```bash
python mlp.py split --dataset data.csv --ratio 0.8 --seed 42
# produces: data_train.csv  data_valid.csv
```

### 2 – Train the model
```bash
python mlp.py train \
  --dataset data_train.csv \
  --valid   data_valid.csv \
  --layer   24 24 \
  --epochs  100 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --optimizer adam \
  --loss categoricalCrossentropy \
  --early_stopping --patience 15
# produces: saved_model.npy  scaler.npy  learning_curves.png
```

### 3 – Predict / evaluate
```bash
python mlp.py predict --dataset data_valid.csv
```

### 4 – Explore the dataset (bonus)
```bash
python mlp.py explore --dataset data.csv
```

---

## Standalone scripts

Each phase can also be run directly:

```bash
python split.py   --dataset data.csv
python train.py   --dataset data_train.csv --valid data_valid.csv
python predict.py --dataset data_valid.csv
```

---

## Implementation Details

### Architecture (default)
```
Input (30) → Dense(30, sigmoid) → Dense(24, sigmoid) → Dense(24, sigmoid) → Dense(2, softmax)
```
- At least **two hidden layers** (mandatory)
- **Softmax** output layer for probabilistic distribution
- Fully configurable via CLI arguments

### Forward pass
Each neuron computes:
```
z = Σ(x_k · w_k) + bias
a = activation(z)
```

### Backpropagation
Gradients flow from the output layer backwards through each layer using the chain rule.  
For the output layer (softmax + categorical cross-entropy):
```
δ = y_pred − y_true
```
For hidden layers:
```
dz = δ_next · W_next.T  ⊙  activation'(z)
dW = x.T @ dz / m
db = mean(dz)
```

### Gradient Descent
Three optimisers are available:

| Optimizer | Description |
|-----------|-------------|
| `sgd`     | Standard mini-batch SGD |
| `rmsprop` | Adaptive learning rate (bonus) |
| `adam`    | Adaptive Moment Estimation (bonus, default) |

### Loss functions
- **Categorical Cross-Entropy** (training, multi-class output)
- **Binary Cross-Entropy** (evaluation, as required by subject):
  ```
  E = -(1/N) Σ [y_n log(p_n) + (1-y_n) log(1-p_n)]
  ```

### Preprocessing
1. Remove the ID column
2. Encode labels: M→[0,1], B→[1,0]
3. Z-score standardisation: `(x − μ) / σ`

---

## Bonus Features Implemented

- ✅ **Adam & RMSprop** optimisers (`--optimizer adam|rmsprop`)
- ✅ **Early stopping** (`--early_stopping --patience N`)
- ✅ **Multiple metrics** tracked: loss, accuracy, precision, recall
- ✅ **Learning curves** saved as PNG (loss + accuracy + recall)
- ✅ **Data exploration** plots (`mlp.py explore`)
- ✅ **Confusion matrix** computation
- ✅ **Reproducible splits** via `--seed`
- ✅ **Metric history** stored in `model.history`

---

## Requirements

```
numpy
matplotlib  (for plots)
```

Install with:
```bash
pip install numpy matplotlib
```

---

## Key Concepts (for defence)

**Feedforward**: Input flows layer-by-layer from input to output. At each neuron, a weighted sum is computed and passed through an activation function.

**Backpropagation**: After a forward pass, the error gradient is computed at the output, then propagated backwards through each layer using the chain rule to compute ∂L/∂W and ∂L/∂b for every parameter.

**Gradient descent**: Parameters are updated in the direction that minimises the loss:
`W ← W − lr · ∂L/∂W`
Mini-batch GD processes a small batch of examples per update, balancing convergence speed and stability.