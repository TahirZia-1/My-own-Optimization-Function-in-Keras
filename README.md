# Custom Hybrid Optimizer for Binary Classification

This repository implements a custom hybrid optimizer combining RMSprop and Adagrad for training a neural network on a binary classification task. The project compares this custom optimizer ("Tahir Model") against standard SGD and Adagrad optimizers using a simple dataset resembling the XOR problem. Built with TensorFlow/Keras, this experiment demonstrates the effectiveness of different optimization strategies.

## Overview

The goal is to classify a small dataset of 4 binary input pairs (`[0,0], [0,1], [1,0], [1,1]`) into two classes (`[1,0]` or `[0,1]`), mimicking XOR-like behavior. A shallow neural network is trained using three optimizers: SGD, Adagrad, and the custom hybrid RMSprop-Adagrad optimizer. Training accuracy is plotted to compare convergence and performance.

### Key Features
- **Custom Optimizer**: A hybrid of RMSprop (learning rate: 0.001) and Adagrad (learning rate: 0.01) for adaptive gradient updates.
- **Dataset**: Simple XOR-like binary classification (4 samples).
- **Network**: Two-layer DNN with ReLU and softmax activations.
- **Comparison**: Evaluates SGD, Adagrad, and the custom optimizer over 1000 epochs.
- **Visualization**: Accuracy plots for all optimizers.

## Implementation

### Network Architecture
- **Input Layer**: 2 units (binary inputs)
- **Hidden Layer**: 4 units, ReLU activation
- **Output Layer**: 2 units, softmax activation

### Optimizers
1. **SGD**:
   - Learning Rate: 0.01
   - Momentum: 0.9
2. **Adagrad**:
   - Learning Rate: 0.01
3. **Tahir Model (Hybrid)**:
   - Combines RMSprop (learning rate: 0.001) and Adagrad (learning rate: 0.01)
   - Conceptually merges adaptive learning rates with momentum-like behavior (note: code correction required for practical use).

### Training
- **Loss**: Categorical Crossentropy
- **Epochs**: 1000
- **Batch Size**: 4 (full dataset per batch)

### Results
- **SGD**: Converges quickly due to momentum, achieving near-perfect accuracy.
- **Adagrad**: Slower convergence but stable, reaching high accuracy.
- **Tahir Model**: Balances adaptability and stability, with performance dependent on implementation details (see notes below).

*Accuracy Plot*:  
![Accuracy Comparison](images/accuracy_comparison.png)  
(Generated from `optimizer_comparison.ipynb` and saved manually as `images/accuracy_comparison.png`.)

*Note*: The original code contains a syntax error in the hybrid optimizer definition (`RMSprop(...) and Adagrad(...)`). This README assumes a conceptual hybrid; for practical use, you’d need to implement a custom `tf.keras.optimizers.Optimizer` class combining RMSprop’s moving average of squared gradients with Adagrad’s sum of squared gradients. Current results reflect RMSprop alone due to this error.

## Installation

### Prerequisites
- Python 3.x
- Required libraries:
  - TensorFlow (`pip install tensorflow`)
  - NumPy (`pip install numpy`)
  - Matplotlib (`pip install matplotlib`)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TahirZia-1/hybrid-optimizer.git
   cd hybrid-optimizer
