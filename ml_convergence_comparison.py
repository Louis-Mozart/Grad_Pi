"""
Simple Machine Learning Model: Linear Regression
Comparing Fixed Learning Rate vs Adaptive Learning Rate (Barzilai-Borwein)
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=100, noise=0.1):
    """Generate synthetic linear data with noise."""
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    y = 2.5 * X + 1.5 + np.random.randn(n_samples) * noise
    return X, y


def compute_loss(X, y, w, b):
    """Compute Mean Squared Error loss."""
    predictions = w * X + b
    loss = np.mean((predictions - y) ** 2)
    return loss


def compute_gradients(X, y, w, b):
    """Compute gradients of loss with respect to w and b."""
    n = len(X)
    predictions = w * X + b
    dw = (2 / n) * np.sum((predictions - y) * X)
    db = (2 / n) * np.sum(predictions - y)
    return dw, db
