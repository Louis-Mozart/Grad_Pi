"""
Polynomial Regression: 2nd Degree Polynomial Fitting
Comparing Fixed Learning Rate vs Adaptive Learning Rate (Barzilai-Borwein)
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_polynomial_data(n_samples=100, noise=5.0):
    """Generate synthetic data for a 2nd degree polynomial: y = ax^2 + bx + c."""
    np.random.seed(42)
    X = np.linspace(-5, 5, n_samples)
    # True polynomial: y = 0.5x^2 + 2x + 1
    y = 0.5 * X**2 + 2.0 * X + 1.0 + np.random.randn(n_samples) * noise
    return X, y


def compute_loss(X, y, a, b, c):
    """Compute Mean Squared Error loss for polynomial y = ax^2 + bx + c."""
    predictions = a * X**2 + b * X + c
    loss = np.mean((predictions - y) ** 2)
    return loss
