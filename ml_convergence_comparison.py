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
