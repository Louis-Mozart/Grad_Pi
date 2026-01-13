"""
Exponential Function Optimization: f(x) = e^x + x + 1
Comparing Fixed Learning Rate vs Adaptive Learning Rate (Barzilai-Borwein)
Finding the minimum of the function
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_function(x):
    """Compute f(x) = e^x + x + 1."""
    return np.exp(x) + x + 1


def compute_gradient(x):
    """Compute gradient df/dx = e^x + 1."""
    return np.exp(x) + 1
