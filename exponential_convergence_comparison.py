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


def gradient_descent_fixed(x_init=0.0, learning_rate=0.01, n_iterations=100):
    """Gradient Descent with Fixed Learning Rate."""
    x = x_init
    x_history = [x]
    f_history = [compute_function(x)]
    
    for i in range(n_iterations):
        grad = compute_gradient(x)
        x = x - learning_rate * grad
        
        x_history.append(x)
        f_history.append(compute_function(x))
    
    return x, x_history, f_history
