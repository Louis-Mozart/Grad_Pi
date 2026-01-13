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


def gradient_descent_bb(x_init=0.0, n_iterations=100, initial_lr=0.01):
    """Gradient Descent with Barzilai-Borwein Adaptive Learning Rate."""
    x = x_init
    x_history = [x]
    f_history = [compute_function(x)]
    
    # First iteration with initial learning rate
    grad_prev = compute_gradient(x)
    x = x - initial_lr * grad_prev
    x_history.append(x)
    f_history.append(compute_function(x))
    
    for i in range(1, n_iterations):
        grad = compute_gradient(x)
        
        # Compute parameter difference and gradient difference
        delta_x = x - x_history[-2]
        grad_diff = grad - grad_prev
        
        # Barzilai-Borwein learning rate (BB method 1)
        if abs(grad_diff) > 1e-10:
            learning_rate = abs(delta_x / grad_diff)
            # Clip learning rate to reasonable bounds
            learning_rate = np.clip(learning_rate, 1e-6, 1.0)
        else:
            learning_rate = initial_lr
        
        # Save current gradient for next iteration
        grad_prev = grad
        
        # Update parameter
        x = x - learning_rate * grad
        x_history.append(x)
        f_history.append(compute_function(x))
    
    return x, x_history, f_history
