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


def compute_gradients(X, y, a, b, c):
    """Compute gradients of loss with respect to a, b, and c."""
    n = len(X)
    predictions = a * X**2 + b * X + c
    error = predictions - y
    
    da = (2 / n) * np.sum(error * X**2)
    db = (2 / n) * np.sum(error * X)
    dc = (2 / n) * np.sum(error)
    
    return da, db, dc


def gradient_descent_fixed(X, y, learning_rate=0.001, n_iterations=1000):
    """Gradient Descent with Fixed Learning Rate."""
    a, b, c = 0.0, 0.0, 0.0
    loss_history = []
    
    for i in range(n_iterations):
        loss = compute_loss(X, y, a, b, c)
        loss_history.append(loss)
        
        da, db, dc = compute_gradients(X, y, a, b, c)
        
        a -= learning_rate * da
        b -= learning_rate * db
        c -= learning_rate * dc
    
    return a, b, c, loss_history


def gradient_descent_bb(X, y, n_iterations=1000, initial_lr=0.001):
    """Gradient Descent with Barzilai-Borwein Adaptive Learning Rate."""
    a, b, c = 0.0, 0.0, 0.0
    loss_history = []
    
    loss = compute_loss(X, y, a, b, c)
    loss_history.append(loss)
    
    da_prev, db_prev, dc_prev = compute_gradients(X, y, a, b, c)
    a -= initial_lr * da_prev
    b -= initial_lr * db_prev
    c -= initial_lr * dc_prev
    
    for i in range(1, n_iterations):
        loss = compute_loss(X, y, a, b, c)
        loss_history.append(loss)
        
        da, db, dc = compute_gradients(X, y, a, b, c)
        
        delta_a = -initial_lr * da_prev if i == 1 else (a - a_prev)
        delta_b = -initial_lr * db_prev if i == 1 else (b - b_prev)
        delta_c = -initial_lr * dc_prev if i == 1 else (c - c_prev)
        
        grad_diff_a = da - da_prev
        grad_diff_b = db - db_prev
        grad_diff_c = dc - dc_prev
        
        numerator = delta_a**2 + delta_b**2 + delta_c**2
        denominator = delta_a * grad_diff_a + delta_b * grad_diff_b + delta_c * grad_diff_c
        
        if abs(denominator) > 1e-10:
            learning_rate = abs(numerator / denominator)
            learning_rate = np.clip(learning_rate, 1e-6, 1.0)
        else:
            learning_rate = initial_lr
        
        a_prev, b_prev, c_prev = a, b, c
        da_prev, db_prev, dc_prev = da, db, dc
        
        a -= learning_rate * da
        b -= learning_rate * db
        c -= learning_rate * dc
    
    return a, b, c, loss_history
