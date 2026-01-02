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


def gradient_descent_fixed_lr(X, y, learning_rate=0.01, n_iterations=100):
    """Gradient descent with fixed learning rate."""
    w, b = 0.0, 0.0
    loss_history = []
    w_history = [w]
    b_history = [b]
    
    for i in range(n_iterations):
        dw, db = compute_gradients(X, y, w, b)
        
        # Update parameters with fixed learning rate
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)
        w_history.append(w)
        b_history.append(b)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, loss_history, w_history, b_history
