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


def gradient_descent_adaptive_lr(X, y, initial_lr=0.01, n_iterations=100):
    """Gradient descent with adaptive learning rate (Barzilai-Borwein method)."""
    w, b = 0.0, 0.0
    loss_history = []
    w_history = [w]
    b_history = [b]
    lr_history = []
    
    # First iteration with initial learning rate
    dw_prev, db_prev = compute_gradients(X, y, w, b)
    w_prev, b_prev = w, b
    w = w - initial_lr * dw_prev
    b = b - initial_lr * db_prev
    
    loss = compute_loss(X, y, w, b)
    loss_history.append(loss)
    w_history.append(w)
    b_history.append(b)
    lr_history.append(initial_lr)
    
    for i in range(1, n_iterations):
        dw, db = compute_gradients(X, y, w, b)
        
        # Compute adaptive learning rate (Barzilai-Borwein)
        # eta_n = |s^T y| / |y^T y| where s = x_new - x_old, y = grad_new - grad_old
        delta_w = w - w_prev
        delta_b = b - b_prev
        delta_grad_w = dw - dw_prev
        delta_grad_b = db - db_prev
        
        s_dot_y = abs(delta_w * delta_grad_w + delta_b * delta_grad_b)
        y_dot_y = delta_grad_w ** 2 + delta_grad_b ** 2
        
        # Compute adaptive learning rate with safeguard
        eta = s_dot_y / (y_dot_y + 1e-8)
        # Clip learning rate to reasonable bounds
        eta = np.clip(eta, 1e-4, 1.0)
        
        # Store previous values
        w_prev, b_prev = w, b
        dw_prev, db_prev = dw, db
        
        # Update parameters with adaptive learning rate
        w = w - eta * dw
        b = b - eta * db
        
        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)
        w_history.append(w)
        b_history.append(b)
        lr_history.append(eta)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}, lr = {eta:.6f}")
    
    return w, b, loss_history, w_history, b_history, lr_history
