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


def plot_results(X, y, results_fixed, results_adaptive):
    """Plot comparison of both methods."""
    w_fixed, b_fixed, loss_fixed, w_hist_fixed, b_hist_fixed = results_fixed
    w_adaptive, b_adaptive, loss_adaptive, w_hist_adaptive, b_hist_adaptive, lr_hist = results_adaptive
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Data and fitted lines
    ax1 = axes[0, 0]
    ax1.scatter(X, y, alpha=0.5, label='Data')
    ax1.plot(X, w_fixed * X + b_fixed, 'r-', linewidth=2, label=f'Fixed LR: y = {w_fixed:.2f}x + {b_fixed:.2f}')
    ax1.plot(X, w_adaptive * X + b_adaptive, 'g-', linewidth=2, label=f'Adaptive LR: y = {w_adaptive:.2f}x + {b_adaptive:.2f}')
    ax1.plot(X, 2.5 * X + 1.5, 'k--', linewidth=1, alpha=0.5, label='True: y = 2.5x + 1.5')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Regression: Data and Fitted Lines')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss convergence
    ax2 = axes[0, 1]
    ax2.plot(loss_fixed, 'r-', label='Fixed Learning Rate', linewidth=2)
    ax2.plot(loss_adaptive, 'g-', label='Adaptive Learning Rate', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Loss Convergence Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Parameter convergence (weight w)
    ax3 = axes[1, 0]
    ax3.plot(w_hist_fixed, 'r-', label='Fixed LR', linewidth=2)
    ax3.plot(w_hist_adaptive, 'g-', label='Adaptive LR', linewidth=2)
    ax3.axhline(y=2.5, color='k', linestyle='--', linewidth=1, alpha=0.5, label='True value (2.5)')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Weight (w)')
    ax3.set_title('Weight Parameter Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate evolution (adaptive method)
    ax4 = axes[1, 1]
    ax4.plot(lr_hist, 'g-', linewidth=2)
    ax4.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='Fixed LR (0.01)')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Evolution (Adaptive Method)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'convergence_comparison.png'")
    plt.show()


def main():
    print("=" * 70)
    print("Machine Learning Model: Linear Regression")
    print("Comparing Fixed Learning Rate vs Adaptive Learning Rate")
    print("=" * 70)
    
    # Generate data
    X, y = generate_data(n_samples=100, noise=10)
    print(f"\nGenerated {len(X)} data points")
    print(f"True parameters: w = 2.5, b = 1.5")
    
    # Method 1: Fixed Learning Rate
    print("\n" + "=" * 70)
    print("METHOD 1: FIXED LEARNING RATE (0.01)")
    print("=" * 70)
    w_fixed, b_fixed, loss_fixed, w_hist_fixed, b_hist_fixed = gradient_descent_fixed_lr(
        X, y, learning_rate=0.01, n_iterations=100
    )
    print(f"\nFinal parameters: w = {w_fixed:.4f}, b = {b_fixed:.4f}")
    print(f"Final loss: {loss_fixed[-1]:.4f}")
    
    # Method 2: Adaptive Learning Rate
    print("\n" + "=" * 70)
    print("METHOD 2: ADAPTIVE LEARNING RATE (Barzilai-Borwein)")
    print("=" * 70)
    w_adaptive, b_adaptive, loss_adaptive, w_hist_adaptive, b_hist_adaptive, lr_hist = gradient_descent_adaptive_lr(
        X, y, initial_lr=0.01, n_iterations=100
    )
    print(f"\nFinal parameters: w = {w_adaptive:.4f}, b = {b_adaptive:.4f}")
    print(f"Final loss: {loss_adaptive[-1]:.4f}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Fixed LR - Final Loss: {loss_fixed[-1]:.6f}")
    print(f"Adaptive LR - Final Loss: {loss_adaptive[-1]:.6f}")
    print(f"Improvement: {((loss_fixed[-1] - loss_adaptive[-1]) / loss_fixed[-1] * 100):.2f}%")
    
    # Plot results
    results_fixed = (w_fixed, b_fixed, loss_fixed, w_hist_fixed, b_hist_fixed)
    results_adaptive = (w_adaptive, b_adaptive, loss_adaptive, w_hist_adaptive, b_hist_adaptive, lr_hist)
    plot_results(X, y, results_fixed, results_adaptive)
