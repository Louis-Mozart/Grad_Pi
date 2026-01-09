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


def plot_results(X, y, a_fixed, b_fixed, c_fixed, a_bb, b_bb, c_bb,
                 loss_history_fixed, loss_history_bb):
    """Create visualization comparing fixed vs BB methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1 = axes[0, 0]
    ax1.scatter(X, y, alpha=0.5, label='Data Points')
    X_plot = np.linspace(X.min(), X.max(), 200)
    
    y_fixed = a_fixed * X_plot**2 + b_fixed * X_plot + c_fixed
    y_bb = a_bb * X_plot**2 + b_bb * X_plot + c_bb
    y_true = 0.5 * X_plot**2 + 2.0 * X_plot + 1.0
    
    ax1.plot(X_plot, y_true, 'g--', linewidth=2, label='True Polynomial')
    ax1.plot(X_plot, y_fixed, 'r-', linewidth=2, label='Fixed LR Fit')
    ax1.plot(X_plot, y_bb, 'b-', linewidth=2, label='BB Method Fit')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Polynomial Fitting Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(loss_history_fixed, 'r-', linewidth=2, label='Fixed LR', alpha=0.7)
    ax2.plot(loss_history_bb, 'b-', linewidth=2, label='BB Method', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Loss Convergence (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3 = axes[1, 0]
    iterations_zoom = min(100, len(loss_history_fixed))
    ax3.plot(loss_history_fixed[:iterations_zoom], 'r-', linewidth=2,
             label='Fixed LR', alpha=0.7)
    ax3.plot(loss_history_bb[:iterations_zoom], 'b-', linewidth=2,
             label='BB Method', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Loss Convergence (First 100 Iterations)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    CONVERGENCE STATISTICS
    
    Fixed Learning Rate:
    • Final Loss: {loss_history_fixed[-1]:.4f}
    • Parameters: a={a_fixed:.4f}, b={b_fixed:.4f}, c={c_fixed:.4f}
    
    Barzilai-Borwein Method:
    • Final Loss: {loss_history_bb[-1]:.4f}
    • Parameters: a={a_bb:.4f}, b={b_bb:.4f}, c={c_bb:.4f}
    
    True Parameters:
    • a=0.5000, b=2.0000, c=1.0000
    
    Improvement:
    • Loss Reduction: {((loss_history_fixed[-1] - loss_history_bb[-1]) / loss_history_fixed[-1] * 100):.2f}%
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('polynomial_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'polynomial_convergence_comparison.png'")
    plt.show()


def main():
    """Main function to run the comparison."""
    print("Polynomial Regression: Fixed LR vs Barzilai-Borwein")
    print("=" * 60)
    
    print("\n1. Generating polynomial data...")
    X, y = generate_polynomial_data(n_samples=100, noise=5.0)
    print(f"   Generated {len(X)} data points")
    
    print("\n2. Running Fixed Learning Rate Gradient Descent...")
    learning_rate = 0.001
    n_iterations = 1000
    a_fixed, b_fixed, c_fixed, loss_history_fixed = gradient_descent_fixed(
        X, y, learning_rate=learning_rate, n_iterations=n_iterations
    )
    print(f"   Final parameters: a={a_fixed:.4f}, b={b_fixed:.4f}, c={c_fixed:.4f}")
    print(f"   Final loss: {loss_history_fixed[-1]:.4f}")
    
    print("\n3. Running Barzilai-Borwein Gradient Descent...")
    a_bb, b_bb, c_bb, loss_history_bb = gradient_descent_bb(
        X, y, n_iterations=n_iterations, initial_lr=learning_rate
    )
    print(f"   Final parameters: a={a_bb:.4f}, b={b_bb:.4f}, c={c_bb:.4f}")
    print(f"   Final loss: {loss_history_bb[-1]:.4f}")
    
    print("\n4. Comparison:")
    print(f"   True parameters: a=0.5000, b=2.0000, c=1.0000")
    improvement = ((loss_history_fixed[-1] - loss_history_bb[-1]) /
                   loss_history_fixed[-1] * 100)
    print(f"   BB Method improvement: {improvement:.2f}%")
    
    print("\n5. Creating visualizations...")
    plot_results(X, y, a_fixed, b_fixed, c_fixed, a_bb, b_bb, c_bb,
                 loss_history_fixed, loss_history_bb)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
