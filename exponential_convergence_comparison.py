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


def plot_results(x_fixed, x_history_fixed, f_history_fixed,
                 x_bb, x_history_bb, f_history_bb):
    """Create visualization comparing fixed vs BB methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Function landscape with optimization paths
    ax1 = axes[0, 0]
    x_range = np.linspace(-3, 2, 500)
    y_range = compute_function(x_range)
    
    ax1.plot(x_range, y_range, 'k-', linewidth=2, label='f(x) = e^x + x + 1')
    ax1.plot(x_history_fixed, f_history_fixed, 'ro-', linewidth=2, 
             markersize=4, label='Fixed LR Path', alpha=0.6)
    ax1.plot(x_history_bb, f_history_bb, 'bo-', linewidth=2, 
             markersize=4, label='BB Method Path', alpha=0.6)
    ax1.plot(x_fixed, compute_function(x_fixed), 'r*', markersize=15, 
             label=f'Fixed LR Final: x={x_fixed:.4f}')
    ax1.plot(x_bb, compute_function(x_bb), 'b*', markersize=15, 
             label=f'BB Final: x={x_bb:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function Landscape and Optimization Paths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Function value convergence
    ax2 = axes[0, 1]
    ax2.plot(f_history_fixed, 'r-', linewidth=2, label='Fixed LR', alpha=0.7)
    ax2.plot(f_history_bb, 'b-', linewidth=2, label='BB Method', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Function Value Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter x convergence
    ax3 = axes[1, 0]
    ax3.plot(x_history_fixed, 'r-', linewidth=2, label='Fixed LR', alpha=0.7)
    ax3.plot(x_history_bb, 'b-', linewidth=2, label='BB Method', alpha=0.7)
    ax3.axhline(y=x_fixed, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=x_bb, color='b', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('x')
    ax3.set_title('Parameter x Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    CONVERGENCE STATISTICS
    
    Fixed Learning Rate:
    • Final x: {x_fixed:.6f}
    • Final f(x): {f_history_fixed[-1]:.6f}
    • Iterations: {len(f_history_fixed) - 1}
    • Improvement from start: {(f_history_fixed[0] - f_history_fixed[-1]):.6f}
    
    Barzilai-Borwein Method:
    • Final x: {x_bb:.6f}
    • Final f(x): {f_history_bb[-1]:.6f}
    • Iterations: {len(f_history_bb) - 1}
    • Improvement from start: {(f_history_bb[0] - f_history_bb[-1]):.6f}
    
    Comparison:
    • BB converged {abs(x_bb - x_fixed):.6f} units closer
    • Final value difference: {abs(f_history_bb[-1] - f_history_fixed[-1]):.6f}
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('exponential_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'exponential_convergence_comparison.png'")
    plt.show()
