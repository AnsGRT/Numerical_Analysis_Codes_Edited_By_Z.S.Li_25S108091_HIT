import numpy as np
import matplotlib.pyplot as plt

# Fonts
plt.rcParams['font.family'] = 'DejaVu Sans'

# Input Define
x_data = np.array([3, 4, 5, 6, 7, 8, 9])
y_data = np.array([2.01, 2.98, 3.50, 5.02, 5.47, 6.02, 7.05])

# Polynomial Degree
degrees = [1, 2, 3, 4, 5]

# Color
colors = ['r', 'g', 'b', 'm', 'c']


x_fit = np.linspace(min(x_data) - 1, max(x_data) + 1, 300)

plt.figure(figsize=(12, 7))
plt.plot(x_data, y_data, 'ko', label='Data points')  # Original data

# Fit Polynomial
for i, deg in enumerate(degrees):
    coeffs = np.polyfit(x_data, y_data, deg)
    y_fit = np.polyval(coeffs, x_fit)
    poly_terms = [f"{coeffs[j]:.4f}*x^{deg - j}" for j in range(deg)]
    poly_expr = " + ".join(poly_terms) + f" + {coeffs[-1]:.4f}"
    plt.plot(x_fit, y_fit, color=colors[i], label=f'{deg}-degree fit: y={poly_expr}')

# Plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fits Comparison (1~5 degree)')
plt.legend(fontsize=8, loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('LeastSquares_result_cp', dpi=300)
plt.show()