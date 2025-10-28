# import numpy as np
# import matplotlib.pyplot as plt
#
# # Use English fonts
# plt.rcParams['font.family'] = 'DejaVu Sans'
#
# # Given data points
# x_data = np.array([3, 4, 5, 6, 7, 8, 9])
# y_data = np.array([2.01, 2.98, 3.50, 5.02, 5.47, 6.02, 7.05])
#
# # User input for polynomial degree
# degree = int(input("Enter the highest degree of the polynomial: "))
#
# # Least squares polynomial fitting
# coeffs = np.polyfit(x_data, y_data, degree)  # coefficients from high to low
# print(f"Fitted polynomial coefficients: {coeffs}")
#
# # Construct polynomial expression string
# poly_terms = [f"{coeffs[i]:.4f}*x^{degree-i}" for i in range(degree)]
# poly_expr = " + ".join(poly_terms) + f" + {coeffs[-1]:.4f}"
# print("Fitted polynomial expression:")
# print(f"y = {poly_expr}")
#
# # Generate fitted curve
# x_fit = np.linspace(min(x_data)-1, max(x_data)+1, 200)
# y_fit = np.polyval(coeffs, x_fit)
#
# # Plotting
# plt.figure(figsize=(8,6))
# plt.plot(x_fit, y_fit, 'r-', label=f'{degree}-degree fit')
# plt.plot(x_data, y_data, 'bo', label='Data points')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Least Squares Polynomial Fit')
# plt.legend()
# plt.grid(True)
#
# # Add polynomial expression inside the plot
# plt.text(0.05, 0.05, f'y = {poly_expr}', transform=plt.gca().transAxes,
#          fontsize=10, color='black', verticalalignment='bottom',
#          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Use English fonts to avoid character issues
plt.rcParams['font.family'] = 'DejaVu Sans'

# Given data points
x_data = np.array([3, 4, 5, 6, 7, 8, 9])
y_data = np.array([2.01, 2.98, 3.50, 5.02, 5.47, 6.02, 7.05])

# Polynomial degrees to compare
degrees = [1, 2, 3, 4, 5]

# Colors for different curves
colors = ['r', 'g', 'b', 'm', 'c']

# Generate x values for smooth curves
x_fit = np.linspace(min(x_data) - 1, max(x_data) + 1, 300)

plt.figure(figsize=(12, 7))
plt.plot(x_data, y_data, 'ko', label='Data points')  # Original data

# Fit polynomials and plot
for i, deg in enumerate(degrees):
    coeffs = np.polyfit(x_data, y_data, deg)
    y_fit = np.polyval(coeffs, x_fit)

    # Construct full polynomial expression string
    poly_terms = [f"{coeffs[j]:.4f}*x^{deg - j}" for j in range(deg)]
    poly_expr = " + ".join(poly_terms) + f" + {coeffs[-1]:.4f}"

    # Plot the polynomial curve with expression in the legend
    plt.plot(x_fit, y_fit, color=colors[i], label=f'{deg}-degree fit: y={poly_expr}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fits Comparison (1~5 degree)')
plt.legend(fontsize=8, loc='upper left')  # Font smaller to accommodate long expressions
plt.grid(True)
plt.tight_layout()
plt.savefig('polynomial_fit_comparison.png', dpi=300)  # Save as PNG with high resolution

plt.show()

