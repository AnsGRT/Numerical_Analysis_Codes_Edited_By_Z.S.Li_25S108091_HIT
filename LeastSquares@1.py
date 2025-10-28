import numpy as np
import matplotlib.pyplot as plt

# Use English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'

# Given data points
x_data = np.array([3, 4, 5, 6, 7, 8, 9])
y_data = np.array([2.01, 2.98, 3.50, 5.02, 5.47, 6.02, 7.05])

# User input for polynomial degree
degree = int(input("Enter the highest degree of the polynomial: "))

# Least squares polynomial fitting
coeffs = np.polyfit(x_data, y_data, degree)  # coefficients from high to low
print(f"Fitted polynomial coefficients: {coeffs}")

# Construct polynomial expression string
poly_terms = [f"{coeffs[i]:.4f}*x^{degree-i}" for i in range(degree)]
poly_expr = " + ".join(poly_terms) + f" + {coeffs[-1]:.4f}"
print("Fitted polynomial expression:")
print(f"y = {poly_expr}")

# Generate fitted curve
x_fit = np.linspace(min(x_data)-1, max(x_data)+1, 200)
y_fit = np.polyval(coeffs, x_fit)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(x_fit, y_fit, 'r-', label=f'{degree}-degree fit')
plt.plot(x_data, y_data, 'bo', label='Data points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Polynomial Fit')
plt.legend()
plt.grid(True)

# Add polynomial expression inside the plot
plt.text(0.05, 0.05, f'y = {poly_expr}', transform=plt.gca().transAxes,
         fontsize=10, color='black', verticalalignment='bottom',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.show()
