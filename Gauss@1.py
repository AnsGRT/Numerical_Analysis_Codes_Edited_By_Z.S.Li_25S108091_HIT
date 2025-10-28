import numpy as np
import matplotlib.pyplot as plt

# Normal Gauss
def gauss_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for k in range(n-1):
        for i in range(k+1, n):
            if A[k,k] == 0:
                raise ValueError("Zero pivot encountered!")
            factor = A[i,k]/A[k,k]
            A[i,k:] -= factor*A[k,k:]
            b[i] -= factor*b[k]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    return x

# Selected Gauss
def gauss_elimination_pivot(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for k in range(n-1):
        max_row = np.argmax(abs(A[k:,k])) + k
        if max_row != k:
            A[[k,max_row]] = A[[max_row,k]]
            b[[k,max_row]] = b[[max_row,k]]
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            A[i,k:] -= factor*A[k,k:]
            b[i] -= factor*b[k]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    return x

# Equations Define
A1 = np.array([[1e-8, 2, 3],
               [-1, 3.712, 4.623],
               [-2, 1.072, 5.643]])
b1 = np.array([1, 2, 3])

A2 = np.array([[4, -2, 4],
               [-2, 17, 10],
               [-4, 10, 9]])
b2 = np.array([10, 3, 7])
x1_gauss = gauss_elimination(A1, b1)
x1_pivot = gauss_elimination_pivot(A1, b1)
x2_gauss = gauss_elimination(A2, b2)
x2_pivot = gauss_elimination_pivot(A2, b2)
print("System 1 Gaussian elimination solution:", x1_gauss)
print("System 1 Gaussian elimination with partial pivoting solution:", x1_pivot)
print("System 2 Gaussian elimination solution:", x2_gauss)
print("System 2 Gaussian elimination with partial pivoting solution:", x2_pivot)

# Plot
fig, ax = plt.subplots(figsize=(10,2))
ax.axis('off')
text = (
    "\n"
    f"         System 1 Gaussian elimination:           {x1_gauss}\n"
    f"         System 1 Gaussian elimination w/ pivot:  {x1_pivot}\n\n\n"
    f"         System 2 Gaussian elimination:           {x2_gauss}\n"
    f"         System 2 Gaussian elimination w/ pivot:  {x2_pivot}"
)
ax.text(0.01, 0.99, text, fontsize=12, va='top', ha='left', family='monospace')
plt.tight_layout()
plt.savefig('Gauss_result.png', dpi=300)
plt.show()
