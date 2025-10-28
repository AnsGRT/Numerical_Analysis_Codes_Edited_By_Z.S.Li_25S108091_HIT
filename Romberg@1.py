import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3

a = 6.0
b = 100.0
tol = 1e-12
max_level = 5   # 列为: 1,2,4,8,16 （可改）

def romberg(f, a, b, tol, max_level):
    T = np.zeros((max_level, max_level), dtype=float)
    # 初始梯形
    T[0,0] = 0.5 * (b - a) * (f(a) + f(b))
    for k in range(1, max_level):
        n_new = 2**k
        h = (b - a) / n_new
        m_count = 2**(k-1)
        # 新增点： a + (2*m-1)*h, m=1..m_count
        xs = a + (2*np.arange(1, m_count+1) - 1) * h
        sum_new = np.sum(f(xs))
        T[k,0] = 0.5 * T[k-1,0] + h * sum_new
        # Richardson 外推
        for j in range(1, k+1):
            T[k,j] = T[k,j-1] + (T[k,j-1] - T[k-1,j-1]) / (4**j - 1)
        # 收敛判断（可选）
        if k > 0 and abs(T[k,k] - T[k-1,k-1]) < tol:
            return T[:k+1, :k+1]
    return T

# 计算表格
T = romberg(f, a, b, tol, max_level)
n = T.shape[0]
theoretical = (b**4 - a**4) / 4.0

# 控制台输出：按你要的“左对齐上三角”格式
labels = ["T", "S", "C", "K", "L", "M"]
print("Romberg 表（行 = T,S,C,K；每行显示 T[k,j] 对于 k=j..n-1）：")
for j in range(n):
    row_label = labels[j] if j < len(labels) else f"R{j}"
    vals = [f"{T[k,j]:.6f}" for k in range(j, n)]
    print(f"{row_label}: " + "  ".join(vals))
print()

best = T[n-1, n-1]
print(f"最精确值（K 最底右） = {best:.12f}")
print(f"解析值                 = {theoretical:.12f}")
print(f"误差                   = {abs(best-theoretical):.3e}")

# 绘图：用 matplotlib.table，左对齐上三角显示
cell_text = []
row_labels = [labels[j] for j in range(n)]
col_labels = [str(2**k) for k in range(n)]
for j in range(n):
    row = [f"{T[k,j]:.6f}" for k in range(j, n)] + [""]*j
    cell_text.append(row)

plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(1.6*n+1.5, 1.0*n+1.2))
ax.axis('off')
ax.set_title("Romberg Table - left-aligned upper-triangle", fontsize=12)
the_table = ax.table(cellText=cell_text,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='left',
                     loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.2)
for (r,c), cell in the_table.get_celld().items():
    cell.set_edgecolor('black')
    if cell.get_text().get_text() == "":
        cell.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.show()
