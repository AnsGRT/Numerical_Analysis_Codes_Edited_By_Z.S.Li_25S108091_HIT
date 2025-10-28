import numpy as np
import matplotlib.pyplot as plt


# 被积函数
def f1(x):
    return x ** 3


def f2(x):
    return np.sinc(x / np.pi)


def f3(x):
    return np.sin(x ** 2)


# Romberg 积分
def romberg(f, a, b, tol=1e-12, max_level=5):
    T = np.zeros((max_level, max_level), dtype=float)
    T[0, 0] = 0.5 * (b - a) * (f(a) + f(b))
    for k in range(1, max_level):
        n_new = 2 ** k
        h = (b - a) / n_new
        m_count = 2 ** (k - 1)
        xs = a + (2 * np.arange(1, m_count + 1) - 1) * h
        T[k, 0] = 0.5 * T[k - 1, 0] + h * np.sum(f(xs))
        for j in range(1, k + 1):
            T[k, j] = T[k, j - 1] + (T[k, j - 1] - T[k - 1, j - 1]) / (4 ** j - 1)
        if k > 0 and abs(T[k, k] - T[k - 1, k - 1]) < tol:
            return T[:k + 1, :k + 1]
    return T


# 测试参数
functions = [f1, f2, f3]
a_values = [6.0, 0.0, 0.0]
b_values = [100.0, 1.0, 1.0]
names = ["x^3", "sin(x)/x", "sin(x^2)"]

for idx, f in enumerate(functions):
    print(f"\n=== Romberg 积分: ∫{names[idx]} dx [{a_values[idx]}, {b_values[idx]}] ===")
    T = romberg(f, a_values[idx], b_values[idx], tol=1e-12, max_level=5)
    n = T.shape[0]

    # 控制台输出
    labels = ["T", "S", "C", "K", "L"]
    print("Romberg 表（左对齐上三角，16位数字）:")
    for j in range(n):
        row_label = labels[j] if j < len(labels) else f"R{j}"
        vals = [f"{T[k, j]:16.8f}" for k in range(j, n)]  # 固定16位显示
        print(f"{row_label}: " + " ".join(vals))

    best = T[n - 1, n - 1]
    print(f"最精确值 = {best:16.12f}")

    # 表格绘制
    cell_text = []
    row_labels = [labels[j] for j in range(n)]
    col_labels = [str(2 ** k) for k in range(n)]
    for j in range(n):
        row = [f"{T[k, j]:16.8f}" for k in range(j, n)] + [""] * j
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(2.0 * n + 1.5, 1.0 * n + 1.2))
    ax.axis('off')
    ax.set_title(f"Romberg Table: {names[idx]}", fontsize=12)
    the_table = ax.table(cellText=cell_text,
                         rowLabels=row_labels,
                         colLabels=col_labels,
                         cellLoc='left',
                         loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.3, 1.3)  # 放大单元格
    for (r, c), cell in the_table.get_celld().items():
        cell.set_edgecolor('black')
        if cell.get_text().get_text() == "":
            cell.set_facecolor('#f0f0f0')
    plt.tight_layout()
    plt.show()


