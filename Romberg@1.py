import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 3


def romberg_integration(f, a, b, tol=1e-6, max_level=10):
    T = np.zeros((max_level, max_level))

    # 初始梯形积分
    h = b - a
    T[0, 0] = h * (f(a) + f(b)) / 2.0

    for k in range(1, max_level):
        h /= 2
        # 计算新增加的点的函数值
        new_points = a + h * np.arange(1, 2 ** k, 2)
        T[k, 0] = 0.5 * T[k - 1, 0] + h * np.sum(f(new_points))

        # Richardson 外推
        for j in range(1, k + 1):
            T[k, j] = T[k, j - 1] + (T[k, j - 1] - T[k - 1, j - 1]) / (4 ** j - 1)

        # 收敛判据
        if abs(T[k, k] - T[k - 1, k - 1]) < tol:
            T = T[:k + 1, :k + 1]
            break

    return T, T[k, k]


if __name__ == "__main__":
    a, b = 6, 100
    tol = float(input("请输入最大误差 tol（如 1e-6）: ") or 1e-6)
    max_level = int(input("请输入最大层数 max_level（如 10）: ") or 10)

    T, result = romberg_integration(f, a, b, tol, max_level)

    print("\nRomberg 积分表 (T 表):")
    for i in range(len(T)):
        row = "\t".join(f"{T[i, j]:.8f}" for j in range(i + 1))
        print(row)

    print(f"\n积分结果 ≈ {result:.8f}")
    print(f"理论值 = {(b ** 4 - a ** 4) / 4:.8f}")

    # 图形化显示 T 表
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Romberg 积分表 (T 数表)", fontsize=14)
    ax.axis('off')

    # 绘制表格
    cell_text = [[f"{T[i, j]:.6f}" if j <= i else "" for j in range(len(T))] for i in range(len(T))]
    table = ax.table(cellText=cell_text, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    plt.show()
