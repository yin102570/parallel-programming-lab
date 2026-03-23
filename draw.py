import pandas as pd
import matplotlib.pyplot as plt

# 读取测试数据
df = pd.read_csv("data.csv")
# 设置图片样式，高清输出
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 1. 矩阵向量乘法时间对比
plt.figure(figsize=(10, 5))
plt.plot(df["n"], df["mat_bad_us"], label="Naive Mat-Vec", marker='o', ms=4)
plt.plot(df["n"], df["mat_fast_us"], label="Optimized Mat-Vec", marker='s', ms=4)
plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (μs)")
plt.title("Matrix-Vector Multiplication Time Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("mat_time.png")
plt.close()

# 2. 矩阵乘法加速比
plt.figure(figsize=(10, 5))
plt.plot(df["n"], df["mat_speedup"], label="Speedup", marker='o', ms=4, color='crimson')
plt.xlabel("Matrix Size (n)")
plt.ylabel("Speedup (Naive/Optimized)")
plt.title("Mat-Vec Multiplication Speedup")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("mat_speedup.png")
plt.close()

# 3. 数组求和时间对比
plt.figure(figsize=(10, 5))
plt.plot(df["n"], df["sum_slow_us"], label="Naive Sum", marker='o', ms=4)
plt.plot(df["n"], df["sum_fast_us"], label="Optimized Sum", marker='s', ms=4)
plt.xlabel("Array Size (n)")
plt.ylabel("Time (μs)")
plt.title("Array Sum Time Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("sum_time.png")
plt.close()

# 4. 数组求和加速比
plt.figure(figsize=(10, 5))
plt.plot(df["n"], df["sum_speedup"], label="Speedup", marker='o', ms=4, color='crimson')
plt.xlabel("Array Size (n)")
plt.ylabel("Speedup (Naive/Optimized)")
plt.title("Array Sum Speedup")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("sum_speedup.png")
plt.close()

print("✅ 四张可视化图已生成：mat_time.png/mat_speedup.png/sum_time.png/sum_speedup.png")