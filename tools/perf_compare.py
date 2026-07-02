"""
perf_compare.py — 跨架构性能对比可视化 🆕

生成全架构性能对比图表: 训练耗时、加速比、精度、功耗
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 全架构性能数据 (来自实验报告 表20)
# ============================================================
data = {
    "Serial":         {"time": 284.6, "speedup": 1.00,  "acc": 97.45, "power": 65,  "difficulty": 1},
    "SIMD AVX2":      {"time": 86.2,  "speedup": 3.30,  "acc": 97.41, "power": 80,  "difficulty": 2},
    "OpenMP 8-core":   {"time": 42.8,  "speedup": 6.65,  "acc": 97.43, "power": 120, "difficulty": 1},
    "MPI 4-node":     {"time": 78.4,  "speedup": 3.63,  "acc": 97.38, "power": 260, "difficulty": 3},
    "CUDA FP16":      {"time": 18.2,  "speedup": 15.64, "acc": 97.44, "power": 200, "difficulty": 2},
    "Hybrid 4-layer": {"time": 5.6,   "speedup": 50.82, "acc": 97.36, "power": 380, "difficulty": 4},
}

names = list(data.keys())
times = [data[n]["time"] for n in names]
speedups = [data[n]["speedup"] for n in names]
accuracies = [data[n]["acc"] for n in names]
powers = [data[n]["power"] for n in names]
difficulties = [data[n]["difficulty"] for n in names]

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

# ============================================================
# 综合性能仪表板
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("ANN Multi-Architecture Performance Dashboard", fontsize=16, fontweight='bold')

# 1. Training Time (Bar)
ax = axes[0, 0]
bars = ax.barh(names, times, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel("Training Time (s)")
ax.set_title("Training Time")
for bar, val in zip(bars, times):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}s', va='center', fontsize=9)

# 2. Speedup (Bar)
ax = axes[0, 1]
bars = ax.bar(names, speedups, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Speedup (×)")
ax.set_title("Speedup vs Serial")
ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}×', ha='center', fontsize=9, fontweight='bold')

# 3. Accuracy
ax = axes[0, 2]
ax.bar(names, [a - 97.0 for a in accuracies], color=colors, edgecolor='black', linewidth=0.5,
       bottom=97.0)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Recall@100 Accuracy")
ax.set_ylim(97.0, 97.6)

# 4. Speedup vs Accuracy scatter
ax = axes[1, 0]
for i, name in enumerate(names):
    ax.scatter(speedups[i], accuracies[i], s=200, c=colors[i], edgecolors='black',
              zorder=5)
    ax.annotate(name.split()[0], (speedups[i], accuracies[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xscale('log')
ax.set_xlabel("Speedup (log scale)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Speedup-Accuracy Trade-off")
ax.grid(True, alpha=0.3)

# 5. Power vs Time (energy efficiency)
ax = axes[1, 1]
for i, name in enumerate(names):
    ax.scatter(times[i], powers[i], s=200, c=colors[i], edgecolors='black', zorder=5)
    ax.annotate(name.split()[0], (times[i], powers[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Power (W)")
ax.set_title("Power-Time Trade-off")
ax.grid(True, alpha=0.3)

# 6. Radar chart: Overall score
ax = axes[1, 2]
ax.axis('off')

score_text = "══════════════════\n"
score_text += " Architecture Ranking\n"
score_text += "══════════════════\n\n"
ranked = sorted(data.items(), key=lambda x: x[1]["speedup"] / (x[1]["power"] * x[1]["difficulty"]**0.5), reverse=True)
for rank, (name, vals) in enumerate(ranked, 1):
    score = vals["speedup"] / (vals["power"]**0.3 * vals["difficulty"]**0.5)
    score_text += f"{rank}. {name:<15s} Score: {score:.2f}\n"

ax.text(0.5, 0.5, score_text, transform=ax.transAxes,
        fontfamily='monospace', fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig("perf_dashboard.png", dpi=150)
print("[SAVED] perf_dashboard.png")

# ============================================================
# 场景化推荐
# ============================================================
print("\n" + "=" * 60)
print("Scenario-Based Architecture Recommendation")
print("=" * 60)

scenarios = [
    ("单机高精度小批量训练", "OpenMP + AVX2 SIMD", "低延迟, 高精度, 易部署"),
    ("单机高吞吐训练", "CUDA 多卡 FP16 + 流并发", "极致吞吐, 15×+ 加速"),
    ("大规模分布式集群", "MPI Async SGD + per-node CUDA", "线性扩展, 无同步瓶颈"),
    ("边缘/移动端推理", "ARM NEON + INT4量化", "超低功耗, 实时推理"),
    ("极致综合性能", "MPI + OpenMP + SIMD + CUDA", "50×+ 加速, 全能型"),
]

for scenario, arch, reason in scenarios:
    print(f"  【{scenario}】")
    print(f"    推荐: {arch}")
    print(f"    理由: {reason}")
    print()
