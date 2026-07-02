"""
draw_plots.py — 性能曲线绘图工具

读取 CSV 格式的性能数据，生成加速比/耗时对比图。
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

def draw_from_csv(csv_path: str, output_dir: str = "."):
    """从 CSV 文件读取性能数据并绘图"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # 自动检测列
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'us' in c.lower()]
    speedup_cols = [c for c in df.columns if 'speedup' in c.lower()]

    if time_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in time_cols:
            if col in df.columns:
                ax.plot(df.index, df[col], marker='o', label=col)
        ax.set_xlabel("Configuration Index")
        ax.set_ylabel("Time (us)")
        ax.set_title("Performance Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "time_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"[SAVED] {path}")

    if speedup_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in speedup_cols:
            if col in df.columns:
                ax.bar(range(len(df)), df[col], label=col)
        ax.set_xlabel("Configuration Index")
        ax.set_ylabel("Speedup (×)")
        ax.set_title("Speedup Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        path = os.path.join(output_dir, "speedup_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"[SAVED] {path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        draw_from_csv(sys.argv[1])
    else:
        csv_file = "../data.csv"
        if os.path.exists(csv_file):
            draw_from_csv(csv_file)
        else:
            print("Usage: python draw_plots.py <performance.csv>")
            print("  Looking for data.csv in parent directory...")
            for f in ["data.csv", "../data.csv", "perf_results.csv"]:
                if os.path.exists(f):
                    draw_from_csv(f)
                    break
            else:
                print("  No CSV file found.")
