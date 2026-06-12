#!/bin/bash
# ============================================================
# run_experiments.sh
# ANN GPU加速 — 一键运行全部实验
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
DATA_DIR="${DATA_DIR:-D:/HuaweiMoveData/Users/asdf1/Desktop/ann数据集}"
QUERIES="${QUERIES:-100}"
BATCH="${BATCH:-64}"
TOP_K="${TOP_K:-10}"
LOG_DIR="logs_$(date +%Y%m%d_%H%M%S)"

banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  $1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_gpu() {
    echo -e "${YELLOW}[CHECK] Detecting GPU...${NC}"
    if command -v rocminfo &>/dev/null; then
        rocminfo | grep -E "Name|Compute Unit" || true
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
    else
        echo -e "${RED}[WARN] No GPU tools detected${NC}"
    fi
    echo ""
}

check_data() {
    echo -e "${YELLOW}[CHECK] Verifying datasets...${NC}"
    for f in "DEEP100K.base.100k.fbin.larkcache" "DEEP100K.query.fbin" "DEEP100K.gt.query.100k.top100.bin"; do
        if [ -f "$DATA_DIR/$f" ]; then
            echo -e "  ${GREEN}✓${NC} $f ($(du -h "$DATA_DIR/$f" | cut -f1))"
        else
            echo -e "  ${RED}✗${NC} $f (NOT FOUND)"
        fi
    done
    echo ""
}

# ==================== Main ====================

banner "ANN GPU 加速实验 — 一键运行脚本"
echo "Config:"
echo "  Data Dir:  $DATA_DIR"
echo "  Queries:   $QUERIES"
echo "  Batch:     $BATCH"
echo "  Top-K:     $TOP_K"
echo ""

check_gpu
check_data

mkdir -p "$LOG_DIR"

# 1. 编译
banner "Step 1: 编译"
echo -e "${BLUE}Compiling...${NC}"
make clean 2>/dev/null || true
make all 2>&1 | tee "$LOG_DIR/build.log"
echo -e "${GREEN}Compilation done.${NC}"

# 2. CPU Baseline
banner "Step 2: CPU Baseline (Mode 5)"
echo -e "${BLUE}Running CPU baseline...${NC}"
./ann_gpu 5 --data "$DATA_DIR" --queries "$QUERIES" --topk "$TOP_K" \
    2>&1 | tee "$LOG_DIR/cpu_baseline.log"

# 3. Block Size Tuning
banner "Step 3: Block Size 调优 (Mode 4)"
echo -e "${BLUE}Scanning optimal block sizes...${NC}"
./ann_gpu 4 --data "$DATA_DIR" --queries 10 --topk "$TOP_K" \
    2>&1 | tee "$LOG_DIR/block_tuning.log"

# 4. Level 1: 单查询GPU
banner "Step 4: Level 1 — 单查询GPU优化 (Mode 0)"
echo -e "${BLUE}Running single query GPU search...${NC}"
./ann_gpu 0 --data "$DATA_DIR" --queries "$QUERIES" --topk "$TOP_K" \
    2>&1 | tee "$LOG_DIR/level1_single.log"

# 5. Level 2: Batch GEMM — Batch Size 扫描
banner "Step 5: Level 2 — Batch矩阵乘法 (核心创新)"

echo ""
echo "| Batch | Per Query (us) | Recall@10 | QPS | TFLOPS |" | tee "$LOG_DIR/level2_batch.log"
echo "| :--- | :--- | :--- | :--- | :--- |" | tee -a "$LOG_DIR/level2_batch.log"

for bs in 1 8 16 32 64 128 200; do
    if [ $bs -le $QUERIES ]; then
        echo -n "  Batch=$bs ... "
        result=$(./ann_gpu 2 --data "$DATA_DIR" --queries "$QUERIES" \
                 --batch $bs --topk "$TOP_K" 2>&1)
        echo "$result" >> "$LOG_DIR/level2_batch_${bs}.log"
        # 提取性能行
        perf_line=$(echo "$result" | grep "Per Query\|│.*[0-9].*│" | tail -1)
        echo "done"
    fi
done

# 6. Level 3: rocBLAS (如果有)
banner "Step 6: Level 3 — rocBLAS 对比 (Mode 3)"
echo -e "${BLUE}Running rocBLAS comparison...${NC}"
./ann_gpu 3 --data "$DATA_DIR" --queries "$QUERIES" --batch "$BATCH" --topk "$TOP_K" \
    2>&1 | tee "$LOG_DIR/level3_rocblas.log" || \
    echo -e "${YELLOW}[SKIP] rocBLAS not available, skipped.${NC}"

# 7. 全模式汇总
banner "Step 7: 全模式汇总 (Mode all)"
echo -e "${BLUE}Running all-mode comparison...${NC}"
./ann_gpu all --data "$DATA_DIR" --queries "$QUERIES" --batch "$BATCH" --topk "$TOP_K" \
    2>&1 | tee "$LOG_DIR/full_comparison.log"

# 完成
banner "✅ 实验完成!"
echo -e "日志保存在: ${GREEN}$LOG_DIR/${NC}"
echo ""
echo "关键文件:"
echo "  $LOG_DIR/full_comparison.log  — 完整对比报告"
echo "  $LOG_DIR/block_tuning.log     — Block Size调优结果"
echo "  $LOG_DIR/level2_batch.log     — Batch矩阵乘法结果"
echo ""
echo -e "${YELLOW}提示: 将实际数据填入上述日志文件对应的表格中${NC}"
