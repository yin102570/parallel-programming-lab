# Makefile for ANN GPU Acceleration (AMD ROCm/HIP)
# 
# 目标:
#   make all     — 编译所有目标
#   make clean   — 清理
#   make run     — 运行Mode 2 (Batch GEMM, 推荐)
#   make runX    — 运行Mode X (0-4)
#   make runall  — 运行全模式对比

# ==================== 编译器配置 ====================
HIPCC ?= hipcc
CXX   ?= g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall

# HIP编译选项 (AMD)
HIPFLAGS = -std=c++17 -O3 --offload-arch=gfx90a -Wall -Wno-unused-result
HIPFLAGS_LINK = -std=c++17 -O3 --offload-arch=gfx90a

# ROCm路径
ROCM_PATH ?= /opt/rocm
HIP_INCLUDE = -I$(ROCM_PATH)/include
HIP_LIB     = -L$(ROCM_PATH)/lib -lamdhip64

# rocBLAS (可选 — 如果有则定义USE_ROCBLAS)
# ROCBLAS_LIB = -lrocblas
# HIPFLAGS += -DUSE_ROCBLAS
# HIPFLAGS_LINK += -lrocblas

# ==================== 目标 ====================
TARGET  = ann_gpu
OBJECTS = main_ann_gpu.o flat_scan_gpu_hip.o

# 数据集路径 (Windows示例)
DATA_DIR = D:/HuaweiMoveData/Users/asdf1/Desktop/ann数据集

.PHONY: all clean run run0 run1 run2 run3 run4 runall

all: $(TARGET)

# HIP源文件编译
flat_scan_gpu_hip.o: flat_scan_gpu_hip.cpp flat_scan_gpu_hip.h
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# 主程序编译 (C++ with HIP runtime)
main_ann_gpu.o: main_ann_gpu.cc flat_scan_gpu_hip.h
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# 链接
$(TARGET): $(OBJECTS)
	$(HIPCC) $(HIPFLAGS_LINK) $^ $(HIP_LIB) $(ROCBLAS_LIB) -o $@

# 纯C++备选编译 (如果HIPCC不可用)
ann_gpu_backup: main_ann_gpu.cc flat_scan_gpu_hip.cpp
	$(CXX) $(CXXFLAGS) -I. -o $@ $^

# ==================== 运行 ====================
RUN_ARGS = --data "$(DATA_DIR)" --queries 100

run: $(TARGET)
	./$(TARGET) 2 --batch 64 $(RUN_ARGS)

run0: $(TARGET)
	./$(TARGET) 0 $(RUN_ARGS)

run1: $(TARGET)
	./$(TARGET) 1 $(RUN_ARGS)

run2: $(TARGET)
	./$(TARGET) 2 --batch 64 $(RUN_ARGS)

run3: $(TARGET)
	./$(TARGET) 3 --batch 64 $(RUN_ARGS)

run4: $(TARGET)
	./$(TARGET) 4 $(RUN_ARGS)

runall: $(TARGET)
	./$(TARGET) all --batch 64 $(RUN_ARGS)

# Block Size调优 (测试不同大小)
tune: $(TARGET)
	./$(TARGET) 4 $(RUN_ARGS)

# ==================== 清理 ====================
clean:
	rm -f $(TARGET) ann_gpu_backup *.o *.obj *.exe
	rm -f *.out *.log
