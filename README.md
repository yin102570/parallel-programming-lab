# parallel-programming-lab

> 并行程序设计实验集合

体系结构相关的并行编程实验，覆盖 SIMD、多线程、MPI、GPU 等范式。

## 📌 项目简介
体系结构相关的并行编程实验，覆盖 SIMD、多线程、MPI、GPU 等范式。

## 🛠 技术栈
`C++ / OpenMP / MPI / CUDA`

## 🚀 快速启动
```bash
g++ -fopenmp demo.cpp    # 或 mpicxx mpi_demo.cpp
```

## 📂 项目结构
```
parallel-programming-lab/
├── include/
│   └── base.h
├── src/
│   └── arr_sum_fast.cpp
│   └── arr_sum_slow.cpp
│   └── main.cpp
│   └── mat_vec_bad.cpp
└── CMakeLists.txt
└── data.csv
└── draw.py
└── perf_ana.sh
└── perf.data
└── program
```

## 📈 开发进度
- [x] 基础框架搭建
- [x] 核心功能实现
- [ ] 单元测试补充
- [ ] 文档与示例完善
- [ ] 性能优化

## 💡 学习收获
实践了多种并行模型，建立了性能分析与加速比评估的直觉。

## 📸 项目截图
> 截图占位：将运行效果图放入 `docs/screenshots/` 并在此处引用。
<!-- ![预览](docs/screenshots/preview.png) -->

## 🔄 持续迭代
本项目是我在 **并行计算 / 体系结构** 方向的实践练习，会持续迭代更新。欢迎提 Issue 与 PR 一起完善。

## 📄 License
基于 [MIT License](./LICENSE) 开源。
