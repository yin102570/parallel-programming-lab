# 数据集说明

## DEEP100K

DEEP100K 是 ANN 搜索标准评测数据集，包含:
- **100,000 条** base 向量 × **96 维** float32
- **2,000 条** query 向量
- **Top-100** ground truth

## 格式

```
base.fbin:  int32(n) + int32(dim) + float32[n*dim]
query.fbin: int32(nq) + int32(dq) + float32[nq*dq]
gt.bin:     int32(n) + int32(k) + int32[n*k]
```

## 数据生成

```bash
cd tools && gcc -O2 -o generate_data generate_data.c && ./generate_data > ../data/random_100k_96.bin
```
