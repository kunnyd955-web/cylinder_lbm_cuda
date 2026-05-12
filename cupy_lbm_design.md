# CuPy-LBM 圆柱绕流 GPU 加速方案设计

## 1. 方案概述

### 核心思路
将 CPU 验证版（numpy）的数组全部迁移到 GPU 显存（cupy），
利用 CuPy 的 numpy 兼容 API，碰撞、宏观量计算等向量化操作
自动在 GPU 上并行执行，无需手写 kernel。

### 改动范围
```
import numpy as np  →  import cupy as cp
np.xxx             →  cp.xxx
固体标记初始化      →  CPU 算完再传入 GPU
结果读取           →  cp.asnumpy() 传回 CPU 做后处理
```

---

## 2. 硬件适配

| 项目 | 参数 | 对本方案的意义 |
|------|------|--------------|
| GPU 架构 | Ada Lovelace | CuPy 完全支持 |
| Compute Capability | 8.9 | CuPy 自动选择最优 kernel |
| 显存 | 8.6 GB | 支持最大 ~4000×4000 格子 |
| nvcc 编译参数 | `-arch=sm_89` | CuPy 安装时自动处理 |

### 显存占用估算（float32，双缓冲）

| D_lat | 域大小 | 显存占用 | 建议 |
|-------|--------|---------|------|
| 20 | 400×400 | ~18 MB | CPU 验证基准 |
| 40 | 800×800 | ~72 MB | GPU 起点，推荐先跑这个 |
| 80 | 1600×1600 | ~288 MB | 精度验证 |
| 160 | 3200×3200 | ~1.15 GB | 高分辨率 |
| 400 | 8000×8000 | ~7.2 GB | 显存上限附近，留 0.4 GB 余量 |


## 4. 模块设计

### 4.1 参数模块
和 CPU 版完全一致，无需改动：

```
Re=100, D_phys=0.1m, nu=1.5e-5 m²/s
U_lat=0.04, tau=0.524, Ma=0.069
D_lat=40（GPU版提升分辨率）
NX=800, NY=800
```

### 4.2 初始化模块
固体标记在 CPU 上计算（只算一次），然后传入 GPU：

```
CPU: 计算圆柱 solid 掩码（bool 数组）
  ↓ cp.asarray()
GPU: solid 常驻显存，全程不变
GPU: f（分布函数）直接在显存初始化
```

### 4.3 流动模块（最核心的改动）

CPU 版用 Numba 逐格点循环，CuPy 版改用向量化索引：

```
对每个方向 i（共9个）：
  用 cp.roll 处理 y 方向周期边界
  用 cp.where 处理 x 方向：
    xs < 0    → 入口平衡态
    xs >= NX  → 出口零梯度（复制最后一列）
    solid     → 反弹（取反方向）
    否则      → 正常 pull
```

注意：`cp.roll` 只用于 y 方向，x 方向显式处理，
避免 CPU 版的出口绕回入口 bug。

### 4.4 碰撞模块
向量化操作，改动最小：

```python
# 计算宏观量（全域一次性计算）
rho = f.sum(axis=0)                          # (NX, NY)
ux  = (EX[:,None,None] * f).sum(axis=0) / rho
uy  = (EY[:,None,None] * f).sum(axis=0) / rho

# 强制入口边界
ux[0, :] = U_lat
uy[0, :] = 0.0
rho[0,:] = 1.0

# BGK 碰撞
fEq = feq(rho, ux, uy)
f[:, ~solid] -= omega * (f[:, ~solid] - fEq[:, ~solid])
```

### 4.5 力计算模块
动量交换法，在 GPU 上用向量化实现：

```
对每个方向 i：
  找到固体表面格点（solid=True 且流体邻居）
  累加动量交换量 → Fx, Fy
Cd = 2*Fx / (0.5 * rho * U_lat² * D_lat)
Cl = 2*Fy / (0.5 * rho * U_lat² * D_lat)
```

### 4.6 数据记录模块
Cl/Cd 每 INTERVAL 步记录一次，
数值从 GPU 传回 CPU 只需一个标量，开销可忽略：

```python
cl_val = float(Cl)   # cp.ndarray → Python float（单次传输极小）
cl_list.append(cl_val)
```

场数据（流速、压力云图）在仿真结束后一次性传回：

```python
ux_cpu = cp.asnumpy(ux)   # 仿真结束后传回，不影响主循环性能
```

---

## 5. 性能预期

### CPU vs GPU 速度对比

| 版本 | 格子规模 | 估计速度 | 30000步耗时 |
|------|---------|---------|------------|
| CPU numpy | 400×400 | ~5 step/s | >1小时 |
| CPU Numba | 400×400 | ~56 step/s | ~9分钟 |
| GPU CuPy | 400×400 | ~500 step/s | ~1分钟 |
| GPU CuPy | 800×800 | ~200 step/s | ~2.5分钟 |
| GPU CuPy | 1600×1600 | ~60 step/s | ~8分钟 |

> 速度估算基于 Ada Lovelace 架构，实际运行后以终端输出为准。

### 瓶颈分析

```
流动步骤（cp.roll + cp.where）  ←  主要瓶颈，约占 60% 时间
碰撞步骤（纯向量化）            ←  约占 30% 时间
力计算（表面积分）              ←  约占 10% 时间
GPU↔CPU 数据传输               ←  可忽略（只传标量）
```

---

## 6. 开发顺序

```
Step 1  安装 CuPy，确认 GPU 可用
        conda install -c conda-forge cupy

Step 2  写 lbm_gpu.py
        先把 CPU 版的参数、D2Q9 常数、feq 函数原样复制
        只改 import 和数组初始化

Step 3  实现流动步骤（向量化 pull 方案）
        这是最需要仔细写的部分

Step 4  实现碰撞 + 力计算

Step 5  主循环 + 进度打印

Step 6  跑 D_lat=40（800×800），对比 CPU 版结果
        Cd、St 偏差应 < 1%（同一物理，只是换了执行设备）



## 8. 安装命令

```bash
# 激活环境
conda activate lbm_cuda

# 安装 CuPy（CUDA 12.x 对应版本）
conda install -c conda-forge cupy

# 验证 GPU 可用
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU found')"
python -c "import cupy as cp; a = cp.ones(1000); print('CuPy OK, device:', cp.cuda.Device())"
```