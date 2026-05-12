"""
LBM 圆柱绕流 — CuPy GPU 加速版

对应 CPU 验证版 lbm_cylinder_revise.py，物理参数完全一致。
设计参考 cupy_lbm_design.md：
  - 数组常驻 GPU 显存 (cp.ndarray, float32)
  - pull-streaming 用 cp.roll 处理 y 周期 + cp.where 处理固体反弹
  - x 方向入口/出口边界显式覆盖，避免周期绕回
  - 仅 Cl/Cd 标量传回 CPU，场数据仿真结束后一次性传回
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time

# ══════════════════════════════════════════
# 物理参数
# ══════════════════════════════════════════
Re      = 100
D_phys  = 0.1
nu_phys = 1.5e-5
V_phys  = Re * nu_phys / D_phys

# ── 格子参数 ──────────────────────────────
D_lat   = 40
Ma_lat  = 0.05
cs      = 1.0 / np.sqrt(3.0)
U_lat   = Ma_lat * cs
nu_lat  = U_lat * D_lat / Re
tau     = nu_lat / (cs**2) + 0.5
omega   = 1.0 / tau

# ── 计算域：30D × 20D ─────────────────────
NX = int(30 * D_lat)        # 上游 8D + 下游 22D
NY = int(20 * D_lat)        # 上下各 10D
cx = int(8  * D_lat)
cy = NY // 2 + 1            # 偏置 1 格，破坏对称、加速涡脱

dx_phys = D_phys / D_lat
dt_phys = dx_phys * U_lat / V_phys

CS2   = 1.0 / 3.0
DTYPE = cp.float32          # 单精度，800×800 域 ~72 MB 双缓冲

# D2Q9 速度集
EX_NP     = np.array([0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
EY_NP     = np.array([0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W_NP      = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36], dtype=np.float32)
BOUNCE_NP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

EX_GPU = cp.asarray(EX_NP, dtype=DTYPE)
EY_GPU = cp.asarray(EY_NP, dtype=DTYPE)
W_GPU  = cp.asarray(W_NP,  dtype=DTYPE)

print(f"D_lat   = {D_lat}, Ma = {Ma_lat}")
print(f"tau     = {tau:.4f}  {'✓ 稳定' if tau > 0.53 else '⚠ 偏低'}")
print(f"域      = {NX} × {NY}  ({NX/D_lat:.0f}D × {NY/D_lat:.0f}D)")
print(f"dt_phys = {dt_phys:.4e} s/step")

dev = cp.cuda.Device(0)
free_mem, total_mem = cp.cuda.runtime.memGetInfo()
print(f"GPU     = device {dev.id}, 显存 {free_mem/1e9:.2f}/{total_mem/1e9:.2f} GB 可用\n")

# ══════════════════════════════════════════
# 初始化
# ══════════════════════════════════════════
# 1) 固体掩码在 CPU 上算一次，再传到 GPU
xx, yy    = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
solid_cpu = (xx - cx)**2 + (yy - cy)**2 <= (D_lat / 2)**2
solid     = cp.asarray(solid_cpu)
fluid_mask = (~solid).astype(DTYPE)         # 1=流体, 0=固体；碰撞时按位掩码

# 2) 入口平衡分布 (常量，预先算好)
f_eq_inlet_cpu = np.zeros(9, dtype=np.float32)
for i in range(9):
    eu = EX_NP[i] * U_lat
    u2 = U_lat ** 2
    f_eq_inlet_cpu[i] = W_NP[i] * (1.0 + eu/CS2 + 0.5*eu**2/(CS2**2) - 0.5*u2/CS2)
f_eq_inlet = cp.asarray(f_eq_inlet_cpu)

# 3) 分布函数：全域均匀来流平衡分布 + 微小扰动
f = cp.empty((9, NX, NY), dtype=DTYPE)
for i in range(9):
    f[i] = f_eq_inlet_cpu[i]

np.random.seed(42)                          # 与 CPU 版同种子，便于交叉对比
perturb_cpu = np.zeros((9, NX, NY), dtype=np.float32)
for i in range(9):
    perturb_cpu[i] = (W_NP[i] * 1e-3 * (np.random.rand(NX, NY) - 0.5)).astype(np.float32)
f += cp.asarray(perturb_cpu)

f_new = cp.empty_like(f)


# ══════════════════════════════════════════
# 流动步骤 (pull-streaming + 边界)
# ══════════════════════════════════════════
def stream(f, f_new):
    """
    f_new[i, x, y] := f[i, x-EX[i], y-EY[i]] 加上边界处理:
      - y 方向：周期 (cp.roll 自动处理)
      - x 方向：入口 (x=0,  ex=+1) → 平衡态
                出口 (x=-1, ex=-1) → 零梯度 (复制 x=-2 列)
      - 固体源：反弹 (取本格 BOUNCE[i] 方向)
    """
    for i in range(9):
        ex = int(EX_NP[i])
        ey = int(EY_NP[i])
        bi = int(BOUNCE_NP[i])

        if ex == 0 and ey == 0:
            f_new[i] = f[i]                 # 静止分量原地复制
        else:
            f_pulled  = cp.roll(f[i],  shift=(ex, ey), axis=(0, 1))
            solid_src = cp.roll(solid, shift=(ex, ey), axis=(0, 1))
            # 源格点是固体 → 反弹 (用本格 BOUNCE[i] 方向的分布)
            f_new[i] = cp.where(solid_src, f[bi], f_pulled)

        # X 方向边界覆盖：纠正 cp.roll 在 x 方向的错误绕回
        if ex == 1:
            f_new[i, 0, :]  = f_eq_inlet[i]                 # 入口平衡态
        elif ex == -1:
            f_new[i, -1, :] = f[i, -2, :]                   # 出口零梯度


# ══════════════════════════════════════════
# 碰撞步骤 (BGK，向量化)
# ══════════════════════════════════════════
def collide(f):
    """f -= ω(f - feq)，固体节点不参与碰撞 (乘 fluid_mask)"""
    rho = f.sum(axis=0)
    ux  = (EX_GPU[:, None, None] * f).sum(axis=0) / rho
    uy  = (EY_GPU[:, None, None] * f).sum(axis=0) / rho

    eu  = EX_GPU[:, None, None] * ux + EY_GPU[:, None, None] * uy
    u2  = ux * ux + uy * uy
    feq = W_GPU[:, None, None] * rho * (
        1.0 + eu/CS2 + 0.5*eu*eu/(CS2**2) - 0.5*u2/CS2
    )

    f -= omega * (f - feq) * fluid_mask[None, :, :]


# ══════════════════════════════════════════
# 力计算 (动量交换法)
# ══════════════════════════════════════════
# 预计算每个方向上 "固体格 & 流体邻居" 的表面掩码 (solid 不变，只算一次)
surface_masks = []
for i in range(1, 9):
    ex = int(EX_NP[i])
    ey = int(EY_NP[i])
    neighbor_fluid = cp.roll(~solid, shift=(-ex, -ey), axis=(0, 1))
    if ex == 1:
        neighbor_fluid[-1, :] = False     # 最右列没有 x+1 邻居
    elif ex == -1:
        neighbor_fluid[0, :]  = False     # 最左列没有 x-1 邻居
    surface_masks.append((solid & neighbor_fluid).astype(DTYPE))


def compute_force(f):
    """
    动量交换法:  F = Σ_i  2 · EX[BOUNCE[i]] · f[BOUNCE[i], x, y]
    其中 (x, y) 是固体表面格、(x+ex, y+ey) 是其流体邻居。
    EX[BOUNCE[i]] = -EX[i]，于是 Fx += -2·ex·f[BOUNCE[i]]·surface
    """
    Fx = cp.zeros((), dtype=DTYPE)
    Fy = cp.zeros((), dtype=DTYPE)
    for idx, i in enumerate(range(1, 9)):
        ex = int(EX_NP[i])
        ey = int(EY_NP[i])
        bi = int(BOUNCE_NP[i])
        contrib = (f[bi] * surface_masks[idx]).sum()
        Fx += -ex * 2.0 * contrib
        Fy += -ey * 2.0 * contrib
    return Fx, Fy


# ══════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════
N_STEPS  = 160000
WARMUP   = 40000
INTERVAL = 100

cl_history, cd_history, step_record = [], [], []

cp.cuda.Stream.null.synchronize()           # 确保初始化数据已就绪
t_start = time.time()

for step in range(N_STEPS):
    stream(f, f_new)
    f, f_new = f_new, f
    collide(f)

    if step % INTERVAL == 0 and step >= WARMUP:
        Fx, Fy = compute_force(f)
        q  = 0.5 * U_lat**2 * D_lat
        cd_history.append(float(Fx) / q)    # 单标量传回 CPU，开销可忽略
        cl_history.append(float(Fy) / q)
        step_record.append(step)

    if step % 5000 == 0 and step > 0:
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - t_start
        speed   = step / elapsed
        eta     = (N_STEPS - step) / speed
        Cd_now  = cd_history[-1] if cd_history else 0.0
        Cl_now  = cl_history[-1] if cl_history else 0.0
        print(f"Step {step:6d}/{N_STEPS}  速度={speed:.0f}步/s  剩余≈{eta/60:.1f}min  "
              f"Cd={Cd_now:.4f}  Cl={Cl_now:.4f}")

cp.cuda.Stream.null.synchronize()
total_time = time.time() - t_start
print(f"\n仿真完成! 总耗时 {total_time/60:.2f} min, 平均 {N_STEPS/total_time:.0f} step/s")


# ══════════════════════════════════════════
# 后处理：Cl/Cd 时序 + FFT 斯特劳哈尔数
# ══════════════════════════════════════════
cl_arr = np.array(cl_history)
cd_arr = np.array(cd_history)
t_arr  = np.array(step_record) * dt_phys

if len(cl_arr) > 10:
    N      = len(cl_arr)
    dt_rec = t_arr[1] - t_arr[0]
    freqs  = fftfreq(N, d=dt_rec)
    amps   = np.abs(fft(cl_arr - cl_arr.mean()))
    idx    = np.argmax(amps[1:N//2]) + 1
    f_s    = freqs[idx]
    St     = f_s * D_phys / V_phys

    print(f"\n{'═'*44}")
    print(f"  Cd 均值 = {cd_arr.mean():.4f}   (文献 ~1.35)")
    print(f"  Cl 振幅 = {(cl_arr.max()-cl_arr.min())/2:.4f}   (文献 ~0.32)")
    print(f"  St      = {St:.4f}   (文献  0.164)")
    print(f"  Cd 误差 = {abs(cd_arr.mean()-1.35)/1.35*100:.1f}%")
    print(f"  St 误差 = {abs(St-0.164)/0.164*100:.1f}%")
    print(f"{'═'*44}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_arr, cl_arr, label='Cl')
    axes[0].plot(t_arr, cd_arr, label='Cd')
    axes[0].set_xlabel("Physical Time [s]")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title(f"Re = {Re} (GPU/CuPy)")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(freqs[1:N//2], amps[1:N//2])
    axes[1].axvline(f_s, color='r', linestyle='--', label=f'St = {St:.4f}')
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Cl Spectrum (GPU)")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("lbm_gpu_validation.png", dpi=150)
    print("→ 已保存 lbm_gpu_validation.png")


# ══════════════════════════════════════════
# 最终流场快照 (一次性 GPU → CPU)
# ══════════════════════════════════════════
rho_end = f.sum(axis=0)
ux_end  = (EX_GPU[:, None, None] * f).sum(axis=0) / rho_end
uy_end  = (EY_GPU[:, None, None] * f).sum(axis=0) / rho_end
umag    = cp.sqrt(ux_end**2 + uy_end**2)

# 流体外+固体内的速度都画出来，固体掩码再叠白色等值线
umag_cpu = cp.asnumpy(umag)

fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(umag_cpu.T, origin='lower', cmap='viridis', vmin=0, vmax=U_lat*1.4)
ax.contour(solid_cpu.T, levels=[0.5], colors='white', linewidths=1)
ax.set_title(f"Velocity Magnitude, step {N_STEPS}  (D_lat={D_lat}, GPU/CuPy)")
ax.set_xlabel("x [lu]")
ax.set_ylabel("y [lu]")
plt.colorbar(im, ax=ax, label="|u| [lu/ts]")
plt.tight_layout()
plt.savefig("lbm_gpu_field.png", dpi=120)
print("→ 已保存 lbm_gpu_field.png")
