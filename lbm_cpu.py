import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from numba import njit, prange
import time

# ══════════════════════════════════════════
# 物理参数
# ══════════════════════════════════════════
Re      = 100
D_phys  = 0.1
nu_phys = 1.5e-5
V_phys  = Re * nu_phys / D_phys

# ── 格子参数 ──────────────────────────────
# D_lat = 30 比 20 好得多：tau 更稳定、圆柱台阶更圆滑
# 想再提升精度可改成 40 或 50（成本随之线性上升）
D_lat   = 30
Ma_lat  = 0.05
cs      = 1.0 / np.sqrt(3.0)
U_lat   = Ma_lat * cs
nu_lat  = U_lat * D_lat / Re
tau     = nu_lat / (cs**2) + 0.5
omega   = 1.0 / tau

# ── 计算域：30D × 16D（关键修正）──────────
# 原版下游只有 15D 太短；30D × 16D 比较经济又能让尾涡正常发展
NX = int(30 * D_lat)        # 上游 8D + 下游 22D
NY = int(16 * D_lat)        # 上下各 8D
cx = int(8  * D_lat)
cy = NY // 2 + 1            # +1 偏置，打破上下对称、加速涡脱

dx_phys = D_phys / D_lat
dt_phys = dx_phys * U_lat / V_phys

CS2 = 1.0 / 3.0

EX     = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
EY     = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W      = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36])
BOUNCE = np.array([ 0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

print(f"D_lat   = {D_lat}, Ma = {Ma_lat}")
print(f"tau     = {tau:.4f}  {'✓' if tau > 0.5 else '✗ 不稳定！'}")
print(f"域      = {NX} × {NY}  ({NX/D_lat:.0f}D × {NY/D_lat:.0f}D)")
print(f"圆柱    = ({cx},{cy})  上游{cx/D_lat:.1f}D, 下游{(NX-cx)/D_lat:.1f}D")
print(f"dt_phys = {dt_phys:.4e} s/step")


# ══════════════════════════════════════════
# LBM 主步骤
# ══════════════════════════════════════════
@njit(cache=True, parallel=True)
def lbm_step(f, f_new, solid, EX, EY, W, BOUNCE,
             NX, NY, omega, U_lat, CS2):
    for x in prange(NX):
        for y in range(NY):

            # ── 1. Pull-streaming + BC ──────────
            for i in range(9):
                xs = x - EX[i]
                ys = (y - EY[i] + NY) % NY     # y 周期

                if xs < 0:
                    # 入口：均匀来流的平衡分布
                    eu = EX[i] * U_lat
                    u2 = U_lat * U_lat
                    f_new[i, x, y] = W[i] * (
                        1.0 + eu/CS2
                        + 0.5*eu*eu/(CS2*CS2)
                        - 0.5*u2/CS2)
                elif xs >= NX:
                    # 出口：零梯度外推
                    f_new[i, x, y] = f[i, NX-2, y]
                elif solid[xs, ys]:
                    # 半步反弹（halfway bounce-back）
                    f_new[i, x, y] = f[BOUNCE[i], x, y]
                else:
                    f_new[i, x, y] = f[i, xs, ys]

            # ── 2. 固体节点不做碰撞 ─────────────
            if solid[x, y]:
                continue

            # ── 3. 宏观量 ──────────────────────
            rho = 0.0
            ux  = 0.0
            uy  = 0.0
            for i in range(9):
                rho += f_new[i, x, y]
                ux  += EX[i] * f_new[i, x, y]
                uy  += EY[i] * f_new[i, x, y]
            ux /= rho
            uy /= rho

            # ── 4. BGK 碰撞 ─────────────────────
            u2 = ux*ux + uy*uy
            for i in range(9):
                eu  = EX[i]*ux + EY[i]*uy
                feq = (W[i] * rho
                       * (1.0 + eu/CS2
                          + 0.5*eu*eu/(CS2*CS2)
                          - 0.5*u2/CS2))
                f_new[i, x, y] -= omega * (f_new[i, x, y] - feq)

    return f_new


# ══════════════════════════════════════════
# 动量交换法（关键修正）
# ══════════════════════════════════════════
# 原理：F = Σ 2·c_α·f*_α(x_fluid)
#   其中 f*_α 是流体节点碰撞后、流向壁面的分布。
# 在 pull-streaming 完成后，f[α, x_solid] 恰好等于上一步流体节点
# 在 α 方向（指向壁）的碰撞后值——这就是 MEM 想要的东西，
# 比从碰撞过的 f[i, x_fluid] 反推干净得多（不会被 BGK 算子吃掉）。
@njit(cache=True, parallel=True)
def compute_force(f, solid, EX, EY, BOUNCE, NX, NY):
    Fx = 0.0
    Fy = 0.0
    for x in prange(NX):
        for y in range(NY):
            if not solid[x, y]:
                continue
            for i in range(1, 9):
                xn = x + EX[i]
                yn = (y + EY[i] + NY) % NY
                if xn < 0 or xn >= NX:
                    continue
                if solid[xn, yn]:
                    continue
                # α = BOUNCE[i] 指向壁内
                # f[α, x_solid] = 流体邻居碰撞后流向壁的分布
                Fx += 2.0 * EX[BOUNCE[i]] * f[BOUNCE[i], x, y]
                Fy += 2.0 * EY[BOUNCE[i]] * f[BOUNCE[i], x, y]
    return Fx, Fy


# ══════════════════════════════════════════
# 初始化
# ══════════════════════════════════════════
solid = np.zeros((NX, NY), dtype=np.bool_)
for x in range(NX):
    for y in range(NY):
        if (x-cx)**2 + (y-cy)**2 <= (D_lat/2)**2:
            solid[x, y] = True

f     = np.zeros((9, NX, NY))
f_new = np.zeros((9, NX, NY))

# 初始：U_lat 均匀流的平衡分布
for i in range(9):
    eu = EX[i] * U_lat
    u2 = U_lat**2
    f[i] = W[i] * (1.0 + eu/CS2 + 0.5*eu**2/CS2**2 - 0.5*u2/CS2)

# 随机扰动（替代正弦扰动）：破对称性更好、不会和 y-周期共振
np.random.seed(42)
for i in range(9):
    f[i] += W[i] * 1e-3 * (np.random.rand(NX, NY) - 0.5)

# ══════════════════════════════════════════
# 预热编译
# ══════════════════════════════════════════
print("\n编译 Numba 函数（首次约 10-30 秒）...")
f_new = lbm_step(f, f_new, solid, EX, EY, W, BOUNCE,
                 NX, NY, omega, U_lat, CS2)
f, f_new = f_new, f
_ = compute_force(f, solid, EX, EY, BOUNCE, NX, NY)
print("编译完成，开始多核仿真...\n")

# ══════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════
N_STEPS  = 60000
WARMUP   = 30000
INTERVAL = 50

cl_history, cd_history, step_record = [], [], []
t_start = time.time()

for step in range(N_STEPS):
    f_new = lbm_step(f, f_new, solid, EX, EY, W, BOUNCE,
                     NX, NY, omega, U_lat, CS2)
    f, f_new = f_new, f

    if step % INTERVAL == 0 and step >= WARMUP:
        Fx, Fy = compute_force(f, solid, EX, EY, BOUNCE, NX, NY)
        q  = 0.5 * U_lat**2 * D_lat       # rho_lat = 1
        cd_history.append(Fx / q)
        cl_history.append(Fy / q)
        step_record.append(step)

    if step % 2000 == 0 and step > 0:
        elapsed = time.time() - t_start
        speed   = step / elapsed
        eta     = (N_STEPS - step) / speed
        Cd_now  = cd_history[-1] if cd_history else 0.0
        Cl_now  = cl_history[-1] if cl_history else 0.0
        print(f"Step {step:6d}/{N_STEPS}  "
              f"速度={speed:.0f}步/s  "
              f"剩余≈{eta/60:.1f}min  "
              f"Cd={Cd_now:.4f}  Cl={Cl_now:.4f}")

print(f"\n总耗时: {(time.time()-t_start)/60:.1f} min")

# ══════════════════════════════════════════
# 后处理
# ══════════════════════════════════════════
cl_arr = np.array(cl_history)
cd_arr = np.array(cd_history)
t_arr  = np.array(step_record) * dt_phys

if len(cl_arr) > 10:
    N      = len(cl_arr)
    dt_rec = t_arr[1] - t_arr[0]
    freqs  = fftfreq(N, d=dt_rec)
    # 减掉直流分量再做 FFT，避免 0 Hz 峰值掩盖真实频率
    amps   = np.abs(fft(cl_arr - cl_arr.mean()))
    idx    = np.argmax(amps[1:N//2]) + 1
    f_s    = freqs[idx]
    St     = f_s * D_phys / V_phys

    print(f"\n{'═'*44}")
    print(f"  Cd 均值 = {cd_arr.mean():.4f}   (文献 ~1.35)")
    print(f"  Cl 振幅 = {(cl_arr.max()-cl_arr.min())/2:.4f}   (文献 ~0.32)")
    print(f"  St      = {St:.4f}   (文献  0.164)")
    print(f"  Cd 偏差 = {abs(cd_arr.mean()-1.35)/1.35*100:.1f}%")
    print(f"  St 偏差 = {abs(St-0.164)/0.164*100:.1f}%")
    print(f"{'═'*44}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_arr, cl_arr, label='Cl')
    axes[0].plot(t_arr, cd_arr, label='Cd')
    axes[0].set_xlabel("Physical Time [s]")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title(f"Re = {Re}")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(freqs[1:N//2], amps[1:N//2])
    axes[1].axvline(f_s, color='r', linestyle='--',
                    label=f'St = {St:.4f}')
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Cl Spectrum")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("lbm_cpu_validation.png", dpi=150)
    print("图像已保存到 lbm_cpu_validation.png")
