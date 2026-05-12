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

# ── 格子参数 (根据建议优化) ────────────────
# 提升 D_lat 到 40，使得 tau 达到约 0.535，大幅提升 BGK 模型的数值稳定性
D_lat   = 40
Ma_lat  = 0.05
cs      = 1.0 / np.sqrt(3.0)
U_lat   = Ma_lat * cs
nu_lat  = U_lat * D_lat / Re
tau     = nu_lat / (cs**2) + 0.5
omega   = 1.0 / tau

# ── 计算域：30D × 20D (根据建议优化) ────────
# 将高度扩大到 20D，消除 Y 方向周期性边界带来的阻塞效应和镜像尾涡干扰
NX = int(30 * D_lat)        # 上游 8D + 下游 22D
NY = int(20 * D_lat)        # 上下各 10D
cx = int(8  * D_lat)
cy = NY // 2 + 1            # 偏置 1 格，打破对称性加速涡脱

dx_phys = D_phys / D_lat
dt_phys = dx_phys * U_lat / V_phys

CS2 = 1.0 / 3.0

EX     = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
EY     = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W      = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36])
BOUNCE = np.array([ 0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

print(f"D_lat   = {D_lat}, Ma = {Ma_lat}")
print(f"tau     = {tau:.4f}  {'✓ 稳定' if tau > 0.53 else '⚠ 偏低'}")
print(f"域      = {NX} × {NY}  ({NX/D_lat:.0f}D × {NY/D_lat:.0f}D)")
print(f"dt_phys = {dt_phys:.4e} s/step")

# ══════════════════════════════════════════
# LBM 核心算子
# ══════════════════════════════════════════
@njit(cache=True, parallel=True)
def lbm_step(f, f_new, solid, EX, EY, W, BOUNCE, NX, NY, omega, U_lat, CS2):
    for x in prange(NX):
        for y in range(NY):
            # ── 1. Pull-streaming + BC ──────────
            for i in range(9):
                xs = x - EX[i]
                ys = (y - EY[i] + NY) % NY     

                if xs < 0:
                    eu = EX[i] * U_lat
                    u2 = U_lat * U_lat
                    f_new[i, x, y] = W[i] * (1.0 + eu/CS2 + 0.5*eu*eu/(CS2*CS2) - 0.5*u2/CS2)
                elif xs >= NX:
                    f_new[i, x, y] = f[i, NX-2, y]
                elif solid[xs, ys]:
                    f_new[i, x, y] = f[BOUNCE[i], x, y]
                else:
                    f_new[i, x, y] = f[i, xs, ys]

            # ── 2. 固体节点跳过碰撞 ─────────────
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
                feq = (W[i] * rho * (1.0 + eu/CS2 + 0.5*eu*eu/(CS2*CS2) - 0.5*u2/CS2))
                f_new[i, x, y] -= omega * (f_new[i, x, y] - feq)

    return f_new


@njit(cache=True, parallel=True)
def compute_force(f, solid, EX, EY, BOUNCE, NX, NY):
    # Numba 的 prange 自动处理这里简单的 += 标量规约，是线程安全的
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
                Fx += 2.0 * EX[BOUNCE[i]] * f[BOUNCE[i], x, y]
                Fy += 2.0 * EY[BOUNCE[i]] * f[BOUNCE[i], x, y]
    return Fx, Fy


# ══════════════════════════════════════════
# 初始化
# ══════════════════════════════════════════
# 根据建议，将废弃的 np.bool_ 修改为原生 bool
solid = np.zeros((NX, NY), dtype=bool)
for x in range(NX):
    for y in range(NY):
        if (x-cx)**2 + (y-cy)**2 <= (D_lat/2)**2:
            solid[x, y] = True

f     = np.zeros((9, NX, NY))
f_new = np.zeros((9, NX, NY))

for i in range(9):
    eu = EX[i] * U_lat
    u2 = U_lat**2
    f[i] = W[i] * (1.0 + eu/CS2 + 0.5*eu**2/CS2**2 - 0.5*u2/CS2)

np.random.seed(42)
for i in range(9):
    f[i] += W[i] * 1e-3 * (np.random.rand(NX, NY) - 0.5)

# ══════════════════════════════════════════
# 预热编译
# ══════════════════════════════════════════
print("\n编译 Numba 函数...")
f_new = lbm_step(f, f_new, solid, EX, EY, W, BOUNCE, NX, NY, omega, U_lat, CS2)
f, f_new = f_new, f
_ = compute_force(f, solid, EX, EY, BOUNCE, NX, NY)
print("编译完成！\n")

# ══════════════════════════════════════════
# 主循环 (根据建议大幅延长仿真时间)
# ══════════════════════════════════════════
# 增加 N_STEPS 以获取极高分辨率的 FFT 频谱
N_STEPS  = 160000 
WARMUP   = 40000    # 前 4 万步让流场和卡门涡街彻底稳定下来
INTERVAL = 100       # 每 100 步采样一次

cl_history, cd_history, step_record = [], [], []
t_start = time.time()

for step in range(N_STEPS):
    f_new = lbm_step(f, f_new, solid, EX, EY, W, BOUNCE, NX, NY, omega, U_lat, CS2)
    f, f_new = f_new, f

    if step % INTERVAL == 0 and step >= WARMUP:
        Fx, Fy = compute_force(f, solid, EX, EY, BOUNCE, NX, NY)
        q  = 0.5 * U_lat**2 * D_lat       
        cd_history.append(Fx / q)
        cl_history.append(Fy / q)
        step_record.append(step)

    if step % 5000 == 0 and step > 0:
        elapsed = time.time() - t_start
        speed   = step / elapsed
        eta     = (N_STEPS - step) / speed
        Cd_now  = cd_history[-1] if cd_history else 0.0
        Cl_now  = cl_history[-1] if cl_history else 0.0
        print(f"Step {step:6d}/{N_STEPS}  速度={speed:.0f}步/s  剩余≈{eta/60:.1f}min  Cd={Cd_now:.4f}  Cl={Cl_now:.4f}")

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
    axes[0].set_title(f"Re = {Re}")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(freqs[1:N//2], amps[1:N//2])
    axes[1].axvline(f_s, color='r', linestyle='--', label=f'St = {St:.4f}')
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Cl Spectrum")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("lbm_ultimate_validation.png", dpi=150)