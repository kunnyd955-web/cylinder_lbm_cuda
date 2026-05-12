import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from numba import njit, prange
import time

# ══════════════════════════════════════════
# 参数
# ══════════════════════════════════════════
Re      = 100
D_phys  = 0.1
nu_phys = 1.5e-5
V_phys  = Re * nu_phys / D_phys

# 建议：如果想得到更精确的文献比对结果，可将 D_lat 提升至 40 或 50
D_lat   = 20
Ma_lat  = 0.05
cs      = 1.0 / np.sqrt(3.0)
U_lat   = Ma_lat * cs
nu_lat  = U_lat * D_lat / Re
tau     = nu_lat / (cs**2) + 0.5
omega   = 1.0 / tau

NX = int(20 * D_lat)   
NY = int(20 * D_lat)   
cx = int(5  * D_lat)   
cy = NY // 2           

dx_phys = D_phys / D_lat
dt_phys = dx_phys * U_lat / V_phys

CS2 = 1.0 / 3.0

EX     = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
EY     = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W      = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36])
BOUNCE = np.array([ 0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

print(f"tau    = {tau:.4f}  {'✓' if tau > 0.5 else '✗ 不稳定！'}")
print(f"域大小 = {NX} × {NY}")
print(f"dt     = {dt_phys:.4e} s/步")

# ══════════════════════════════════════════
# Numba 加速的核心函数 (已开启 parallel=True 并使用 prange)
# ══════════════════════════════════════════

@njit(cache=True, parallel=True)
def lbm_step(f, f_new, solid, EX, EY, W, BOUNCE,
             NX, NY, omega, U_lat, CS2):
    """
    单步 LBM：流动 + 碰撞合并在一个函数里。
    使用 parallel=True 和 prange 自动进行多核并行计算。
    """
    for x in prange(NX):
        for y in range(NY):

            # ── 1. 流动（pull）──────────────────────
            for i in range(9):
                xs = x - EX[i]
                ys = (y - EY[i] + NY) % NY   

                if xs < 0:
                    rho_in = 1.0
                    eu = EX[i] * U_lat
                    u2 = U_lat * U_lat
                    f_new[i, x, y] = (W[i] * rho_in
                        * (1.0 + eu/CS2
                           + 0.5*eu*eu/(CS2*CS2)
                           - 0.5*u2/CS2))
                elif xs >= NX:
                    f_new[i, x, y] = f[i, NX-2, y]
                elif solid[xs, ys]:
                    f_new[i, x, y] = f[BOUNCE[i], x, y]
                else:
                    f_new[i, x, y] = f[i, xs, ys]

            # ── 2. 固体节点不做碰撞 ──────────────────
            if solid[x, y]:
                continue

            # ── 3. 计算宏观量 ────────────────────────
            rho = 0.0
            ux  = 0.0
            uy  = 0.0
            for i in range(9):
                rho += f_new[i, x, y]
                ux  += EX[i] * f_new[i, x, y]
                uy  += EY[i] * f_new[i, x, y]
            ux /= rho
            uy /= rho

            #if x == 0:
             #   ux  = U_lat
              #  uy  = 0.0
              #  rho = 1.0

            # ── 4. 碰撞（BGK）───────────────────────
            u2 = ux*ux + uy*uy
            for i in range(9):
                eu  = EX[i]*ux + EY[i]*uy
                feq = (W[i] * rho
                       * (1.0 + eu/CS2
                          + 0.5*eu*eu/(CS2*CS2)
                          - 0.5*u2/CS2))
                f_new[i, x, y] -= omega * (f_new[i, x, y] - feq)

    return f_new


@njit(cache=True, parallel=True)
def compute_force(f, solid, EX, EY, W, BOUNCE, NX, NY, U_lat):
    """动量交换法计算升阻力 (已修复公式并开启并行)。"""
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
                # 修复后的动量交换公式：提取反弹方向的分布函数之和
                Fx += EX[BOUNCE[i]] * (f[i, xn, yn] + f[BOUNCE[i], x, y])
                Fy += EY[BOUNCE[i]] * (f[i, xn, yn] + f[BOUNCE[i], x, y])
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

for i in range(9):
    eu = EX[i] * U_lat
    u2 = U_lat**2
    f[i] = W[i] * (1.0 + eu/CS2 + 0.5*eu**2/CS2**2 - 0.5*u2/CS2)

Y_grid = np.tile(np.arange(NY), (NX, 1))
for i in range(9):
    f[i] += W[i] * 1e-4 * np.sin(2 * np.pi * Y_grid / NY)

# ══════════════════════════════════════════
# 预热编译
# ══════════════════════════════════════════
print("\n正在编译 Numba 函数（首次运行需要约 10-30 秒）...")
f_new = lbm_step(f, f_new, solid, EX, EY, W, BOUNCE,
                 NX, NY, omega, U_lat, CS2)
f, f_new = f_new, f
print("编译完成，开始多核并行仿真...\n")

# ══════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════
N_STEPS  = 40000
WARMUP   = 20000
INTERVAL = 100

cl_history  = []
cd_history  = []
step_record = []

t_start = time.time()

for step in range(N_STEPS):
    f_new = lbm_step(f, f_new, solid, EX, EY, W, BOUNCE,
                     NX, NY, omega, U_lat, CS2)
    f, f_new = f_new, f   

    if step % INTERVAL == 0 and step >= WARMUP:
        Fx, Fy = compute_force(f, solid, EX, EY, W, BOUNCE,
                               NX, NY, U_lat)
        q  = 0.5 * 1.0 * U_lat**2 * D_lat
        Cd = Fx / q
        Cl = Fy / q
        cl_history.append(Cl)
        cd_history.append(Cd)
        step_record.append(step)

    if step % 2000 == 0 and step > 0:
        elapsed  = time.time() - t_start
        speed    = step / elapsed          
        eta      = (N_STEPS - step) / speed
        Cd_now   = cd_history[-1] if cd_history else 0.0
        Cl_now   = cl_history[-1] if cl_history else 0.0
        print(f"Step {step:6d}/{N_STEPS}  "
              f"速度={speed:.0f}步/s  "
              f"剩余≈{eta/60:.1f}分钟  "
              f"Cd={Cd_now:.4f}  Cl={Cl_now:.4f}")

print(f"\n总耗时：{(time.time()-t_start)/60:.1f} 分钟")

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
    amps   = np.abs(fft(cl_arr))
    idx    = np.argmax(amps[1:N//2]) + 1
    f_s    = freqs[idx]
    St     = f_s * D_phys / V_phys

    print(f"\n{'═'*40}")
    print(f"LBM CPU 并行结果（D={D_lat} 格点）")
    print(f"  Cd 均值 = {cd_arr.mean():.4f}  （文献: ~1.35）")
    print(f"  Cl 振幅 = {(cl_arr.max()-cl_arr.min())/2:.4f}")
    print(f"  St      = {St:.4f}  （文献: 0.164）")
    print(f"  Cd 偏差 = {abs(cd_arr.mean()-1.35)/1.35*100:.1f}%")
    print(f"  St 偏差 = {abs(St-0.164)/0.164*100:.1f}%")
    print(f"{'═'*40}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_arr, cl_arr)
    axes[0].set_xlabel("Physical Time [s]")
    axes[0].set_ylabel("Cl")
    axes[0].set_title(f"Lift Coefficient Re={Re}")
    axes[0].grid(True)

    axes[1].plot(freqs[1:N//2], amps[1:N//2])
    axes[1].axvline(f_s, color='r', linestyle='--',
                    label=f'St={St:.4f}')
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Cl Spectrum")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("lbm_cpu_validation.png", dpi=150)
    print("图像已保存到 lbm_cpu_validation.png")