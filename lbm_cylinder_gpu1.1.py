"""
LBM 圆柱绕流 — CuPy RawKernel 算子融合版 (FP64)
访存合并优化: 内存布局 (9, NY, NX)，x 维度连续，同 warp 线程 stride=1
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time

# ══════════════════════════════════════════
# 1. 物理参数配置
# ══════════════════════════════════════════
Re      = 100
D_phys  = 0.1
nu_phys = 1.5e-5
V_phys  = Re * nu_phys / D_phys

D_lat   = 40
Ma_lat  = 0.05
cs      = 1.0 / np.sqrt(3.0)
U_lat   = Ma_lat * cs
nu_lat  = U_lat * D_lat / Re
tau     = nu_lat / (cs**2) + 0.5
omega   = 1.0 / tau

NX = int(30 * D_lat)
NY = int(20 * D_lat)
cx = int(8  * D_lat)
cy = NY // 2 + 1

dx_phys = D_phys / D_lat
dt_phys = dx_phys * U_lat / V_phys
CS2     = 1.0 / 3.0

print(f"D_lat   = {D_lat}, Ma = {Ma_lat}")
print(f"域      = {NX} x {NY}  (节点数: {NX*NY/10000:.1f} 万)")
print(f"tau     = {tau:.4f}\n")

# ══════════════════════════════════════════
# 2. CUDA Kernel 融合代码 (C++ 源码)
# ══════════════════════════════════════════
# 内存布局: (9, NY, NX) — x 维度在内存中连续
# f[i, y, x] 偏移 = i*NY*NX + y*NX + x
# 同 warp 内线程 x 连续 → 访存 stride=1 → 完美合并访问
lbm_kernel_code = r'''
extern "C" __global__
void lbm_fused_step(const double* f, double* f_new, const unsigned char* solid,
                    int NX, int NY, double omega, double U_lat, double CS2) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    constexpr int EX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    constexpr int EY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    constexpr double W[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                             1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
    constexpr int BOUNCE[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    // 布局 (9, NY, NX): f[i, y, x] 偏移 = i*NY*NX + y*NX + x
    auto idx = [&](int i, int xx, int yy) {
        return i * (NY * NX) + yy * NX + xx;
    };

    double local_f[9];

    // -- A. Pull-Streaming & Boundary Conditions --
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int xs = x - EX[i];
        int ys = y - EY[i];

        if (ys < 0) ys += NY;
        else if (ys >= NY) ys -= NY;

        if (xs < 0) {
            double eu_in = EX[i] * U_lat;
            double u2_in = U_lat * U_lat;
            local_f[i] = W[i] * (1.0 + eu_in/CS2 + 0.5*eu_in*eu_in/(CS2*CS2) - 0.5*u2_in/CS2);
        } else if (xs >= NX) {
            local_f[i] = f[idx(i, NX - 2, y)];
        } else if (solid[ys * NX + xs]) {
            local_f[i] = f[idx(BOUNCE[i], x, y)];
        } else {
            local_f[i] = f[idx(i, xs, ys)];
        }
    }

    // -- B. BGK Collision (only fluid nodes) --
    if (!solid[y * NX + x]) {
        double rho = 0.0, ux = 0.0, uy = 0.0;

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            rho += local_f[i];
            ux += EX[i] * local_f[i];
            uy += EY[i] * local_f[i];
        }
        ux /= rho;
        uy /= rho;

        double u2 = ux * ux + uy * uy;

        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            double eu = EX[i] * ux + EY[i] * uy;
            double feq = W[i] * rho * (1.0 + eu / CS2 + 0.5 * eu * eu / (CS2 * CS2) - 0.5 * u2 / CS2);
            local_f[i] -= omega * (local_f[i] - feq);
        }
    }

    // -- C. Write Back --
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        f_new[idx(i, x, y)] = local_f[i];
    }
}
'''
# 编译并加载 CUDA Kernel
lbm_fused_step = cp.RawKernel(lbm_kernel_code, 'lbm_fused_step')

# ══════════════════════════════════════════
# 3. 初始化数组 (布局: 9, NY, NX)
# ══════════════════════════════════════════
yy, xx    = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
solid_cpu = (xx - cx)**2 + (yy - cy)**2 <= (D_lat / 2)**2       # shape (NY, NX)
solid     = cp.asarray(solid_cpu, dtype=cp.uint8)

f     = cp.zeros((9, NY, NX), dtype=cp.float64)
f_new = cp.zeros((9, NY, NX), dtype=cp.float64)

EX_NP = np.array([0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
EY_NP = np.array([0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W_NP  = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36], dtype=np.float64)

EX_GPU = cp.asarray(EX_NP, dtype=cp.float64)[:, None, None]
EY_GPU = cp.asarray(EY_NP, dtype=cp.float64)[:, None, None]

# 入口平衡分布 (CPU 预计算，仅用于初始化)
f_eq_inlet_cpu = np.zeros(9, dtype=np.float64)
for i in range(9):
    eu = EX_NP[i] * U_lat
    u2 = U_lat ** 2
    f_eq_inlet_cpu[i] = W_NP[i] * (1.0 + eu/CS2 + 0.5*eu**2/(CS2**2) - 0.5*u2/CS2)

for i in range(9):
    f[i] = f_eq_inlet_cpu[i]

np.random.seed(42)
perturb_cpu = np.zeros((9, NY, NX), dtype=np.float64)
for i in range(9):
    perturb_cpu[i] = W_NP[i] * 1e-3 * (np.random.rand(NY, NX) - 0.5)
f += cp.asarray(perturb_cpu)

# ══════════════════════════════════════════
# 力计算掩码 (solid: NY x NX, axis0=y, axis1=x)
# ══════════════════════════════════════════
surface_masks = []
BOUNCE_NP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
for i in range(1, 9):
    ex = int(EX_NP[i])
    ey = int(EY_NP[i])
    # solid shape (NY, NX): axis0=y, axis1=x
    # shift=(-ey, -ex): 沿 y 移 -ey, 沿 x 移 -ex
    neighbor_fluid = cp.roll(~solid, shift=(-ey, -ex), axis=(0, 1))
    if ex == 1:
        neighbor_fluid[:, -1] = False      # 最右列没有 x+1 邻居
    elif ex == -1:
        neighbor_fluid[:, 0]  = False      # 最左列没有 x-1 邻居
    surface_masks.append((solid & neighbor_fluid).astype(cp.float64))

def compute_force(f):
    Fx = cp.float64(0.0)
    Fy = cp.float64(0.0)
    for idx, i in enumerate(range(1, 9)):
        ex = int(EX_NP[i])
        ey = int(EY_NP[i])
        bi = int(BOUNCE_NP[i])
        contrib = (f[bi] * surface_masks[idx]).sum()
        Fx += -ex * 2.0 * contrib
        Fy += -ey * 2.0 * contrib
    return float(Fx), float(Fy)

# ══════════════════════════════════════════
# 4. 主循环
# ══════════════════════════════════════════
N_STEPS  = 160000
WARMUP   = 40000
INTERVAL = 100

cl_history, cd_history, step_record = [], [], []

block_size = (16, 16)
grid_size  = ((NX + 15) // 16, (NY + 15) // 16)

print("开始 GPU RawKernel 计算 (访存合并优化)...")
cp.cuda.Stream.null.synchronize()
t_start = time.time()

for step in range(N_STEPS):
    lbm_fused_step(
        grid_size, block_size,
        (f, f_new, solid, NX, NY, cp.float64(omega), cp.float64(U_lat), cp.float64(CS2))
    )
    f, f_new = f_new, f

    if step % INTERVAL == 0 and step >= WARMUP:
        Fx, Fy = compute_force(f)
        q  = 0.5 * U_lat**2 * D_lat
        cd_history.append(float(Fx) / q)
        cl_history.append(float(Fy) / q)
        step_record.append(step)

    if step % 5000 == 0 and step > 0:
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - t_start
        speed   = step / elapsed
        eta     = (N_STEPS - step) / speed
        Cd_now  = cd_history[-1] if cd_history else 0.0
        Cl_now  = cl_history[-1] if cl_history else 0.0
        print(f"Step {step:6d}/{N_STEPS} | 速度={speed:.0f} 步/s | 剩余~{eta/60:.1f} min | Cd={Cd_now:.4f} | Cl={Cl_now:.4f}")

cp.cuda.Stream.null.synchronize()
total_time = time.time() - t_start
print(f"\n仿真完成! 总耗时 {total_time/60:.2f} min, 平均 {N_STEPS/total_time:.0f} step/s")

# ══════════════════════════════════════════
# 5. 后处理: Cl/Cd 时序 + FFT Strouhal 数
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

    print(f"\n{'='*44}")
    print(f"  Cd 均值 = {cd_arr.mean():.4f}   (文献 ~1.35)")
    print(f"  Cl 振幅 = {(cl_arr.max()-cl_arr.min())/2:.4f}   (文献 ~0.32)")
    print(f"  St      = {St:.4f}   (文献  0.164)")
    print(f"  Cd 误差 = {abs(cd_arr.mean()-1.35)/1.35*100:.1f}%")
    print(f"  St 误差 = {abs(St-0.164)/0.164*100:.1f}%")
    print(f"{'='*44}")

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
    print("Saved lbm_gpu_validation.png")

# ══════════════════════════════════════════
# 6. 最终流场快照
# ══════════════════════════════════════════
# f shape: (9, NY, NX), sum over axis 0 → (NY, NX)
rho_end = f.sum(axis=0)
ux_end  = (EX_GPU * f).sum(axis=0) / rho_end
uy_end  = (EY_GPU * f).sum(axis=0) / rho_end
umag    = cp.sqrt(ux_end**2 + uy_end**2)

umag_cpu = cp.asnumpy(umag)          # shape (NY, NX) — 无需转置

fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(umag_cpu, origin='lower', cmap='viridis', vmin=0, vmax=U_lat*1.4)
ax.contour(solid_cpu, levels=[0.5], colors='white', linewidths=1)
ax.set_title(f"Velocity Magnitude, step {N_STEPS}  (D_lat={D_lat}, GPU/CuPy)")
ax.set_xlabel("x [lu]")
ax.set_ylabel("y [lu]")
plt.colorbar(im, ax=ax, label="|u| [lu/ts]")
plt.tight_layout()
plt.savefig("lbm_gpu_field.png", dpi=120)
print("Saved lbm_gpu_field.png")
