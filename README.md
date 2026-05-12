# cylinder_lbm_cuda

D2Q9 Lattice Boltzmann simulation of flow past a cylinder at Re=100, implemented in Python (Numba CPU) and CuPy GPU. Validates against benchmark values: Cd ≈ 1.35, St ≈ 0.164.

## Results

| Version | Domain | Dtype | Backend |
|---------|--------|-------|---------|
| CPU (Numba) | 900×480 (30D×16D) | float64 | multi-core |
| GPU v1.0 | 1200×800 (30D×20D) | float32 | CuPy vectorized |
| GPU v1.1 | 1200×800 (30D×20D) | float64 | CuPy RawKernel fused |

Validation plots are in `results/`.

## Physics

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Reynolds number | Re | 100 |
| Cylinder diameter | D_phys | 0.1 m |
| Kinematic viscosity | ν | 1.5×10⁻⁵ m²/s |
| Freestream velocity | U_phys | 0.015 m/s |
| Lattice Mach number | Ma | 0.05 |
| Relaxation time | τ | ~0.535 |

## File Structure

```
.
├── lbm_cpu.py              # CPU version (Numba + parallel)
├── lbm_gpu_cupy.py         # GPU v1.0 (CuPy, FP32, vectorized)
├── lbm_gpu_rawkernel.py    # GPU v1.1 (CuPy RawKernel, FP64, fused stream+collide)
├── src/
│   └── device_check.cu     # CUDA device info utility
├── results/                # Validation plots (Cd/Cl time series, flow field)
├── docs/
│   └── cupy_design.md      # GPU implementation design notes
└── archive/                # Earlier draft versions
```

## Environment

- OS: WSL2 (Ubuntu)
- GPU: RTX 4060, Compute Capability 8.9
- Python: 3.13 (Miniconda, env `lbm_cuda`)
- Dependencies: `numpy scipy matplotlib numba cupy`

```bash
conda activate lbm_cuda
```

## Running

```bash
# CPU version (~10 min, D_lat=30)
python lbm_cpu.py

# GPU v1.0 (FP32, D_lat=40)
python lbm_gpu_cupy.py

# GPU v1.1 (FP64, RawKernel, D_lat=40)
python lbm_gpu_rawkernel.py

# CUDA device check
nvcc -arch=sm_89 -O3 src/device_check.cu -o device_check && ./device_check
```

## Benchmark Targets

| Metric | Literature | Target error |
|--------|-----------|-------------|
| Cd | 1.35 | < 15% |
| St | 0.164 | < 10% |
