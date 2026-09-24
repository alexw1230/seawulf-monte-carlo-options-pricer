import time

import cupy as cp
import numpy as np

# ---- 1. What GPU do we have? -----------------------------------------
n_gpus = cp.cuda.runtime.getDeviceCount()
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"GPUs visible: {n_gpus}")
print(f"GPU 0: {props['name'].decode()}, {props['totalGlobalMem'] / 1e9:.0f} GB memory")

# ---- 2. Same math on CPU and GPU should agree ------------------------
x_cpu = np.linspace(-3, 3, 1_000_000)
x_gpu = cp.asarray(x_cpu)          # copy host (CPU) memory -> device (GPU) memory
cpu_result = float(np.exp(x_cpu).sum())
gpu_result = float(cp.exp(x_gpu).sum())   # float() copies the one number back
print(f"CPU sum: {cpu_result:.10f}")
print(f"GPU sum: {gpu_result:.10f}   match: {np.isclose(cpu_result, gpu_result)}")

# ---- 3. Timing rules -------------------------------------------------
a = cp.random.standard_normal(100_000_000)     # 1e8 float64 values, 800 MB, made on the GPU

# Rule 1: the first call to a GPU function compiles it. Never time the first call.
t0 = time.perf_counter()
cp.exp(a)
cp.cuda.Device().synchronize()
t_first = time.perf_counter() - t0

# Rule 2: GPU calls return immediately. Without synchronize() you time the queueing.
t0 = time.perf_counter()
b = cp.exp(a)
t_nosync = time.perf_counter() - t0
cp.cuda.Device().synchronize()

t0 = time.perf_counter()
b = cp.exp(a)
cp.cuda.Device().synchronize()                 # wait for the GPU to actually finish
t_sync = time.perf_counter() - t0

print(f"\nexp() over 1e8 values:")
print(f"  first call (includes compile): {t_first * 1e3:8.2f} ms")
print(f"  without synchronize (WRONG):   {t_nosync * 1e3:8.2f} ms")
print(f"  with synchronize (real):       {t_sync * 1e3:8.2f} ms")

# exp reads 8 bytes and writes 8 bytes per value, so this is memory traffic / time
gb_moved = 100_000_000 * 16 / 1e9
print(f"  effective memory bandwidth:    {gb_moved / t_sync:8.0f} GB/s")