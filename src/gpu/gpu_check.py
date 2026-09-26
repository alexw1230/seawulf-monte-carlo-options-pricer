import time

import cupy as cp
import numpy as np

n_gpus = cp.cuda.runtime.getDeviceCount()
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"GPUs visible: {n_gpus}")
print(f"GPU 0: {props['name'].decode()}, {props['totalGlobalMem'] / 1e9:.0f} GB memory")

#cpu gpu agree?
x_cpu = np.linspace(-3, 3, 1000000)
x_gpu = cp.asarray(x_cpu)
cpu_result = float(np.exp(x_cpu).sum())
gpu_result = float(cp.exp(x_gpu).sum())
print(f"CPU sum: {cpu_result:.10f}")
print(f"GPU sum: {gpu_result:.10f}   match: {np.isclose(cpu_result, gpu_result)}")

a = cp.random.standard_normal(100000000)

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
cp.cuda.Device().synchronize() # wait for the GPU to finish
t_sync = time.perf_counter() - t0

print(f"\nexp() over 1e8 values:")
print(f"  first call (includes compile): {t_first * 1e3:8.2f} ms")
print(f"  without synchronize (WRONG):   {t_nosync * 1e3:8.2f} ms")
print(f"  with synchronize (real):       {t_sync * 1e3:8.2f} ms")

gb_moved = 100000000 * 16 / 1e9
print(f"  effective memory bandwidth:    {gb_moved / t_sync:8.0f} GB/s")