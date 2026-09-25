"""Monte Carlo option pricer as ONE fused GPU kernel.

Run from src/:
    python3 -m gpu.fused_pricer --n 10000000000          # readable output
    python3 -m gpu.fused_pricer --n 10000000000 --csv    # one CSV row
    python3 -m gpu.fused_pricer --check                  # compare to CPU reference

Every thread generates its own random numbers (Philox, counter = trial pair
index), computes both payoffs, and keeps four running sums in registers.
GPU memory is touched only once per block, to write the block's sums. The
algorithm is identical to philox_ref.py, which serves as the CPU reference.
"""
import argparse
import math
import time

import cupy as cp
import numpy as np

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
THREADS = 256           # threads per block (a multiple of the 32-thread warp)

KERNEL_SRC = r"""
extern "C" __global__
void mc_fused(const unsigned long long pair_start,   // first trial pair for this launch
              const unsigned long long pair_end,     // one past the last pair
              const unsigned long long n_total,      // total trials (cuts off an odd last trial)
              const unsigned int key0, const unsigned int key1,
              const double s0, const double k, const double drift,
              const double vol, const double disc,
              double* partial)                       // 4 sums per block
{
    double cs = 0.0, cq = 0.0, ps = 0.0, pq = 0.0;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    // Grid-stride loop: thread i handles pairs i, i + stride, i + 2*stride, ...
    for (unsigned long long p = pair_start + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < pair_end; p += stride) {

        // ---- Philox4x32-10: 4 random 32-bit words from (key, counter = p) ----
        unsigned int c0 = (unsigned int)p, c1 = (unsigned int)(p >> 32), c2 = 0u, c3 = 0u;
        unsigned int k0 = key0, k1 = key1;
        #pragma unroll
        for (int r = 0; r < 10; ++r) {
            const unsigned int hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
            const unsigned int hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
            const unsigned int n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
            c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
            k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
        }

        // ---- 2 uniform doubles (53 bits each) -> Box-Muller -> 2 normals ----
        const double u1 = ((c0 >> 5) * 67108864.0 + (c1 >> 6) + 1.0) / 9007199254740992.0;  // (0,1]
        const double u2 = ((c2 >> 5) * 67108864.0 + (c3 >> 6)) / 9007199254740992.0;        // [0,1)
        const double rad = sqrt(-2.0 * log(u1));
        double sn, cn;
        sincospi(2.0 * u2, &sn, &cn);

        // ---- trial 2p ----
        double st = s0 * exp(drift + vol * (rad * cn));
        double call = disc * fmax(st - k, 0.0), put = disc * fmax(k - st, 0.0);
        cs += call; cq += call * call; ps += put; pq += put * put;

        // ---- trial 2p+1 (skipped if N is odd and this is the last pair) ----
        if (2 * p + 1 < n_total) {
            st = s0 * exp(drift + vol * (rad * sn));
            call = disc * fmax(st - k, 0.0); put = disc * fmax(k - st, 0.0);
            cs += call; cq += call * call; ps += put; pq += put * put;
        }
    }

    // ---- Block reduction: warp shuffles, then one warp combines the warps ----
    for (int off = 16; off > 0; off >>= 1) {
        cs += __shfl_down_sync(0xffffffffu, cs, off);
        cq += __shfl_down_sync(0xffffffffu, cq, off);
        ps += __shfl_down_sync(0xffffffffu, ps, off);
        pq += __shfl_down_sync(0xffffffffu, pq, off);
    }
    __shared__ double sh[4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sh[0][warp] = cs; sh[1][warp] = cq; sh[2][warp] = ps; sh[3][warp] = pq; }
    __syncthreads();
    if (warp == 0) {
        const int n_warps = blockDim.x >> 5;
        cs = lane < n_warps ? sh[0][lane] : 0.0;
        cq = lane < n_warps ? sh[1][lane] : 0.0;
        ps = lane < n_warps ? sh[2][lane] : 0.0;
        pq = lane < n_warps ? sh[3][lane] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            cs += __shfl_down_sync(0xffffffffu, cs, off);
            cq += __shfl_down_sync(0xffffffffu, cq, off);
            ps += __shfl_down_sync(0xffffffffu, ps, off);
            pq += __shfl_down_sync(0xffffffffu, pq, off);
        }
        if (lane == 0) {   // fixed write order -> results are bit-reproducible
            partial[4 * blockIdx.x + 0] = cs; partial[4 * blockIdx.x + 1] = cq;
            partial[4 * blockIdx.x + 2] = ps; partial[4 * blockIdx.x + 3] = pq;
        }
    }
}
"""

_kernel = cp.RawKernel(KERNEL_SRC, "mc_fused")   # compiled on first launch


def fused_sums(n, seed, pair_start=0, pair_end=None):
    """Four sums over trial pairs [pair_start, pair_end). Defaults to all of 0..n-1."""
    if pair_end is None:
        pair_end = (n + 1) // 2
    sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]
    blocks = sm_count * 8
    partial = cp.zeros(4 * blocks, dtype=cp.float64)
    drift, vol, disc = (R - 0.5 * SIG**2) * T, SIG * math.sqrt(T), math.exp(-R * T)
    _kernel((blocks,), (THREADS,),
            (np.uint64(pair_start), np.uint64(pair_end), np.uint64(n),
             np.uint32(seed & 0xFFFFFFFF), np.uint32((seed >> 32) & 0xFFFFFFFF),
             np.float64(S0), np.float64(K), np.float64(drift), np.float64(vol),
             np.float64(disc), partial))
    return tuple(float(x) for x in partial.reshape(blocks, 4).sum(axis=0))


def combine_stats(n, call_sum, call_sumsq, put_sum, put_sumsq):
    call_mean, put_mean = call_sum / n, put_sum / n
    call_var = (call_sumsq - n * call_mean**2) / (n - 1)
    put_var = (put_sumsq - n * put_mean**2) / (n - 1)
    return call_mean, math.sqrt(call_var / n), put_mean, math.sqrt(put_var / n)


def check_against_reference(seed):
    """GPU and CPU run the identical algorithm, so sums must agree to ~1e-12."""
    from gpu.philox_ref import price as ref_price
    ok = True
    for n in (1_000_000, 1_000_001, 4_000_003):
        gpu, cpu = fused_sums(n, seed), ref_price(n, seed)
        worst = max(abs(g - c) / abs(c) for g, c in zip(gpu, cpu))
        good = worst < 1e-10
        ok &= good
        print(f"check n={n:>9}: max relative difference {worst:.2e}  {'OK' if good else 'MISMATCH'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--check", action="store_true", help="compare to CPU reference and exit")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(0 if check_against_reference(args.seed) else 1)

    fused_sums(1_000_000, 0)          # warm-up: compiles the kernel outside the timed region
    cp.cuda.Device().synchronize()

    start = time.perf_counter()
    sums = fused_sums(args.n, args.seed)
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start

    call, call_se, put, put_se = combine_stats(args.n, *sums)
    if args.csv:
        print(f"1,{args.n},{call:.6f},{call_se:.6f},{put:.6f},{put_se:.6f},"
              f"{elapsed:.6f},0.000000,{elapsed:.6f}")
    else:
        print(f"N = {args.n}")
        print(f"Call = {call:.6f}  (SE {call_se:.6f})")
        print(f"Put  = {put:.6f}  (SE {put_se:.6f})")
        print(f"Time = {elapsed:.4f} s   ({args.n / elapsed / 1e9:.2f} billion trials/s)")


if __name__ == "__main__":
    main()
