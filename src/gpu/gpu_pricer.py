import argparse
import math
import time

import cupy as cp

from Simulations.single import calculate_st, calculate_payoffs

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
BYTES_PER_TRIAL = 232
#naive version

def run(n, batch, rng):
    # Running sums live on the GPU as 0-d arrays. Converting to a Python float
    # every batch would force a synchronize + copy each time; instead we pull
    # the four numbers back once, at the very end.
    call_sum = cp.zeros((), dtype=cp.float64)
    call_sumsq = cp.zeros((), dtype=cp.float64)
    put_sum = cp.zeros((), dtype=cp.float64)
    put_sumsq = cp.zeros((), dtype=cp.float64)

    remaining = n
    while remaining > 0:
        b = min(batch, remaining)
        Z = rng.standard_normal(b, dtype=cp.float64)   # random numbers made on the GPU
        st = calculate_st(S0, K, R, SIG, T, Z)
        call_disc, put_disc = calculate_payoffs(K, st, R, T)
        call_sum += call_disc.sum()
        call_sumsq += (call_disc * call_disc).sum()
        put_sum += put_disc.sum()
        put_sumsq += (put_disc * put_disc).sum()
        remaining -= b

    return float(call_sum), float(call_sumsq), float(put_sum), float(put_sumsq)


def combine_stats(n, call_sum, call_sumsq, put_sum, put_sumsq):
    call_mean, put_mean = call_sum / n, put_sum / n
    call_var = (call_sumsq - n * call_mean**2) / (n - 1)
    put_var = (put_sumsq - n * put_mean**2) / (n - 1)
    return call_mean, math.sqrt(call_var / n), put_mean, math.sqrt(put_var / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000)
    parser.add_argument("--batch", type=int, default=100_000_000,
                        help="trials per batch; GPUs want large batches")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    rng = cp.random.RandomState(args.seed)   # cuRAND generator, lives on the GPU

    probe = calculate_st(S0, K, R, SIG, T, cp.zeros(4))
    assert isinstance(probe, cp.ndarray), "single.py fell back to the CPU"

    #Warmup
    run(min(args.batch, 1_000_000), args.batch, cp.random.RandomState(0))
    cp.cuda.Device().synchronize()

    start = time.perf_counter()
    sums = run(args.n, args.batch, rng)   # float() at the end already waits for the GPU
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start

    call, call_se, put, put_se = combine_stats(args.n, *sums)
    if args.csv:
        print(f"1,{args.n},{call:.6f},{call_se:.6f},{put:.6f},{put_se:.6f},"
              f"{elapsed:.6f},0.000000,{elapsed:.6f}")
    else:
        print(f"N = {args.n}   batch = {args.batch}")
        print(f"Call = {call:.6f}  (SE {call_se:.6f})")
        print(f"Put  = {put:.6f}  (SE {put_se:.6f})")
        print(f"Time = {elapsed:.4f} s   ({args.n / elapsed / 1e9:.2f} billion trials/s)")
        print(f"Estimated memory traffic: {BYTES_PER_TRIAL * args.n / elapsed / 1e9:.0f} GB/s")


if __name__ == "__main__":
    main()