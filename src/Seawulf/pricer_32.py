import time
import numpy as np
import argparse
from mpi4py import MPI
from Simulations.single import calculate_st, calculate_payoffs


def chunk_size(n_total, size, rank):
    base = n_total // size
    remainder = n_total % size
    return base + 1 if rank < remainder else base


def run_chunk(so, k, u, sig, t, n, rng, batch=1000000):
    total = 0
    call_sum = call_sumsq = 0.0
    put_sum = put_sumsq = 0.0

    remaining = n
    while remaining > 0:
        b = min(batch, remaining)
        Z = rng.standard_normal(b, dtype=np.float32)          # <-- the actual change
        st = calculate_st(so, k, u, sig, t, Z)                 # stays float32, unmodified function
        call_disc, put_disc = calculate_payoffs(k, st, u, t)   # stays float32, unmodified function

        total += b
        call_sum += float(np.sum(call_disc, dtype=np.float64))
        call_sumsq += float(np.sum(call_disc.astype(np.float64) ** 2))
        put_sum += float(np.sum(put_disc, dtype=np.float64))
        put_sumsq += float(np.sum(put_disc.astype(np.float64) ** 2))
        remaining -= b

    return {
        "n": total,
        "call_sum": call_sum,
        "call_sumsq": call_sumsq,
        "put_sum": put_sum,
        "put_sumsq": put_sumsq,
    }


def combine_stats(n, call_sum, call_sumsq, put_sum, put_sumsq):
    call_mean = call_sum / n
    put_mean = put_sum / n
    call_var = (call_sumsq - n * call_mean**2) / (n - 1)
    put_var = (put_sumsq - n * put_mean**2) / (n - 1)
    call_se = np.sqrt(call_var / n)
    put_se = np.sqrt(put_var / n)
    return call_mean, call_se, put_mean, put_se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000000)
    parser.add_argument("--batch", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = args.n
    S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0

    seed_seq = np.random.SeedSequence(args.seed)
    child_seeds = seed_seq.spawn(size)
    rng = np.random.default_rng(child_seeds[rank])

    n_chunk = chunk_size(N, size, rank)

    comm.Barrier()
    start = time.perf_counter()
    local = run_chunk(S0, K, R, SIG, T, n_chunk, rng, batch=args.batch)
    elapsed_compute = time.perf_counter() - start

    comm.Barrier()
    start_comm = time.perf_counter()
    total_n = comm.reduce(local["n"], op=MPI.SUM, root=0)
    total_call_sum = comm.reduce(local["call_sum"], op=MPI.SUM, root=0)
    total_call_sumsq = comm.reduce(local["call_sumsq"], op=MPI.SUM, root=0)
    total_put_sum = comm.reduce(local["put_sum"], op=MPI.SUM, root=0)
    total_put_sumsq = comm.reduce(local["put_sumsq"], op=MPI.SUM, root=0)
    elapsed_comm = time.perf_counter() - start_comm

    max_compute_t = comm.reduce(elapsed_compute, op=MPI.MAX, root=0)
    max_comm_t = comm.reduce(elapsed_comm, op=MPI.MAX, root=0)

    if rank == 0:
        call_price, call_se, put_price, put_se = combine_stats(
            total_n, total_call_sum, total_call_sumsq, total_put_sum, total_put_sumsq
        )
        call_ci = (call_price - 1.96 * call_se, call_price + 1.96 * call_se)
        put_ci = (put_price - 1.96 * put_se, put_price + 1.96 * put_se)
        total_t = max_compute_t + max_comm_t

        if args.csv:
            print(f"{size},{total_n},{call_price:.6f},{call_se:.6f},"
                  f"{put_price:.6f},{put_se:.6f},"
                  f"{max_compute_t:.6f},{max_comm_t:.6f},{total_t:.6f}")
        else:
            print(f"Ranks (P) = {size}")
            print(f"N = {total_n}")
            print(f"Call price = {call_price:.4f}  (SE={call_se:.4f})  "
                  f"95% CI: [{call_ci[0]:.4f}, {call_ci[1]:.4f}]")
            print(f"Put price  = {put_price:.4f}  (SE={put_se:.4f})  "
                  f"95% CI: [{put_ci[0]:.4f}, {put_ci[1]:.4f}]")
            print(f"Compute Time = {max_compute_t:.4f} sec")
            print(f"Communication Time = {max_comm_t:.4f} sec")
            print(f"Total Time = {total_t:.4f} sec")


if __name__ == "__main__":
    main()