import time
import numpy as np
from mpi4py import MPI
from Simulations.single import calculate_st, calculate_payoffs


def chunk_size(n_total, size, rank):
    #Best split total number of sims into chunks
    base = n_total // size
    remainder = n_total % size
    return base + 1 if rank < remainder else base


def run_chunk(so, k, u, sig, t, n, rng):
    #See single py for logic
    Z = rng.standard_normal(n)
    st = calculate_st(so, k, u, sig, t, Z)
    call_disc, put_disc = calculate_payoffs(k, st, u, t)
    return {
        "n": n,
        "call_sum": float(call_disc.sum()),
        "call_sumsq": float(np.sum(call_disc**2)),
        "put_sum": float(put_disc.sum()),
        "put_sumsq": float(np.sum(put_disc**2)),
    }


def combine_stats(n, call_sum, call_sumsq, put_sum, put_sumsq):
    #Assemble total mean and std err
    call_mean = call_sum / n
    put_mean = put_sum / n
    call_var = (call_sumsq - n * call_mean**2) / (n - 1)
    put_var = (put_sumsq - n * put_mean**2) / (n - 1)
    call_se = np.sqrt(call_var / n)
    put_se = np.sqrt(put_var / n)
    return call_mean, call_se, put_mean, put_se


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Test Parameters
    S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
    N = 10000000
    BASE_SEED = 30

    # Seed sequence to ensure a distinct random seed is always used for each ranks
    seed_seq = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_seq.spawn(size)
    rng = np.random.default_rng(child_seeds[rank])

    #Chunk size
    n_chunk = chunk_size(N, size, rank)

    comm.Barrier() #Wait for completed operation
    start = time.perf_counter() #Start the clock
    local = run_chunk(S0, K, R, SIG, T, n_chunk, rng) #Execute
    comm.Barrier() #Wait for completed operation
    elapsed = time.perf_counter() - start #Calculate time

    #gather up all the data from the different workers back into the root process
    total_n = comm.reduce(local["n"], op=MPI.SUM, root=0)
    total_call_sum = comm.reduce(local["call_sum"], op=MPI.SUM, root=0)
    total_call_sumsq = comm.reduce(local["call_sumsq"], op=MPI.SUM, root=0)
    total_put_sum = comm.reduce(local["put_sum"], op=MPI.SUM, root=0)
    total_put_sumsq = comm.reduce(local["put_sumsq"], op=MPI.SUM, root=0)

    #We care about the worst time
    max_elapsed = comm.reduce(elapsed, op=MPI.MAX, root=0)

    #Only do these computations and prints on rank 0
    if rank == 0:
        call_price, call_se, put_price, put_se = combine_stats(
            total_n, total_call_sum, total_call_sumsq, total_put_sum, total_put_sumsq
        )
        call_ci = (call_price - 1.96 * call_se, call_price + 1.96 * call_se)
        put_ci = (put_price - 1.96 * put_se, put_price + 1.96 * put_se)

        print(f"Ranks (P) = {size}")
        print(f"N = {total_n}")
        print(f"Call price = {call_price:.4f}  (SE={call_se:.4f})  "
              f"95% CI: [{call_ci[0]:.4f}, {call_ci[1]:.4f}]")
        print(f"Put price  = {put_price:.4f}  (SE={put_se:.4f})  "
              f"95% CI: [{put_ci[0]:.4f}, {put_ci[1]:.4f}]")
        print(f"Runtime = {max_elapsed:.4f} sec")


if __name__ == "__main__":
    main()