import time
import numpy as np

from single import calculate_payoffs, calculate_st



if __name__ == "__main__":
    #Test Parameters
    S0 = 100.0 
    K = 100.0 
    U = 0.05 
    SIG = 0.2
    T = 1.0

    #Independent trials
    N = 10000000
    rng = np.random.default_rng(seed=300)

    call_payoffs = []
    put_payoffs = []

    #timing
    start = time.perf_counter()

    #for n sims
    for i in range(N):
        #calculate a new random val
        Z = rng.standard_normal()
        #calculate payoffs
        ST = calculate_st(S0, K,U,SIG,T,Z)
        call_disc,put_disc = calculate_payoffs(K,ST,U,T)
        #append payoffs
        call_payoffs.append(call_disc)
        put_payoffs.append(put_disc)
    #calculate final time for entire sim
    t = time.perf_counter() - start

    #find prices
    call_price = np.mean(call_payoffs)
    put_price = np.mean(put_payoffs)

    print(f"Call Price = {call_price:.4f}")
    print(f"Put Price = {put_price:.4f}")
    print(f"Execution Time = {t:.4f}")