import time
import numpy as np

from single import calculate_payoffs, calculate_st



def run_serial(n, s0, k, u, sig, t):
    rng = np.random.default_rng(seed=30)

    call_payoffs = []
    put_payoffs = []

    #timing
    start = time.perf_counter()

    #for n sims
    for i in range(N):
        #calculate a new random val
        z = rng.standard_normal()
        #calculate payoffs
        st = calculate_st(s0,k,u,sig,t,z)
        call_disc,put_disc = calculate_payoffs(k,st,u,t)
        #append payoffs
        call_payoffs.append(call_disc)
        put_payoffs.append(put_disc)
    
    #calculate final time for entire sim
    exec_t = time.perf_counter() - start

    #find prices
    call_price = np.mean(call_payoffs)
    put_price = np.mean(put_payoffs)

    #std err = sample std / sqrt(n)
    call_err = np.std(call_payoffs, ddof=1)/np.sqrt(n)
    put_err = np.std(put_payoffs, ddof=1)/np.sqrt(n)

    #95% confidence interval
    call_ci = (call_price - 1.96*call_err,call_price+1.96*call_err)
    put_ci = (put_price-1.96*put_err,put_price+1.96*put_err)

    return call_ci, put_ci, exec_t
    


if __name__ == "__main__":
    #Test Parameters
    S0 = 100.0 
    K = 100.0 
    U = 0.05 
    SIG = 0.2
    T = 1.0
    N = 1000000
    call_ci,put_ci,exec_t = run_serial(N, S0, K, U, SIG, T)

    print(f"Call Price Interval = {call_ci[0]:.4f} - {call_ci[1]:.4f}")
    print(f"Put Price Interval = {put_ci[0]:.4f} - {put_ci[1]:.4f}")
    print(f"Execution Time = {exec_t:.4f}")