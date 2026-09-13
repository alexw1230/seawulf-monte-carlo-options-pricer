import time
import numpy as np

from single import calculate_payoffs, calculate_st

def run_vector(n, s0, k, u, sig, t):
        rng = np.random.default_rng(seed=30)

        #Instead of using a for loop, lets use numpy vectors to just do the whole thing is one go
        start = time.perf_counter()
        #Its the same logic as in serial runs and single though
        z = rng.standard_normal(n)
        st = calculate_st(s0,k,u,sig,t,z)
        call_disc,put_disc = calculate_payoffs(k,st,u,t)
        
        call_price = call_disc.mean()
        put_price = put_disc.mean()
        
        exec_t = time.perf_counter() - start
        call_err = np.std(call_disc, ddof=1)/np.sqrt(n)
        put_err = np.std(put_disc, ddof=1)/np.sqrt(n)
    
        #95% confidence interval
        call_ci = (call_price-1.96*call_err,call_price+1.96*call_err)
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
    call_ci,put_ci,exec_t = run_vector(N, S0, K, U, SIG, T)

    print(f"Call Price Interval - NP Vector = {call_ci[0]:.4f} - {call_ci[1]:.4f}")
    print(f"Put Price Interval - NP Vector = {put_ci[0]:.4f} - {put_ci[1]:.4f}")
    print(f"Execution Time - NP Vector = {exec_t:.4f}")