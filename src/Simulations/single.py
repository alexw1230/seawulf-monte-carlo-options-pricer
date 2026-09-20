import numpy as np
import math

def calculate_st(so, k, u, sig, t, z):
    # s0 = original stock price
    # k = strike price
    # u = risk free rate
    # sig = volatility
    # t = exp time (years)
    # St = simulated price
    # drift = (r - 0.5sig^2)t
    drift = (u-0.5*sig**2)*t
    # expected direction (risk neutral since if sig = 0, drift = rt)
    # diffusion = sig*sqrt(t)*z
    # brownian motion, linear variance
    diff = sig*math.sqrt(t)*z

    #st = s0*exp((u-0.5*sig^2)*t+sig*sqrt(t)*z)

    return so*np.exp(drift+diff)

def calculate_payoffs(k, st, u, t):
    #Calculates expected payoff, including discount factor

    #Payoffs
    call = np.maximum(st-k,0.0) #Excersise the call option vs don't
    put = np.maximum(k-st,0.0) #Excersise the put option vs don't

    #discount factor
    disc_fact = math.exp(-u*t)
    call_disc = disc_fact*call
    put_disc = disc_fact*put
    return (call_disc,put_disc)

if __name__ == "main":

    #Test Parameters
    S0 = 100.0 
    K = 100.0 
    U = 0.05 
    SIG = 0.2
    T = 1.0

    #Individual rng object
    #Fixed seed for now, tbc later
    rng = np.random.default_rng(seed=30)
    Z = rng.standard_normal()
    ST = calculate_st(S0, K, U, SIG, T, Z)

    (CALL, PUT) = calculate_payoffs(K,ST, U, T)

    print (f"CV = {CALL:.4f}")
    print (f"PV = {PUT:.4f}")