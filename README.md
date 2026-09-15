# 📈🐺 Monte Carlo Simulation Based Options Pricer for Seawulf 🐺📈 #

AMS 530 Project 1, Stony Brook Graduate School

Supervisor: Professor Yuefan Deng, AMS Department, Stony Brook University

## Project Description ##

This project implements a Monte Carlo simulator for pricing European options, and investigates how the method can be parallelized on the SeaWulf supercomputer cluster.

Monte Carlo option pricing works by simulating a large number of independent, random stock price outcomes, computing the option's payoff for each simulated outcome, and averaging the discounted payoffs across all trials to estimate the option's present-day price.

In this project, I implemented three versions of the same underlying model, so that the cost and benefit of each optimization can be measured in isolation.

- **Naive Serial** — a simple unoptimized for loop over individual trials.
- **Vectorized Serial** — the same model expressed as NumPy array operations instead of a Python loop.
- **Parallel** — using `mpi4py`, splitting the total number of simulations across ranks running on SeaWulf.

## Algorithm Structure ##

### Core Math ###

Every implementation is built on the same two core pricing functions (see `src/Simulations/single.py`). The code works in both single operation cases and multi-operation cases since it's built entirely out of elementwise
NumPy operations, with no Python functions that assumes a scalar input.

```
FUNCTION calculate_st(S0, K, r, sigma, T, Z):
    # Geometric Brownian Motion terminal price under the risk-neutral measure
    drift = (r - 0.5 * sigma^2) * T
    diffusion = sigma * sqrt(T) * Z
    RETURN S0 * exp(drift + diffusion)

FUNCTION calculate_payoffs(K, S_T, r, T):
    call_payoff = max(S_T - K, 0)
    put_payoff = max(K - S_T, 0)
    discount = exp(-r * T)
    RETURN discount * call_payoff, discount * put_payoff
```

One trial is one draw `Z ~ N(0,1)` fed through both functions. The Monte Carlo price
estimate is the mean of the discounted payoff across N independent trials, with standard error
`sample_std / sqrt(N)`, where `N` is the number of total simulations to run.

### Serial Baselines ###

- **Naive**: loop `N` times, drawing one `Z` and calling both functions per iteration.
- **Vectorized**: draw all `N` values of `Z` as one array and call both functions once on the whole array.

### Parallel ###

![Dataflow1](/Images/Dataflow1.png)

```
P = number of ranks
rank = this process's rank

n_chunk = share of N for this rank (remainder distributed across the first few ranks so chunk sizes always sum to exactly N)

# every rank derives the same list of P child seeds from one base seed, then picks out only its own entry. This guarantees an independent stream per rank without any seed needing to be communicated, only base_seed

child_seed = SeedSequence(base_seed).spawn(P)[rank]
rng = default_rng(child_seed)

# process n_chunk trials in fixed-size batches, accumulating only summary statistics which bounds peak memory per rank regardless of how large n_chunk is

FOR each batch (size <= batch_size) until n_chunk done:
    Z = rng.standard_normal(batch_size)
    ST = calculate_st(..., Z)
    call, put = calculate_payoffs(..., ST)
    accumulate count, sum(call), sum(call^2), sum(put), sum(put^2)

# combine every rank's partial statistics onto rank 0
total_*** = REDUCE(SUM, local partial statistics, root=0)

IF rank == 0:
    mean = total_sum / total_n
    variance = (total_sumsq - total_n * mean^2) / (total_n - 1)      
    #ddof=1
    se = sqrt(variance / total_n)
    print price, 95% CI, runtime
```

### Repo Structure ###

```
seawulf-monte-carlo-options-pricer/
├── src/
│   ├── MPIBasicTest/
│   │   └── mpi_test.py        # mpi4py baseline test / Seawulf environment check
│   ├── Seawulf/
│   │   └── pricer.py          # parallel pricer
│   ├── Simulations/
│   │   ├── single.py          # core model: calculate_st, calculate_payoffs
│   │   ├── serial_runs.py     # naive serial baseline
│   │   └── vectorized.py      # vectorized serial baseline
│   └── Slurm/
│       ├── hello_mpi.sh       # Bash script for the mpi_test
│       └── pricer.sh          # Bash script for the parallel pricer
├── HPC_Logs/                  # SLURM stdout/stderr logs gitignored
│                               
├── README.md
├── .gitignore
└── requirements.txt
```

> **Note on `HPC_Logs/`**: the contents of this directory are gitignored

### Compilation & execution ###

**Dependencies**: `numpy`, `mpi4py` (SeaWulf via `module load mpi4py/latest`).
```bash
pip install numpy
```

**Local (no MPI needed)**:
```bash
python3 src/Simulations/serial_runs.py
python3 src/Simulations/vectorized.py
```

**On SeaWulf, environment check**:
> **Seawulf Notice🐺:** All bash scripts should be run from repo root
```bash
sbatch src/Slurm/hello_mpi.sh
```

**On SeaWulf, the real parallel pricer**:
```bash
sbatch src/Slurm/pricer.sh
```
`pricer.sh` loads the required modules and launches the pricer via:
```bash
module purge
module load slurm
module load mpi4py/latest
cd src
mpirun -n $SLURM_NTASKS python3 -m Seawulf.pricer --n <N> --batch <BATCH> --seed <SEED>
```
`--n`, `--batch`, and `--seed` are optional (default to 10,000,000 / 1,000,000 / 30 respectively)

## Quantitative Results ##

N/A

## Computing Performance Metrics ##

N/A
