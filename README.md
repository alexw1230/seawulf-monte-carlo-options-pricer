# 📈🐺 Monte Carlo Options Pricer: Parallel Accelerator 🐺📈 #

AMS 530 Project 1, Stony Brook Graduate School

Supervisor: Professor Yuefan Deng, AMS Department, Stony Brook University

## Project Description ##

This project implements a Monte Carlo simulator for pricing European options, and investigates how the method can be parallelized on the SeaWulf supercomputer cluster.

Monte Carlo option pricing works by simulating a large number of independent, random stock price outcomes, computing the option's payoff for each simulated outcome, and averaging the discounted payoffs across all trials to estimate the option's present-day price.

In this project, I implemented three versions of the same underlying model, so that the cost and benefit of each optimization can be measured in isolation.

- **Naive Serial** — a simple unoptimized for loop over individual trials.
- **Vectorized Serial** — the same model expressed as NumPy array operations instead of a Python loop.
- **Parallel** — using `mpi4py`, splitting the total number of simulations across ranks running on SeaWulf.

Beyond the baseline scaling, I also investigate the effects of memory bandwidth, cache-sized batching, and the effects of CPU clock throttling on parallel efficiency.

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

To keep everything standardized, I used the same test parameters throughout all experiments:
```
S0 = 100.0 Original Stock Price (USD)
K = 100.0 Strike Price (USD)
R = 0.05 Risk Free Rate
SIG = 0.20 Volitility
T = 1.0 Time (Years)
```

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
│   │   ├── pricer.py          # parallel pricer
│   │   └── pricer_32.py       # float32 variant, for memory bandwidth experiments
│   ├── Simulations/
│   │   ├── single.py          # core math: calculate_st, calculate_payoffs
│   │   ├── serial_runs.py     # naive serial baseline
│   │   └── vectorized.py      # vectorized serial baseline
│   └── Slurm/
│       └── *.sh               # Bash files for various experiments
│       
├── HPC_Logs/                  # Seawulf Logs
├── Images/                    # Images for README
├── Plotting/                  # Plotting scripts
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
> **Seawulf Notice:** All bash scripts should be run from repo root
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
`--n`, `--batch`, and `--seed` are optional (default to 10,000,000 / 30,000 / 30 respectively)

## Experiment Structure ##
During the course of the experiment, I used many different nodes and configuations, but for the final experiments reported, the experimental setup and hardware are as follows:

| | |
| --- | --- |
| Nodes | `hbm-short-96core` Intel Sapphire Rapids |
| Cores | 96 core capacity |
| Memory | 256 GB DDR5, 128 GB HBM (Independently Addressible) |
| Repeats | 5 for each experiment |
| Timing | Maximum Compute Time amongst cores|

> Note all nodes used are not shared and exclusively used to run the experiment when reserved

Unless stated otherwise, results use `batch=30000` and DDR5 memory.

## Final Results ##

### Black-Scholes Validation ###

$P = 1$

| N | Call Price ($) | Call SE ($) | Put Price ($) | Put SE ($) |
| -------- | -------- | -------- | -------- | -------- |
| $10^5$ | 10.478727 | 0.046493	| 5.538411	| 0.027340
| $10^6$ | 10.454772 | 0.014720	| 5.557732	| 0.008652
| $10^7$ | 10.447164 | 0.004654	| 5.571562	| 0.002737
| $10^8$ | 10.450550 | 0.001472	| 5.572333	| 0.000866
| $10^9$ | 10.450518 | 0.000465	| 5.573629	| 0.000274
| $10^{10}$ | 10.450502 | 0.000147 | 5.573508 | 0.000087

The standard error falls by ~3x for each 10x in N. For every value of N, the calculated value of both the call and put option is well within the error bars.

### Single Node Scaling ###

> Note these specific values of P were chosen to show the effect of doubling core count up to 32, then using half a node, and a full node for 48 and 96 respectively.

$N = 10^{10}$

$Efficiency = \frac{\frac{T(1)}{T(P)}}{P}, Efficiency >= 0 *$

> *Most of the time Efficiency < 1, as 1 means linear scaling

| P | Time (s) | Efficiency |
| -------- | -------- | -------- |
| 1 | 150.46 | 1.00
| 2 | 74.49 | 1.01
| 4 | 40.05 | 0.94
| 8 | 19.83 | 0.95
| 16 | 9.95 | 0.95
| 32 | 4.95 | 0.95
| 48 | 3.35 | 0.94
| 96 | 2.23 | 0.70

From this data, it is clear that for P values up and to 48, we achieve near-linear scaling (94%+). However, when the full allocation of cores is utilized, efficiency appears to do far worse, only ~70% efficiency.

> Note that throughout these runs, I have also measured communication time independently, and it has never exceeded more than 0.7% of runtime even in the worst cases. This makes sense since there is only communication at the beginning and end of the program, as it is an Embarrassignly Parallel task.

Other lower values of N were also tested at various P

Efficiency at various N

| P | $10^{10}$ | $10^9$ | $10^8$ | $10^7$ |
| -------- | -------- | -------- | -------- | -------- |
| 1 | 1.00 | 1.00 | 1.00 | 1.00
| 16 | 0.95 | 0.93 | 0.89 | 0.79
| 32 | 0.95 | 0.93 | 0.87 | 0.64
| 48 | 0.94 | 0.91 | 0.83 | 0.56
| 96 | 0.70 | 0.69 | 0.60 | 0.30

From this data, we can observe that efficency degrades sharply as N gets smaller, as overhead fixed costs become a larger portion of the total compute time.

After implementing the lessons learned from the experiments in **Memory Bandwidth & Batching** and **Full Load & CPU Clock Throttling**, the final, best results are as follows:

Once again, these are all using $N = 10^{10}$, batch = $3 * 10^4$, DDR5, `hbm-short-96core`. The efficiency score is calculated against T(1) with these same parameters.

### Most Efficient (Excluding the trivial $P=1$) ###
| Layout (Node x Core/Node) | Total Cores P | Time (s) | Efficiency | 
| -------- | -------- | -------- | -------- |
| 2 x 48 | 96 | 1.68 | **0.95**

### Overall Fastest ###
| Layout (Node x Core/Node) | Total Cores P | Time (s) | Efficiency | 
| -------- | -------- | -------- | -------- |
| 4 x 96 | 384 | **0.58** | 0.69

### Best High Core Count ($>= 2$ full nodes) ###
| Layout (Node x Core/Node) | Total Cores P | Time (s) | Efficiency | 
| -------- | -------- | -------- | -------- |
| 4 x 48 | 192 | **0.86** | **0.92**

## Performance Analysis & Problems Faced ##

### Memory Bandwidth & Batching ###

When originally testing, I implemented batching solely as a way to constrain maximum memory usage. I arbitrarily chose $10^6$ as my batch size, however I realized there was an issue with this: an array of this size will not fit into cache, and so every memory operation done has to read from and write to main memory, which is obviously much slower.

In order to test this hypothesis, I utilized the hbm memory and compared to an identical run using DDR5

$P = 96$, $N = 10^{10}$

| Batch Size | DDR5 (s) | HBM(s) | HBM Speedup | DDR5 Speedup vs Original | 
| -------- | -------- | -------- | -------- | -------- |
| $10^3$ | 3.80 | 3.80 | 1.00x | 1.95x
| $10^4$ | 2.24 | 2.24 | 1.00x | 3.3x
| $3 * 10^4$ | 2.23 | 2.21 | 1.01x | 3.3x
| $10^5$ | 4.48 | 2.78 | 1.61x | 1.65x
| $10^6$ | 7.41 | 3.54 | 2.10x | 1x
| $10^7$ | 9.51 | 3.91 | 2.43x | 0.78x

> Note the value $3 * 10^4$ was randomly selected by me between $10^4$ and $10^5$. There is probably a better value out there, but a local minimum of this level of precision is sufficent here.

At the original batch size, the HBM more than doubled its speed. We also can notice that at smaller batch sizes, whether we use HBM or DDR5 becomes irrelevant because memory bandwidth is no longer the limiting factor. However, notice that at very small batch sizes, ($<= 10^3$), there is also a significant slowdown due to overhead running too often. Using an semi-optimal batch size alone give us a ~3.3x speedup compared to the original batch size, as well as an efficiency of around 70%. This is why in **Final Results** and in experiments after this section, all batches are $3 * 10^4$ in size. Also note that since HBM didn't provide any meaningful speedup compared to DDR5, DDR5 is used everywhere else.

### Full Load & CPU Clock Throttling ###

Notice in **Single Node Scaling**, efficiency sharply dropped when $P > 48$ from ~94% to ~70%. Since everything else remains the same, this means that using the maximum cores of a node seems to be throttling it down. To test this, I ran the following layout test.

| Layout (Node x Core/Node) | Total Cores P | Time (s) | Efficiency |
| -------- | -------- | -------- | -------- |
| 1 x 48 | 48 | 3.36 | 0.95 
| 2 x 48 | 96 | 1.68 | 0.95
| 4 x 48 | 192 | 0.86 | 0.92 
| 1 x 96 | 96 | 2.24 | 0.70
| 2 x 96 | 192 | 1.13 | 0.71
| 4 x 96 | 384 | 0.58 | 0.69

Notice how as the number of nodes increases, the efficiency barely changes. However, whenever we use full nodes it drops ~15-20% in terms of efficiency compared to half nodes of the same number of cores.

I also ran a script to test the operational clock frequency at various values of P.

| P | Freq (GHz) |
| -------- | -------- |
| 24 | 3.41
| 48 | 3.06
| 96 | 2.43

This frequency drop explains the poorer performance of the larger values of P. This decrease is likely due to core temperature and/or power consumption getting too much and causing the CPU to throttle down as a safety measure. By splitting the load among more nodes, each core can run at a higher frequency and thus produce a higher speed. This explains the drop in efficency when more cores are used as in the benchmark, with the fewest number of cores, will have access to more node-level resources and thus will run better.