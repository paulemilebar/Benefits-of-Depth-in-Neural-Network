# Theorical Principles of Deep Learning Course Project — Practical Experiments on “Benefits of depth in Neural Network” (Telgarsky 2016)

This repository contains **experiments** that aim to illustrate the theorical results of *Benefits of depth in neural networks* (Telgarsky, 2016).

The repo focuses on two main ideas:

1. **Depth vs Width (representation power)**  
   A function computed by a deeper network can be **hard to approximate** with shallower networks **unless** their **width grows exponentially** (for a fixed target error).

2. **Limits of depth (statistical limitation)**  
   Even expressive deep networks cannot “learn anything”: with **random labels** and a **fixed architecture**, training error stops improving when the dataset size grows (qualitative illustration of Theorem 1.2).

---

## Organization of the repo

### Notebooks (3 experiments)

- **`triangle_wave_approx.ipynb`**  
  Approximates an oscillatory target built from the **triangle wave** function composed **L = 4** times (we use a practical, controllable depth). 
  We train 1D fully-connected **ReLU** networks with:
  - depth `k` varying (e.g. `k = 1 … 6`)
  - width `w` varying (e.g. `w = 2 … 256`)
  
  For each `(k, w)` we measure:
  - **MAE** (as an L1 approximation proxy)
  - parameter count / model size
  - and identify an **empirical optimal width** `w*(k)` as the smallest width reaching a target error threshold `τ` (e.g. `τ = 1/32`).

- **`approximation_polynoms.ipynb`**  
  Same experimental protocol, but the target is a **degree-6 polynomial** (a smooth baseline).  
  The goal is to compare how depth/width trade-offs behave on a non-oscillatory function and look for qualitative links between target complexity and the best depth/width regime.

- **`limitation_of_depth.ipynb`**  
  Simple classification experiment illustrating the limitation stated in Theorem 1.2:
  - We fix a network architecture (e.g. depth = 3, width = 3)
  - sample `n` input points and assign **random labels**
  - train multiple runs with controlled seeds
  - report best training error statistics as `n` increases (mean/std/min over runs)


## Results structure

Each experiment writes its outputs to a dedicated folder under `results/` (plots + saved metrics).

## Reference

- Telgarsky, M. (2016). *Benefits of depth in neural networks.*
