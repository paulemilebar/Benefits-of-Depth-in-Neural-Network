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

- **trial_of_approximation_triangle_wave_with_theorical_results.ipynb`**  
  Here, we wanted naively to reproduce the results of Theorem 1.1 with exactly the same hypotheses. We faced directly to the issue that it is unfeasible in pratice.

- **`approximation_polynoms_and_trianglewave.ipynb`**  

  We aim to approximate 2 oscillatory target function built from  a **degree-6 polynomial** and the **triangle wave** function composed **L = 4** times.

  We train 1D fully-connected **ReLU** networks with:

  - depth `k` varying (e.g. `k = 2 … 6`)
  - width `w` varying (e.g. `w = 2 … 256`)
  
  For each `(k, w)` we measure:

  - **MAE** (L1 approximation error)

  - When for a depth = k and a width = w, the MAE error goes below the theorical threshold (1/64), we stop and report the width as the **optimal width** (w*(k)) for approximating our target function given a depth k.

  - Then, with those different pair (k, w*(k)),, we compute the number of parameters and the number of total nodes of the network.

  NOTE : In this notebook, we test for our 2 target functions : the polynomial function and the wave triangle function composed L = 4 times.

- **`limitation_of_depth.ipynb`**  
  Simple classification experiment illustrating the limitation stated in Theorem 1.2:
  - We fix a network architecture (e.g. depth = 3, width = 3)
  - sample `n` input points and assign **random labels**
  - train multiple runs with controlled seeds
  - report best training error statistics as `n` increases (mean/std/min over runs)


## Results structure

We did the experiments on GOOGLE COLAB using a GPU T4 for better results (by better, we mean getting closer to the theorical results). So we saved all the results in a dedicated folder : `Results/` (plots + saved metrics)
But, of course, the code presented in this repo will give the exact same results. 


## Reference

- Telgarsky, M. (2016). **Benefits of depth in neural networks.**
