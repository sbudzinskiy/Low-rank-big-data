# Entrywise low-rank approximation
This repository contains a Julia implementation of the method of alternating projections (AP) for the entrywise low-rank approximation of matrices and tensors as described in
- Budzinskiy S. Quasioptimal alternating projections and their use in low-rank approximation of matrices and tensors. arXiv: [2308.16097](https://arxiv.org/abs/2308.16097) (2023).

## Function-generated matrices and tensors
This code was used (with Julia 1.8.2) to carry out the numerical experiments in
- Budzinskiy S. When big data actually are low-rank, or entrywise approximation of certain function-generated matrices. arXiv: [2407.03250](https://arxiv.org/abs/2407.03250) (2024).

#### Examples of usage
The executable files `lrma_exponential.jl`, `lrma_quartic_gaussian.jl` and `lrta_sinh.jl`correspond to the three test functions used in [2407.03250](https://arxiv.org/abs/2407.03250). Their command-line arguments specify 
- name of the output NPZ file,
- number of modes `D` of the function-generated `N x ... x N`  dataset, 
- list of values of `N` (number of samples per mode) for which to generate data,
- list of values of `M`(latent dimension) for which to generate data,
- list of values of `R` (rank) for which to approximate data,
- number of random trials,
- random sampling scheme (independent or symmetric),
- method of low-rank approximation (AP or truncated SVD).
