import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def conditional_b(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, resid, beta, out):
    # b_k = X_k' r / n + beta_k for every pair (the single-variant statistic given all other effects)
    n_genes = resid.shape[0]
    n = resid.shape[1]
    for g in prange(n_genes):
        sum_r = 0.0
        for i in range(n):
            sum_r += resid[g, i]
        for k in range(gene_var_ptr[g], gene_var_ptr[g + 1]):
            v = gene_var_idx[k]
            dot = 0.0
            for i in range(n):
                dot += G[v, i] * resid[g, i]
            out[k] = (dot - var_mean[v] * sum_r) / (var_sd[v] * n) + beta[k]


def eb_weights(b, v, grid_var, pi_init, n_em=500, tol=1e-8):
    # argmax_pi sum_k log sum_c pi_c N(b_k; 0, grid_var_c + v_k) by EM (concave in pi), warm-started at pi_init
    total = grid_var[None, :] + v[:, None]
    log_l = -0.5 * np.log(total) - 0.5 * b[:, None] ** 2 / total
    log_l -= np.max(log_l, axis=1, keepdims=True)
    L = np.exp(log_l)
    pi = np.maximum(pi_init, 1e-12); pi /= pi.sum()
    prev = -np.inf
    for it in range(n_em):
        mix = L @ pi
        ll = np.sum(np.log(mix))
        pi = pi * (L.T @ (1.0 / mix)) / len(b)
        if ll - prev < tol * len(b):
            break
        prev = ll
    return pi
