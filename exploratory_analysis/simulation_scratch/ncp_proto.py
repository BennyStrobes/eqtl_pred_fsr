import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _delta_loglik(G, var_mean, var_sd, gene_var_idx, resid, sigma2_e, flip_gene_ptr, flip_pair, flip_dbeta, flip_genes, out_dX):
    # Per gene with flips: change of the log-likelihood when beta changes by flip_dbeta on its flipped pairs; out_dX[f] = X dbeta of gene f
    n = resid.shape[1]
    out = np.zeros(len(flip_genes))
    for f in prange(len(flip_genes)):
        g = flip_genes[f]
        for i in range(n):
            out_dX[f, i] = 0.0
        for t in range(flip_gene_ptr[f], flip_gene_ptr[f + 1]):
            v = gene_var_idx[flip_pair[t]]
            scale = flip_dbeta[t] / var_sd[v]
            for i in range(n):
                out_dX[f, i] += (G[v, i] - var_mean[v]) * scale
        cross = 0.0
        sq = 0.0
        for i in range(n):
            cross += resid[g, i] * out_dX[f, i]
            sq += out_dX[f, i] * out_dX[f, i]
        out[f] = (2.0 * cross - sq) / (2.0 * sigma2_e[g])
    return out


def ncp_weight_moves(model, G, var_mean, var_sd, gene_var_idx, n_moves, step, alpha, rng, target_flips=200):
    # Non-centred Metropolis-Hastings moves of the grid weights (README.md, section 5k): move mass delta between components a and b; every
    # variant of the losing component switches to the gaining one with probability |delta| / pi_loser, keeping its standardized effect
    # (beta' = beta sigma_new / sigma_old; spike -> slab draws beta' = sigma_new N(0, 1); slab -> spike sets 0); accepted with the change of the
    # likelihood and of the Dirichlet(alpha) prior. Exact (the uniform position of each variant within its component and the standard normal
    # effect of spike variants are auxiliary variables drawn from their conditionals). Returns the number accepted
    K1 = len(model.grid_var)
    sd = np.sqrt(model.grid_var)
    n_acc = 0
    z = model.z
    for move in range(n_moves):
        a, b = rng.choice(K1, size=2, replace=False)
        pa, pb = model.pi[a], model.pi[b]
        # delta sized so that about target_flips variants switch
        scale = target_flips / model.n_pairs
        delta = rng.normal() * scale * step[a, b]
        if not (-pa < delta < pb):
            continue
        if delta > 0:
            loser, gainer, prob = b, a, delta / pb
        else:
            loser, gainer, prob = a, b, -delta / pa
        cand = np.flatnonzero(z == loser)
        flips = cand[rng.random(len(cand)) < prob]
        if len(flips) == 0:
            continue
        old = model.beta[flips]
        if loser == 0:
            new = sd[gainer] * rng.normal(size=len(flips))
        elif gainer == 0:
            new = np.zeros(len(flips))
        else:
            new = old * sd[gainer] / sd[loser]
        dbeta = new - old
        genes = model.gene_of_pair[flips]
        order = np.argsort(genes, kind='stable')
        flips, dbeta, genes = flips[order], dbeta[order], genes[order]
        flip_genes, starts = np.unique(genes, return_index=True)
        flip_gene_ptr = np.append(starts, len(flips)).astype(np.int64)
        dX = np.empty((len(flip_genes), model.resid.shape[1]))
        dll = _delta_loglik(G, var_mean, var_sd, gene_var_idx, model.resid, model.sigma2_e, flip_gene_ptr, flips, dbeta, flip_genes, dX)
        new_pi = model.pi.copy()
        new_pi[a] += delta
        new_pi[b] -= delta
        log_ratio = np.sum(dll) + (alpha - 1.0) * (np.log(new_pi[a]) + np.log(new_pi[b]) - np.log(pa) - np.log(pb))
        if np.log(rng.random()) < log_ratio:
            model.resid[flip_genes] -= dX
            model.beta[flips] += dbeta
            z[flips] = gainer
            np.add.at(model.class_counts, (genes, np.full(len(genes), loser)), -1.0)
            np.add.at(model.class_counts, (genes, np.full(len(genes), gainer)), 1.0)
            model.pi = new_pi
            n_acc += 1
            step[a, b] *= 1.1
            step[b, a] *= 1.1
        else:
            step[a, b] *= 0.95
            step[b, a] *= 0.95
    return n_acc
