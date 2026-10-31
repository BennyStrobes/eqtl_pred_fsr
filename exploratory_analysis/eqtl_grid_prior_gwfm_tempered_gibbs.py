import numpy as np
import time
from numba import njit, prange
from eqtl_grid_prior_gibbs import EQTL_GRID_PRIOR_GIBBS, _grid_gibbs_sweep_numba
from eqtl_mixture_prior_tempered_gibbs import build_sparse_ld_neighbours


###############################################################################
# GWFM-style tempered Gibbs for the fixed-grid prior of eqtl_grid_prior_gibbs.py (README.md, section 5e): the same scheme as
# eqtl_mixture_prior_gwfm_tempered_gibbs.py (section 5c) with K + 1 classes in place of 3. The standard Gibbs sweep of eqtl_grid_prior_gibbs.py
# every iteration, plus, every gwfm_interval iterations, tempered Gibbs (TGS) blocks on the variants with running PIP >= gwfm_pip_threshold
# that have a gene variant in training-fold LD r^2 >= gwfm_r2_threshold with them (set S = the variant and those partners).
#   - Fully tempered TGS with K + 1 classes, as GCTB does for its ndist classes (sampleFromTGS_eigen): member a is selected with probability
#     proportional to p_a = 1 / ((K + 1) P_a(z_a)), its class is drawn uniformly from the K other classes, and its effect from the exact
#     conditional given the class. Weight of a sub-state: 1 / Z, Z = mean_a p_a. max(gwfm_min_steps, gwfm_steps_per_snp x |S|) sub-steps.
#   - The members' b_a are computed exactly at the start of the block and kept exact through their pairwise LD; the residual is updated once,
#     at the end.
#   - The block ends in one of its sub-states drawn with probability proportional to 1 / Z, and the 1 / Z - weighted Rao-Blackwellized group
#     probabilities and effects over the block replace the members' Gibbs contributions of this iteration.
# As in section 5c, a block runs on the running PIP alone, not on the focal variant's current class (GCTB's condition biases the chain).
###############################################################################


@njit(cache=True)
def _grid_conditional(a, b2, const, half_inv_var, P):
    # P[a, :] = full conditional class probabilities of a variant with single-variant coefficient b (b2 = b^2)
    n_comp = const.shape[0]
    mx = -np.inf
    for c in range(n_comp):
        P[a, c] = const[c] - half_inv_var[c] * b2
        if P[a, c] > mx:
            mx = P[a, c]
    tot = 0.0
    for c in range(n_comp):
        P[a, c] = np.exp(P[a, c] - mx)
        tot += P[a, c]
    for c in range(n_comp):
        P[a, c] /= tot


@njit(parallel=True, cache=True)
def _grid_gwfm_tgs_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, fr_ptr, fr_idx, resid, beta, z, sigma2_e, log_pi, grid_var, comp_group,
                         running_pip, pip_threshold, steps_per_snp, min_steps, it_p_group, it_beta, class_counts, stats):
    # TGS blocks of every gene (see top of file). On entry resid[g] = y_g - X_g beta_g; z, beta and resid are updated in place.
    # fr_ptr / fr_idx: CSR lists (per pair, itself included; local indices within the gene) of the variants in LD r^2 >= gwfm_r2_threshold
    # it_p_group / it_beta: this iteration's contributions to the group probabilities and effects; overwritten for the members of every block
    # class_counts: recomputed for the genes with at least one block
    # stats (n_genes x 2): number of blocks and number of members whose class changed between the start and the end of their block
    n_genes = resid.shape[0]
    n = resid.shape[1]
    n_comp = grid_var.shape[0]
    for g in prange(n_genes):
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        v_g = sigma2_e[g] / n
        const = np.empty(n_comp)
        half_inv_var = np.empty(n_comp)
        shrink = np.empty(n_comp)
        post_sd = np.empty(n_comp)
        for c in range(n_comp):
            total_var = grid_var[c] + v_g
            const[c] = log_pi[c] - 0.5 * np.log(total_var)
            half_inv_var[c] = 0.5 / total_var
            shrink[c] = grid_var[c] / total_var
            post_sd[c] = np.sqrt(grid_var[c] * v_g / total_var)
        n_blocks = 0
        n_changes = 0
        for jf in range(end - start):
            kf = start + jf
            if running_pip[kf] < pip_threshold:
                continue
            m_set = fr_ptr[kf + 1] - fr_ptr[kf]
            if m_set < 2:
                continue
            n_blocks += 1
            # Members (local indices), their exact b_a and pairwise LD
            S = np.empty(m_set, dtype=np.int64)
            for a in range(m_set):
                S[a] = fr_idx[fr_ptr[kf] + a]
            sum_r = 0.0
            for i in range(n):
                sum_r += resid[g, i]
            b_S = np.empty(m_set)
            R = np.empty((m_set, m_set))
            for a in range(m_set):
                va = gene_var_idx[start + S[a]]
                dot = 0.0
                for i in range(n):
                    dot += G[va, i] * resid[g, i]
                b_S[a] = (dot - var_mean[va] * sum_r) / (var_sd[va] * n) + beta[start + S[a]]
                R[a, a] = 1.0
                for c in range(a):
                    vc = gene_var_idx[start + S[c]]
                    acc = 0.0
                    for i in range(n):
                        acc += (G[va, i] - var_mean[va]) * (G[vc, i] - var_mean[vc])
                    R[a, c] = acc / (n * var_sd[va] * var_sd[vc])
                    R[c, a] = R[a, c]
            beta_start = np.empty(m_set)
            z_start = np.empty(m_set, dtype=np.int64)
            beta_S = np.empty(m_set)
            z_S = np.empty(m_set, dtype=np.int64)
            for a in range(m_set):
                beta_start[a] = beta[start + S[a]]
                z_start[a] = z[start + S[a]]
                beta_S[a] = beta_start[a]
                z_S[a] = z_start[a]
            P = np.empty((m_set, n_comp))
            for a in range(m_set):
                _grid_conditional(a, b_S[a] * b_S[a], const, half_inv_var, P)

            # TGS sub-steps, with the weighted reservoir (res_*) and the weighted Rao-Blackwellized sums
            n_steps = max(min_steps, int(steps_per_snp * m_set + 0.5))
            W = 0.0
            sum_p_small = np.zeros(m_set)
            sum_p_large = np.zeros(m_set)
            sum_b = np.zeros(m_set)
            res_W = 0.0
            res_beta = beta_S.copy()
            res_z = z_S.copy()
            p_sel = np.empty(m_set)
            for step in range(n_steps):
                s = 0.0
                for a in range(m_set):
                    p_sel[a] = 1.0 / (n_comp * max(P[a, z_S[a]], 1e-300))
                    s += p_sel[a]
                w = m_set / s
                W += w
                for a in range(m_set):
                    p_small = 0.0
                    p_large = 0.0
                    for c in range(1, n_comp):
                        if comp_group[c] == 0:
                            p_small += P[a, c]
                        else:
                            p_large += P[a, c]
                    sum_p_small[a] += w * p_small
                    sum_p_large[a] += w * p_large
                    sum_b[a] += w * beta_S[a]
                res_W += w
                if np.random.random() * res_W < w:
                    for a in range(m_set):
                        res_beta[a] = beta_S[a]
                        res_z[a] = z_S[a]
                # Select a member, draw its class uniformly from the K others and its effect from the exact conditional
                u = np.random.random() * s
                sel = m_set - 1
                acc = 0.0
                for a in range(m_set):
                    acc += p_sel[a]
                    if u < acc:
                        sel = a
                        break
                c_new = np.random.randint(0, n_comp - 1)
                if c_new >= z_S[sel]:
                    c_new += 1
                if c_new == 0:
                    b_new = 0.0
                else:
                    b_new = shrink[c_new] * b_S[sel] + post_sd[c_new] * np.random.normal()
                d = b_new - beta_S[sel]
                beta_S[sel] = b_new
                z_S[sel] = c_new
                if d != 0.0:
                    for a in range(m_set):
                        if a != sel:
                            b_S[a] -= d * R[a, sel]
                            _grid_conditional(a, b_S[a] * b_S[a], const, half_inv_var, P)

            # End in the resampled sub-state: residual, chain state, and this iteration's contributions of the members
            for a in range(m_set):
                k = start + S[a]
                dd = res_beta[a] - beta_start[a]
                if dd != 0.0:
                    v = gene_var_idx[k]
                    scale = dd / var_sd[v]
                    for i in range(n):
                        resid[g, i] -= (G[v, i] - var_mean[v]) * scale
                if res_z[a] != z_start[a]:
                    n_changes += 1
                beta[k] = res_beta[a]
                z[k] = res_z[a]
                it_p_group[k, 0] = sum_p_small[a] / W
                it_p_group[k, 1] = sum_p_large[a] / W
                it_beta[k] = sum_b[a] / W
        if n_blocks > 0:
            for c in range(n_comp):
                class_counts[g, c] = 0.0
            for k in range(start, end):
                class_counts[g, z[k]] += 1.0
        stats[g, 0] = n_blocks
        stats[g, 1] = n_changes


class EQTL_GRID_PRIOR_GWFM_TEMPERED_GIBBS(EQTL_GRID_PRIOR_GIBBS):
    def __init__(self, gwfm_interval=10, gwfm_r2_threshold=0.95, gwfm_pip_threshold=0.05, gwfm_steps_per_snp=10.0, gwfm_min_steps=20, **kwargs):
        # kwargs: the arguments of EQTL_GRID_PRIOR_GIBBS (same meaning)
        # gwfm_*: as in EQTL_MIXTURE_PRIOR_GWFM_TEMPERED_GIBBS
        super().__init__(**kwargs)
        if gwfm_interval < 1:
            raise ValueError('gwfm_interval must be at least 1')
        if not (0.0 < gwfm_r2_threshold <= 1.0):
            raise ValueError('gwfm_r2_threshold must be in (0, 1]')
        self.gwfm_interval = int(gwfm_interval)
        self.gwfm_r2_threshold = float(gwfm_r2_threshold)
        self.gwfm_pip_threshold = float(gwfm_pip_threshold)
        self.gwfm_steps_per_snp = float(gwfm_steps_per_snp)
        self.gwfm_min_steps = int(gwfm_min_steps)

    def sweep(self, itera, G, var_mean, var_sd, gene_var_ptr, gene_var_idx):
        # The Gibbs sweep; on TGS iterations its contributions go to it_p_group / it_beta first, so that the blocks can replace their
        # members' contributions
        if not (itera > 0 and itera % self.gwfm_interval == 0):
            return super().sweep(itera, G, var_mean, var_sd, gene_var_ptr, gene_var_idx)
        self.it_p_group[:] = 0.0
        self.it_beta[:] = 0.0
        log_pi = np.log(self.pi)
        _grid_gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, log_pi, self.grid_var,
                                self.comp_group, self.ig_a, self.ig_b, self.it_p_group, self.it_beta, self.class_counts)
        running_pip = np.sum(self.sum_p_group, axis=1) / itera
        _grid_gwfm_tgs_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.fr_ptr, self.fr_idx, self.resid, self.beta, self.z, self.sigma2_e,
                             log_pi, self.grid_var, self.comp_group, running_pip, self.gwfm_pip_threshold, self.gwfm_steps_per_snp, self.gwfm_min_steps,
                             self.it_p_group, self.it_beta, self.class_counts, self.stats)
        self.sum_p_group += self.it_p_group
        self.sum_beta += self.it_beta
        return ' tgs_blocks=%d tgs_class_changes=%d' % (np.sum(self.stats[:, 0]), np.sum(self.stats[:, 1]))

    def initialize(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y):
        super().initialize(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y)
        if np.max(np.diff(gene_var_ptr)) > 32767:
            raise ValueError('a gene has more than 32767 variants (int16 neighbour indices)')
        self.it_p_group = np.zeros((self.n_pairs, 2))
        self.it_beta = np.zeros(self.n_pairs)
        self.stats = np.zeros((self.n_genes, 2))
        t0 = time.time()
        self.fr_ptr, self.fr_idx, _ = build_sparse_ld_neighbours(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.gwfm_r2_threshold)
        set_sizes = np.diff(self.fr_ptr)
        print('GWFM tempered Gibbs: interval=' + str(self.gwfm_interval) + ' r2_threshold=' + str(self.gwfm_r2_threshold) + ' pip_threshold=' + str(self.gwfm_pip_threshold) + ' steps_per_snp=' + str(self.gwfm_steps_per_snp) + ' min_steps=' + str(self.gwfm_min_steps) + '; %.1f%% of variants have an LD partner, mean set size %.1f among them (built in %.0f sec)' % (100.0 * np.mean(set_sizes > 1), np.mean(set_sizes[set_sizes > 1]) if np.any(set_sizes > 1) else 0.0, time.time() - t0), flush=True)
