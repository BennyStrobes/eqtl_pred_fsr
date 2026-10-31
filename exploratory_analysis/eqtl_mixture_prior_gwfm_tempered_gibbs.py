import numpy as np
import time
from numba import njit, prange
from eqtl_mixture_prior_gibbs import EQTL_MIXTURE_PRIOR_GIBBS, _seed_numba, _recompute_residuals_numba, _gibbs_sweep_numba
from eqtl_mixture_prior_tempered_gibbs import _tgs_conditional, build_sparse_ld_neighbours


###############################################################################
# GWFM-style tempered Gibbs alternative to eqtl_mixture_prior_gibbs.py (README.md, section 5c). The model, priors, inputs and outputs are
# identical. It follows the --gwfm analysis of GCTB (Wu et al. 2025, Nature Genetics; algorithm "TGS_thin" of ApproxBayesRC in
# github.com/jianzeng/GCTB, scr/model.cpp, sampleFromTGS_eigen): the standard Gibbs sweep of eqtl_mixture_prior_gibbs.py every iteration,
# plus, every gwfm_interval iterations, tempered Gibbs (TGS) blocks on small sets of variants in near-perfect LD, which is where single-site
# Gibbs fails to move an effect between LD partners. Unlike eqtl_mixture_prior_tempered_gibbs.py, every other variant and every
# hyper-parameter is sampled by plain Gibbs, so the cost is close to the standard sampler's.
#
# TGS block (every gwfm_interval iterations, after the Gibbs sweep): for every variant j of a gene, in order, whose running PIP (mean over
# the previous iterations of its Rao-Blackwellized PIP) is >= gwfm_pip_threshold, and that has at least one variant of the gene in
# training-fold LD r^2 >= gwfm_r2_threshold with it: the set S = {j and those variants}. GCTB also requires j's current class to be
# non-zero; that makes whether the block runs depend on the state the block resamples, which biases the chain (on an exact test, the PIPs
# of an LD pair were off by 0.015, more with longer blocks), so it is not done here.
#   - Fully tempered TGS, as in GCTB: member a is selected with probability proportional to p_a = 1 / (3 P_a(z_a)), its class is drawn
#     uniformly from the two other classes, and its effect from the exact conditional given the class. Weight of a sub-state: 1 / Z,
#     Z = mean_a p_a. max(gwfm_min_steps, gwfm_steps_per_snp x |S|) sub-steps (GCTB uses |S|; more sub-steps make the resampled end
#     state below closer to a posterior draw).
#   - The members' single-variant regression coefficients b_a are computed exactly from the residual at the start of the block and kept
#     exact through the members' pairwise LD (b_a -= d R_ac after a move of member c by d); the residual is updated once, at the end.
#   - The block ends in one of its sub-states drawn with probability proportional to 1 / Z (GCTB continues from the last sub-state, which is
#     distributed as the tempered target rather than the posterior), and the 1 / Z - weighted Rao-Blackwellized slab probabilities and
#     effects over the block replace the members' Gibbs contributions of this iteration to the PIPs, slab probabilities and posterior means
#     (as GCTB replaces the members' PIPs).
###############################################################################


@njit(parallel=True, cache=True)
def _gwfm_tgs_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, fr_ptr, fr_idx, resid, beta, z, sigma2_e, log_pi, slab_var,
                    running_pip, pip_threshold, steps_per_snp, min_steps, it_p_slab, it_beta, class_counts, class_ss, stats):
    # TGS blocks of every gene (see top of file). On entry resid[g] = y_g - X_g beta_g; z, beta and resid are updated in place.
    # fr_ptr / fr_idx: CSR lists (per pair, itself included; local indices within the gene) of the variants in LD r^2 >= gwfm_r2_threshold
    # it_p_slab / it_beta: this iteration's contributions to the slab probabilities and effects; overwritten for the members of every block
    # class_counts / class_ss: recomputed for the genes with at least one block
    # stats (n_genes x 2): number of blocks and number of members whose class changed between the start and the end of their block
    n_genes = resid.shape[0]
    n = resid.shape[1]
    for g in prange(n_genes):
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        v_g = sigma2_e[g] / n
        total_var_1 = slab_var[0] + v_g
        total_var_2 = slab_var[1] + v_g
        dL1 = log_pi[1] - log_pi[0] - 0.5 * np.log(total_var_1 / v_g)
        dL2 = log_pi[2] - log_pi[0] - 0.5 * np.log(total_var_2 / v_g)
        a1 = 0.5 * (1.0 / v_g - 1.0 / total_var_1)
        a2 = 0.5 * (1.0 / v_g - 1.0 / total_var_2)
        shrink_1 = slab_var[0] / total_var_1
        shrink_2 = slab_var[1] / total_var_2
        post_sd_1 = np.sqrt(slab_var[0] * v_g / total_var_1)
        post_sd_2 = np.sqrt(slab_var[1] * v_g / total_var_2)
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
            P = np.empty((m_set, 3))
            for a in range(m_set):
                _tgs_conditional(a, b_S[a] * b_S[a], dL1, dL2, a1, a2, P)

            # TGS sub-steps, with the weighted reservoir (res_*) and the weighted Rao-Blackwellized sums
            n_steps = max(min_steps, int(steps_per_snp * m_set + 0.5))
            W = 0.0
            sum_p1 = np.zeros(m_set)
            sum_p2 = np.zeros(m_set)
            sum_b = np.zeros(m_set)
            res_W = 0.0
            res_beta = beta_S.copy()
            res_z = z_S.copy()
            p_sel = np.empty(m_set)
            for step in range(n_steps):
                s = 0.0
                for a in range(m_set):
                    p_sel[a] = 1.0 / (3.0 * max(P[a, z_S[a]], 1e-300))
                    s += p_sel[a]
                w = m_set / s
                W += w
                for a in range(m_set):
                    sum_p1[a] += w * P[a, 1]
                    sum_p2[a] += w * P[a, 2]
                    sum_b[a] += w * beta_S[a]
                res_W += w
                if np.random.random() * res_W < w:
                    for a in range(m_set):
                        res_beta[a] = beta_S[a]
                        res_z[a] = z_S[a]
                # Select a member, draw its class uniformly from the two others and its effect from the exact conditional
                u = np.random.random() * s
                sel = m_set - 1
                acc = 0.0
                for a in range(m_set):
                    acc += p_sel[a]
                    if u < acc:
                        sel = a
                        break
                c_new = (z_S[sel] + (1 if np.random.random() < 0.5 else 2)) % 3
                b = b_S[sel]
                if c_new == 0:
                    b_new = 0.0
                elif c_new == 1:
                    b_new = shrink_1 * b + post_sd_1 * np.random.normal()
                else:
                    b_new = shrink_2 * b + post_sd_2 * np.random.normal()
                d = b_new - beta_S[sel]
                beta_S[sel] = b_new
                z_S[sel] = c_new
                if d != 0.0:
                    for a in range(m_set):
                        if a != sel:
                            b_S[a] -= d * R[a, sel]
                            _tgs_conditional(a, b_S[a] * b_S[a], dL1, dL2, a1, a2, P)

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
                it_p_slab[k, 0] = sum_p1[a] / W
                it_p_slab[k, 1] = sum_p2[a] / W
                it_beta[k] = sum_b[a] / W
        if n_blocks > 0:
            for c in range(3):
                class_counts[g, c] = 0.0
                class_ss[g, c] = 0.0
            for k in range(start, end):
                class_counts[g, z[k]] += 1.0
                class_ss[g, z[k]] += beta[k] * beta[k]
        stats[g, 0] = n_blocks
        stats[g, 1] = n_changes


class EQTL_MIXTURE_PRIOR_GWFM_TEMPERED_GIBBS(EQTL_MIXTURE_PRIOR_GIBBS):
    def __init__(self, gwfm_interval=10, gwfm_r2_threshold=0.95, gwfm_pip_threshold=0.05, gwfm_steps_per_snp=10.0, gwfm_min_steps=20, **kwargs):
        # kwargs: the arguments of EQTL_MIXTURE_PRIOR_GIBBS (same meaning)
        # gwfm_interval: TGS blocks every this many iterations; gwfm_r2_threshold: training-fold LD r^2 defining a variant's set;
        # gwfm_pip_threshold: running PIP a variant needs for its block; gwfm_steps_per_snp / gwfm_min_steps: sub-steps per block,
        # max(gwfm_min_steps, gwfm_steps_per_snp x set size)
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

    def fit(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y):
        # Same inputs as EQTL_MIXTURE_PRIOR_GIBBS.fit, and the same attributes afterwards
        if self.seed is not None:
            np.random.seed(self.seed)
            _seed_numba(self.seed)
        self.initialize(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y)
        retained_iters = [itera for itera in range(self.burn_in, self.n_iters) if (itera - self.burn_in) % self.thin == 0]
        self.n_samples = len(retained_iters)
        self.hyperparameter_trace = np.zeros((self.n_iters, len(self.hyperparameter_names)))
        self.retained = np.zeros(self.n_iters, dtype=bool)
        self.retained[retained_iters] = True
        self.sigma2_e_samples = np.zeros((self.n_samples, self.n_genes), dtype=np.float32)
        # sum_p_slab / sum_beta accumulate every iteration (the running PIP needs the burn-in too); the posterior summaries are their
        # increase after burn-in. On TGS iterations the Gibbs contributions go to it_p_slab / it_beta first, so that the blocks can
        # replace their members' contributions.
        it_p_slab = np.zeros((self.n_pairs, 2))
        it_beta = np.zeros(self.n_pairs)
        stats = np.zeros((self.n_genes, 2))
        nz_pair_list = []
        nz_beta_list = []
        sample_iter = 0
        for itera in range(self.n_iters):
            t0 = time.time()
            if itera > 0 and itera % self.residual_recompute_interval == 0:
                _recompute_residuals_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, self.beta, self.resid)
            if itera == self.burn_in:
                burn_in_p_slab = self.sum_p_slab.copy()
                burn_in_beta = self.sum_beta.copy()
            tgs_iteration = itera > 0 and itera % self.gwfm_interval == 0
            if tgs_iteration:
                it_p_slab[:] = 0.0
                it_beta[:] = 0.0
                _gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, np.log(self.pi), self.slab_var,
                                   self.ig_a, self.ig_b, True, it_p_slab, it_beta, self.class_counts, self.class_ss)
                running_pip = np.sum(self.sum_p_slab, axis=1) / itera
                _gwfm_tgs_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.fr_ptr, self.fr_idx, self.resid, self.beta, self.z, self.sigma2_e,
                                np.log(self.pi), self.slab_var, running_pip, self.gwfm_pip_threshold, self.gwfm_steps_per_snp, self.gwfm_min_steps,
                                it_p_slab, it_beta, self.class_counts, self.class_ss, stats)
                self.sum_p_slab += it_p_slab
                self.sum_beta += it_beta
            else:
                _gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, np.log(self.pi), self.slab_var,
                                   self.ig_a, self.ig_b, True, self.sum_p_slab, self.sum_beta, self.class_counts, self.class_ss)
            guards = self.sample_hyperparameters()
            self.hyperparameter_trace[itera] = self.current_hyperparameters()
            if self.retained[itera]:
                nz = np.flatnonzero(self.beta)
                nz_pair_list.append(nz.astype(np.int32))
                nz_beta_list.append(self.beta[nz].astype(np.float32))
                self.sigma2_e_samples[sample_iter] = self.sigma2_e
                sample_iter += 1
            if self.verbose:
                n_class = np.sum(self.class_counts, axis=0)
                tgs_string = ' tgs_blocks=%d tgs_class_changes=%d' % (np.sum(stats[:, 0]), np.sum(stats[:, 1])) if tgs_iteration else ''
                print('iter ' + str(itera) + ': ' + ' '.join(name + '=%.3e' % value for name, value in zip(self.hyperparameter_names, self.current_hyperparameters())) + ' n_slab1=%d n_slab2=%d mean_sigma2_e=%.3e' % (n_class[1], n_class[2], np.mean(self.sigma2_e)) + tgs_string + ' sec=%.1f' % (time.time() - t0) + guards, flush=True)

        # Posterior summaries over the post burn-in iterations
        n_post = self.n_iters - self.burn_in
        self.p_slab = (self.sum_p_slab - burn_in_p_slab) / n_post
        self.pip = np.sum(self.p_slab, axis=1)
        self.posterior_mean = (self.sum_beta - burn_in_beta) / n_post

        # Non-zero effect samples, sorted by pair then sample
        nz_pair = np.concatenate(nz_pair_list)
        nz_sample = np.concatenate([np.full(len(nz), s, dtype=np.int16) for s, nz in enumerate(nz_pair_list)])
        nz_beta = np.concatenate(nz_beta_list)
        order = np.lexsort((nz_sample, nz_pair))
        self.nz_pair = nz_pair[order]
        self.nz_sample = nz_sample[order]
        self.nz_beta = nz_beta[order]
        print(str(self.n_samples) + ' retained samples, ' + str(len(self.nz_beta)) + ' non-zero effect samples (' + '%.1f' % (len(self.nz_beta) / self.n_samples) + ' per sample)', flush=True)
        return self

    def initialize(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y):
        super().initialize(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y)
        if np.max(np.diff(gene_var_ptr)) > 32767:
            raise ValueError('a gene has more than 32767 variants (int16 neighbour indices)')
        t0 = time.time()
        self.fr_ptr, self.fr_idx, _ = build_sparse_ld_neighbours(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.gwfm_r2_threshold)
        set_sizes = np.diff(self.fr_ptr)
        print('GWFM tempered Gibbs: interval=' + str(self.gwfm_interval) + ' r2_threshold=' + str(self.gwfm_r2_threshold) + ' pip_threshold=' + str(self.gwfm_pip_threshold) + ' steps_per_snp=' + str(self.gwfm_steps_per_snp) + ' min_steps=' + str(self.gwfm_min_steps) + '; %.1f%% of variants have an LD partner, mean set size %.1f among them (built in %.0f sec)' % (100.0 * np.mean(set_sizes > 1), np.mean(set_sizes[set_sizes > 1]) if np.any(set_sizes > 1) else 0.0, time.time() - t0), flush=True)
