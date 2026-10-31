import numpy as np
import time
from numba import njit, prange
from eqtl_mixture_prior_gibbs import EQTL_MIXTURE_PRIOR_GIBBS, _seed_numba, _recompute_residuals_numba


###############################################################################
# Tempered Gibbs alternative to eqtl_mixture_prior_gibbs.py (README.md, section 5b). The model, priors, inputs and outputs are identical;
# only the sampling of the variant classes changes, following the tempered Gibbs sampler (TGS; Zanella & Roberts 2019, "Scalable importance
# tempering and Bayesian variable selection", JRSS-B) as implemented for colocalization in eb_coloc/multi_causal_variant/sims/ld/mcvc_tempered.py.
#
# Single-site Gibbs mixes poorly among variants in near-perfect LD: once a variant carries a strong signal, moving the effect to an LD
# partner requires first switching it off, which its full conditional makes essentially impossible. One TGS sub-step, for gene g:
#   1. every variant j has its full conditional class probabilities P_j(.) (effect integrated out, as in the plain sampler) and a selection
#      weight p_j = g_j(z_j) / P_j(z_j), with g_j = (1 - eps) P_j + eps / 3 the tempered conditional (eps = tgs_epsilon);
#   2. variant j is selected with probability proportional to p_j, its class is drawn from g_j and its effect from the exact normal
#      conditional given the class (as in the plain sampler), and the residual is updated.
# The TGS chain is reversible with respect to pi_TGS proportional to pi * Z, Z = mean_j p_j, so posterior expectations under pi are averages
# over the visited sub-states weighted by 1 / Z. Per gene and sweep (tgs_sub_steps_per_snp x p_g sub-steps):
#   - PIPs, slab probabilities and posterior means: 1 / Z - weighted Rao-Blackwellized conditionals and effects, pooled over every post
#     burn-in sub-state (normalized per gene by the total weight);
#   - retained samples: one sub-state per gene, drawn with probability proportional to 1 / Z among the sub-states visited since the
#     previous retained iteration (a weighted reservoir), so the saved effect samples are (approximately) draws from the posterior and
#     the downstream evaluation uses them unchanged. Its residual variance sample is drawn from Inverse-Gamma(a + n / 2, b + r'r / 2) given
#     that sub-state's residual;
#   - hyper-parameters (pi, sigma2_1, sigma2_2, sigma2_e_g): conditioned on importance-weighted EXPECTED sufficient statistics (class counts,
#     sums of squared effects, residual sums of squares, per gene) pooled over the last tgs_window_sweeps sweeps, rather than on the raw chain
#     state, which is distributed as pi_TGS and over-represents transient states such as a null variant just switched on. The TGS chain
#     itself continues from its last state. The self-normalized weights bias each sweep's expected counts slightly upward (states with
#     spurious inclusions have larger Z and so smaller weights, a negative weight-count covariance), and the bias feeds back through pi:
#     in simulations, pooling 1 sweep sent pi_1 from 0.008 to 0.2 with a near-zero sigma2_1, 3 sweeps drifted slowly, 10 was stable over
#     the run and 30 stayed at the true values. Pooling costs only memory (tgs_window_sweeps x genes x 7 floats).
#
# Individual-level data. A sub-step needs every variant's conditional, which depends on the data only through its single-variant regression
# coefficient b_a = X_a' r / n + beta_a. The sweep keeps b_a for every variant: exact at the start of the sweep; after a move of variant j by d,
# b_a -= d R_aj for the variants a in LD with j at r^2 >= tgs_r2_update_threshold (b_j itself does not change), with the training-fold LD R_aj
# precomputed per gene and stored sparsely. The other variants' b_a (used only for their selection weights) are stale by less than
# |d| sqrt(threshold) per move until the next sweep; threshold 0 is exact TGS (up to the int16 quantization of R, relative error 1.5e-5).
# The selected variant's b_j is recomputed exactly from the residual, unless every move since its last exact computation was by j or one of
# its LD neighbours (then b_cur is already exact). The residual itself is updated lazily: effect changes are applied to it only before the
# next exact b_j and at the end of the sweep, so a variant switched on and back off costs nothing. The residual sum of squares is updated
# in O(1): r'r <- r'r - 2 d X_j' r + d^2 n.
# Sparse LD storage (per pair, CSR): nbr_ptr[k]:nbr_ptr[k + 1] are the neighbours of pair k (itself included), nbr_idx their index within
# the gene (int16) and nbr_r their LD R_aj quantized to int16 (R_aj = nbr_r / 32767).
#
# Speed. A move changes the b_a of a few hundred LD neighbours, and their conditionals (two exponentials each) are the main cost. The
# conditional depends on b_a only through t_a = b_a^2: the log odds of slab c against the spike are dL_c + a_c t_a, with 0 < a_1 < a_2. A
# neighbour's b_a is always updated, but its conditional and selection weight are recomputed only once t_a has moved far enough from its
# value at the last recomputation that the selection weight may have changed by a relative tol = tgs_selection_tolerance (exact bound,
# _tgs_set_selection; tol = 0 recomputes every neighbour). The selected variant's conditional is always exact. The variant is selected
# through per-block sums of the selection weights (blocks of SELECTION_BLOCK variants, recomputed only when one of their weights changed),
# rather than by scanning every weight. Genes run in parallel (numba prange), so a job given more cores runs proportionally faster.
###############################################################################

LD_QUANTIZATION = 32767.0
SELECTION_BLOCK = 32


@njit(cache=True)
def _tgs_conditional(jj, t, dL1, dL2, a1, a2, P):
    # Full conditional class probabilities of local variant jj given t = b^2 (b its single-variant regression coefficient) -> P[jj, :];
    # the log odds of slab c against the spike are dL_c + a_c t
    L1 = dL1 + a1 * t
    L2 = dL2 + a2 * t
    if L1 < 700.0 and L2 < 700.0:
        e1 = np.exp(L1)
        e2 = np.exp(L2)
        P0 = 1.0 / (1.0 + e1 + e2)
        P[jj, 0] = P0
        P[jj, 1] = e1 * P0
        P[jj, 2] = e2 * P0
    else:
        mx = max(L1, L2)
        w0 = np.exp(-mx)
        w1 = np.exp(L1 - mx)
        w2 = np.exp(L2 - mx)
        tot = w0 + w1 + w2
        P[jj, 0] = w0 / tot
        P[jj, 1] = w1 / tot
        P[jj, 2] = w2 / tot


@njit(cache=True)
def _tgs_set_selection(jj, zc, P, p_sel, dt_max, block_dirty, one_minus_eps, eps_over_K, a1, a2, tol):
    # Selection weight of local variant jj in class zc, and the change of t = b^2 it tolerates before its conditional must be recomputed:
    # with the log odds of class c moving at rate alpha_c relative to class zc, the selection weight changes by a relative factor of at most
    # sum_{c != zc} P_c (exp(alpha_c |dt|) - 1) <= S |dt| / (1 - alpha_max |dt|), S = sum_{c != zc} P_c alpha_c, which is <= tol for
    # |dt| <= tol / (S + tol alpha_max)
    p_sel[jj] = one_minus_eps + eps_over_K / max(P[jj, zc], 1e-300)
    if zc == 0:
        S = P[jj, 1] * a1 + P[jj, 2] * a2
        alpha_max = a2
    elif zc == 1:
        S = P[jj, 0] * a1 + P[jj, 2] * (a2 - a1)
        alpha_max = max(a1, a2 - a1)
    else:
        S = P[jj, 0] * a2 + P[jj, 1] * (a2 - a1)
        alpha_max = a2
    dt_max[jj] = tol / (S + tol * alpha_max) if tol > 0.0 else 0.0
    block_dirty[jj // SELECTION_BLOCK] = True


@njit(cache=True)
def _tgs_apply_pending(g, start, G, var_mean, var_sd, gene_var_idx, resid, pend_list, n_pend, pend_d, in_pend):
    # Subtract from the residual of gene g the effect changes not yet applied to it (pend_d of the n_pend local variants in pend_list)
    n = resid.shape[1]
    for pp in range(n_pend):
        jj = pend_list[pp]
        if pend_d[jj] != 0.0:
            v = gene_var_idx[start + jj]
            m = var_mean[v]
            scale = pend_d[jj] / var_sd[v]
            for i in range(n):
                resid[g, i] -= (G[v, i] - m) * scale
        pend_d[jj] = 0.0
        in_pend[jj] = False


@njit(cache=True)
def _tgs_flush(jj, k, W, last_W, beta, P, accumulate, sum_beta, sum_p_slab):
    # Lazy accumulation of the 1 / Z - weighted effect and slab probabilities of local variant jj (pair k): they only change when jj is
    # updated, so the weight accumulated since its last change is applied once, before the change
    if accumulate:
        dw = W - last_W[jj]
        sum_beta[k] += dw * beta[k]
        sum_p_slab[k, 0] += dw * P[jj, 1]
        sum_p_slab[k, 1] += dw * P[jj, 2]
    last_W[jj] = W


@njit(parallel=True, cache=True)
def _tgs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, nbr_ptr, nbr_idx, nbr_r, resid, beta, z, sigma2_e, log_pi, slab_var,
                     sub_steps_per_snp, epsilon, tol, accumulate, reset_reservoir, sum_p_slab, sum_beta, sum_weight, res_weight, res_beta, res_rss,
                     sweep_stats):
    # One tempered-Gibbs sweep over every gene (see top of file). On entry resid[g] = y_g - X_g beta_g; the chain state (z, beta, resid) is
    # updated in place and left at the last sub-state. tol: tgs_selection_tolerance.
    # accumulate (post burn-in): add the 1 / Z - weighted conditional slab probabilities and effects to sum_p_slab and sum_beta, the total
    #     weight to sum_weight, and update the reservoir (res_weight: total weight of the sub-states it has seen; res_beta / res_rss: the
    #     effects and residual sum of squares of the sub-state it holds). reset_reservoir: start a new reservoir with this sweep.
    # sweep_stats (n_genes x 9), 1 / Z - weighted sums over this sweep's sub-states: [0] total weight, [1] sum of squared weights,
    #     [2:5] class counts, [5] / [6] sums of squared effects in slab 1 / 2, [7] residual sum of squares, [8] number of class changes
    n_genes = resid.shape[0]
    n = resid.shape[1]
    eps_over_K = epsilon / 3.0
    one_minus_eps = 1.0 - epsilon
    for g in prange(n_genes):
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        p = end - start
        # Per-gene constants of the class probabilities (log odds dL_c + a_c b^2 of slab c against the spike) and effect conditionals
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

        # Exact X_j' r, conditionals and selection weights of every variant; residual sum of squares; class counts and sums of squares
        P = np.empty((p, 3))
        b_cur = np.empty(p)
        t_last = np.empty(p)
        dt_max = np.empty(p)
        p_sel = np.empty(p)
        last_W = np.zeros(p)
        n_blocks = (p + SELECTION_BLOCK - 1) // SELECTION_BLOCK
        block_sum = np.zeros(n_blocks)
        block_dirty = np.ones(n_blocks, dtype=np.bool_)
        # Lazy residual: effect changes not yet applied to resid[g] (applied before the next exact X_j' r, and at the end of the sweep)
        pend_d = np.zeros(p)
        pend_list = np.empty(p, dtype=np.int64)
        in_pend = np.zeros(p, dtype=np.bool_)
        n_pend = 0
        # Exactness of b_cur: b_cur[j] is exact if every move since it was last computed from the residual (when the gene had made
        # base_moves[j] moves and j had received base_hits[j] neighbour updates) was by j itself or one of its LD neighbours
        n_moves = 0
        nbr_hits = np.zeros(p, dtype=np.int64)
        base_moves = np.zeros(p, dtype=np.int64)
        base_hits = np.zeros(p, dtype=np.int64)
        sum_r = 0.0
        rss = 0.0
        for i in range(n):
            sum_r += resid[g, i]
            rss += resid[g, i] * resid[g, i]
        n_class = np.zeros(3)
        ss_class = np.zeros(3)
        for jj in range(p):
            k = start + jj
            v = gene_var_idx[k]
            dot = 0.0
            for i in range(n):
                dot += G[v, i] * resid[g, i]
            b_cur[jj] = (dot - var_mean[v] * sum_r) / (var_sd[v] * n) + beta[k]
            t_last[jj] = b_cur[jj] * b_cur[jj]
            _tgs_conditional(jj, t_last[jj], dL1, dL2, a1, a2, P)
            _tgs_set_selection(jj, z[k], P, p_sel, dt_max, block_dirty, one_minus_eps, eps_over_K, a1, a2, tol)
            n_class[z[k]] += 1.0
            ss_class[z[k]] += beta[k] * beta[k]

        if accumulate and reset_reservoir:
            res_weight[g] = 0.0
        W = 0.0
        sum_w_sq = 0.0
        w_class = np.zeros(3)
        w_ss_1 = 0.0
        w_ss_2 = 0.0
        w_rss = 0.0
        n_changes = 0
        n_sub_steps = max(1, int(sub_steps_per_snp * p + 0.5))
        for step in range(n_sub_steps):
            # Weight of the current sub-state: 1 / Z, Z = mean_j p_j (block sums recomputed where a selection weight changed)
            s = 0.0
            for bb in range(n_blocks):
                if block_dirty[bb]:
                    acc = 0.0
                    for jj in range(bb * SELECTION_BLOCK, min(p, (bb + 1) * SELECTION_BLOCK)):
                        acc += p_sel[jj]
                    block_sum[bb] = acc
                    block_dirty[bb] = False
                s += block_sum[bb]
            w = p / s
            W += w
            sum_w_sq += w * w
            for c in range(3):
                w_class[c] += w * n_class[c]
            w_ss_1 += w * ss_class[1]
            w_ss_2 += w * ss_class[2]
            w_rss += w * rss
            if accumulate:
                # Weighted reservoir: the held sub-state is replaced by the current one with probability w / (total weight seen)
                res_weight[g] += w
                if np.random.random() * res_weight[g] < w:
                    for kk in range(start, end):
                        res_beta[kk] = beta[kk]
                    res_rss[g] = rss

            # Informed choice of the variant to update: j with probability proportional to p_j (block, then variant within the block)
            u = np.random.random() * s
            bb_sel = n_blocks - 1
            acc = 0.0
            for bb in range(n_blocks):
                if u < acc + block_sum[bb]:
                    bb_sel = bb
                    break
                acc += block_sum[bb]
            u -= acc
            block_end = min(p, (bb_sel + 1) * SELECTION_BLOCK)
            j = block_end - 1
            acc = 0.0
            for jj in range(bb_sel * SELECTION_BLOCK, block_end):
                acc += p_sel[jj]
                if u < acc:
                    j = jj
                    break
            k = start + j
            v = gene_var_idx[k]
            m = var_mean[v]
            sd = var_sd[v]

            # Exact b_j and conditional of the selected variant: b_j is recomputed from the (updated) residual unless it is already exact
            if n_moves - base_moves[j] != nbr_hits[j] - base_hits[j]:
                _tgs_apply_pending(g, start, G, var_mean, var_sd, gene_var_idx, resid, pend_list, n_pend, pend_d, in_pend)
                n_pend = 0
                dot = 0.0
                sum_r = 0.0
                for i in range(n):
                    dot += G[v, i] * resid[g, i]
                    sum_r += resid[g, i]
                b_cur[j] = (dot - m * sum_r) / (sd * n) + beta[k]
                base_moves[j] = n_moves
                base_hits[j] = nbr_hits[j]
            _tgs_flush(j, k, W, last_W, beta, P, accumulate, sum_beta, sum_p_slab)
            b = b_cur[j]
            xr_j = (b - beta[k]) * n
            t_last[j] = b * b
            _tgs_conditional(j, t_last[j], dL1, dL2, a1, a2, P)

            # Tempered draw of the class from g_j = (1 - eps) P_j + eps / 3, effect from its exact conditional given the class
            g0 = one_minus_eps * P[j, 0] + eps_over_K
            g1 = one_minus_eps * P[j, 1] + eps_over_K
            u = np.random.random()
            if u < g0:
                c = 0
                b_new = 0.0
            elif u < g0 + g1:
                c = 1
                b_new = shrink_1 * b + post_sd_1 * np.random.normal()
            else:
                c = 2
                b_new = shrink_2 * b + post_sd_2 * np.random.normal()
            c_old = z[k]
            if c != c_old:
                n_changes += 1
            n_class[c_old] -= 1.0
            ss_class[c_old] -= beta[k] * beta[k]
            n_class[c] += 1.0
            ss_class[c] += b_new * b_new

            # Apply the move: residual (lazily), residual sum of squares, b_a of j's LD neighbours (b_j itself does not change), and the
            # conditionals and selection weights of the neighbours whose weight may have moved by more than tol
            d = b_new - beta[k]
            beta[k] = b_new
            z[k] = c
            if d != 0.0:
                if not in_pend[j]:
                    pend_list[n_pend] = j
                    n_pend += 1
                    in_pend[j] = True
                pend_d[j] += d
                rss += -2.0 * d * xr_j + d * d * n
                n_moves += 1
                nbr_hits[j] += 1
                d_r = d / LD_QUANTIZATION
                for idx in range(nbr_ptr[k], nbr_ptr[k + 1]):
                    a = nbr_idx[idx]
                    if a == j:
                        continue
                    nbr_hits[a] += 1
                    b_a = b_cur[a] - d_r * nbr_r[idx]
                    b_cur[a] = b_a
                    t_a = b_a * b_a
                    if abs(t_a - t_last[a]) > dt_max[a]:
                        ka = start + a
                        _tgs_flush(a, ka, W, last_W, beta, P, accumulate, sum_beta, sum_p_slab)
                        t_last[a] = t_a
                        _tgs_conditional(a, t_a, dL1, dL2, a1, a2, P)
                        _tgs_set_selection(a, z[ka], P, p_sel, dt_max, block_dirty, one_minus_eps, eps_over_K, a1, a2, tol)
            _tgs_set_selection(j, c, P, p_sel, dt_max, block_dirty, one_minus_eps, eps_over_K, a1, a2, tol)

        _tgs_apply_pending(g, start, G, var_mean, var_sd, gene_var_idx, resid, pend_list, n_pend, pend_d, in_pend)
        for jj in range(p):
            _tgs_flush(jj, start + jj, W, last_W, beta, P, accumulate, sum_beta, sum_p_slab)
        if accumulate:
            sum_weight[g] += W
        sweep_stats[g, 0] = W
        sweep_stats[g, 1] = sum_w_sq
        for c in range(3):
            sweep_stats[g, 2 + c] = w_class[c]
        sweep_stats[g, 5] = w_ss_1
        sweep_stats[g, 6] = w_ss_2
        sweep_stats[g, 7] = w_rss
        sweep_stats[g, 8] = n_changes


def build_sparse_ld_neighbours(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, r2_threshold):
    # Per gene, the training-fold LD R = X_g' X_g / n of its variants; returns the CSR neighbour lists (r^2 >= r2_threshold, the variant
    # itself included) described at the top of the file
    n = G.shape[1]
    n_genes = len(gene_var_ptr) - 1
    counts = np.zeros(len(gene_var_idx), dtype=np.int64)
    idx_list = []
    r_list = []
    for g in range(n_genes):
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        rows = gene_var_idx[start:end]
        X = (G[rows].astype(np.float32) - var_mean[rows, None].astype(np.float32)) / var_sd[rows, None].astype(np.float32)
        R = np.dot(X, X.T) / n
        np.fill_diagonal(R, 1.0)
        R = np.clip(R, -1.0, 1.0)
        keep = np.square(R) >= r2_threshold
        np.fill_diagonal(keep, True)
        counts[start:end] = np.sum(keep, axis=1)
        nbr_rows, nbr_cols = np.nonzero(keep)
        idx_list.append(nbr_cols.astype(np.int16))
        r_list.append(np.round(R[nbr_rows, nbr_cols] * LD_QUANTIZATION).astype(np.int16))
    nbr_ptr = np.zeros(len(gene_var_idx) + 1, dtype=np.int64)
    nbr_ptr[1:] = np.cumsum(counts)
    return nbr_ptr, np.concatenate(idx_list), np.concatenate(r_list)


class EQTL_MIXTURE_PRIOR_TEMPERED_GIBBS(EQTL_MIXTURE_PRIOR_GIBBS):
    def __init__(self, tgs_epsilon=0.5, tgs_sub_steps_per_snp=1.0, tgs_r2_update_threshold=0.01, tgs_selection_tolerance=1e-3, tgs_window_sweeps=30, **kwargs):
        # kwargs: the arguments of EQTL_MIXTURE_PRIOR_GIBBS (same meaning; thin also sets how many sweeps' sub-states each retained sample
        #     is drawn from)
        # tgs_epsilon: weight of the uniform component of the tempered conditional (1 = fully tempered)
        # tgs_sub_steps_per_snp: sub-steps per sweep and gene, as a multiple of the gene's number of variants
        # tgs_r2_update_threshold: after a move, only the variants in training-fold LD (r^2) with the moved variant above this have X_a' r and
        #     their selection weight updated within the sweep (0 = every variant of the gene: exact TGS; see top of file)
        # tgs_selection_tolerance: relative error allowed on the selection weights of the LD neighbours of a moved variant before their
        #     conditionals are recomputed (0 = always recomputed; see top of file)
        # tgs_window_sweeps: the hyper-parameter updates use importance-weighted expected sufficient statistics pooled over this many sweeps
        #     (small windows let pi_1 drift upward; see top of file)
        super().__init__(**kwargs)
        if not (0.0 < tgs_epsilon <= 1.0):
            raise ValueError('tgs_epsilon must be in (0, 1]')
        if not (0.0 <= tgs_r2_update_threshold <= 1.0):
            raise ValueError('tgs_r2_update_threshold must be in [0, 1]')
        if tgs_selection_tolerance < 0.0:
            raise ValueError('tgs_selection_tolerance must be non-negative')
        if tgs_window_sweeps < 1:
            raise ValueError('tgs_window_sweeps must be at least 1')
        self.tgs_epsilon = float(tgs_epsilon)
        self.tgs_sub_steps_per_snp = float(tgs_sub_steps_per_snp)
        self.tgs_r2_update_threshold = float(tgs_r2_update_threshold)
        self.tgs_selection_tolerance = float(tgs_selection_tolerance)
        self.tgs_window_sweeps = int(tgs_window_sweeps)

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
        n = Y.shape[1]
        nz_pair_list = []
        nz_beta_list = []
        sample_iter = 0
        for itera in range(self.n_iters):
            t0 = time.time()
            if itera > 0 and itera % self.residual_recompute_interval == 0:
                _recompute_residuals_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, self.beta, self.resid)
            accumulate = itera >= self.burn_in
            # Each retained sample is drawn from the sub-states since the previous retained iteration (only this sweep's at burn_in)
            reset_reservoir = itera <= self.burn_in or bool(self.retained[itera - 1])
            _tgs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.nbr_ptr, self.nbr_idx, self.nbr_r, self.resid, self.beta, self.z,
                             self.sigma2_e, np.log(self.pi), self.slab_var, self.tgs_sub_steps_per_snp, self.tgs_epsilon, self.tgs_selection_tolerance, accumulate, reset_reservoir,
                             self.sum_p_slab, self.sum_beta, self.sum_weight, self.res_weight, self.res_beta, self.res_rss, self.sweep_stats)
            self.update_window_statistics()
            # Residual variances, then the shared hyper-parameters, from the expected statistics (the parent's sample_hyperparameters reads
            # class_counts / class_ss)
            self.sigma2_e = 1.0 / np.random.gamma(self.ig_a + 0.5 * n, 1.0 / (self.ig_b + 0.5 * self.expected_rss))
            guards = self.sample_hyperparameters()
            if '[slabs relabeled]' in guards:
                self.buf_stats[:, :, [2, 3]] = self.buf_stats[:, :, [3, 2]]
                self.buf_stats[:, :, [4, 5]] = self.buf_stats[:, :, [5, 4]]
            self.hyperparameter_trace[itera] = self.current_hyperparameters()
            if self.retained[itera]:
                nz = np.flatnonzero(self.res_beta)
                nz_pair_list.append(nz.astype(np.int32))
                nz_beta_list.append(self.res_beta[nz].astype(np.float32))
                self.sigma2_e_samples[sample_iter] = 1.0 / np.random.gamma(self.ig_a + 0.5 * n, 1.0 / (self.ig_b + 0.5 * self.res_rss))
                sample_iter += 1
            if self.verbose:
                n_class = np.sum(self.class_counts, axis=0)
                ess = self.sweep_stats[:, 0] ** 2 / self.sweep_stats[:, 1]
                print('iter ' + str(itera) + ': ' + ' '.join(name + '=%.3e' % value for name, value in zip(self.hyperparameter_names, self.current_hyperparameters())) + ' expected_n_slab1=%.1f expected_n_slab2=%.1f mean_sigma2_e=%.3e' % (n_class[1], n_class[2], np.mean(self.sigma2_e)) + ' tgs_changes_per_gene=%.1f tgs_mean_ess=%.1f' % (np.mean(self.sweep_stats[:, 8]), np.mean(ess)) + ' sec=%.1f' % (time.time() - t0) + guards, flush=True)

        # Posterior summaries: 1 / Z - weighted averages over every post burn-in sub-state, per gene
        pair_weight = np.repeat(self.sum_weight, np.diff(gene_var_ptr))
        self.p_slab = self.sum_p_slab / pair_weight[:, None]
        self.pip = np.sum(self.p_slab, axis=1)
        self.posterior_mean = self.sum_beta / pair_weight

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
        self.nbr_ptr, self.nbr_idx, self.nbr_r = build_sparse_ld_neighbours(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.tgs_r2_update_threshold)
        print('tempered Gibbs: epsilon=' + str(self.tgs_epsilon) + ' sub_steps_per_snp=' + str(self.tgs_sub_steps_per_snp) + ' r2_update_threshold=' + str(self.tgs_r2_update_threshold) + ' selection_tolerance=' + str(self.tgs_selection_tolerance) + ' window_sweeps=' + str(self.tgs_window_sweeps) + '; LD neighbours: %.1f per variant on average (%.1f%% of its gene\'s variants), %.2f GB, built in %.0f sec' % (len(self.nbr_idx) / self.n_pairs, 100.0 * np.mean(np.diff(self.nbr_ptr) / np.repeat(np.diff(gene_var_ptr), np.diff(gene_var_ptr))), (self.nbr_idx.nbytes + self.nbr_r.nbytes + self.nbr_ptr.nbytes) / 1e9, time.time() - t0), flush=True)
        self.sum_weight = np.zeros(self.n_genes)
        self.res_weight = np.zeros(self.n_genes)
        self.res_beta = np.zeros(self.n_pairs)
        self.res_rss = np.zeros(self.n_genes)
        self.sweep_stats = np.zeros((self.n_genes, 9))
        # Circular buffer of the per-sweep weighted statistics of every gene (columns 0 and 2:8 of sweep_stats: total weight, class counts 0 / 1 / 2,
        # sums of squares in slab 1 / 2, residual sum of squares)
        self.buf_stats = np.zeros((self.tgs_window_sweeps, self.n_genes, 7))
        self.sweeps_done = 0

    def update_window_statistics(self):
        # Store this sweep's weighted statistics in the window, and set the expected class counts / sums of squares per gene (class_counts,
        # class_ss, read by sample_hyperparameters) and residual sums of squares (expected_rss) pooled over the filled window
        slot = self.sweeps_done % self.tgs_window_sweeps
        self.buf_stats[slot] = self.sweep_stats[:, [0, 2, 3, 4, 5, 6, 7]]
        self.sweeps_done += 1
        pooled = np.sum(self.buf_stats[:min(self.sweeps_done, self.tgs_window_sweeps)], axis=0)
        total_w = pooled[:, 0]
        self.class_counts = pooled[:, 1:4] / total_w[:, None]
        self.class_ss = np.zeros((self.n_genes, 3))
        self.class_ss[:, 1:3] = pooled[:, 4:6] / total_w[:, None]
        self.expected_rss = pooled[:, 6] / total_w
