import numpy as np
import time
from numba import njit, prange
from eqtl_mixture_prior_gibbs import _seed_numba, _recompute_residuals_numba, sample_log_dirichlet


###############################################################################
# eQTL effect prediction with a genome-wide shared fixed-grid mixture prior (README.md, section 5e). Same data, likelihood and outputs as
# eqtl_mixture_prior_gibbs.py; only the prior on the effects differs.
#
# Model (per gene g; n training individuals, p_g cis variants):
#   y_g = X_g beta_g + eps_g,  eps_g ~ N(0, sigma2_e_g I)
#   beta_gj ~ pi_0 delta_0 + sum_{k=1..K} pi_k N(0, grid_var_k),  grid_var_1 < ... < grid_var_K FIXED (as in BayesR, Erbe et al. 2012)
# Shared across all genes: pi = (pi_0, ..., pi_K). Per gene: sigma2_e_g.
# Hyper-priors: pi ~ Dirichlet(alpha, ..., alpha); sigma2_e_g ~ Inverse-Gamma(a, b)
# With the component variances fixed there is no slab variance to collapse and no label switching; a component the data do not support
# just gets a small weight.
#
# Gibbs sampler, one iteration:
#   1. every gene (in parallel over genes): random-scan single-site updates of (z_gj, beta_gj) given the residual, over the K + 1 classes,
#      then sigma2_e_g given the residual
#   2. pi given the class counts
#
# Compatibility with the downstream code of the spike and two slab model: the components are split into a "small" group
# (grid_var_k < small_large_cutoff) and a "large" group. The PIP is the probability of any non-zero component, and the slab 1 / slab 2
# probabilities (pair_p_slab1 / pair_p_slab2 of the samples file, expected_n_slab1 / 2) are the probabilities of the small / large groups.
# The hyper-parameters pi0, pi1, pi2, sigma2_1, sigma2_2 are the collapsed two-group mixture: pi1 = total weight of the small group and
# sigma2_1 = its weight-averaged variance (the variance of an effect drawn from the group), and likewise for the large group, so that the
# prior-implied cis-h2, n_variants x (pi1 sigma2_1 + pi2 sigma2_2), is exact. The weights of the individual components are kept in
# grid_weight_trace.
###############################################################################

# Default grid: half-decades from 1e-4 to 1 (per-variant variance explained; expression has unit variance)
DEFAULT_GRID_VARIANCES = tuple(10.0 ** (-4.0 + 0.5 * k) for k in range(9))


@njit(parallel=True, cache=True)
def _grid_gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, resid, beta, z, sigma2_e, log_pi, grid_var, comp_group, ig_a, ig_b,
                            sum_p_group, sum_beta, class_counts):
    # One sweep over every gene: (z, beta) of each of its variants in random order, then its residual variance.
    # grid_var (K + 1): component variances, grid_var[0] = 0 (spike); comp_group (K + 1): 0 = small group, 1 = large group (spike: -1)
    # On entry resid[g] = y_g - X_g beta_g; it is kept up to date through every move.
    # Adds the conditional small / large group probabilities (Rao-Blackwellized; PIP = their sum) and the new effects to sum_p_group and sum_beta.
    # class_counts (n_genes x (K + 1)): number of variants in each component, per gene, after the sweep.
    n_genes = resid.shape[0]
    n = resid.shape[1]
    n_comp = grid_var.shape[0]
    for g in prange(n_genes):
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        # Per-gene constants: log pi_c + log N(b_j; 0, grid_var_c + v_g) up to -0.5 log(2 pi), and the posterior of beta_j given component c,
        # N(shrink_c b_j, post_sd_c^2)
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
        w = np.empty(n_comp)
        order = np.random.permutation(end - start)
        for oo in range(end - start):
            k = start + order[oo]
            v = gene_var_idx[k]
            m = var_mean[v]
            s = var_sd[v]
            # b_j = X_j' (r + X_j beta_j) / n, with X_j' r = (G_j' r - m_j sum(r)) / s_j
            dot = 0.0
            sum_r = 0.0
            for i in range(n):
                dot += G[v, i] * resid[g, i]
                sum_r += resid[g, i]
            b = (dot - m * sum_r) / (s * n) + beta[k]
            b2 = b * b
            mx = -np.inf
            for c in range(n_comp):
                w[c] = const[c] - half_inv_var[c] * b2
                if w[c] > mx:
                    mx = w[c]
            tot = 0.0
            for c in range(n_comp):
                w[c] = np.exp(w[c] - mx)
                tot += w[c]
            u = np.random.random() * tot
            c_new = n_comp - 1
            acc = 0.0
            for c in range(n_comp):
                acc += w[c]
                if u < acc:
                    c_new = c
                    break
            if c_new == 0:
                b_new = 0.0
            else:
                b_new = shrink[c_new] * b + post_sd[c_new] * np.random.normal()
            p_small = 0.0
            p_large = 0.0
            for c in range(1, n_comp):
                if comp_group[c] == 0:
                    p_small += w[c]
                else:
                    p_large += w[c]
            sum_p_group[k, 0] += p_small / tot
            sum_p_group[k, 1] += p_large / tot
            sum_beta[k] += b_new
            # Apply the move to the residual (a null variant that stays null leaves it unchanged)
            d = b_new - beta[k]
            if d != 0.0:
                scale = d / s
                for i in range(n):
                    resid[g, i] -= (G[v, i] - m) * scale
            beta[k] = b_new
            z[k] = c_new
        # Residual variance: Inverse-Gamma(a + n / 2, b + r'r / 2)
        rss = 0.0
        for i in range(n):
            rss += resid[g, i] * resid[g, i]
        sigma2_e[g] = 1.0 / np.random.gamma(ig_a + 0.5 * n, 1.0 / (ig_b + 0.5 * rss))
        for c in range(n_comp):
            class_counts[g, c] = 0.0
        for k in range(start, end):
            class_counts[g, z[k]] += 1.0


class EQTL_GRID_PRIOR_GIBBS(object):
    def __init__(self, n_iters=3000, burn_in=500, thin=5, dirichlet_alpha=0.1, inverse_gamma_a=1e-10, inverse_gamma_b=1e-10,
                 grid_variances=DEFAULT_GRID_VARIANCES, small_large_cutoff=1e-3, pi0_init=0.99, residual_recompute_interval=100, seed=None, verbose=True):
        # n_iters: total iterations; burn_in: iterations discarded; thin: every thin-th iteration after burn-in is a retained sample
        # (beta samples are saved); PIPs and posterior means average every post burn-in iteration
        # dirichlet_alpha: Dirichlet(alpha, ..., alpha) prior on (pi_0, ..., pi_K); inverse_gamma_a / _b: Inverse-Gamma(a, b) prior on every
        #     residual variance
        # grid_variances: the K fixed non-zero component variances (per-variant variance explained, expression has unit variance)
        # small_large_cutoff: components with variance below it form the "small" group, the others the "large" group (see top of file)
        # pi0_init: initial spike weight; the rest is shared equally by the K components
        # residual_recompute_interval: residuals are recomputed from scratch every this many iterations
        if burn_in >= n_iters:
            raise ValueError('burn_in must be smaller than n_iters')
        grid_variances = np.sort(np.asarray(grid_variances, dtype=float))
        if np.any(grid_variances <= 0.0) or len(np.unique(grid_variances)) != len(grid_variances):
            raise ValueError('grid_variances must be positive and distinct')
        if not (grid_variances[0] < small_large_cutoff <= grid_variances[-1]):
            raise ValueError('small_large_cutoff must leave at least one grid variance in each of the small and large groups')
        if len(grid_variances) + 1 > 127:
            raise ValueError('at most 126 grid variances (int8 classes)')
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.thin = thin
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.ig_a = float(inverse_gamma_a)
        self.ig_b = float(inverse_gamma_b)
        self.grid_var = np.concatenate([[0.0], grid_variances])
        self.n_comp = len(self.grid_var)
        self.small_large_cutoff = float(small_large_cutoff)
        self.comp_group = np.where(self.grid_var < small_large_cutoff, 0, 1).astype(np.int64)
        self.comp_group[0] = -1
        self.pi = np.concatenate([[pi0_init], np.full(len(grid_variances), (1.0 - pi0_init) / len(grid_variances))])
        self.residual_recompute_interval = residual_recompute_interval
        self.seed = seed
        self.verbose = verbose
        self.hyperparameter_names = ['pi0', 'pi1', 'pi2', 'sigma2_1', 'sigma2_2']
        self.grid_weight_names = ['pi_spike'] + ['pi_var_%.3g' % value for value in grid_variances]

    def fit(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y):
        # Same inputs as EQTL_MIXTURE_PRIOR_GIBBS.fit, and the same attributes afterwards, plus grid_weight_trace (n_iters x (K + 1))
        if self.seed is not None:
            np.random.seed(self.seed)
            _seed_numba(self.seed)
        self.initialize(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y)
        retained_iters = [itera for itera in range(self.burn_in, self.n_iters) if (itera - self.burn_in) % self.thin == 0]
        self.n_samples = len(retained_iters)
        self.hyperparameter_trace = np.zeros((self.n_iters, len(self.hyperparameter_names)))
        self.grid_weight_trace = np.zeros((self.n_iters, self.n_comp))
        self.retained = np.zeros(self.n_iters, dtype=bool)
        self.retained[retained_iters] = True
        self.sigma2_e_samples = np.zeros((self.n_samples, self.n_genes), dtype=np.float32)
        nz_pair_list = []
        nz_beta_list = []
        sample_iter = 0
        # sum_p_group / sum_beta accumulate every iteration (the GWFM subclass needs a running PIP during burn-in too); the posterior
        # summaries are their increase after burn-in
        for itera in range(self.n_iters):
            t0 = time.time()
            if itera > 0 and itera % self.residual_recompute_interval == 0:
                _recompute_residuals_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, self.beta, self.resid)
            if itera == self.burn_in:
                burn_in_p_group = self.sum_p_group.copy()
                burn_in_beta = self.sum_beta.copy()
            sweep_string = self.sweep(itera, G, var_mean, var_sd, gene_var_ptr, gene_var_idx)
            guards = self.sample_hyperparameters()
            self.hyperparameter_trace[itera] = self.current_hyperparameters()
            self.grid_weight_trace[itera] = self.pi
            if self.retained[itera]:
                nz = np.flatnonzero(self.beta)
                nz_pair_list.append(nz.astype(np.int32))
                nz_beta_list.append(self.beta[nz].astype(np.float32))
                self.sigma2_e_samples[sample_iter] = self.sigma2_e
                sample_iter += 1
            if self.verbose:
                n_class = np.sum(self.class_counts, axis=0)
                print('iter ' + str(itera) + ': ' + ' '.join(name + '=%.3e' % value for name, value in zip(self.hyperparameter_names, self.current_hyperparameters())) + ' n_small=%d n_large=%d mean_sigma2_e=%.3e' % (np.sum(n_class[self.comp_group == 0]), np.sum(n_class[self.comp_group == 1]), np.mean(self.sigma2_e)) + sweep_string + ' sec=%.1f' % (time.time() - t0) + guards, flush=True)

        # Posterior summaries over the post burn-in iterations
        n_post = self.n_iters - self.burn_in
        self.p_slab = (self.sum_p_group - burn_in_p_group) / n_post
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

    def sweep(self, itera, G, var_mean, var_sd, gene_var_ptr, gene_var_idx):
        # One Gibbs sweep, accumulating into sum_p_group / sum_beta; returns a string for the verbose output
        _grid_gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, np.log(self.pi), self.grid_var,
                                self.comp_group, self.ig_a, self.ig_b, self.sum_p_group, self.sum_beta, self.class_counts)
        return ''

    def initialize(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y):
        if self.n_iters - self.burn_in > 32767 * self.thin:
            raise ValueError('too many retained samples for int16 sample indices')
        self.n_genes = Y.shape[0]
        self.n_pairs = len(gene_var_idx)
        self.beta = np.zeros(self.n_pairs)
        self.z = np.zeros(self.n_pairs, dtype=np.int8)
        self.resid = Y.copy()
        self.sigma2_e = np.var(Y, axis=1)
        self.sum_p_group = np.zeros((self.n_pairs, 2))
        self.sum_beta = np.zeros(self.n_pairs)
        self.class_counts = np.zeros((self.n_genes, self.n_comp))
        print(str(self.n_genes) + ' genes, ' + str(self.n_pairs) + ' variant-gene pairs, ' + str(G.shape[0]) + ' variants, ' + str(Y.shape[1]) + ' training individuals', flush=True)
        print('grid prior: ' + str(self.n_comp - 1) + ' components, variances ' + ', '.join('%.3g' % value for value in self.grid_var[1:]) + '; small group: variance < ' + str(self.small_large_cutoff), flush=True)

    def current_hyperparameters(self):
        # Collapsed two-group mixture (see top of file)
        small = self.comp_group == 0
        large = self.comp_group == 1
        pi1 = np.sum(self.pi[small])
        pi2 = np.sum(self.pi[large])
        return np.asarray([self.pi[0], pi1, pi2, np.sum(self.pi[small] * self.grid_var[small]) / pi1, np.sum(self.pi[large] * self.grid_var[large]) / pi2])

    def sample_hyperparameters(self):
        # pi | z: Dirichlet(alpha + n_0, ..., alpha + n_K), drawn in log space. Each pi_k is floored only at the smallest positive double, so
        # that log pi stays finite: unlike the two slab model's floor of 1 / n_pairs, a floor that large would hold the weights of the empty
        # large-variance components well above their posterior and inflate the prior's variance E[beta^2] (by about 15% in a 164k-pair
        # simulation). Returns a string naming the numerical guards that fired (empty if none)
        guards = ''
        n_class = np.sum(self.class_counts, axis=0)
        pi = np.exp(sample_log_dirichlet(self.dirichlet_alpha + n_class))
        floor = np.finfo(float).tiny
        if np.any(pi < floor):
            pi = np.maximum(pi, floor)
            pi = pi / np.sum(pi)
            guards += ' [pi floored]'
        self.pi = pi
        return guards
