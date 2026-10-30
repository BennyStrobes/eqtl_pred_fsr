import numpy as np
import time
from numba import njit, prange


###############################################################################
# eQTL effect prediction with a genome-wide shared mixture prior (README.md, sections 4 and 5).
#
# Model (per gene g; n training individuals, p_g cis variants):
#   y_g = X_g beta_g + eps_g,  eps_g ~ N(0, sigma2_e_g I)
#   y_g: expression centered on its training mean; X_g: alt-allele dosages standardized in the training individuals (mean 0, variance 1)
#   beta_gj ~ pi_0 delta_0 + pi_1 N(0, sigma2_1) + pi_2 N(0, sigma2_2),  sigma2_1 < sigma2_2
# Shared across all genes: pi = (pi_0, pi_1, pi_2), sigma2_1, sigma2_2. Per gene: sigma2_e_g.
# Hyper-priors (weakly informative): pi ~ Dirichlet(alpha, alpha, alpha); sigma2_1, sigma2_2, sigma2_e_g ~ Inverse-Gamma(a, b)
#
# Gibbs sampler, one iteration:
#   1. every gene (in parallel over genes, numba prange; genes are independent given the shared parameters): random-scan single-site
#      updates of (z_gj, beta_gj) given the residual r = y_g - X_g beta_g, then sigma2_e_g given r
#   2. pi given the class counts, sigma2_c given the effects in class c, then the two slabs relabeled if sigma2_1 > sigma2_2 (the prior
#      is symmetric in the two slabs, so relabeling the draws targets the posterior under the ordering constraint)
# Genotypes are int8 alt-allele dosages G (one row per variant, training individuals only), standardized on the fly:
#   X_j = (G_j - m_j) / s_j, with m_j and s_j the training mean and standard deviation (1/n convention, so X_j' X_j = n).
#
# Variant-gene pairs are indexed k = 0, ..., n_pairs - 1; the pairs of gene g are gene_var_ptr[g]:gene_var_ptr[g + 1] and
# gene_var_idx[k] is the row of G of pair k (a variant in several genes' windows has one row and several pairs).
###############################################################################


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


@njit(parallel=True, cache=True)
def _gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, resid, beta, z, sigma2_e, log_pi, slab_var, ig_a, ig_b,
                       accumulate, sum_p_slab, sum_beta, class_counts, class_ss):
    # One sweep over every gene: (z, beta) of each of its variants in random order, then its residual variance.
    # On entry resid[g] = y_g - X_g beta_g; it is kept up to date through every move.
    # accumulate: add the conditional slab probabilities (Rao-Blackwellized; PIP = their sum) and the new effects to sum_p_slab and sum_beta.
    # class_counts / class_ss (n_genes x 3): number of variants and sum of squared effects in each class, per gene, after the sweep.
    n_genes = resid.shape[0]
    n = resid.shape[1]
    for g in prange(n_genes):
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        # Sampling variance of the single-variant regression coefficient b_j, and the per-gene constants of the class probabilities:
        # log pi_c + log N(b_j; 0, var_c) up to -0.5 log(2 pi), with var_0 = v_g (spike) and var_c = sigma2_c + v_g (slab c)
        v_g = sigma2_e[g] / n
        total_var_1 = slab_var[0] + v_g
        total_var_2 = slab_var[1] + v_g
        const_0 = log_pi[0] - 0.5 * np.log(v_g)
        const_1 = log_pi[1] - 0.5 * np.log(total_var_1)
        const_2 = log_pi[2] - 0.5 * np.log(total_var_2)
        # Posterior of beta_j given slab c: N(shrink_c b_j, post_sd_c^2)
        shrink_1 = slab_var[0] / total_var_1
        shrink_2 = slab_var[1] / total_var_2
        post_sd_1 = np.sqrt(slab_var[0] * v_g / total_var_1)
        post_sd_2 = np.sqrt(slab_var[1] * v_g / total_var_2)
        order = np.random.permutation(end - start)
        for oo in range(end - start):
            k = start + order[oo]
            v = gene_var_idx[k]
            m = var_mean[v]
            s = var_sd[v]
            # b_j = X_j' (r + X_j beta_j) / n, with X_j' r = (G_j' r - m_j sum(r)) / s_j (sum(r) is 0 up to rounding)
            dot = 0.0
            sum_r = 0.0
            for i in range(n):
                dot += G[v, i] * resid[g, i]
                sum_r += resid[g, i]
            b = (dot - m * sum_r) / (s * n) + beta[k]
            half_b2 = 0.5 * b * b
            l0 = const_0 - half_b2 / v_g
            l1 = const_1 - half_b2 / total_var_1
            l2 = const_2 - half_b2 / total_var_2
            mx = max(l0, max(l1, l2))
            w0 = np.exp(l0 - mx)
            w1 = np.exp(l1 - mx)
            w2 = np.exp(l2 - mx)
            tot = w0 + w1 + w2
            p1 = w1 / tot
            p2 = w2 / tot
            u = np.random.random()
            if u < p1:
                c = 1
                b_new = shrink_1 * b + post_sd_1 * np.random.normal()
            elif u < p1 + p2:
                c = 2
                b_new = shrink_2 * b + post_sd_2 * np.random.normal()
            else:
                c = 0
                b_new = 0.0
            if accumulate:
                sum_p_slab[k, 0] += p1
                sum_p_slab[k, 1] += p2
                sum_beta[k] += b_new
            # Apply the move to the residual (a null variant that stays null leaves it unchanged)
            d = b_new - beta[k]
            if d != 0.0:
                scale = d / s
                for i in range(n):
                    resid[g, i] -= (G[v, i] - m) * scale
            beta[k] = b_new
            z[k] = c
        # Residual variance: Inverse-Gamma(a + n / 2, b + r'r / 2)
        rss = 0.0
        for i in range(n):
            rss += resid[g, i] * resid[g, i]
        sigma2_e[g] = 1.0 / np.random.gamma(ig_a + 0.5 * n, 1.0 / (ig_b + 0.5 * rss))
        for c in range(3):
            class_counts[g, c] = 0.0
            class_ss[g, c] = 0.0
        for k in range(start, end):
            class_counts[g, z[k]] += 1.0
            class_ss[g, z[k]] += beta[k] * beta[k]


@njit(parallel=True, cache=True)
def _recompute_residuals_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, beta, resid):
    # resid[g] = y_g - X_g beta_g from scratch (removes the rounding error accumulated by the incremental updates)
    n_genes = Y.shape[0]
    n = Y.shape[1]
    for g in prange(n_genes):
        for i in range(n):
            resid[g, i] = Y[g, i]
        for k in range(gene_var_ptr[g], gene_var_ptr[g + 1]):
            if beta[k] != 0.0:
                v = gene_var_idx[k]
                m = var_mean[v]
                scale = beta[k] / var_sd[v]
                for i in range(n):
                    resid[g, i] -= (G[v, i] - m) * scale


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def sample_log_dirichlet(alphas):
    # log of a Dirichlet(alphas) draw, through log-Gamma variates: Gamma(a) = Gamma(a + 1) U^(1 / a), so small shapes (alpha + 0 for an
    # empty class, with alpha = 0.1) do not underflow to 0
    log_gammas = np.log(np.random.gamma(alphas + 1.0)) + np.log(np.random.random(len(alphas))) / alphas
    return log_gammas - logsumexp(log_gammas)


def sample_inverse_gamma(shape, rate):
    return 1.0 / np.random.gamma(shape, 1.0 / rate)


class EQTL_MIXTURE_PRIOR_GIBBS(object):
    def __init__(self, n_iters=3000, burn_in=500, thin=5, dirichlet_alpha=0.1, inverse_gamma_a=1e-10, inverse_gamma_b=1e-10,
                 pi_init=(0.99, 0.009, 0.001), slab_var_init=(1e-4, 1e-2), residual_recompute_interval=100, seed=None, verbose=True):
        # n_iters: total iterations; burn_in: iterations discarded; thin: every thin-th iteration after burn-in is a retained sample
        # (beta samples are saved); PIPs and posterior means average every post burn-in iteration
        # dirichlet_alpha: Dirichlet(alpha, alpha, alpha) prior on (pi_0, pi_1, pi_2); inverse_gamma_a / _b: Inverse-Gamma(a, b) prior on
        #     sigma2_1, sigma2_2 and every residual variance
        # pi_init, slab_var_init: initial (pi_0, pi_1, pi_2) and (sigma2_1, sigma2_2) (per-variant variance explained, expression has unit variance)
        # residual_recompute_interval: residuals are recomputed from scratch every this many iterations
        if burn_in >= n_iters:
            raise ValueError('burn_in must be smaller than n_iters')
        if slab_var_init[0] >= slab_var_init[1]:
            raise ValueError('slab_var_init must be increasing (sigma2_1 < sigma2_2)')
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.thin = thin
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.ig_a = float(inverse_gamma_a)
        self.ig_b = float(inverse_gamma_b)
        self.pi = np.asarray(pi_init, dtype=float) / np.sum(pi_init)
        self.slab_var = np.asarray(slab_var_init, dtype=float)
        self.residual_recompute_interval = residual_recompute_interval
        self.seed = seed
        self.verbose = verbose
        self.hyperparameter_names = ['pi0', 'pi1', 'pi2', 'sigma2_1', 'sigma2_2']

    def fit(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y):
        # G: int8 alt-allele dosages (n_variants x n training individuals); var_mean / var_sd: their training mean and standard deviation
        # gene_var_ptr (n_genes + 1), gene_var_idx (n_pairs): variant rows of each gene (see top of file)
        # Y: expression (n_genes x n), each gene centered on its training mean
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
        nz_pair_list = []
        nz_beta_list = []
        sample_iter = 0
        for itera in range(self.n_iters):
            t0 = time.time()
            if itera > 0 and itera % self.residual_recompute_interval == 0:
                _recompute_residuals_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, self.beta, self.resid)
            accumulate = itera >= self.burn_in
            _gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, np.log(self.pi), self.slab_var,
                               self.ig_a, self.ig_b, accumulate, self.sum_p_slab, self.sum_beta, self.class_counts, self.class_ss)
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
                print('iter ' + str(itera) + ': ' + ' '.join(name + '=%.3e' % value for name, value in zip(self.hyperparameter_names, self.current_hyperparameters())) + ' n_slab1=%d n_slab2=%d mean_sigma2_e=%.3e' % (n_class[1], n_class[2], np.mean(self.sigma2_e)) + ' sec=%.1f' % (time.time() - t0) + guards, flush=True)

        # Posterior summaries over the post burn-in iterations
        n_post = self.n_iters - self.burn_in
        self.p_slab = self.sum_p_slab / n_post
        self.pip = np.sum(self.p_slab, axis=1)
        self.posterior_mean = self.sum_beta / n_post

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
        if self.n_iters - self.burn_in > 32767 * self.thin:
            raise ValueError('too many retained samples for int16 sample indices')
        self.n_genes = Y.shape[0]
        self.n_pairs = len(gene_var_idx)
        self.beta = np.zeros(self.n_pairs)
        self.z = np.zeros(self.n_pairs, dtype=np.int8)
        self.resid = Y.copy()
        self.sigma2_e = np.var(Y, axis=1)
        self.sum_p_slab = np.zeros((self.n_pairs, 2))
        self.sum_beta = np.zeros(self.n_pairs)
        self.class_counts = np.zeros((self.n_genes, 3))
        self.class_ss = np.zeros((self.n_genes, 3))
        print(str(self.n_genes) + ' genes, ' + str(self.n_pairs) + ' variant-gene pairs, ' + str(G.shape[0]) + ' variants, ' + str(Y.shape[1]) + ' training individuals', flush=True)

    def current_hyperparameters(self):
        return np.asarray([self.pi[0], self.pi[1], self.pi[2], self.slab_var[0], self.slab_var[1]])

    def sample_hyperparameters(self):
        # Returns a string naming the numerical guards that fired (empty if none)
        guards = ''
        n_class = np.sum(self.class_counts, axis=0)
        ss_class = np.sum(self.class_ss, axis=0)
        # pi | z: Dirichlet(alpha + n_0, alpha + n_1, alpha + n_2), each pi_c floored at 1 / n_pairs (no class becomes impossible)
        pi = np.exp(sample_log_dirichlet(self.dirichlet_alpha + n_class))
        floor = 1.0 / self.n_pairs
        if np.any(pi < floor):
            pi = np.maximum(pi, floor)
            pi = pi / np.sum(pi)
            guards += ' [pi floored]'
        self.pi = pi
        # sigma2_c | beta: Inverse-Gamma(a + n_c / 2, b + sum of squares / 2), kept at its previous value if slab c is empty (its
        # conditional is then the near-improper prior)
        for c in range(2):
            if n_class[c + 1] > 0:
                self.slab_var[c] = sample_inverse_gamma(self.ig_a + 0.5 * n_class[c + 1], self.ig_b + 0.5 * ss_class[c + 1])
            else:
                guards += ' [slab ' + str(c + 1) + ' empty]'
        # Ordering constraint sigma2_1 < sigma2_2: relabel the slabs
        if self.slab_var[0] > self.slab_var[1]:
            self.slab_var = self.slab_var[::-1].copy()
            self.pi = self.pi[[0, 2, 1]]
            in_1 = self.z == 1
            in_2 = self.z == 2
            self.z[in_1] = 2
            self.z[in_2] = 1
            self.class_counts = self.class_counts[:, [0, 2, 1]]
            self.class_ss = self.class_ss[:, [0, 2, 1]]
            guards += ' [slabs relabeled]'
        return guards
