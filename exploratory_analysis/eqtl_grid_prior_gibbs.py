import numpy as np
import time
from numba import njit, prange
from scipy.special import gammaln
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
# Variant-count bins (n_variant_bins = B, README.md, section 5g): genes are split into B equal-count bins of their number of cis variants M_g,
# and each bin has its own inclusion probability, so that a gene's prior number of non-zero effects need not grow linearly with M_g:
#   beta_gj ~ (1 - q_b(g)) delta_0 + q_b(g) sum_k w_k N(0, grid_var_k)
# Shared: w = (w_1, ..., w_K), the weights of the non-zero components. Hyper-priors: q_b ~ Beta(alpha, alpha), w ~ Dirichlet(alpha, ..., alpha).
# The prior on the classes factorizes as prod_b q_b^k_b (1 - q_b)^(M_b - k_b) x prod_k w_k^n_k (k_b, M_b: non-zero effects and variants of
# the genes in bin b; n_k: variants in component k), so both updates are exact and conjugate. pi (the grid weights written to grid_weight_trace
# and collapsed into pi0, pi1, ...) is then the prior averaged over all variant-gene pairs, pi_0 = sum_b M_b (1 - q_b) / M, pi_k = (1 - pi_0) w_k.
#
# Variant categories (variant_category_labels, README.md, section 5j): every variant-gene pair belongs to one of A categories fixed before
# fitting (a bin of its training-fold MAF, or of its distance to the gene's TSS), and each category has its own grid weights:
#   beta_gj ~ pi_{a(gj), 0} delta_0 + sum_k pi_{a(gj), k} N(0, grid_var_k),  pi_a ~ Dirichlet(alpha, ..., alpha)
# so that both the inclusion probability and the distribution of effect sizes can differ between categories; pi_a | z ~ Dirichlet(alpha + n_a0,
# ..., alpha + n_aK), with n_ac the pairs of category a in component c.
#
# Latent gene classes (n_gene_classes = L, README.md, section 5j): every gene belongs to one of L classes, each with its own grid weights, and
# the classes are shared by the genes:
#   c_g ~ Categorical(rho),  beta_gj ~ pi_{c_g, 0} delta_0 + sum_k pi_{c_g, k} N(0, grid_var_k)
# Hyper-priors: rho ~ Dirichlet(gene_class_alpha, ...), pi_l ~ Dirichlet(alpha, ..., alpha). Given the component assignments, c_g | z ~
# rho_l prod_c pi_{l, c}^n_gc (n_gc: variants of gene g in component c), then pi_l | z, c ~ Dirichlet(alpha + sum_{g: c_g = l} n_g) and
# rho | c ~ Dirichlet(gene_class_alpha + genes per class); all exact and conjugate. Because c_g given the labels mixes slowly (a gene keeps the
# small effects of its class), every iteration also makes one non-centred move of every gene's class (_gene_class_ncp_numba). The classes are exchangeable (labels can switch, which does
# not change any prediction); they start with inclusion probabilities spread geometrically over 1e-3 to 1e-1 to break the symmetry, every
# gene in the class closest to 1 - pi0_init.
#
# Implementation of the four priors (shared, variant-count bins, variant categories, gene classes): the sweep takes a table of log prior
# weights, one row per prior, and the row of every variant-gene pair (the gene's bin, the pair's category, or the gene's current class).
# pi (grid_weight_trace, pi0, pi1, ...) is always the prior averaged over all variant-gene pairs.
#
# Non-centred weight moves (ncp_moves, shared prior only; README.md, section 5k): the Dirichlet update of pi given the class counts mixes very
# slowly along directions the data barely inform, such as more near-null effects against smaller residual variances (each gene's sigma2_e
# trades against its small effects), and the chain then wanders far from the posterior of pi. In simulations this made the genetic FSR
# over-confident by 0.03 at a mean FSR of 0.11. After it, ncp_moves Metropolis-Hastings moves of pi each iteration: mass delta moves between
# two components a and b; every variant in the losing component switches to the gaining one with probability |delta| / pi_loser, keeping its
# standardized effect (beta' = beta sigma_new / sigma_old; spike to slab: beta' = sigma_new N(0, 1); slab to spike: 0); the move is accepted
# with the change of the likelihood and of the Dirichlet prior. It is exact (a variant's uniform position within its component, and the
# standard normal effect of a spike variant, are auxiliary variables drawn from their conditionals; ancillarity-sufficiency interweaving,
# Yu and Meng 2011). delta has about ncp_target_flips variants switch; its scale per pair of components adapts during burn-in only.
#
# Hierarchical residual variances (hierarchical_sigma2_e, README.md, section 5k): sigma2_e_g ~ Inverse-Gamma(a, b) with (a, b) learned instead
# of fixed at (inverse_gamma_a, inverse_gamma_b). A fixed prior multiplies over genes: lowering every gene's sigma2_e by a fraction f changes the
# log prior by about n_genes f (for the default near-1 / sigma2 prior), and the data barely distinguish that from more near-null effects, so the
# fixed prior sets the weight of the smallest components (too many with 1 / sigma2, none with a flat prior; both made the genetic FSR
# over-confident in simulations). A learned (a, b) follows such a shift, so the data decide. Updates: b | a ~ Gamma(n_genes a + 1,
# sum_g 1 / sigma2_e_g) (flat prior on b), then 5 random-walk Metropolis steps on log a (prior proportional to 1 / a).
#
# Gibbs sampler, one iteration:
#   1. every gene (in parallel over genes): random-scan single-site updates of (z_gj, beta_gj) given the residual, over the K + 1 classes,
#      then sigma2_e_g given the residual
#   2. pi (or q_b and w; or pi_a; or c_g, then pi_l and rho) given the class counts
#   3. with ncp_moves, the non-centred moves of pi
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
def _grid_gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, resid, beta, z, sigma2_e, log_pi, pair_row, grid_var, comp_group, ig_a,
                            ig_b, sample_sigma2_e, sum_p_group, sum_beta, class_counts):
    # One sweep over every gene: (z, beta) of each of its variants in random order, then its residual variance (unless sample_sigma2_e is
    # False: held fixed).
    # log_pi (n_rows x (K + 1)): log prior weights of the components, one row per prior; pair_row (n_pairs): the row of each variant-gene pair
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
        # Per-gene constants: log N(b_j; 0, grid_var_c + v_g) up to -0.5 log(2 pi) (the pair's log pi_c is added per variant), and the
        # posterior of beta_j given component c, N(shrink_c b_j, post_sd_c^2)
        v_g = sigma2_e[g] / n
        const = np.empty(n_comp)
        half_inv_var = np.empty(n_comp)
        shrink = np.empty(n_comp)
        post_sd = np.empty(n_comp)
        for c in range(n_comp):
            total_var = grid_var[c] + v_g
            const[c] = -0.5 * np.log(total_var)
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
            row = pair_row[k]
            mx = -np.inf
            for c in range(n_comp):
                w[c] = log_pi[row, c] + const[c] - half_inv_var[c] * b2
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
        if sample_sigma2_e:
            rss = 0.0
            for i in range(n):
                rss += resid[g, i] * resid[g, i]
            sigma2_e[g] = 1.0 / np.random.gamma(ig_a + 0.5 * n, 1.0 / (ig_b + 0.5 * rss))
        for c in range(n_comp):
            class_counts[g, c] = 0.0
        for k in range(start, end):
            class_counts[g, z[k]] += 1.0


@njit(parallel=True, cache=True)
def _ncp_delta_loglik_numba(G, var_mean, var_sd, gene_var_idx, resid, sigma2_e, flip_gene_ptr, flip_pair, flip_dbeta, flip_genes, dX):
    # Change of the log-likelihood of every gene in flip_genes when its pairs flip_pair[flip_gene_ptr[f]:flip_gene_ptr[f + 1]] change by
    # flip_dbeta; dX[f] = X_g dbeta (applied to the residual if the move is accepted)
    n = resid.shape[1]
    out = np.zeros(len(flip_genes))
    for f in prange(len(flip_genes)):
        g = flip_genes[f]
        for i in range(n):
            dX[f, i] = 0.0
        for t in range(flip_gene_ptr[f], flip_gene_ptr[f + 1]):
            v = gene_var_idx[flip_pair[t]]
            scale = flip_dbeta[t] / var_sd[v]
            for i in range(n):
                dX[f, i] += (G[v, i] - var_mean[v]) * scale
        cross = 0.0
        sq = 0.0
        for i in range(n):
            cross += resid[g, i] * dX[f, i]
            sq += dX[f, i] * dX[f, i]
        out[f] = (2.0 * cross - sq) / (2.0 * sigma2_e[g])
    return out


@njit(parallel=True, cache=True)
def _gene_class_ncp_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, resid, beta, z, sigma2_e, row_log_pi, gene_class, log_class_weight, grid_sd,
                          class_counts, accepted):
    # One non-centred Metropolis-Hastings move of every gene's class (README.md, section 5k): a proposed class l' (uniform among the others);
    # each variant keeps its uniform position u within its component under the current class's weights (u drawn from its conditional) and
    # takes the component of u under l', and keeps its standardized effect (a spike variant draws one from N(0, 1)); accepted with the change
    # of the likelihood and of log rho. accepted[g] = 1 if gene g moved
    n_genes = resid.shape[0]
    n = resid.shape[1]
    n_rows = row_log_pi.shape[0]
    n_comp = row_log_pi.shape[1]
    for g in prange(n_genes):
        accepted[g] = 0
        if n_rows < 2:
            continue
        l = gene_class[g]
        lp = np.random.randint(0, n_rows - 1)
        if lp >= l:
            lp += 1
        # Cumulative weights of both classes (component order: spike, then increasing variance)
        cum = np.zeros(n_comp + 1)
        cum_p = np.zeros(n_comp + 1)
        for c in range(n_comp):
            cum[c + 1] = cum[c] + np.exp(row_log_pi[l, c])
            cum_p[c + 1] = cum_p[c] + np.exp(row_log_pi[lp, c])
        start = gene_var_ptr[g]
        end = gene_var_ptr[g + 1]
        new_z = np.empty(end - start, dtype=np.int64)
        new_beta = np.empty(end - start)
        dX = np.zeros(n)
        any_change = False
        for k in range(start, end):
            c = z[k]
            u = cum[c] + np.random.random() * (cum[c + 1] - cum[c])
            cp = n_comp - 1
            for c2 in range(n_comp):
                if u < cum_p[c2 + 1]:
                    cp = c2
                    break
            new_z[k - start] = cp
            if cp == c:
                new_beta[k - start] = beta[k]
                continue
            if cp == 0:
                b_new = 0.0
            elif c == 0:
                b_new = grid_sd[cp] * np.random.normal()
            else:
                b_new = beta[k] * grid_sd[cp] / grid_sd[c]
            new_beta[k - start] = b_new
            d = b_new - beta[k]
            if d != 0.0:
                any_change = True
                v = gene_var_idx[k]
                scale = d / var_sd[v]
                for i in range(n):
                    dX[i] += (G[v, i] - var_mean[v]) * scale
        cross = 0.0
        sq = 0.0
        if any_change:
            for i in range(n):
                cross += resid[g, i] * dX[i]
                sq += dX[i] * dX[i]
        log_ratio = (2.0 * cross - sq) / (2.0 * sigma2_e[g]) + log_class_weight[lp] - log_class_weight[l]
        if np.log(np.random.random()) < log_ratio:
            for i in range(n):
                resid[g, i] -= dX[i]
            for c in range(n_comp):
                class_counts[g, c] = 0.0
            for k in range(start, end):
                beta[k] = new_beta[k - start]
                z[k] = new_z[k - start]
                class_counts[g, new_z[k - start]] += 1.0
            gene_class[g] = lp
            accepted[g] = 1


class EQTL_GRID_PRIOR_GIBBS(object):
    def __init__(self, n_iters=3000, burn_in=500, thin=5, dirichlet_alpha=0.1, inverse_gamma_a=1e-10, inverse_gamma_b=1e-10,
                 grid_variances=DEFAULT_GRID_VARIANCES, small_large_cutoff=1e-3, pi0_init=0.99, residual_recompute_interval=100, n_variant_bins=None,
                 variant_category_labels=None, n_gene_classes=None, gene_class_alpha=1.0, fixed_weights=None, ncp_moves=0, ncp_target_flips=200,
                 hierarchical_sigma2_e=False, gene_class_moves=True, seed=None, verbose=True):
        # n_iters: total iterations; burn_in: iterations discarded; thin: every thin-th iteration after burn-in is a retained sample
        # (beta samples are saved); PIPs and posterior means average every post burn-in iteration
        # dirichlet_alpha: Dirichlet(alpha, ..., alpha) prior on (pi_0, ..., pi_K); inverse_gamma_a / _b: Inverse-Gamma(a, b) prior on every
        #     residual variance
        # grid_variances: the K fixed non-zero component variances (per-variant variance explained, expression has unit variance)
        # small_large_cutoff: components with variance below it form the "small" group, the others the "large" group (see top of file)
        # pi0_init: initial spike weight; the rest is shared equally by the K components
        # residual_recompute_interval: residuals are recomputed from scratch every this many iterations
        # n_variant_bins: None for one prior shared by every gene; B for an inclusion probability per bin of the number of cis variants (see top)
        # variant_category_labels: None, or the label of each of the A variant categories, with grid weights per category (fit then takes
        #     pair_category, the category of every variant-gene pair; see top)
        # n_gene_classes: None, or L latent gene classes with grid weights per class (see top); gene_class_alpha: Dirichlet prior on their weights
        # At most one of n_variant_bins, variant_category_labels and n_gene_classes
        # fixed_weights: None, or the K + 1 grid weights (spike first) held fixed instead of sampled (shared prior only; README.md, section 5k)
        # ncp_moves: non-centred Metropolis-Hastings moves of the weights per iteration (shared prior only; see top); ncp_target_flips: variants
        #     switching component in a typical move
        # gene_class_moves: with gene classes, the non-centred class move every iteration (see top; False: the conditional draw of c_g only)
        # hierarchical_sigma2_e: learn the Inverse-Gamma prior of the residual variances (see top); inverse_gamma_a / _b are then its start
        #     (if inverse_gamma_a < 1: a = 10 with mean the mean variance of the expression)
        if burn_in >= n_iters:
            raise ValueError('burn_in must be smaller than n_iters')
        grid_variances = np.sort(np.asarray(grid_variances, dtype=float))
        if np.any(grid_variances <= 0.0) or len(np.unique(grid_variances)) != len(grid_variances):
            raise ValueError('grid_variances must be positive and distinct')
        if not (grid_variances[0] < small_large_cutoff <= grid_variances[-1]):
            raise ValueError('small_large_cutoff must leave at least one grid variance in each of the small and large groups')
        if len(grid_variances) + 1 > 127:
            raise ValueError('at most 126 grid variances (int8 classes)')
        if n_variant_bins is not None and n_variant_bins < 1:
            raise ValueError('n_variant_bins must be at least 1')
        if n_gene_classes is not None and not (1 <= n_gene_classes <= 127):
            raise ValueError('n_gene_classes must be between 1 and 127 (int8 classes)')
        if np.sum([n_variant_bins is not None, variant_category_labels is not None, n_gene_classes is not None]) > 1:
            raise ValueError('at most one of n_variant_bins, variant_category_labels and n_gene_classes')
        self.n_variant_bins = n_variant_bins
        self.variant_category_labels = list(variant_category_labels) if variant_category_labels is not None else None
        self.n_gene_classes = n_gene_classes
        self.gene_class_alpha = float(gene_class_alpha)
        if n_variant_bins is not None:
            self.prior_mode = 'variant_count_bins'
        elif variant_category_labels is not None:
            self.prior_mode = 'variant_categories'
        elif n_gene_classes is not None:
            self.prior_mode = 'gene_classes'
        else:
            self.prior_mode = 'shared'
        self.pi0_init = float(pi0_init)
        if fixed_weights is not None:
            if self.prior_mode != 'shared' or len(fixed_weights) != len(grid_variances) + 1:
                raise ValueError('fixed_weights needs the shared prior and K + 1 weights')
        self.fixed_weights = None if fixed_weights is None else np.asarray(fixed_weights, dtype=float) / np.sum(fixed_weights)
        if ncp_moves > 0 and (self.prior_mode != 'shared' or fixed_weights is not None):
            raise ValueError('ncp_moves needs the shared prior with sampled weights')
        self.ncp_moves = int(ncp_moves)
        self.hierarchical_sigma2_e = bool(hierarchical_sigma2_e)
        self.gene_class_moves_on = bool(gene_class_moves)
        self.ncp_target_flips = float(ncp_target_flips)
        self.sigma2_e_fixed = None
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
        if self.fixed_weights is not None:
            self.pi = self.fixed_weights.copy()
        self.residual_recompute_interval = residual_recompute_interval
        self.seed = seed
        self.verbose = verbose
        self.hyperparameter_names = ['pi0', 'pi1', 'pi2', 'sigma2_1', 'sigma2_2']
        self.grid_weight_names = ['pi_spike'] + ['pi_var_%.3g' % value for value in grid_variances]

    def fit(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, pair_category=None, sigma2_e_fixed=None):
        # Same inputs as EQTL_MIXTURE_PRIOR_GIBBS.fit, and the same attributes afterwards, plus grid_weight_trace (n_iters x (K + 1))
        # pair_category (n_pairs): with variant_category_labels, the category of every variant-gene pair (0 to A - 1)
        # sigma2_e_fixed (n_genes): residual variances held fixed instead of sampled (stage 1 of the two-stage fit; README.md, section 5k)
        # With variant categories or gene classes, also row_weight_trace (n_iters x n_rows x (K + 1)): the grid weights of every category or
        # class after every iteration; with gene classes, gene_class_samples (n_samples x n_genes) and gene_class_weight_trace (n_iters x L)
        if self.seed is not None:
            np.random.seed(self.seed)
            _seed_numba(self.seed)
        self.sigma2_e_fixed = None if sigma2_e_fixed is None else np.asarray(sigma2_e_fixed, dtype=float)
        self.initialize(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, pair_category)
        retained_iters = [itera for itera in range(self.burn_in, self.n_iters) if (itera - self.burn_in) % self.thin == 0]
        self.n_samples = len(retained_iters)
        self.hyperparameter_trace = np.zeros((self.n_iters, len(self.hyperparameter_names)))
        self.grid_weight_trace = np.zeros((self.n_iters, self.n_comp))
        self.sigma2_e_prior_trace = np.zeros((self.n_iters, 2))
        if self.n_variant_bins is not None:
            # q_b after every iteration, and the mean over retained samples of each bin's fraction of variants in the large group (an
            # identified quantity, unlike q_b, whose mass in the near-spike components is not; README.md, section 5e)
            self.bin_q_trace = np.zeros((self.n_iters, self.n_variant_bins))
            self.bin_large_fraction = np.zeros(self.n_variant_bins)
        if self.prior_mode in ['variant_categories', 'gene_classes']:
            self.row_weight_trace = np.zeros((self.n_iters, self.n_rows, self.n_comp), dtype=np.float32)
        if self.prior_mode == 'gene_classes':
            self.gene_class_samples = np.zeros((self.n_samples, self.n_genes), dtype=np.int8)
            self.gene_class_weight_trace = np.zeros((self.n_iters, self.n_gene_classes))
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
            if self.hierarchical_sigma2_e and self.sigma2_e_fixed is None:
                self.sample_sigma2_e_prior()
                self.sigma2_e_prior_trace[itera] = [self.ig_a, self.ig_b]
            if self.ncp_moves > 0:
                guards += self.ncp_weight_moves(G, var_mean, var_sd, gene_var_idx, adapt=itera < self.burn_in)
            self.hyperparameter_trace[itera] = self.current_hyperparameters()
            self.grid_weight_trace[itera] = self.pi
            if self.n_variant_bins is not None:
                self.bin_q_trace[itera] = self.bin_q
                if self.retained[itera]:
                    gene_n_large = np.sum(self.class_counts[:, self.comp_group == 1], axis=1)
                    self.bin_large_fraction += np.bincount(self.gene_bin, weights=gene_n_large, minlength=self.n_variant_bins) / self.bin_n_pairs
            if self.prior_mode in ['variant_categories', 'gene_classes']:
                self.row_weight_trace[itera] = np.exp(self.row_log_pi)
            if self.prior_mode == 'gene_classes':
                self.gene_class_weight_trace[itera] = self.gene_class_weight
                if self.retained[itera]:
                    self.gene_class_samples[sample_iter] = self.gene_class
            if self.retained[itera]:
                nz = np.flatnonzero(self.beta)
                nz_pair_list.append(nz.astype(np.int32))
                nz_beta_list.append(self.beta[nz].astype(np.float32))
                self.sigma2_e_samples[sample_iter] = self.sigma2_e
                sample_iter += 1
            if self.verbose:
                n_class = np.sum(self.class_counts, axis=0)
                print('iter ' + str(itera) + ': ' + ' '.join(name + '=%.3e' % value for name, value in zip(self.hyperparameter_names, self.current_hyperparameters())) + ' n_small=%d n_large=%d mean_sigma2_e=%.3e' % (np.sum(n_class[self.comp_group == 0]), np.sum(n_class[self.comp_group == 1]), np.mean(self.sigma2_e)) + (' q_bin_first=%.3e q_bin_last=%.3e' % (self.bin_q[0], self.bin_q[-1]) if self.n_variant_bins is not None else '') + self.row_string() + sweep_string + ' sec=%.1f' % (time.time() - t0) + guards, flush=True)

        # Posterior summaries over the post burn-in iterations
        if self.n_variant_bins is not None:
            self.bin_large_fraction /= self.n_samples
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
        log_pi, pair_row = self.prior_table()
        _grid_gibbs_sweep_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, log_pi, pair_row, self.grid_var,
                                self.comp_group, self.ig_a, self.ig_b, self.sigma2_e_fixed is None, self.sum_p_group, self.sum_beta, self.class_counts)
        return ''

    def prior_table(self):
        # Log prior weights of the components, one row per prior (n_rows x (K + 1)), and the row of every variant-gene pair (see top)
        if self.prior_mode == 'shared':
            return np.log(self.pi)[None, :], self.pair_row
        if self.prior_mode == 'variant_count_bins':
            return np.ascontiguousarray(np.column_stack([self.bin_log_1mq, self.bin_log_q[:, None] + self.log_w[None, :]])), self.pair_row
        return self.row_log_pi, self.pair_row

    def row_string(self):
        # Verbose output of the variant categories (inclusion probability 1 - pi_a0 of each) or of the non-empty gene classes (genes, inclusion)
        if self.prior_mode == 'variant_categories':
            return ' q_categories=' + ','.join('%.2e' % value for value in -np.expm1(self.row_log_pi[:, 0]))
        if self.prior_mode == 'gene_classes':
            n_genes_class = np.bincount(self.gene_class, minlength=self.n_gene_classes)
            used = np.flatnonzero(n_genes_class > 0)
            used = used[np.argsort(self.row_log_pi[used, 0])]
            return ' gene_classes=' + ','.join('%d:%.2e' % (n_genes_class[l], -np.expm1(self.row_log_pi[l, 0])) for l in used) + ' class_moves=%d' % self.gene_class_moves
        return ''

    def initialize(self, G, var_mean, var_sd, gene_var_ptr, gene_var_idx, Y, pair_category=None):
        if self.n_iters - self.burn_in > 32767 * self.thin:
            raise ValueError('too many retained samples for int16 sample indices')
        self.n_genes = Y.shape[0]
        self.n_pairs = len(gene_var_idx)
        self.beta = np.zeros(self.n_pairs)
        self.z = np.zeros(self.n_pairs, dtype=np.int8)
        self.resid = Y.copy()
        self.sigma2_e = np.var(Y, axis=1) if self.sigma2_e_fixed is None else self.sigma2_e_fixed.copy()
        if self.hierarchical_sigma2_e and self.ig_a < 1.0:
            # Start of the learned prior: a = 10 and mean b / (a - 1) = the mean variance of the expression
            self.ig_a = 10.0
            self.ig_b = 9.0 * np.mean(np.var(Y, axis=1))
        self.sum_p_group = np.zeros((self.n_pairs, 2))
        self.sum_beta = np.zeros(self.n_pairs)
        self.class_counts = np.zeros((self.n_genes, self.n_comp))
        self.gene_of_pair = np.repeat(np.arange(self.n_genes, dtype=np.int64), np.diff(gene_var_ptr))
        self.ncp_step = np.ones((self.n_comp, self.n_comp))
        self.ncp_n_accepted = 0
        self.ncp_n_proposed = 0
        self.pair_row = np.zeros(self.n_pairs, dtype=np.int64)
        self.n_rows = 1
        if (pair_category is not None) != (self.prior_mode == 'variant_categories'):
            raise ValueError('pair_category is given exactly when variant_category_labels is')
        if self.prior_mode == 'variant_categories':
            self.n_rows = len(self.variant_category_labels)
            self.pair_row = np.asarray(pair_category, dtype=np.int64)
            if len(self.pair_row) != self.n_pairs or np.min(self.pair_row) < 0 or np.max(self.pair_row) >= self.n_rows:
                raise ValueError('pair_category must give every variant-gene pair a category 0 to ' + str(self.n_rows - 1))
            self.row_n_pairs = np.bincount(self.pair_row, minlength=self.n_rows)
            if np.any(self.row_n_pairs == 0):
                raise ValueError('a variant category has no variant-gene pairs: ' + ', '.join(self.variant_category_labels[a] for a in np.flatnonzero(self.row_n_pairs == 0)))
            self.row_log_pi = np.tile(np.log(self.pi), (self.n_rows, 1))
            print(str(self.n_rows) + ' variant categories: ' + ', '.join(label + ' (' + str(n) + ' pairs)' for label, n in zip(self.variant_category_labels, self.row_n_pairs)), flush=True)
        self.ncp_class_G = (G, var_mean, var_sd, gene_var_ptr, gene_var_idx) if self.prior_mode == 'gene_classes' and self.gene_class_moves_on else None
        self.gene_class_moves = 0
        if self.prior_mode == 'gene_classes':
            # Initial inclusion probabilities spread geometrically over 1e-3 to 1e-1, the rest of each class's weight shared equally by the K
            # components; every gene in the class closest to 1 - pi0_init
            self.n_rows = self.n_gene_classes
            q_init = np.geomspace(1e-3, 1e-1, self.n_gene_classes) if self.n_gene_classes > 1 else np.asarray([1.0 - self.pi0_init])
            K = self.n_comp - 1
            self.row_log_pi = np.log(np.column_stack([1.0 - q_init] + [q_init / K] * K))
            self.gene_class = np.full(self.n_genes, np.argmin(np.abs(np.log(q_init) - np.log(1.0 - self.pi0_init))), dtype=np.int8)
            self.gene_class_weight = np.full(self.n_gene_classes, 1.0 / self.n_gene_classes)
            self.pair_row = self.gene_class[self.gene_of_pair].astype(np.int64)
            print(str(self.n_gene_classes) + ' latent gene classes, initial inclusion probabilities ' + ', '.join('%.2e' % value for value in q_init), flush=True)
        if self.n_variant_bins is not None:
            # Equal-count bins of the genes' numbers of cis variants (genes with the same number share a bin, so bin sizes can differ slightly)
            gene_n_vars = np.diff(gene_var_ptr)
            self.bin_edges = np.quantile(gene_n_vars, np.linspace(0.0, 1.0, self.n_variant_bins + 1))
            self.gene_bin = np.searchsorted(self.bin_edges[1:-1], gene_n_vars, side='right')
            self.bin_n_genes = np.bincount(self.gene_bin, minlength=self.n_variant_bins)
            self.bin_n_pairs = np.bincount(self.gene_bin, weights=gene_n_vars, minlength=self.n_variant_bins)
            if np.any(self.bin_n_genes == 0):
                raise ValueError('a variant-count bin has no genes (ties in the number of cis variants); use fewer bins')
            self.bin_q = np.full(self.n_variant_bins, 1.0 - self.pi[0])
            self.bin_log_q = np.log(self.bin_q)
            self.bin_log_1mq = np.log1p(-self.bin_q)
            self.log_w = np.log(self.pi[1:] / (1.0 - self.pi[0]))
            self.n_rows = self.n_variant_bins
            self.pair_row = self.gene_bin[self.gene_of_pair].astype(np.int64)
            print(str(self.n_variant_bins) + ' variant-count bins, edges ' + ', '.join('%d' % value for value in self.bin_edges) + '; genes per bin ' + str(np.min(self.bin_n_genes)) + ' to ' + str(np.max(self.bin_n_genes)), flush=True)
        print(str(self.n_genes) + ' genes, ' + str(self.n_pairs) + ' variant-gene pairs, ' + str(G.shape[0]) + ' variants, ' + str(Y.shape[1]) + ' training individuals', flush=True)
        print('grid prior: ' + str(self.n_comp - 1) + ' components, variances ' + ', '.join('%.3g' % value for value in self.grid_var[1:]) + '; small group: variance < ' + str(self.small_large_cutoff), flush=True)

    def sample_sigma2_e_prior(self):
        # (a, b) of the residual variances' Inverse-Gamma prior given the current residual variances (see top)
        n_genes = len(self.sigma2_e)
        sum_inv = np.sum(1.0 / self.sigma2_e)
        sum_log = np.sum(np.log(self.sigma2_e))
        self.ig_b = np.random.gamma(n_genes * self.ig_a + 1.0, 1.0 / sum_inv)

        def log_target(a):
            # log likelihood of the residual variances under IG(a, b); the 1 / a prior cancels the Jacobian of the log-scale proposal
            return n_genes * (a * np.log(self.ig_b) - gammaln(a)) - (a + 1.0) * sum_log

        for step in range(5):
            proposal = self.ig_a * np.exp(0.1 * np.random.normal())
            if np.log(np.random.random()) < log_target(proposal) - log_target(self.ig_a):
                self.ig_a = proposal

    def ncp_weight_moves(self, G, var_mean, var_sd, gene_var_idx, adapt):
        # The non-centred moves of the weights (see top); updates pi, z, beta, resid and class_counts in place. Returns a string for the
        # verbose output
        sd = np.sqrt(self.grid_var)
        n_accepted = 0
        for move in range(self.ncp_moves):
            a, b = np.random.choice(self.n_comp, size=2, replace=False)
            delta = np.random.normal() * self.ncp_target_flips / self.n_pairs * self.ncp_step[a, b]
            if not (-self.pi[a] < delta < self.pi[b]):
                continue
            self.ncp_n_proposed += 1
            if delta > 0:
                loser, gainer, prob = b, a, delta / self.pi[b]
            else:
                loser, gainer, prob = a, b, -delta / self.pi[a]
            candidates = np.flatnonzero(self.z == loser)
            flips = candidates[np.random.random(len(candidates)) < prob]
            if len(flips) == 0:
                continue
            old = self.beta[flips]
            if loser == 0:
                new = sd[gainer] * np.random.normal(size=len(flips))
            elif gainer == 0:
                new = np.zeros(len(flips))
            else:
                new = old * sd[gainer] / sd[loser]
            genes = self.gene_of_pair[flips]
            order = np.argsort(genes, kind='stable')
            flips, dbeta, genes = flips[order], (new - old)[order], genes[order]
            flip_genes, starts = np.unique(genes, return_index=True)
            flip_gene_ptr = np.append(starts, len(flips)).astype(np.int64)
            dX = np.empty((len(flip_genes), self.resid.shape[1]))
            delta_loglik = _ncp_delta_loglik_numba(G, var_mean, var_sd, gene_var_idx, self.resid, self.sigma2_e, flip_gene_ptr, flips, dbeta, flip_genes, dX)
            new_pi = self.pi.copy()
            new_pi[a] += delta
            new_pi[b] -= delta
            log_ratio = np.sum(delta_loglik) + (self.dirichlet_alpha - 1.0) * (np.log(new_pi[a]) + np.log(new_pi[b]) - np.log(self.pi[a]) - np.log(self.pi[b]))
            if np.log(np.random.random()) < log_ratio:
                self.resid[flip_genes] -= dX
                self.beta[flips] += dbeta
                self.z[flips] = gainer
                np.add.at(self.class_counts, (genes, np.full(len(genes), loser)), -1.0)
                np.add.at(self.class_counts, (genes, np.full(len(genes), gainer)), 1.0)
                self.pi = new_pi
                n_accepted += 1
                if adapt:
                    self.ncp_step[a, b] *= 1.1
                    self.ncp_step[b, a] *= 1.1
            elif adapt:
                self.ncp_step[a, b] *= 0.95
                self.ncp_step[b, a] *= 0.95
        self.ncp_n_accepted += n_accepted
        return ' ncp_accepted=%d/%d' % (n_accepted, self.ncp_moves)

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
        # With variant-count bins: q_b | z ~ Beta(alpha + k_b, alpha + M_b - k_b) and w | z ~ Dirichlet(alpha + n_1, ..., alpha + n_K), the
        # latter with the same floor
        # With variant categories: pi_a | z ~ Dirichlet(alpha + n_a0, ..., alpha + n_aK). With gene classes: c_g | z, then pi_l and rho (see top)
        guards = ''
        floor = np.finfo(float).tiny
        if self.fixed_weights is not None:
            return guards
        if self.prior_mode == 'variant_categories':
            counts = np.bincount(self.pair_row * self.n_comp + self.z.astype(np.int64), minlength=self.n_rows * self.n_comp).reshape(self.n_rows, self.n_comp)
            for a in range(self.n_rows):
                self.row_log_pi[a], floored = floored_log_dirichlet(self.dirichlet_alpha + counts[a])
                if floored:
                    guards = ' [pi floored]'
            self.pi = self.row_n_pairs @ np.exp(self.row_log_pi) / self.n_pairs
            return guards
        if self.prior_mode == 'gene_classes':
            # Classes of the genes (Gumbel-max draw from log rho_l + sum_c n_gc log pi_lc), then a non-centred move of every gene's class (see
            # _gene_class_ncp_numba), then the weights of every class and of the classes
            log_post = np.log(self.gene_class_weight)[None, :] + self.class_counts @ self.row_log_pi.T
            self.gene_class = np.argmax(log_post - np.log(-np.log(np.random.random(log_post.shape))), axis=1).astype(np.int8)
            if self.ncp_class_G is not None:
                accepted = np.zeros(self.n_genes, dtype=np.int64)
                G, var_mean, var_sd, gene_var_ptr, gene_var_idx = self.ncp_class_G
                _gene_class_ncp_numba(G, var_mean, var_sd, gene_var_ptr, gene_var_idx, self.resid, self.beta, self.z, self.sigma2_e, self.row_log_pi, self.gene_class,
                                      np.log(self.gene_class_weight), np.sqrt(self.grid_var), self.class_counts, accepted)
                self.gene_class_moves = int(np.sum(accepted))
            membership = np.zeros((self.n_genes, self.n_gene_classes))
            membership[np.arange(self.n_genes), self.gene_class] = 1.0
            counts = membership.T @ self.class_counts
            for l in range(self.n_gene_classes):
                self.row_log_pi[l], floored = floored_log_dirichlet(self.dirichlet_alpha + counts[l])
                if floored:
                    guards = ' [pi floored]'
            self.gene_class_weight = np.exp(sample_log_dirichlet(self.gene_class_alpha + np.sum(membership, axis=0)))
            self.pair_row = self.gene_class[self.gene_of_pair].astype(np.int64)
            self.pi = np.sum(counts, axis=1) @ np.exp(self.row_log_pi) / self.n_pairs
            return guards
        if self.n_variant_bins is not None:
            gene_n_nonzero = np.sum(self.class_counts[:, 1:], axis=1)
            bin_n_nonzero = np.bincount(self.gene_bin, weights=gene_n_nonzero, minlength=self.n_variant_bins)
            for b in range(self.n_variant_bins):
                self.bin_log_q[b], self.bin_log_1mq[b] = sample_log_dirichlet(self.dirichlet_alpha + np.asarray([bin_n_nonzero[b], self.bin_n_pairs[b] - bin_n_nonzero[b]]))
            self.bin_q = np.exp(self.bin_log_q)
            self.log_w = sample_log_dirichlet(self.dirichlet_alpha + np.sum(self.class_counts[:, 1:], axis=0))
            if np.any(self.log_w < np.log(floor)):
                w = np.maximum(np.exp(self.log_w), floor)
                self.log_w = np.log(w / np.sum(w))
                guards += ' [w floored]'
            pi0 = np.sum(self.bin_n_pairs * np.exp(self.bin_log_1mq)) / np.sum(self.bin_n_pairs)
            self.pi = np.concatenate([[pi0], (1.0 - pi0) * np.exp(self.log_w)])
            return guards
        n_class = np.sum(self.class_counts, axis=0)
        pi = np.exp(sample_log_dirichlet(self.dirichlet_alpha + n_class))
        if np.any(pi < floor):
            pi = np.maximum(pi, floor)
            pi = pi / np.sum(pi)
            guards += ' [pi floored]'
        self.pi = pi
        return guards


def floored_log_dirichlet(alphas):
    # log of a Dirichlet(alphas) draw with every weight floored at the smallest positive double (as sample_hyperparameters), and whether the
    # floor fired
    floor = np.finfo(float).tiny
    log_w = sample_log_dirichlet(alphas)
    if np.all(log_w >= np.log(floor)):
        return log_w, False
    w = np.maximum(np.exp(log_w), floor)
    return np.log(w / np.sum(w)), True
