import sys
import os
import argparse
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posterior_beta_samples import load_posterior_beta_samples

# Self-consistency check of the shared effect prior of one fit (run_eqtl_mixture_prior_gibbs.py or run_eqtl_grid_prior_gibbs.py, any sampler;
# README.md, section 5f). If the effects of every gene were drawn from the fitted prior, one posterior sample per gene, pooled over genes, would
# be distributed as draws from the prior (the identity behind simulation-based calibration). Two per-gene statistics of one posterior sample are
# compared with the same statistics simulated from the prior, for the gene's number of cis variants and the hyper-parameters of that sample:
#   K_g(t): number of variants with |beta| > t (t in --thresholds). Under the prior K_g ~ Binomial(M_g, q), q = sum_c pi_c 2 Phi(-t / sigma_c)
#   S_g:    sum of squared standardized effects (an LD-free cis-h2)
# A shared prior that is too homogeneous across genes shows up as over-dispersion of K_g (more genes with no and with several large effects than
# the binomial allows) and a heavier right tail of S_g. Genes with little information have posterior close to the prior and match it by
# construction, so a mismatch is a muted (conservative) signal. Use every gene: restricting to genes chosen by their data (e.g. heritable genes)
# breaks the identity; excluding genes by position (--exclude_regions, e.g. the MHC) does not.
# Every replicate draws a new posterior sample per gene and a new prior simulation. The observed statistic is the mean over replicates; the prior
# interval is over replicates and is the spread of the statistic for this many genes under the prior (the reference for a single observed draw),
# z = (observed - prior mean) / prior sd.
# Output (<output_stem>_*):
#   summary.txt: per statistic, observed, prior mean and 95% interval, z. Dispersion statistics of K_g:
#       pearson_dispersion  sum (K - M q)^2 / sum M q (1 - q); 1 under the prior
#       overdispersion_rho  moment estimate of the beta-binomial intra-gene correlation; 0 under the prior. Converted to a Beta(a, b) prior on a
#                           gene-specific q with a + b = (1 - rho) / rho (implied_beta_a_plus_b, observed rho > 0 only; a lower bound on the
#                           heterogeneity, so an upper bound on a + b)
#       slope_on_expected   least-squares slope of K_g on its expected count M_g q over genes; 1 under the prior. Below 1: counts rise more
#                           slowly with the number of cis variants than the prior assumes
#       dispersion_around_slope  pearson_dispersion around that fitted line instead of M_g q: the over-dispersion not explained by the slope
#   count_histogram.txt: fraction of genes with K_g = 0, 1, ..., --max_count (last bin: >= --max_count)
#   sum_sq_histogram.txt: fraction of genes per bin of log10 S_g (first bin: S_g < 1e-4, including 0)


def load_variant_bins(gibbs_results_stem, n_vars):
    # Fits with --n_variant_bins: q_b per retained sample (n_samples x B) and the bin of every gene, with the same equal-count bins as
    # eqtl_grid_prior_gibbs.py; None for fits without
    bin_trace_file = gibbs_results_stem + '_variant_bin_trace.txt'
    if not os.path.exists(bin_trace_file):
        return None, None
    trace = np.loadtxt(bin_trace_file, skiprows=1, ndmin=2)
    bin_q = trace[trace[:, 1] == 1, 2:]
    n_bins = bin_q.shape[1]
    bin_edges = np.quantile(n_vars, np.linspace(0.0, 1.0, n_bins + 1))
    return bin_q, np.searchsorted(bin_edges[1:-1], n_vars, side='right')


def load_prior_samples(gibbs_results_stem, n_samples):
    # Per retained sample, the weights (n_samples x K) and variances (n_samples x K) of the non-zero components of the prior (for fits with
    # variant-count bins, the weights averaged over all variant-gene pairs)
    grid_trace_file = gibbs_results_stem + '_grid_weight_trace.txt'
    if os.path.exists(grid_trace_file):
        model = 'grid'
        variances = []
        f = open(gibbs_results_stem + '_grid_weight_summary.txt')
        head_count = 0
        for line in f:
            data = line.rstrip('\n').split('\t')
            if head_count == 0:
                head_count += 1
                continue
            if data[2] != 'spike':
                variances.append(float(data[1]))
        f.close()
        trace = np.loadtxt(grid_trace_file, skiprows=1, ndmin=2)
        weights = trace[trace[:, 1] == 1, 3:]  # columns: iteration, retained, pi_spike, pi_var_*
        variances = np.tile(np.asarray(variances), (weights.shape[0], 1))
    else:
        model = 'two_slab'
        trace = np.loadtxt(gibbs_results_stem + '_hyperparameter_trace.txt', skiprows=1, ndmin=2)
        retained = trace[trace[:, 1] == 1]  # columns: iteration, retained, pi0, pi1, pi2, sigma2_1, sigma2_2
        weights = retained[:, 3:5]
        variances = retained[:, 5:7]
    if weights.shape[0] != n_samples:
        raise ValueError(str(weights.shape[0]) + ' retained iterations in the trace, ' + str(n_samples) + ' retained samples')
    return model, weights, variances


def simulate_prior_statistics(rng, n_vars, weights, variances, exceed_prob):
    # K_g for every threshold (n_thresholds x n_genes) and S_g (n_genes) from the prior; weights, variances: n_genes x K, the hyper-parameters of
    # each gene's sample; exceed_prob: n_thresholds x n_genes, q of each gene
    K = rng.binomial(n_vars[None, :], exceed_prob)
    # Component counts by sequential binomials (the spike takes the rest), then S_g = sum_c sigma2_c chi^2_{n_c}
    remaining = n_vars.copy()
    remaining_prob = np.ones(len(n_vars))
    S = np.zeros(len(n_vars))
    for c in range(weights.shape[1]):
        n_c = rng.binomial(remaining, np.clip(weights[:, c] / np.maximum(remaining_prob, 1e-300), 0.0, 1.0))
        remaining = remaining - n_c
        remaining_prob = remaining_prob - weights[:, c]
        S += rng.gamma(n_c / 2.0, 2.0 * variances[:, c])
    return K, S


def count_statistics(K, n_vars, exceed_prob, max_count):
    # Summary statistics and histogram of K (n_genes) given each gene's q
    expected = n_vars * exceed_prob
    binomial_var = n_vars * exceed_prob * (1.0 - exceed_prob)
    excess = np.sum(np.square(K - expected)) - np.sum(binomial_var)
    # Least-squares line of K on the expected count M q, and the dispersion left around it
    centered_expected = expected - np.mean(expected)
    slope = np.sum(centered_expected * (K - np.mean(K))) / np.sum(np.square(centered_expected))
    residual = K - np.mean(K) - slope * centered_expected
    out = {
        'mean': np.mean(K),
        'fraction_zero': np.mean(K == 0),
        'fraction_3_or_more': np.mean(K >= 3),
        'pearson_dispersion': np.sum(np.square(K - expected)) / np.sum(binomial_var),
        'overdispersion_rho': excess / np.sum(n_vars * (n_vars - 1.0) * exceed_prob * (1.0 - exceed_prob)),
        'slope_on_expected': slope,
        'dispersion_around_slope': np.sum(np.square(residual)) / np.sum(binomial_var),
    }
    histogram = np.bincount(np.minimum(K, max_count), minlength=max_count + 1) / len(K)
    return out, histogram


def sum_sq_statistics(S, bin_edges):
    out = {
        'median': np.median(S),
        'quantile_0.9': np.quantile(S, 0.9),
        'quantile_0.99': np.quantile(S, 0.99),
        'fraction_above_0.1': np.mean(S > 0.1),
        'fraction_above_0.3': np.mean(S > 0.3),
    }
    log_S = np.log10(np.maximum(S, 1e-300))
    histogram = np.bincount(np.searchsorted(bin_edges, log_S, side='right'), minlength=len(bin_edges) + 1) / len(S)
    return out, histogram


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--gibbs_results_stem', type=str)  # --output_stem of run_eqtl_mixture_prior_gibbs.py or run_eqtl_grid_prior_gibbs.py
parser.add_argument('--output_stem', type=str)
parser.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.2])  # |beta| thresholds of K_g (0.1: beta^2 > 0.01)
parser.add_argument('--n_reps', type=int, default=100)
parser.add_argument('--max_count', type=int, default=6)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--exclude_regions', type=str, nargs='*', default=[])  # chrom:start-end (TSS, e.g. 6:28510120-33480577 for the GRCh38 MHC)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
samples = load_posterior_beta_samples(args.gibbs_results_stem + '_posterior_beta_samples.npz')
n_samples = samples['n_samples']
n_vars = np.diff(samples['gene_pair_ptr']).astype(np.int64)
n_genes = len(n_vars)
model, weights, variances = load_prior_samples(args.gibbs_results_stem, n_samples)
bin_q, gene_bin = load_variant_bins(args.gibbs_results_stem, n_vars)
if bin_q is not None:
    # Each gene's prior is q_b(g) w, with w the weights of the non-zero components normalized to sum to 1
    weights = weights / np.sum(weights, axis=1)[:, None]
    model += ' with ' + str(bin_q.shape[1]) + ' variant-count bins'
thresholds = np.asarray(args.thresholds)
# Genes whose statistics are compared (all genes are kept in the simulation, so the arrays stay aligned)
compared = np.ones(n_genes, dtype=bool)
for region in args.exclude_regions:
    chrom, interval = region.split(':')
    start, end = [int(value) for value in interval.split('-')]
    compared &= ~((samples['gene_chroms'].astype(int) == int(chrom.replace('chr', ''))) & (samples['gene_tss'] >= start) & (samples['gene_tss'] <= end))
if len(args.exclude_regions) > 0:
    print(str(n_genes - np.sum(compared)) + ' genes in ' + ' '.join(args.exclude_regions) + ' excluded from the comparison', flush=True)
print(model + ' prior: ' + str(n_genes) + ' genes, ' + str(np.sum(n_vars)) + ' variant-gene pairs, ' + str(n_samples) + ' retained samples, ' + str(len(samples['nz_beta'])) + ' non-zero effect samples', flush=True)

# Gene of every non-zero effect sample
gene_of_pair = np.repeat(np.arange(n_genes, dtype=np.int32), n_vars)
gene_of_nz = gene_of_pair[samples['nz_pair']]
nz_sample = samples['nz_sample'].astype(np.int32)
nz_beta = samples['nz_beta'].astype(np.float64)
del gene_of_pair

bin_edges = np.arange(-4.0, 0.5001, 0.25)
count_names = ['mean', 'fraction_zero', 'fraction_3_or_more', 'pearson_dispersion', 'overdispersion_rho', 'slope_on_expected', 'dispersion_around_slope']
sum_sq_names = ['median', 'quantile_0.9', 'quantile_0.99', 'fraction_above_0.1', 'fraction_above_0.3']
observed_count = np.zeros((args.n_reps, len(thresholds), len(count_names)))
prior_count = np.zeros((args.n_reps, len(thresholds), len(count_names)))
observed_count_hist = np.zeros((args.n_reps, len(thresholds), args.max_count + 1))
prior_count_hist = np.zeros((args.n_reps, len(thresholds), args.max_count + 1))
observed_sum_sq = np.zeros((args.n_reps, len(sum_sq_names)))
prior_sum_sq = np.zeros((args.n_reps, len(sum_sq_names)))
observed_sum_sq_hist = np.zeros((args.n_reps, len(bin_edges) + 1))
prior_sum_sq_hist = np.zeros((args.n_reps, len(bin_edges) + 1))
for rep in range(args.n_reps):
    # One posterior sample per gene, and the hyper-parameters of that sample
    chosen = rng.integers(n_samples, size=n_genes)
    keep = nz_sample == chosen[gene_of_nz]
    gene_kept = gene_of_nz[keep]
    beta_kept = nz_beta[keep]
    gene_weights = weights[chosen]
    if bin_q is not None:
        gene_weights = gene_weights * bin_q[chosen, gene_bin][:, None]
    gene_variances = variances[chosen]
    exceed_prob = np.sum(gene_weights[None, :, :] * 2.0 * stats.norm.sf(thresholds[:, None, None] / np.sqrt(gene_variances[None, :, :])), axis=2)

    prior_K, prior_S = simulate_prior_statistics(rng, n_vars, gene_weights, gene_variances, exceed_prob)
    for i, threshold in enumerate(thresholds):
        K = np.bincount(gene_kept[np.abs(beta_kept) > threshold], minlength=n_genes)
        out, observed_count_hist[rep, i] = count_statistics(K[compared], n_vars[compared], exceed_prob[i, compared], args.max_count)
        observed_count[rep, i] = [out[name] for name in count_names]
        out, prior_count_hist[rep, i] = count_statistics(prior_K[i, compared], n_vars[compared], exceed_prob[i, compared], args.max_count)
        prior_count[rep, i] = [out[name] for name in count_names]
    S = np.bincount(gene_kept, weights=np.square(beta_kept), minlength=n_genes)
    out, observed_sum_sq_hist[rep] = sum_sq_statistics(S[compared], bin_edges)
    observed_sum_sq[rep] = [out[name] for name in sum_sq_names]
    out, prior_sum_sq_hist[rep] = sum_sq_statistics(prior_S[compared], bin_edges)
    prior_sum_sq[rep] = [out[name] for name in sum_sq_names]


def summary_line(statistic, threshold, observed, prior):
    # observed, prior: values over replicates
    prior_sd = np.std(prior)
    z = (np.mean(observed) - np.mean(prior)) / prior_sd if prior_sd > 0.0 else np.nan
    return statistic + '\t' + threshold + '\t' + str(np.mean(observed)) + '\t' + str(np.mean(prior)) + '\t' + str(np.quantile(prior, 0.025)) + '\t' + str(np.quantile(prior, 0.975)) + '\t' + str(z) + '\n'


t = open(args.output_stem + '_summary.txt', 'w')
t.write('statistic\tthreshold\tobserved\tprior_mean\tprior_95_lb\tprior_95_ub\tz\n')
for i, threshold in enumerate(thresholds):
    for j, name in enumerate(count_names):
        line = summary_line('K_' + name, str(threshold), observed_count[:, i, j], prior_count[:, i, j])
        t.write(line)
        print(line.rstrip('\n'), flush=True)
    rho = np.mean(observed_count[:, i, count_names.index('overdispersion_rho')])
    t.write('K_implied_beta_a_plus_b\t' + str(threshold) + '\t' + (str((1.0 - rho) / rho) if rho > 0.0 else 'NA') + '\tNA\tNA\tNA\tNA\n')
for j, name in enumerate(sum_sq_names):
    line = summary_line('S_' + name, 'NA', observed_sum_sq[:, j], prior_sum_sq[:, j])
    t.write(line)
    print(line.rstrip('\n'), flush=True)
t.close()

t = open(args.output_stem + '_count_histogram.txt', 'w')
t.write('threshold\tcount\tobserved_fraction\tprior_fraction\tprior_95_lb\tprior_95_ub\n')
for i, threshold in enumerate(thresholds):
    for count in range(args.max_count + 1):
        prior = prior_count_hist[:, i, count]
        t.write(str(threshold) + '\t' + str(count) + '\t' + str(np.mean(observed_count_hist[:, i, count])) + '\t' + str(np.mean(prior)) + '\t' + str(np.quantile(prior, 0.025)) + '\t' + str(np.quantile(prior, 0.975)) + '\n')
t.close()

t = open(args.output_stem + '_sum_sq_histogram.txt', 'w')
t.write('log10_lower\tlog10_upper\tobserved_fraction\tprior_fraction\tprior_95_lb\tprior_95_ub\n')
lower_edges = np.concatenate([[-np.inf], bin_edges])
upper_edges = np.concatenate([bin_edges, [np.inf]])
for b in range(len(bin_edges) + 1):
    prior = prior_sum_sq_hist[:, b]
    t.write(str(lower_edges[b]) + '\t' + str(upper_edges[b]) + '\t' + str(np.mean(observed_sum_sq_hist[:, b])) + '\t' + str(np.mean(prior)) + '\t' + str(np.quantile(prior, 0.025)) + '\t' + str(np.quantile(prior, 0.975)) + '\n')
t.close()
print('written to ' + args.output_stem + '_*', flush=True)
