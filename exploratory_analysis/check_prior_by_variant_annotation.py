import sys
import os
import argparse
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posterior_beta_samples import load_posterior_beta_samples

# Does the shared effect prior of one fit hold for every kind of variant? (README.md, section 5h.) Every variant-gene pair gets the same prior
# (or, for fits with --n_variant_bins, the prior of its gene's bin). If the effects were drawn from it, then in any category of pairs defined
# without the expression data, the posterior rate of |beta| > t averaged over the category equals the prior rate (the identity behind
# simulation-based calibration, as in check_prior_self_consistency.py), and so does the mean beta^2 (per-variant cis-h2). Categories, all from
# genotypes and positions only:
#   maf: training-fold minor allele frequency, fixed bins (--maf_edges)
#   tss_distance: |variant position - gene TSS|, fixed bins (--tss_distance_edges)
#   ld_score: in-window LD score of the pair, sum over the gene's cis variants of the bias-corrected r^2 in the training individuals (with
#             --gibbs_input_summary_file, --fold_file, --held_out_fold: the inputs of the fit), equal-count bins (--n_ld_bins)
#   maf_x_ld_score: the two crossed (MAF and LD score are correlated)
# Enrichment = observed / prior; it is 1 under the prior. Its overall level over all pairs need not be exactly 1 for a finite chain (the prior
# mean beta^2 is dominated by large-variance components with small, noisy weights), so the test is of the relative enrichment, enrichment over
# that of all pairs: a prior stratified by the annotation is called for when categories differ from each other. Genes with little information
# have posterior close to the prior, which pulls both toward 1 (conservative). Standard errors are over genes (clustered by gene, delta method).
# Output (<output_stem>_*):
#   enrichment.txt: annotation (all: every pair), category, n_pairs, statistic (exceed_<t> for |beta| > t, beta2 for the mean beta^2),
#       observed, prior, enrichment and its se, relative_enrichment and its se, z of relative_enrichment - 1
#   gene_composition.txt: per gene, the number of cis variants, the fraction with MAF < 0.05 (fraction_low_maf), the mean MAF and (with genotypes) the mean LD
#       score: covariates for check_expression_fsr_excess.py --gene_composition_file


def load_in_fold_file(fold_file):
    # genotype index (row of the plink .fam file) -> (individual id, fold)
    genotype_index_to_fold = {}
    f = open(fold_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        genotype_index_to_fold[int(data[0])] = (data[1], int(data[2]))
    f.close()
    return genotype_index_to_fold


def load_variant_bins(gibbs_results_stem, n_vars):
    # As in check_prior_self_consistency.py: q_b per retained sample (n_samples x B) and the bin of every gene; None for fits without bins
    bin_trace_file = gibbs_results_stem + '_variant_bin_trace.txt'
    if not os.path.exists(bin_trace_file):
        return None, None
    trace = np.loadtxt(bin_trace_file, skiprows=1, ndmin=2)
    bin_q = trace[trace[:, 1] == 1, 2:]
    n_bins = bin_q.shape[1]
    bin_edges = np.quantile(n_vars, np.linspace(0.0, 1.0, n_bins + 1))
    return bin_q, np.searchsorted(bin_edges[1:-1], n_vars, side='right')


def load_prior_samples(gibbs_results_stem, n_samples):
    # As in check_prior_self_consistency.py: per retained sample, the weights (n_samples x K) and variances (n_samples x K) of the non-zero
    # components of the prior
    grid_trace_file = gibbs_results_stem + '_grid_weight_trace.txt'
    if os.path.exists(grid_trace_file):
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
        trace = np.loadtxt(gibbs_results_stem + '_hyperparameter_trace.txt', skiprows=1, ndmin=2)
        retained = trace[trace[:, 1] == 1]  # columns: iteration, retained, pi0, pi1, pi2, sigma2_1, sigma2_2
        weights = retained[:, 3:5]
        variances = retained[:, 5:7]
    if weights.shape[0] != n_samples:
        raise ValueError(str(weights.shape[0]) + ' retained iterations in the trace, ' + str(n_samples) + ' retained samples')
    return weights, variances


def in_window_ld_scores(samples, gibbs_input_summary_file, genotype_index_to_fold, held_out_fold):
    # LD score of every pair within its gene's cis variants: sum_k r^2_jk - (1 - r^2_jk) / (n - 2), r from the training-fold dosages
    # standardized as in the fit (pair_train_mean, pair_train_sd)
    ptr = samples['gene_pair_ptr']
    pair_ids = samples['pair_variant_ids'].astype(str)
    gene_chroms = samples['gene_chroms'].astype(int)
    train_ids = samples['train_individual_ids'].astype(str)
    ld = np.full(len(pair_ids), np.nan)
    f = open(gibbs_input_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        chrom_num = int(data[0].replace('chr', ''))
        genes = np.where(gene_chroms == chrom_num)[0]
        if len(genes) == 0:
            continue
        gibbs_input = np.load(data[1])
        folds = np.asarray([genotype_index_to_fold[genotype_index][1] for genotype_index in gibbs_input['genotype_indices']])
        train = folds != held_out_fold
        chrom_train_ids = np.asarray([genotype_index_to_fold[genotype_index][0] for genotype_index in gibbs_input['genotype_indices'][train]])
        if np.array_equal(chrom_train_ids, train_ids) == False:
            raise ValueError('training individuals of chr' + str(chrom_num) + ' differ from those of the fit: check --held_out_fold')
        genotype_train = gibbs_input['genotype'][:, train]
        row_of = {variant_id: row for row, variant_id in enumerate(gibbs_input['variant_ids'].astype(str))}
        n = genotype_train.shape[1]
        for g in genes:
            lo, hi = ptr[g], ptr[g + 1]
            rows = np.asarray([row_of[variant_id] for variant_id in pair_ids[lo:hi]])
            X = (genotype_train[rows].astype(np.float64) - samples['pair_train_mean'][lo:hi, None]) / samples['pair_train_sd'][lo:hi, None]
            r2 = np.square(X @ X.T / n)
            ld[lo:hi] = np.sum(r2 - (1.0 - r2) / (n - 2.0), axis=1)
        gibbs_input.close()
        print('chr' + str(chrom_num) + ': LD scores of ' + str(len(genes)) + ' genes', flush=True)
    f.close()
    return ld


def bin_labels(values, edges, unit):
    # Category of each value for bins [edges[i], edges[i + 1]) (the last bin includes values above it), and the label of each bin
    category = np.clip(np.searchsorted(edges, values, side='right') - 1, 0, len(edges) - 2)
    labels = [unit + ' ' + ('%g' % edges[i]) + ' to ' + ('%g' % edges[i + 1]) for i in range(len(edges) - 1)]
    return category, labels


def enrichment_rows(category, n_categories, observed, prior, gene_of_pair, n_genes):
    # Per category: n_pairs, observed and prior means over its pairs, their ratio and its standard error over genes, and the ratio relative to
    # that of all pairs with its standard error (influence of each gene on log ratio_c - log ratio_all)
    index = gene_of_pair.astype(np.int64) * n_categories + category
    O = np.bincount(index, weights=observed, minlength=n_genes * n_categories).reshape(n_genes, n_categories)
    E = np.bincount(index, weights=prior, minlength=n_genes * n_categories).reshape(n_genes, n_categories)
    n_pairs = np.bincount(category, minlength=n_categories)
    O_all, E_all = np.sum(O, axis=1), np.sum(E, axis=1)
    ratio_all = np.sum(O_all) / np.sum(E_all)
    influence_all = (O_all - ratio_all * E_all) / np.sum(O_all)
    out = []
    for c in range(n_categories):
        if n_pairs[c] == 0:
            continue
        ratio = np.sum(O[:, c]) / np.sum(E[:, c])
        se = np.sqrt(np.sum(np.square(O[:, c] - ratio * E[:, c]))) / np.sum(E[:, c])
        relative = ratio / ratio_all
        relative_se = relative * np.sqrt(np.sum(np.square((O[:, c] - ratio * E[:, c]) / np.sum(O[:, c]) - influence_all)))
        out.append((c, n_pairs[c], np.sum(O[:, c]) / n_pairs[c], np.sum(E[:, c]) / n_pairs[c], ratio, se, relative, relative_se))
    return out


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--gibbs_results_stem', type=str)  # --output_stem of run_eqtl_mixture_prior_gibbs.py or run_eqtl_grid_prior_gibbs.py
parser.add_argument('--output_stem', type=str)
parser.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1])  # |beta| thresholds (as check_prior_self_consistency.py)
parser.add_argument('--maf_edges', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
parser.add_argument('--tss_distance_edges', type=float, nargs='+', default=[0, 2000, 5000, 10000, 20000, 50000, 100000])
parser.add_argument('--n_ld_bins', type=int, default=5)
# Optional, for the LD scores: the inputs of the fit
parser.add_argument('--gibbs_input_summary_file', type=str, default=None)
parser.add_argument('--fold_file', type=str, default=None)
parser.add_argument('--held_out_fold', type=int, default=None)
args = parser.parse_args()

samples = load_posterior_beta_samples(args.gibbs_results_stem + '_posterior_beta_samples.npz')
n_samples = samples['n_samples']
ptr = samples['gene_pair_ptr']
n_vars = np.diff(ptr).astype(np.int64)
n_genes = len(n_vars)
gene_of_pair = np.repeat(np.arange(n_genes, dtype=np.int64), n_vars)
print(str(n_genes) + ' genes, ' + str(len(gene_of_pair)) + ' variant-gene pairs, ' + str(n_samples) + ' retained samples', flush=True)

# Prior rates of every gene: mean over retained samples of q_t = sum_c w_c 2 Phi(-t / sigma_c) and of E[beta^2] = sum_c w_c sigma2_c
weights, variances = load_prior_samples(args.gibbs_results_stem, n_samples)
bin_q, gene_bin = load_variant_bins(args.gibbs_results_stem, n_vars)
thresholds = np.asarray(args.thresholds)
sample_exceed = np.sum(weights[None, :, :] * 2.0 * stats.norm.sf(thresholds[:, None, None] / np.sqrt(variances[None, :, :])), axis=2)  # n_thresholds x n_samples
sample_beta2 = np.sum(weights * variances, axis=1)
if bin_q is None:
    gene_prior = np.tile(np.concatenate([np.mean(sample_exceed, axis=1), [np.mean(sample_beta2)]])[:, None], (1, n_genes))
else:
    # Each gene's prior is q_b(g) w, with w the weights of the non-zero components normalized to sum to 1
    total = np.sum(weights, axis=1)
    gene_prior = np.concatenate([np.mean((sample_exceed / total[None, :])[:, :, None] * bin_q[None, :, gene_bin], axis=1),
                                 np.mean((sample_beta2 / total)[:, None] * bin_q[:, gene_bin], axis=0)[None, :]], axis=0)
    print(str(bin_q.shape[1]) + ' variant-count bins', flush=True)
statistic_names = ['exceed_' + ('%g' % t) for t in thresholds] + ['beta2']

# Posterior rates of every pair: fraction of retained samples with |beta| > t, and mean beta^2
nz_pair = samples['nz_pair'].astype(np.int64)
nz_beta = samples['nz_beta'].astype(np.float64)
pair_observed = [np.bincount(nz_pair[np.abs(nz_beta) > t], minlength=len(gene_of_pair)) / n_samples for t in thresholds]
pair_observed.append(np.bincount(nz_pair, weights=np.square(nz_beta), minlength=len(gene_of_pair)) / n_samples)
del nz_pair, nz_beta

# Annotations of every pair
af = samples['pair_train_mean'] / 2.0
maf = np.minimum(af, 1.0 - af)
position = np.asarray([int(variant_id.split('_')[1]) for variant_id in samples['pair_variant_ids'].astype(str)])
tss_distance = np.abs(position - samples['gene_tss'][gene_of_pair])
annotations = []
category, labels = bin_labels(maf, np.asarray(args.maf_edges), 'MAF')
annotations.append(('maf', category, labels))
maf_category, maf_labels = category, labels
category, labels = bin_labels(tss_distance, np.asarray(args.tss_distance_edges), 'TSS distance')
annotations.append(('tss_distance', category, labels))
ld = None
if args.gibbs_input_summary_file is not None:
    ld = in_window_ld_scores(samples, args.gibbs_input_summary_file, load_in_fold_file(args.fold_file), args.held_out_fold)
    ld_edges = np.quantile(ld, np.linspace(0.0, 1.0, args.n_ld_bins + 1))
    category, labels = bin_labels(ld, ld_edges, 'LD score')
    annotations.append(('ld_score', category, labels))
    # MAF x LD score, with LD score terciles
    ld_tercile, ld_tercile_labels = bin_labels(ld, np.quantile(ld, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]), 'LD score')
    annotations.append(('maf_x_ld_score', maf_category * 3 + ld_tercile, [m + ', ' + l for m in maf_labels for l in ld_tercile_labels]))

t = open(args.output_stem + '_enrichment.txt', 'w')
t.write('annotation\tcategory\tn_pairs\tstatistic\tobserved\tprior\tenrichment\tse\trelative_enrichment\trelative_se\tz\n')
annotations.insert(0, ('all', np.zeros(len(gene_of_pair), dtype=np.int64), ['all pairs']))
for name, category, labels in annotations:
    for s, statistic in enumerate(statistic_names):
        for c, n_pairs, observed, prior, ratio, se, relative, relative_se in enrichment_rows(category, len(labels), pair_observed[s], gene_prior[s][gene_of_pair], gene_of_pair, n_genes):
            z = (relative - 1.0) / relative_se if relative_se > 0.0 else np.nan
            line = name + '\t' + labels[c] + '\t' + str(n_pairs) + '\t' + statistic + '\t' + str(observed) + '\t' + str(prior) + '\t' + ('%.4f' % ratio) + '\t' + ('%.4f' % se) + '\t' + ('%.4f' % relative) + '\t' + ('%.4f' % relative_se) + '\t' + ('%.1f' % z) + '\n'
            t.write(line)
            print(line.rstrip('\n'), flush=True)
t.close()

t = open(args.output_stem + '_gene_composition.txt', 'w')
t.write('gene_id\tn_cis_variants\tfraction_low_maf\tmean_maf' + ('\tmean_ld_score' if ld is not None else '') + '\n')
fraction_rare = np.bincount(gene_of_pair, weights=(maf < 0.05).astype(float), minlength=n_genes) / n_vars
mean_maf = np.bincount(gene_of_pair, weights=maf, minlength=n_genes) / n_vars
mean_ld = np.bincount(gene_of_pair, weights=ld, minlength=n_genes) / n_vars if ld is not None else None
for g, gene_id in enumerate(samples['gene_ids'].astype(str)):
    t.write(gene_id + '\t' + str(n_vars[g]) + '\t' + str(fraction_rare[g]) + '\t' + str(mean_maf[g]) + ('\t' + str(mean_ld[g]) if ld is not None else '') + '\n')
t.close()
print('written to ' + args.output_stem + '_*', flush=True)
