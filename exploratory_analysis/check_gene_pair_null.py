import argparse
import numpy as np
from scipy import stats

# Summary of the gene-pair null of evaluate_gene_pair_null.py (README.md, section 5i), joined with the per-gene evaluation:
#   null dispersion D = mean over partners of (n_test - 1) r_AB^2, 1 if the held-out individuals are exchangeable: over all genes, by decile of
#       n_cis_variants, by n_cis_variants tercile x training LRT p-value, and regressed on log10 n_cis_variants and the LRT strata (clustered by
#       gene). A D above 1 that grows with n_cis_variants and with weak training evidence would explain the expression FSR excess
#       (check_expression_fsr_excess.py) as residual structure shared by expression and genotypes
#   predicted excess: does that D account for the size of the excess? Over the rows with expression_FSR in --fsr_range, the FSR of a normal
#       approximation to the posterior predictive distribution of r, Phi(-expected_r / expected_r_sd), recomputed with its noise variance
#       (about (1 - expected_r^2)^2 / (n_test - 1)) inflated by D - 1, D the mean dispersion of the row's n_cis_variants decile x LRT stratum
#       (a gene's own D is too noisy); predicted excess = inflated FSR - FSR, against the observed excess 1[observed_r < 0] - expression_FSR
# Standard errors are clustered by block of --block_mb Mb of the TSS: genes with overlapping windows have similar predictions, so their
# dispersions are correlated (clustering by gene understated the standard errors in simulations without structure)
# Input: the gene_pair_null.txt of every fold and the <organized stem>_per_gene_fold.txt of the same fit


def cluster_sums(values, cluster_index, n_clusters):
    # Same as check_expression_fsr_excess.py
    out = np.zeros((n_clusters,) + values.shape[1:])
    np.add.at(out, cluster_index, values)
    return out


def mean_with_se(e, cluster_index):
    # Same as check_expression_fsr_excess.py: mean of e and its standard error clustered by gene
    _, local_index = np.unique(cluster_index, return_inverse=True)
    sums = cluster_sums(e - np.mean(e), local_index, np.max(local_index) + 1)
    return np.mean(e), np.sqrt(np.sum(np.square(sums))) / len(e)


def clustered_ols(X, y, cluster_index):
    # Same as check_expression_fsr_excess.py: least-squares coefficients and their standard errors clustered by gene
    _, local_index = np.unique(cluster_index, return_inverse=True)
    bread = np.linalg.inv(X.T @ X)
    coef = bread @ X.T @ y
    scores = cluster_sums(X * (y - X @ coef)[:, None], local_index, np.max(local_index) + 1)
    cov = bread @ (scores.T @ scores) @ bread
    return coef, np.sqrt(np.diag(cov))


def print_row(label, n, mean, se, reference):
    # z of mean - reference
    print(label + '\t' + str(n) + '\t' + '%.4f' % mean + '\t' + '%.4f' % se + '\t' + '%.1f' % ((mean - reference) / se if se > 0 else np.nan))


parser = argparse.ArgumentParser()
parser.add_argument('--gene_pair_null_files', type=str, nargs='+')  # output of evaluate_gene_pair_null.py, one per fold
parser.add_argument('--per_gene_fold_file', type=str)  # <organized stem>_per_gene_fold.txt of the same fit
parser.add_argument('--fsr_range', type=float, nargs=2, default=[0.1, 0.4])  # rows with expression_FSR in (lower, upper]
parser.add_argument('--block_mb', type=float, default=5.0)  # TSS block (Mb) of the clustered standard errors
args = parser.parse_args()

null_dispersion = {}
gene_block = {}
for null_file in args.gene_pair_null_files:
    table = np.genfromtxt(null_file, delimiter='\t', names=True, dtype=None, encoding=None)
    for gene_id, chrom, tss, fold, value in zip(np.asarray(table['gene_id']).astype(str), np.asarray(table['chrom']), np.asarray(table['tss']), np.asarray(table['fold']), np.asarray(table['null_dispersion'], dtype=float)):
        null_dispersion[gene_id + ':' + str(fold)] = value
        gene_block[gene_id] = str(chrom) + ':' + str(int(tss // (args.block_mb * 1e6)))
table = np.genfromtxt(args.per_gene_fold_file, delimiter='\t', names=True, dtype=None, encoding=None)
gene_ids = np.asarray(table['gene_id']).astype(str)
D = np.asarray([null_dispersion.get(gene_id + ':' + str(fold), np.nan) for gene_id, fold in zip(gene_ids, np.asarray(table['fold']))])
fsr = np.asarray(table['expression_FSR'], dtype=float)
observed_r = np.asarray(table['observed_r'], dtype=float)
expected_r = np.asarray(table['expected_r'], dtype=float)
expected_r_sd = np.asarray(table['expected_r_sd'], dtype=float)
n_vars = np.asarray(table['n_cis_variants'], dtype=float)
n_test = np.asarray(table['n_test'], dtype=float)
pvalue = np.asarray(table['cis_snp_h2_pvalue_train'], dtype=float)
# Cluster of each row: its gene's TSS block (rows of genes without a null dispersion are not used)
_, gene_index = np.unique(np.asarray([gene_block.get(gene_id, 'none') for gene_id in gene_ids]), return_inverse=True)
print(str(np.sum(np.isfinite(D))) + ' of ' + str(len(D)) + ' gene-fold rows with a null dispersion')

valid = np.isfinite(D) & np.isfinite(pvalue)
decile_breaks = np.quantile(n_vars[valid], np.linspace(0, 1, 11))
decile = np.clip(np.searchsorted(decile_breaks, n_vars, side='right') - 1, 0, 9)
tercile_breaks = np.quantile(n_vars[valid], [1.0 / 3.0, 2.0 / 3.0])
tercile = np.searchsorted(tercile_breaks, n_vars, side='right')
tercile_labels = ['n_cis_variants <= %d' % tercile_breaks[0], 'n_cis_variants %d to %d' % (tercile_breaks[0], tercile_breaks[1]), 'n_cis_variants > %d' % tercile_breaks[1]]
p_stratum = np.searchsorted([0.05, 0.5], pvalue, side='right')
p_labels = ['LRT p < 0.05', 'LRT p 0.05 to 0.5', 'LRT p >= 0.5']

print(str(len(np.unique(gene_index[valid]))) + ' TSS blocks of ' + ('%g' % args.block_mb) + ' Mb')
print('columns: set, n, mean null dispersion, clustered se, z of dispersion - 1\n')
m, se = mean_with_se(D[valid], gene_index[valid])
print_row('all', np.sum(valid), m, se, 1.0)
print('\nDeciles of n_cis_variants')
for d in range(10):
    rows = valid & (decile == d)
    m, se = mean_with_se(D[rows], gene_index[rows])
    print_row('n_cis_variants %d to %d' % (decile_breaks[d], decile_breaks[d + 1]), np.sum(rows), m, se, 1.0)
print('\nTercile of n_cis_variants x training LRT p-value')
for t in range(3):
    for s in range(3):
        rows = valid & (tercile == t) & (p_stratum == s)
        if np.sum(rows) < 2:
            continue
        m, se = mean_with_se(D[rows], gene_index[rows])
        print_row(tercile_labels[t] + ', ' + p_labels[s], np.sum(rows), m, se, 1.0)

print('\nRegression of the null dispersion (coefficient, clustered se, z)')
log_n = np.log10(n_vars[valid])
X = np.column_stack([np.ones(np.sum(valid)), log_n - np.mean(log_n)] + [(p_stratum[valid] == s).astype(float) for s in [1, 2] if np.any(p_stratum[valid] == s)])
names = ['intercept (LRT p < 0.05, mean log10 n_cis_variants)', 'log10 n_cis_variants'] + [p_labels[s] for s in [1, 2] if np.any(p_stratum[valid] == s)]
coef, coef_se = clustered_ols(X, D[valid], gene_index[valid])
for name, c, s in zip(names, coef, coef_se):
    print(name + '\t' + '%.4f' % c + '\t' + '%.4f' % s + '\t' + '%.1f' % (c / s))

# Predicted against observed excess, with D the mean of the row's decile x LRT stratum
cell = decile * 3 + p_stratum
cell_D = np.full(30, np.nan)
for c in range(30):
    if np.sum(valid & (cell == c)) > 0:
        cell_D[c] = np.mean(D[valid & (cell == c)])
in_range = valid & np.isfinite(fsr) & np.isfinite(observed_r) & np.isfinite(expected_r_sd) & (expected_r_sd > 0.0) & (fsr > args.fsr_range[0]) & (fsr <= args.fsr_range[1])
noise_variance = np.square(1.0 - np.square(expected_r)) / (n_test - 1.0)
inflated_sd = np.sqrt(np.square(expected_r_sd) + np.maximum(cell_D[cell] - 1.0, -1.0) * noise_variance)
predicted = stats.norm.cdf(-expected_r / inflated_sd) - stats.norm.cdf(-expected_r / expected_r_sd)
e = (observed_r < 0.0).astype(float) - fsr
print('\nObserved against predicted expression FSR excess, expression_FSR in (' + str(args.fsr_range[0]) + ', ' + str(args.fsr_range[1]) + ']')
print('columns: set, n, observed excess, clustered se, z of observed - predicted, predicted excess')
for label, rows in [('all', in_range)] + [('n_cis_variants %d to %d' % (decile_breaks[d], decile_breaks[d + 1]), in_range & (decile == d)) for d in range(10)]:
    m, se = mean_with_se(e[rows], gene_index[rows])
    p = np.mean(predicted[rows])
    print(label + '\t' + str(np.sum(rows)) + '\t' + '%.4f' % m + '\t' + '%.4f' % se + '\t' + '%.1f' % ((m - p) / se) + '\t' + '%.4f' % p)
