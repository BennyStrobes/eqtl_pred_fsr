import argparse
import numpy as np

# Expression FSR excess of each gene-fold row, e = 1[observed_r < 0] - expression_FSR. If the model is right, e has mean 0 in any set of genes
# chosen on training-fold quantities (README.md, section 8, plot 3e). The stratifiers of plot 3e (number of cis variants, training LRT p-value)
# are correlated with each other and with the FSR itself; this separates them, over the rows with expression_FSR in --fsr_range:
#   deciles of n_cis_variants: mean excess (is the gradient monotone, and linear in log n_cis_variants?)
#   two-way table: mean excess by tercile of n_cis_variants x training LRT p-value stratum
#   regression: e on log10 n_cis_variants, expression_FSR (both centered) and the LRT p-value stratum (reference p < 0.05)
# Standard errors are clustered by gene (a gene's rows in different folds share training individuals).
# With --gene_composition_file (the gene_composition.txt of check_prior_by_variant_annotation.py, README.md, section 5h), the regression is
# repeated with the variant composition of each gene's cis window added (fraction of variants with MAF < 0.05, log10 mean in-window LD score;
# from one fold's training genotypes, nearly the same in every fold): if these take over the log10 n_cis_variants coefficient, the gradient
# with the number of variants is a gradient with the kind of variants, and a prior stratified by variant annotation is the natural fix.
# Input: the <organized stem>_per_gene_fold.txt of organize_cross_validation_results.py


def cluster_sums(values, cluster_index, n_clusters):
    # Sum of the rows of values (n x k, or n) within each cluster
    out = np.zeros((n_clusters,) + values.shape[1:])
    np.add.at(out, cluster_index, values)
    return out


def mean_with_se(e, cluster_index):
    # Mean of e and its standard error clustered by gene
    _, local_index = np.unique(cluster_index, return_inverse=True)
    sums = cluster_sums(e - np.mean(e), local_index, np.max(local_index) + 1)
    return np.mean(e), np.sqrt(np.sum(np.square(sums))) / len(e)


def clustered_ols(X, y, cluster_index):
    # Least-squares coefficients and their standard errors clustered by gene (sandwich estimator)
    _, local_index = np.unique(cluster_index, return_inverse=True)
    bread = np.linalg.inv(X.T @ X)
    coef = bread @ X.T @ y
    scores = cluster_sums(X * (y - X @ coef)[:, None], local_index, np.max(local_index) + 1)
    cov = bread @ (scores.T @ scores) @ bread
    return coef, np.sqrt(np.diag(cov))


def print_row(label, n, mean, se):
    print(label + '\t' + str(n) + '\t' + '%.4f' % mean + '\t' + '%.4f' % se + '\t' + '%.1f' % (mean / se if se > 0 else np.nan))


parser = argparse.ArgumentParser()
parser.add_argument('--per_gene_fold_file', type=str)  # <organized stem>_per_gene_fold.txt
parser.add_argument('--fsr_range', type=float, nargs=2, default=[0.1, 0.4])  # rows with expression_FSR in (lower, upper]
parser.add_argument('--gene_composition_file', type=str, default=None)  # gene_composition.txt of check_prior_by_variant_annotation.py
args = parser.parse_args()

table = np.genfromtxt(args.per_gene_fold_file, delimiter='\t', names=True, dtype=None, encoding=None)
gene_ids = np.asarray(table['gene_id']).astype(str)
fsr = np.asarray(table['expression_FSR'], dtype=float)
observed_r = np.asarray(table['observed_r'], dtype=float)
n_vars = np.asarray(table['n_cis_variants'], dtype=float)
pvalue = np.asarray(table['cis_snp_h2_pvalue_train'], dtype=float)
_, gene_index = np.unique(gene_ids, return_inverse=True)

# Terciles of n_cis_variants over all rows (as plot 3e), then the rows in the FSR range
tercile_breaks = np.quantile(n_vars, [1.0 / 3.0, 2.0 / 3.0])
valid = np.isfinite(fsr) & np.isfinite(observed_r) & np.isfinite(pvalue) & (fsr > args.fsr_range[0]) & (fsr <= args.fsr_range[1])
composition = None
if args.gene_composition_file is not None:
    # Covariates of each row's gene (rows of genes not in the file are dropped from every analysis below)
    composition_table = np.genfromtxt(args.gene_composition_file, delimiter='\t', names=True, dtype=None, encoding=None)
    composition_names = [name for name in ['fraction_low_maf', 'mean_ld_score'] if name in composition_table.dtype.names]
    row_of_gene = {gene_id: row for row, gene_id in enumerate(np.asarray(composition_table['gene_id']).astype(str))}
    composition_row = np.asarray([row_of_gene.get(gene_id, -1) for gene_id in gene_ids])
    valid &= composition_row >= 0
    composition = np.column_stack([np.asarray(composition_table[name], dtype=float)[composition_row[valid]] for name in composition_names])
    print(str(np.sum(composition_row < 0)) + ' rows of genes not in the composition file dropped')
fsr, observed_r, n_vars, pvalue, gene_index = fsr[valid], observed_r[valid], n_vars[valid], pvalue[valid], gene_index[valid]
e = (observed_r < 0.0).astype(float) - fsr
print(str(len(e)) + ' gene-fold rows with expression FSR in (' + str(args.fsr_range[0]) + ', ' + str(args.fsr_range[1]) + '], ' + str(len(np.unique(gene_index))) + ' genes')
print('columns: set, n, mean excess, clustered se, z\n')
m, se = mean_with_se(e, gene_index)
print_row('all', len(e), m, se)

print('\nDeciles of n_cis_variants (within the FSR range)')
decile_breaks = np.quantile(n_vars, np.linspace(0, 1, 11))
decile = np.clip(np.searchsorted(decile_breaks, n_vars, side='right') - 1, 0, 9)
for d in range(10):
    rows = decile == d
    m, se = mean_with_se(e[rows], gene_index[rows])
    print_row('n_cis_variants %d to %d (mean FSR %.3f)' % (decile_breaks[d], decile_breaks[d + 1], np.mean(fsr[rows])), np.sum(rows), m, se)

print('\nTercile of n_cis_variants x training LRT p-value')
tercile = np.searchsorted(tercile_breaks, n_vars, side='right')
tercile_labels = ['n_cis_variants <= %d' % tercile_breaks[0], 'n_cis_variants %d to %d' % (tercile_breaks[0], tercile_breaks[1]), 'n_cis_variants > %d' % tercile_breaks[1]]
p_stratum = np.searchsorted([0.05, 0.5], pvalue, side='right')
p_labels = ['LRT p < 0.05', 'LRT p 0.05 to 0.5', 'LRT p >= 0.5']
for t in range(3):
    for s in range(3):
        rows = (tercile == t) & (p_stratum == s)
        if np.sum(rows) < 2:
            continue
        m, se = mean_with_se(e[rows], gene_index[rows])
        print_row(tercile_labels[t] + ', ' + p_labels[s] + ' (mean FSR %.3f)' % np.mean(fsr[rows]), np.sum(rows), m, se)

print('\nRegression of the excess (coefficient, clustered se, z)')
log_n = np.log10(n_vars)
X = np.column_stack([np.ones(len(e)), log_n - np.mean(log_n), fsr - np.mean(fsr), (p_stratum == 1).astype(float), (p_stratum == 2).astype(float)])
names = ['intercept (LRT p < 0.05, mean log10 n_cis_variants and FSR)', 'log10 n_cis_variants', 'expression_FSR', 'LRT p 0.05 to 0.5', 'LRT p >= 0.5']
coef, coef_se = clustered_ols(X, e, gene_index)
for name, c, s in zip(names, coef, coef_se):
    print(name + '\t' + '%.4f' % c + '\t' + '%.4f' % s + '\t' + '%.1f' % (c / s))

if composition is not None:
    # The same regression with the variant composition of the gene's window added (log10 of the mean LD score), each covariate centered
    print('\nRegression of the excess with the variant composition of the window (coefficient, clustered se, z)')
    columns = [X]
    for j, name in enumerate(composition_names):
        values = np.log10(composition[:, j]) if name == 'mean_ld_score' else composition[:, j]
        columns.append((values - np.mean(values))[:, None])
        names = names + [('log10 ' if name == 'mean_ld_score' else '') + name]
        print('  correlation of ' + names[-1] + ' with log10 n_cis_variants: %.2f' % np.corrcoef(values, log_n)[0, 1])
    coef, coef_se = clustered_ols(np.column_stack(columns), e, gene_index)
    for name, c, s in zip(names, coef, coef_se):
        print(name + '\t' + '%.4f' % c + '\t' + '%.4f' % s + '\t' + '%.1f' % (c / s))
