import argparse
import numpy as np

# Gene-by-gene agreement of two fits of the same model with different samplers (e.g. ${tissue}_grid and ${tissue}_grid_gwfm). If both chains
# explore the posterior well, their predictions y_hat differ only by Monte Carlo noise, so their observed held-out correlations agree to about
#   sd(r_A - r_B) ~ sqrt((1 - r^2) * (y_hat_mc_variance_A / V_A + y_hat_mc_variance_B / V_B) / n_test)
# (V = predicted_expression_variance; y_hat_mc_variance is the batch-means Monte Carlo variance of y_hat, README.md, section 7). The ratio of the
# observed root-mean-square difference to this is about 1 under good mixing (it also picks up chains that settle on different hyper-parameters).
# Reported over all matched gene-fold rows and by decile of n_cis_variants: a ratio that grows with n_cis_variants would mean the chains explore
# genes with many (correlated) variants poorly, which would explain the expression FSR excess that grows with n_cis_variants
# (check_expression_fsr_excess.py) but did not move when the prior's inclusion probability was binned by n_cis_variants (README.md, section 5g).
# If only one file has the y_hat_mc_variance column (evaluations made before it was added), its Monte Carlo variance is used for both fits.
# Input: the <organized stem>_per_gene_fold.txt of the two fits


def load(file_name):
    table = np.genfromtxt(file_name, delimiter='\t', names=True, dtype=None, encoding=None)
    keys = [g + ':' + str(f) for g, f in zip(np.asarray(table['gene_id']).astype(str), np.asarray(table['fold']))]
    columns = {name: np.asarray(table[name], dtype=float) for name in ['observed_r', 'expected_r', 'expression_FSR', 'predicted_expression_variance', 'n_cis_variants', 'n_test']}
    columns['mc_fraction'] = np.asarray(table['y_hat_mc_variance'], dtype=float) / columns['predicted_expression_variance'] if 'y_hat_mc_variance' in table.dtype.names else None
    return keys, columns


def print_row(label, rows):
    d_r = a['observed_r'][rows] - b['observed_r'][rows]
    rms = np.sqrt(np.mean(np.square(d_r)))
    mc_rms = np.sqrt(np.mean(mc_var[rows])) if mc_var is not None else np.nan
    d_fsr = a['expression_FSR'][rows] - b['expression_FSR'][rows]
    d_er = a['expected_r'][rows] - b['expected_r'][rows]
    print(label + '\t' + str(np.sum(rows)) + '\t' + '\t'.join('%.4f' % v for v in [rms, mc_rms, rms / mc_rms, np.mean(np.abs(d_er)), np.mean(d_fsr), np.mean(np.abs(d_fsr))]))


parser = argparse.ArgumentParser()
parser.add_argument('--per_gene_fold_file_a', type=str)
parser.add_argument('--per_gene_fold_file_b', type=str)
args = parser.parse_args()

keys_a, a = load(args.per_gene_fold_file_a)
keys_b, b = load(args.per_gene_fold_file_b)
index_b = {k: i for i, k in enumerate(keys_b)}
matched = np.array([k in index_b for k in keys_a])
order_b = np.array([index_b[k] for k in np.array(keys_a)[matched]], dtype=int)
a = {name: (v[matched] if v is not None else None) for name, v in a.items()}
b = {name: (v[order_b] if v is not None else None) for name, v in b.items()}

# Monte Carlo variance of r_A - r_B for each row
mc_a, mc_b = a['mc_fraction'], b['mc_fraction']
if mc_a is None and mc_b is None:
    mc_sum = None
    print('neither file has y_hat_mc_variance: the Monte Carlo reference and the ratio are NA')
else:
    mc_sum = (mc_a if mc_a is not None else mc_b) + (mc_b if mc_b is not None else mc_a)
valid = np.isfinite(a['observed_r']) & np.isfinite(b['observed_r']) & np.isfinite(a['expression_FSR']) & np.isfinite(b['expression_FSR'])
mc_var = (1.0 - np.square(a['observed_r'])) * mc_sum / a['n_test'] if mc_sum is not None else None
if mc_var is not None:
    valid &= np.isfinite(mc_var)
print(str(np.sum(valid)) + ' matched gene-fold rows (' + str(len(keys_a)) + ' in a, ' + str(len(keys_b)) + ' in b)')
print('columns: set, n, rms(r_a - r_b), Monte Carlo rms, ratio, mean |expected_r_a - expected_r_b|, mean (FSR_a - FSR_b), mean |FSR_a - FSR_b|\n')
print_row('all', valid)

print('\nDeciles of n_cis_variants')
n_vars = a['n_cis_variants']
decile_breaks = np.quantile(n_vars[valid], np.linspace(0, 1, 11))
decile = np.clip(np.searchsorted(decile_breaks, n_vars, side='right') - 1, 0, 9)
for d in range(10):
    print_row('n_cis_variants %d to %d' % (decile_breaks[d], decile_breaks[d + 1]), valid & (decile == d))
