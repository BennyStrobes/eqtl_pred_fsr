import sys, os, argparse
import numpy as np
sys.path.insert(0, '/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis')
import posterior_beta_samples as pbs

# Metrics of one fit of a simulated scenario, on held-out fold --fold:
#   prediction: mean observed r; mean true genetic r = corr(y_hat, u_true) in the held-out individuals
#   expression FSR calibration: mean excess 1[r < 0] - expression_FSR over rows with FSR in (0.1, 0.4], by gene; and over all genes
#   genetic FSR calibration (possible only in simulation): fraction with cov(y_hat, u_true) < 0 against mean genetic_FSR, over genes with
#       genetic_FSR in (0.01, 0.5] and a non-zero true genetic expression (the sign is undefined for null genes)
#   by stratum of n_cis_variants terciles and training LRT p (< 0.05 / >= 0.05): the mid-range expression FSR excess
#   gene classes (scenario gene_classes): per true class, mean posterior large-group weight of the gene's prior

parser = argparse.ArgumentParser()
parser.add_argument('--sim_dir', type=str)
parser.add_argument('--stem', type=str)  # fit stem (posterior samples) ; evaluation file = stem + '_eval.txt'
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--label', type=str)
args = parser.parse_args()

ev = np.genfromtxt(args.stem + '_eval.txt', delimiter='\t', names=True, dtype=None, encoding=None)
samples = pbs.load_posterior_beta_samples(args.stem + '_posterior_beta_samples.npz')
truth = np.load(args.sim_dir + 'truth.npz')
folds = np.loadtxt(args.sim_dir + 'folds.txt', skiprows=1, usecols=2, dtype=int)
test = folds == args.fold
gene_ids = samples['gene_ids'].astype(str)
ptr = samples['gene_pair_ptr']
beta_mean = np.bincount(samples['nz_pair'], weights=samples['nz_beta'].astype(np.float64), minlength=ptr[-1]) / samples['n_samples']
truth_index = {g: i for i, g in enumerate(truth['gene_ids'].astype(str))}
true_r = np.full(len(gene_ids), np.nan)
true_cov_negative = np.full(len(gene_ids), np.nan)
for chrom in np.unique(samples['gene_chroms']):
    gi = np.load(args.sim_dir + 'chr' + str(chrom) + '.npz')
    G = gi['genotype']
    vid = gi['variant_ids'].astype(str)
    row_of = {v: r for r, v in enumerate(vid)}
    chrom_gene_index = {g: i for i, g in enumerate(gi['gene_ids'].astype(str))}
    dm, ds = truth['dosage_mean_chr' + str(chrom)], truth['dosage_sd_chr' + str(chrom)]
    for g in np.flatnonzero(samples['gene_chroms'] == chrom):
        rows = np.asarray([row_of[v] for v in pbs.gene_variant_ids(samples, g)])
        Xt = (G[rows][:, test].T.astype(float) - samples['pair_train_mean'][ptr[g]:ptr[g + 1]]) / samples['pair_train_sd'][ptr[g]:ptr[g + 1]]
        y_hat = Xt @ beta_mean[ptr[g]:ptr[g + 1]]
        cg = chrom_gene_index[gene_ids[g]]
        wrows = np.arange(gi['gene_variant_start'][cg], gi['gene_variant_end'][cg])
        ti = truth_index[gene_ids[g]]
        bt = truth['beta_true'][truth['gene_pair_ptr'][ti]:truth['gene_pair_ptr'][ti + 1]]
        u = ((G[wrows][:, test].T.astype(float) - dm[wrows]) / ds[wrows]) @ bt
        if np.std(y_hat) > 0 and np.std(u) > 0:
            true_r[g] = np.corrcoef(y_hat, u)[0, 1]
            true_cov_negative[g] = float(true_r[g] < 0)
        elif np.std(y_hat) > 0:
            true_cov_negative[g] = 0.5
    gi.close()

key = {g: i for i, g in enumerate(gene_ids)}
order = np.asarray([key[g] for g in np.asarray(ev['gene_id']).astype(str)])
true_r, true_cov_negative = true_r[order], true_cov_negative[order]
fsr = np.asarray(ev['expression_FSR'], dtype=float)
gfsr = np.asarray(ev['genetic_FSR'], dtype=float)
r = np.asarray(ev['observed_r'], dtype=float)
nv = np.asarray(ev['n_cis_variants'], dtype=float)
pv = np.asarray(ev['cis_snp_h2_pvalue_train'], dtype=float)


def mean_se(x):
    x = x[np.isfinite(x)]
    return np.mean(x), np.std(x) / np.sqrt(len(x)), len(x)


out = [args.label]
ok = np.isfinite(r)
out.append('r=%.4f' % np.mean(r[ok]))
out.append('true_r=%.4f' % np.nanmean(true_r))
mid = ok & (fsr > 0.1) & (fsr <= 0.4)
m, se, k = mean_se((r[mid] < 0) - fsr[mid])
out.append('excess_mid=%+.4f(%.4f,n=%d)' % (m, se, k))
m, se, k = mean_se((r[ok] < 0) - fsr[ok])
out.append('excess_all=%+.4f(%.4f)' % (m, se))
gm = np.isfinite(gfsr) & np.isfinite(true_r) & (gfsr > 0.01) & (gfsr <= 0.5)  # genes with a non-zero true genetic expression only
m, se, k = mean_se(true_cov_negative[gm] - gfsr[gm])
out.append('genetic_excess=%+.4f(%.4f,n=%d,meanFSR=%.3f)' % (m, se, k, np.mean(gfsr[gm])))
# Calibration of the whole posterior predictive distribution of r (every gene): coverage of the 80% and 95% intervals and mean z^2,
# z = (observed_r - expected_r) / expected_r_sd (1 if calibrated, above 1 if over-confident)
er, esd = np.asarray(ev['expected_r'], dtype=float), np.asarray(ev['expected_r_sd'], dtype=float)
cov_ok = ok & np.isfinite(esd) & (esd > 0)
for level in [80, 95]:
    lb, ub = np.asarray(ev['expected_r_lb_' + str(level)], dtype=float), np.asarray(ev['expected_r_ub_' + str(level)], dtype=float)
    inside = (r[cov_ok] >= lb[cov_ok]) & (r[cov_ok] <= ub[cov_ok])
    out.append('cov%d=%.4f(%.4f)' % (level, np.mean(inside), np.std(inside) / np.sqrt(inside.sum() + (~inside).sum())))
z2 = np.square((r[cov_ok] - er[cov_ok]) / esd[cov_ok])
out.append('z2=%.3f(%.3f)' % (np.mean(z2), np.std(z2) / np.sqrt(len(z2))))
breaks = np.quantile(nv, [1 / 3, 2 / 3])
terc = np.searchsorted(breaks, nv, side='right')
strata = []
for t in range(3):
    for s, sel in [('h', pv < 0.05), ('w', pv >= 0.05)]:
        rows = mid & (terc == t) & sel
        m, se, k = mean_se((r[rows] < 0) - fsr[rows])
        strata.append('M%d%s:%+.3f(%.3f)' % (t + 1, s, m, se))
out.append(' '.join(strata))
if os.path.exists(args.stem + '_prior_rows.npz') and 'gene_class_samples' in np.load(args.stem + '_prior_rows.npz').files:
    pr = np.load(args.stem + '_prior_rows.npz')
    w = pr['row_weight_trace'][pr['retained']].astype(float)
    grid_var = pr['grid_var']
    large = np.sum(w[:, :, grid_var >= 1e-3], axis=2)  # n_samples x classes
    gcs = pr['gene_class_samples'].astype(int)
    gene_large = np.mean(large[np.arange(gcs.shape[0])[:, None], gcs], axis=0)
    tc = truth['gene_classes'][[truth_index[g] for g in gene_ids]]
    out.append('class_large_weight_by_true_class=' + ','.join('%.4f' % np.mean(gene_large[tc == c]) for c in range(3)))
print('  '.join(out), flush=True)
