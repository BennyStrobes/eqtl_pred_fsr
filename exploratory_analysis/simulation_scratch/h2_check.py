import sys, argparse
import numpy as np
# cis-h2 in the training individuals against the truth (variance of the true genetic expression in the training individuals): the posterior
# (posterior_cis_h2_train, 95% interval) and the model-free ML and HE estimates of the evaluation file. Per estimator: mean bias, slope of the
# truth on the estimate (1 if calibrated as a predictor; below 1 = estimates too spread), correlation, and by bins of true h2 the mean estimate;
# bins are of the absolute true Var(u); ML and HE are compared with the proportion Var(u) / (Var(u) + 1); for the posterior also the coverage of its 95% interval
parser = argparse.ArgumentParser()
parser.add_argument('--sim_dir', type=str)
parser.add_argument('--stem', type=str)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--label', type=str)
args = parser.parse_args()

ev = np.genfromtxt(args.stem + '_eval.txt', delimiter='\t', names=True, dtype=None, encoding=None)
truth = np.load(args.sim_dir + 'truth.npz')
train = np.loadtxt(args.sim_dir + 'folds.txt', skiprows=1, usecols=2, dtype=int) != args.fold
ti = {g: i for i, g in enumerate(truth['gene_ids'].astype(str))}
true_h2 = {}
for chrom in np.unique(truth['gene_chroms']):
    gi = np.load(args.sim_dir + 'chr' + str(chrom) + '.npz')
    G = gi['genotype']; dm, ds = truth['dosage_mean_chr' + str(chrom)], truth['dosage_sd_chr' + str(chrom)]
    for c, gname in enumerate(gi['gene_ids'].astype(str)):
        w = np.arange(gi['gene_variant_start'][c], gi['gene_variant_end'][c]); k = ti[gname]
        u = ((G[w][:, train].T.astype(float) - dm[w]) / ds[w]) @ truth['beta_true'][truth['gene_pair_ptr'][k]:truth['gene_pair_ptr'][k + 1]]
        true_h2[gname] = np.var(u)
    gi.close()
h = np.asarray([true_h2[g] for g in np.asarray(ev['gene_id']).astype(str)])
print('== ' + args.label + ' (%d genes, mean true h2 %.4f)' % (len(h), h.mean()))
edges = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 1.0]
h_abs = h
for name in ['posterior_cis_h2_train', 'cis_snp_h2_train', 'cis_snp_h2_he_train']:
    est = np.asarray(ev[name], dtype=float)
    # the posterior is Var(X beta) on the expression scale (noise variance 1); ML and HE estimate the proportion Var(u) / (Var(u) + 1)
    h = h_abs if name == 'posterior_cis_h2_train' else h_abs / (h_abs + 1.0)
    ok = np.isfinite(est)
    slope = np.cov(h[ok], est[ok])[0, 1] / np.var(est[ok], ddof=1)
    line = '  %-24s bias %+.4f slope %.3f corr %.3f' % (name, np.mean(est[ok] - h[ok]), slope, np.corrcoef(h[ok], est[ok])[0, 1])
    if name == 'posterior_cis_h2_train':
        lb, ub = np.asarray(ev['posterior_cis_h2_train_lb'], dtype=float), np.asarray(ev['posterior_cis_h2_train_ub'], dtype=float)
        line += ' cov95 %.3f' % np.mean((h >= lb) & (h <= ub))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (h_abs >= lo) & (h_abs < hi)
        if sel.sum() > 0:
            extra = ''
            if name == 'posterior_cis_h2_train':
                extra = ' cov %.2f' % np.mean((h[sel] >= lb[sel]) & (h[sel] <= ub[sel]))
            bins.append('[%g,%g) n=%d est %.4f true %.4f%s' % (lo, hi, sel.sum(), est[sel].mean(), h[sel].mean(), extra))
    print(line)
    print('      ' + '; '.join(bins))
