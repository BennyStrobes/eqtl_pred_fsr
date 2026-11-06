import sys, numpy as np
sys.path.insert(0, '/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis')
import posterior_beta_samples as pbs
# Genetic FSR variants against the truth: (a) as the evaluation (y_hat from all samples), (b) y_hat from the first half of the samples and FSR
# from the second (no self-inclusion), (c) as (b) with all-zero samples counted 1/2
d, stem = sys.argv[1], sys.argv[2]
samples = pbs.load_posterior_beta_samples(stem + '_posterior_beta_samples.npz')
truth = np.load(d + 'truth.npz')
test = np.loadtxt(d + 'folds.txt', skiprows=1, usecols=2, dtype=int) == 1
ptr = samples['gene_pair_ptr']; gid = samples['gene_ids'].astype(str); S = samples['n_samples']
ti = {g: i for i, g in enumerate(truth['gene_ids'].astype(str))}
rows_out = []
for chrom in np.unique(samples['gene_chroms']):
    gi = np.load(d + 'chr%d.npz' % chrom); G = gi['genotype']; row_of = {v: r for r, v in enumerate(gi['variant_ids'].astype(str))}
    cgi = {g: i for i, g in enumerate(gi['gene_ids'].astype(str))}; dm, ds = truth['dosage_mean_chr%d' % chrom], truth['dosage_sd_chr%d' % chrom]
    for g in np.flatnonzero(samples['gene_chroms'] == chrom):
        rows = np.asarray([row_of[v] for v in pbs.gene_variant_ids(samples, g)])
        X = (G[rows][:, test].T.astype(float) - samples['pair_train_mean'][ptr[g]:ptr[g+1]]) / samples['pair_train_sd'][ptr[g]:ptr[g+1]]
        U = pbs.gene_beta_sample_matrix(samples, g) @ X.T  # S x n_test
        c = cgi[gid[g]]; w = np.arange(gi['gene_variant_start'][c], gi['gene_variant_end'][c]); k = ti[gid[g]]
        u = ((G[w][:, test].T.astype(float) - dm[w]) / ds[w]) @ truth['beta_true'][truth['gene_pair_ptr'][k]:truth['gene_pair_ptr'][k+1]]
        if np.std(u) == 0: continue
        zero = np.all(U == 0, axis=1)
        out = []
        for fit_idx, ev_idx in [(np.arange(S), np.arange(S)), (np.arange(S // 2), np.arange(S // 2, S))]:
            yh = np.mean(U[fit_idx], axis=0); yh = yh - yh.mean()
            if np.std(yh) == 0: out += [np.nan, np.nan, np.nan]; continue
            cov = (U[ev_idx] - U[ev_idx].mean(1, keepdims=True)) @ yh
            fsr = np.mean(cov < 0); fsr_half = np.mean(cov < 0) + 0.5 * np.mean(zero[ev_idx])
            out += [fsr, fsr_half, float(np.dot(u - u.mean(), yh) < 0)]
        rows_out.append(out + [np.mean(zero)])
    gi.close()
R = np.asarray(rows_out)
print('genes with non-zero truth', len(R), ' mean fraction of all-zero samples %.4f' % np.nanmean(R[:, 6]))
for label, f, t in [('(a) as evaluation', R[:, 0], R[:, 2]), ('(b) split', R[:, 3], R[:, 5]), ('(c) split, zero = 1/2', R[:, 4], R[:, 5])]:
    sel = np.isfinite(f) & (f > 0.01) & (f <= 0.5)
    e = t[sel] - f[sel]
    print('%-24s n=%d mean FSR %.3f observed %.3f excess %+.4f (se %.4f)' % (label, sel.sum(), f[sel].mean(), t[sel].mean(), e.mean(), e.std() / np.sqrt(sel.sum())))
    for lo, hi in [(0.01, 0.05), (0.05, 0.15), (0.15, 0.5)]:
        s2 = np.isfinite(f) & (f > lo) & (f <= hi)
        print('    FSR %.2f-%.2f: n=%d mean FSR %.3f observed %.3f' % (lo, hi, s2.sum(), f[s2].mean(), t[s2].mean()))
