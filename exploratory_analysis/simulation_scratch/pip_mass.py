import sys
import numpy as np
sys.path.insert(0, '/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis')
import posterior_beta_samples as pbs
# Where the summed-PIP excess sits: expected (summed posterior probability) against true number of pairs with |beta| > t, for a ladder of t;
# and the summed PIP split by the pair's PIP bin, against the true causal count in that bin
d, stem = sys.argv[1], sys.argv[2]
s = pbs.load_posterior_beta_samples(stem + '_posterior_beta_samples.npz')
truth = np.load(d + 'truth.npz')
bt = np.abs(truth['beta_true'])
S = s['n_samples']; P = s['gene_pair_ptr'][-1]
nzb = np.abs(s['nz_beta'].astype(float))
print('== ' + stem)
for t in [0, 0.005, 0.01, 0.02, 0.03, 0.05]:
    print('  |b|>%-5g expected %8.0f  true %6d' % (t, np.sum(nzb > t) / S, np.sum(bt > t)))
pip = s['pair_pip'].astype(float)
edges = [0, 0.01, 0.05, 0.1, 0.5, 1.01]
print('  PIP mass by bin: ' + '; '.join('[%g,%g) n=%d sum=%.0f' % (lo, hi, ((pip >= lo) & (pip < hi)).sum(), pip[(pip >= lo) & (pip < hi)].sum()) for lo, hi in zip(edges[:-1], edges[1:])))
