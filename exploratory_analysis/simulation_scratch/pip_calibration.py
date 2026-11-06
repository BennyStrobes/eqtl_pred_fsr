import sys, argparse
import numpy as np
sys.path.insert(0, '/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis')
import posterior_beta_samples as pbs

# Calibration of per-pair posterior probabilities against the truth of a simulated scenario:
#   PIP = P(beta != 0) (the sampler's Rao-Blackwellized pair_pip) against 1[beta_true != 0]
#   P(|beta| > t) from the retained samples against 1[|beta_true| > t], t = 0.05, 0.1 (on the training-standardized scale of the fit; the
#       truth's all-individual standardization is converted with the ratio of the two sds)
# For each, bins of the posterior probability: pairs, mean probability, observed fraction (with binomial se), and the expected and observed
# numbers of true signals over pairs with probability >= 0.5 (a fine-mapping-style summary)
parser = argparse.ArgumentParser()
parser.add_argument('--sim_dir', type=str)
parser.add_argument('--stem', type=str)
parser.add_argument('--label', type=str)
args = parser.parse_args()

samples = pbs.load_posterior_beta_samples(args.stem + '_posterior_beta_samples.npz')
truth = np.load(args.sim_dir + 'truth.npz')
ptr = samples['gene_pair_ptr']
gene_ids = samples['gene_ids'].astype(str)
pair_ids = samples['pair_variant_ids'].astype(str)
S = samples['n_samples']
truth_index = {g: i for i, g in enumerate(truth['gene_ids'].astype(str))}
true_beta = np.full(ptr[-1], np.nan)
for chrom in np.unique(samples['gene_chroms']):
    gi = np.load(args.sim_dir + 'chr' + str(chrom) + '.npz')
    vid = gi['variant_ids'].astype(str)
    chrom_gene_index = {g: i for i, g in enumerate(gi['gene_ids'].astype(str))}
    ds_all = truth['dosage_sd_chr' + str(chrom)]
    for g in np.flatnonzero(samples['gene_chroms'] == chrom):
        cg = chrom_gene_index[gene_ids[g]]
        wstart, wend = gi['gene_variant_start'][cg], gi['gene_variant_end'][cg]
        pos_of = {v: r - wstart for r, v in enumerate(vid[wstart:wend], start=wstart)}
        ti = truth_index[gene_ids[g]]
        bt = truth['beta_true'][truth['gene_pair_ptr'][ti]:truth['gene_pair_ptr'][ti + 1]]
        for k in range(ptr[g], ptr[g + 1]):
            j = pos_of[pair_ids[k]]
            # effect per training sd: beta_all-standardized x (training sd / all-individual sd)
            true_beta[k] = bt[j] * samples['pair_train_sd'][k] / ds_all[wstart + j]
    gi.close()

nz_pair, nz_beta = samples['nz_pair'], np.abs(samples['nz_beta'].astype(float))
quantities = [('PIP', samples['pair_pip'].astype(float), true_beta != 0)]
for t in [0.05, 0.1]:
    quantities.append(('P(|b|>%g)' % t, np.bincount(nz_pair[nz_beta > t], minlength=ptr[-1]) / S, np.abs(true_beta) > t))
edges = np.asarray([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0000001])
print('== ' + args.label)
for name, prob, truth_flag in quantities:
    line = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (prob >= lo) & (prob < hi)
        if sel.sum() == 0:
            continue
        obs = truth_flag[sel].mean()
        line.append('[%.2f,%.2f) n=%d mean=%.3f obs=%.3f(%.3f)' % (lo, min(hi, 1), sel.sum(), prob[sel].mean(), obs, np.sqrt(max(obs * (1 - obs), 1e-12) / sel.sum())))
    hi_sel = prob >= 0.5
    print('  %s: total expected %.1f observed %d; prob >= 0.5: expected %.1f observed %d' % (name, prob.sum(), truth_flag.sum(), prob[hi_sel].sum(), truth_flag[hi_sel].sum()))
    for l in line:
        print('    ' + l)
