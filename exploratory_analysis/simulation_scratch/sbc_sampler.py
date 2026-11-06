import sys, argparse, numpy as np
sys.path.insert(0, '/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis')
from eqtl_grid_prior_gibbs import EQTL_GRID_PRIOR_GIBBS
src = open('/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis/evaluate_held_out_expression_prediction.py').read()
ns = {'__file__': 'x'}; exec(src[:src.index('##########################\n# Command line arguments')], ns); cq = ns['compute_expected_quantities']
# Calibration of the grid sampler on genes drawn from a known prior (independent variants). --fixed: weights held at the truth
p_ = argparse.ArgumentParser()
p_.add_argument('--fixed', action='store_true'); p_.add_argument('--fixed_sigma2e', action='store_true')
p_.add_argument('--n_genes', type=int, default=3000); p_.add_argument('--p', type=int, default=200); p_.add_argument('--q', type=float, default=0.02)
p_.add_argument('--n_iters', type=int, default=2000); p_.add_argument('--burn_in', type=int, default=1000); p_.add_argument('--seed', type=int, default=1)
p_.add_argument('--split', action='store_true')
p_.add_argument('--model_grid', type=float, nargs='+', default=None)  # grid of the fit (default: the true grid)
p_.add_argument('--cutoff', type=float, default=5e-3)
p_.add_argument('--ig', type=float, nargs=2, default=None); p_.add_argument('--hier', action='store_true'); p_.add_argument('--ncp', type=int, default=0); p_.add_argument('--two_stage', action='store_true'); p_.add_argument('--stage1_iters', type=int, default=2000); p_.add_argument('--start_at_truth', action='store_true'); p_.add_argument('--trace', action='store_true'); p_.add_argument('--eb', action='store_true'); p_.add_argument('--eb_interval', type=int, default=5)
a = p_.parse_args()
rng = np.random.default_rng(a.seed)
grid = np.asarray([1e-3, 1e-2, 1e-1]); w_true = np.asarray([0.5, 0.35, 0.15])
pi_true = np.concatenate([[1 - a.q], a.q * w_true])
n_tr, n_te = 412, 138; n = n_tr + n_te
G_all = []; Y = []; Yte = []; Xte_list = []; beta_list = []
for g in range(a.n_genes):
    maf = rng.uniform(0.05, 0.5, a.p)
    Gg = rng.binomial(2, maf[:, None], size=(a.p, n)).astype(np.int8)
    Gg = Gg[np.std(Gg[:, :n_tr], axis=1) > 0]
    m, s = Gg[:, :n_tr].mean(1), Gg[:, :n_tr].std(1)
    X = (Gg.T - m) / s
    c = rng.choice(4, size=len(m), p=pi_true)
    b = np.where(c > 0, rng.normal(size=len(m)) * np.sqrt(np.concatenate([[0], grid])[c]), 0.0)
    y = X @ b + rng.normal(size=n)
    G_all.append(Gg); Y.append(y[:n_tr] - y[:n_tr].mean()); Yte.append(y[n_tr:]); beta_list.append(b)
p_g = np.asarray([x.shape[0] for x in G_all])
G = np.ascontiguousarray(np.concatenate([x[:, :n_tr] for x in G_all]))
var_mean = G.mean(1).astype(float); var_sd = G.std(1).astype(float)
ptr = np.concatenate([[0], np.cumsum(p_g)]); idx = np.arange(ptr[-1])
model_grid = np.asarray(a.model_grid) if a.model_grid is not None else grid
model = EQTL_GRID_PRIOR_GIBBS(n_iters=a.n_iters, burn_in=a.burn_in, thin=5, grid_variances=model_grid, small_large_cutoff=a.cutoff, seed=a.seed, verbose=False, **({'inverse_gamma_a': 1e6, 'inverse_gamma_b': 1e6} if a.fixed_sigma2e else ({'inverse_gamma_a': a.ig[0], 'inverse_gamma_b': a.ig[1]} if a.ig is not None else {})))
if a.fixed:
    model.pi = pi_true.copy(); model.sample_hyperparameters = lambda: ''
if a.hier:
    # Hierarchical residual-variance prior sigma2_e_g ~ IG(a, b), (a, b) learned: b | a ~ Gamma(G a + 1, sum 1 / sigma2_e) (flat prior on b),
    # log a by random-walk Metropolis (prior p(a) proportional to 1 / a); the sweep uses the current (a, b)
    from scipy.special import gammaln
    model.ig_a, model.ig_b = 10.0, 10.0
    hier_rng = np.random.default_rng(a.seed + 11)
    hier_trace = []
    _prev = model.sample_hyperparameters
    def with_hier():
        out = _prev()
        s2 = model.sigma2_e
        Gn = len(s2); sum_inv = np.sum(1.0 / s2); sum_log = np.sum(np.log(s2))
        model.ig_b = hier_rng.gamma(Gn * model.ig_a + 1.0, 1.0 / sum_inv)
        def logp(av):
            return Gn * (av * np.log(model.ig_b) - gammaln(av)) - (av + 1.0) * sum_log - np.log(av) + np.log(av)  # 1/a prior, log-scale proposal Jacobian
        for _ in range(5):
            prop = model.ig_a * np.exp(0.1 * hier_rng.normal())
            if np.log(hier_rng.random()) < logp(prop) - logp(model.ig_a):
                model.ig_a = prop
        hier_trace.append((model.ig_a, model.ig_b))
        return out
    model.sample_hyperparameters = with_hier
if a.ncp > 0:
    from ncp_proto import ncp_weight_moves
    ncp_rng = np.random.default_rng(a.seed + 7)
    step = np.ones((len(model_grid) + 1, len(model_grid) + 1))
    ncp_state = {'it': 0, 'acc': 0, 'tried': 0, 'trace': []}
    _cp = model.sample_hyperparameters
    def cp_then_ncp():
        out = _cp()
        st = step if ncp_state['it'] < a.burn_in else step.copy()  # adapt the step sizes during burn-in only
        ncp_state['acc'] += ncp_weight_moves(model, G, var_mean, var_sd, idx, a.ncp, st, 0.1, ncp_rng)
        ncp_state['tried'] += a.ncp; ncp_state['it'] += 1; ncp_state['trace'].append(model.pi.copy())
        return out
    model.sample_hyperparameters = cp_then_ncp
if a.start_at_truth or a.trace:
    if a.start_at_truth:
        model.pi = pi_true.copy()
    trace = []
    _orig = model.sample_hyperparameters
    def traced():
        out = _orig(); trace.append(model.pi.copy()); return out
    model.sample_hyperparameters = traced
if a.eb:
    from eb_proto import conditional_b, eb_weights
    state = {'it': 0}
    def eb_update():
        it = state['it']; state['it'] += 1
        if it < a.burn_in and it % a.eb_interval == 0:
            b = np.empty(model.n_pairs)
            conditional_b(G, var_mean, var_sd, ptr, idx, model.resid, model.beta, b)
            v = (model.sigma2_e / model.resid.shape[1])[model.gene_of_pair]
            model.pi = eb_weights(b, v, model.grid_var, model.pi)
        return ''
    model.sample_hyperparameters = eb_update
import contextlib, io
if a.two_stage:
    # Stage 1: residual variances fixed at var(y) (1 - ML cis-h2 from the gene's GRM), weights learned; stage 2: weights fixed at the mean of
    # the second half of stage 1, everything else sampled
    s2_hat = np.empty(a.n_genes)
    for g in range(a.n_genes):
        grm = ns['compute_cis_grm'](G_all[g][:, :n_tr].T.astype(float))
        h2, _ = ns['estimate_cis_snp_heritability_with_lrt'](grm, Y[g])
        s2_hat[g] = np.var(Y[g]) * (1.0 - h2)
    stage1 = EQTL_GRID_PRIOR_GIBBS(n_iters=a.stage1_iters, burn_in=a.stage1_iters - 5, thin=5, grid_variances=model_grid, small_large_cutoff=a.cutoff, seed=a.seed + 100, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        stage1.fit(G, var_mean, var_sd, ptr, idx, np.asarray(Y), sigma2_e_fixed=s2_hat)
    pi_hat = np.mean(stage1.grid_weight_trace[a.stage1_iters // 2:], axis=0)
    T = stage1.grid_weight_trace
    print('  stage 1: 1-pi0 at iterations 100, 500, 1000, end: ' + ', '.join('%.4f' % (1 - T[i, 0]) for i in [100, 500, min(1000, len(T) - 1), len(T) - 1]) + '; pi_hat=' + np.array2string(pi_hat, precision=4) + '; truth=' + np.array2string(pi_true, precision=4))
    model = EQTL_GRID_PRIOR_GIBBS(n_iters=a.n_iters, burn_in=a.burn_in, thin=5, grid_variances=model_grid, small_large_cutoff=a.cutoff, seed=a.seed, verbose=False, fixed_weights=pi_hat)
with contextlib.redirect_stdout(io.StringIO()):
    model.fit(G, var_mean, var_sd, ptr, idx, np.asarray(Y))
S = model.n_samples
rows = []
order = np.argsort(model.nz_pair, kind='stable')
for g in range(a.n_genes):
    lo, hi = ptr[g], ptr[g + 1]
    sel = (model.nz_pair >= lo) & (model.nz_pair < hi)
    B = np.zeros((S, hi - lo)); B[model.nz_sample[sel], model.nz_pair[sel] - lo] = model.nz_beta[sel]
    Gg = G_all[g]; Xte = (Gg[:, n_tr:].T - var_mean[lo:hi]) / var_sd[lo:hi]
    U = B @ Xte.T
    out = cq(U, model.sigma2_e_samples[:, g].astype(float), Yte[g], np.arange(S) // 20, a.split)
    Xte_true = (Gg[:, n_tr:].T - Gg.mean(1)) / Gg.std(1)
    u = ((Gg[:, n_tr:].T - var_mean[lo:hi]) / var_sd[lo:hi]) @ beta_list[g]
    yh = U.mean(0)
    rows.append([out['observed_r'], out['expected_r'], out['expected_r_sd'], out['expected_r_lb_80'], out['expected_r_ub_80'], out['expression_FSR'], out['genetic_FSR'],
                 float(np.cov(u, yh)[0, 1] < 0) if np.std(u) > 0 else np.nan, np.mean(model.sigma2_e_samples[:, g])])
R = np.asarray(rows, dtype=float)
r, er, sd, lb, ub, fsr, gfsr, gneg, s2 = R.T
ok = np.isfinite(r) & np.isfinite(sd) & (sd > 0)
mid = ok & (fsr > 0.1) & (fsr <= 0.4); gm = ok & np.isfinite(gneg) & (gfsr > 0.01) & (gfsr <= 0.5)
print('hier=%s ig=%s ncp=%d two_stage=%s eb=%s grid=%s fixed=%s split=%s q=%.3f p=%d: pi_end=%s cov80=%.4f(%.4f) z2=%.3f excess_mid=%+.4f(%.4f,n=%d) genetic_excess=%+.4f(%.4f,n=%d) mean_sigma2e=%.4f mean_r=%.3f' % (
    a.hier, a.ig, a.ncp, a.two_stage, a.eb, 'true' if a.model_grid is None else len(a.model_grid), a.fixed, a.split, a.q, a.p, np.array2string(model.pi, precision=4), np.mean((r[ok] >= lb[ok]) & (r[ok] <= ub[ok])), np.sqrt(0.16 / ok.sum()), np.mean(((r[ok] - er[ok]) / sd[ok]) ** 2),
    np.mean((r[mid] < 0) - fsr[mid]), np.std((r[mid] < 0) - fsr[mid]) / np.sqrt(mid.sum()), mid.sum(), np.mean(gneg[gm] - gfsr[gm]), np.std(gneg[gm] - gfsr[gm]) / np.sqrt(gm.sum()), gm.sum(), np.mean(s2), np.nanmean(r)), flush=True)
if a.start_at_truth or a.trace:
    T = np.asarray(trace)
    for it in [0, 100, 300, 1000, 2000, len(T) - 1]:
        if it < len(T):
            print('  iter %d: 1-pi0=%.4f weights=%s' % (it, 1 - T[it, 0], np.array2string(T[it, 1:], precision=4)))
if a.ncp > 0:
    T = np.asarray(ncp_state['trace'])
    print('  ncp acceptance %.3f; 1-pi0 / weights at iterations: ' % (ncp_state['acc'] / ncp_state['tried']) + '; '.join('%d: %.4f %s' % (it, 1 - T[it, 0], np.array2string(T[it, 1:], precision=4)) for it in [0, 100, 300, 1000, len(T) // 2, len(T) - 1] if it < len(T)))
if a.hier:
    H = np.asarray(hier_trace)
    print('  hierarchical IG(a, b) at iterations 100, 1000, end: ' + ', '.join('(%.1f, %.1f; mean %.3f)' % (H[i, 0], H[i, 1], H[i, 1] / (H[i, 0] - 1)) for i in [100, 1000, len(H) - 1]))
