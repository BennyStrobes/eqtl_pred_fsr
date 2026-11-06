import numpy as np, sys
# Evaluation math against an exact posterior: conjugate genes, beta ~ N(0, h2 / p I), sigma2_e = 1 known, exact posterior draws
src = open('/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis/evaluate_held_out_expression_prediction.py').read()
ns = {'__file__': '/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis/evaluate_held_out_expression_prediction.py'}
exec(src[:src.index('##########################\n# Command line arguments')], ns)
cq = ns['compute_expected_quantities']
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
np.random.seed(5)
n_tr, n_te, p, S, G = 412, 138, 50, 300, 6000
rows = []
for g in range(G):
    h2 = rng.uniform(0.0, 0.1)
    X = rng.normal(size=(n_tr + n_te, p)); X = (X - X.mean(0)) / X.std(0)
    b = rng.normal(size=p) * np.sqrt(h2 / p)
    y = X @ b + rng.normal(size=n_tr + n_te)
    Xtr, Xte = X[:n_tr], X[n_tr:]
    prec = Xtr.T @ Xtr + (p / h2) * np.eye(p)
    L = np.linalg.cholesky(np.linalg.inv(prec))
    mu = np.linalg.solve(prec, Xtr.T @ (y[:n_tr] - y[:n_tr].mean()))
    B = mu[None, :] + rng.normal(size=(S, p)) @ L.T
    U = B @ Xte.T
    out = cq(U, np.ones(S), y[n_tr:], np.arange(S) // 25, False)
    u_true = Xte @ b
    rows.append([out['observed_r'], out['expected_r'], out['expected_r_sd'], out['expected_r_lb_80'], out['expected_r_ub_80'], out['expression_FSR'], out['genetic_FSR'], float(np.cov(u_true, U.mean(0))[0, 1] < 0)])
R = np.asarray(rows)
r, er, sd, lb, ub, fsr, gfsr, gneg = R.T
mid = (fsr > 0.1) & (fsr <= 0.4)
gm = (gfsr > 0.01) & (gfsr <= 0.5)
print('exact posterior: cov80=%.4f(%.4f) z2=%.3f excess_mid=%+.4f(%.4f,n=%d) genetic_excess=%+.4f(%.4f,n=%d)' % (np.mean((r >= lb) & (r <= ub)), np.sqrt(0.16 / G), np.mean(((r - er) / sd) ** 2),
      np.mean((r[mid] < 0) - fsr[mid]), np.std((r[mid] < 0) - fsr[mid]) / np.sqrt(mid.sum()), mid.sum(), np.mean(gneg[gm] - gfsr[gm]), np.std(gneg[gm] - gfsr[gm]) / np.sqrt(gm.sum()), gm.sum()))
