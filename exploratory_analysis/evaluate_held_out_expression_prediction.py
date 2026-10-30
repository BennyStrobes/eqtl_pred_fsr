import sys
import os
import argparse
import shutil
import tempfile
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posterior_beta_samples as pbs

# Evaluate the expression predictions of one fit of run_eqtl_mixture_prior_gibbs.py on its held-out fold (README.md, section 7), one line per gene.
# Every quantity is for one gene in one fold: the posterior fit without the held-out fold, evaluated on the held-out individuals.
# u^(s) = X_test beta^(s) is the held-out genetic expression under retained effect sample s (computed with posterior_beta_samples.py, held-out
# dosages standardized with the training mean and standard deviation), y_hat = mean_s u^(s) the prediction, y the held-out expression.
#   observed_r: corr(y_hat, y)
#   predicted_expression_variance: V = Var_i(y_hat); expected_r_plugin: sqrt(V)
#   expected_r, expected_r_sd, expected_r_{lb,ub}_{80,90,95,99}: mean, standard deviation and quantile intervals over s of corr(y_hat, u^(s) + eps^(s)),
#       eps^(s) ~ N(0, sigma2_e^(s) I) (posterior predictive distribution of the observed correlation)
#   expected_cis_h2: mean_s Var_i(u^(s)); expected_genetic_r: mean_s corr(y_hat, u^(s)) (0 for samples with u^(s) constant)
#   predicted_r_variance: Var_s[Cov_i(y_hat, u^(s)) / sqrt(V)]
#   genetic_FSR: mean_s 1[Cov_i(y_hat, u^(s)) < 0]; expression_FSR: mean_s 1[corr(y_hat, u^(s) + eps^(s)) < 0]
#   posterior_cis_h2_train (_lb, _ub): mean (95% interval) over s of Var_i(X_train beta^(s)) in the training individuals
#   cis_snp_h2 (ML, LRT p-value) and cis_snp_h2_he (Haseman-Elston, standard error, p-value) in the training (_train) and held-out (_test)
#       individuals, from genotypes and expression only (no model)
# Genes whose prediction is constant in the held-out individuals (V = 0) get nan for the quantities that need V > 0.

coverage_levels = [80, 90, 95, 99]


def load_in_fold_file(fold_file):
    # Same as run_eqtl_mixture_prior_gibbs.py
    genotype_index_to_fold = {}
    f = open(fold_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        genotype_index_to_fold[int(data[0])] = (data[1], int(data[2]))
    f.close()
    return genotype_index_to_fold


def load_in_gibbs_input_summary(gibbs_input_summary_file):
    # chromosome number -> gibbs input npz file
    chrom_to_file = {}
    f = open(gibbs_input_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        chrom_to_file[int(data[0].replace('chr', ''))] = data[1]
    f.close()
    return chrom_to_file


def compute_cis_grm(genotype_mat):
    # Same as gdl_uncertainty_paper/expression_correlations/personalized_expression_correlations_per_tissue.py
    # Standardize genotypes in sample (discarding variants with no variance) and return the cis GRM X X^T / m
    X = np.asarray(genotype_mat, dtype=float)
    X = X - np.mean(X, axis=0)
    snp_sdevs = np.std(X, axis=0)
    valid_snps = np.isfinite(snp_sdevs) & (snp_sdevs > 0.0)
    if np.sum(valid_snps) == 0:
        return None
    X = X[:, valid_snps] / snp_sdevs[valid_snps][None, :]
    return np.dot(X, np.transpose(X)) / X.shape[1]


def estimate_cis_snp_heritability_with_lrt(grm, expr_vec):
    # Same as estimate_cis_snp_heritability_with_lrt of gdl_uncertainty_paper/expression_correlations/personalized_expression_correlations_per_tissue.py,
    # but takes the cis GRM (shared with the Haseman-Elston estimate)
    # Cis-SNP heritability via maximum likelihood on a single variance component model (y ~ N(0, h2*GRM + (1-h2)*I) up to a scale), with a
    # likelihood ratio test against h2 = 0. The p-value uses the 50:50 mixture of chi2(0) and chi2(1) since h2 = 0 is on the boundary.
    if grm is None:
        return np.nan, np.nan
    y = np.asarray(expr_vec, dtype=float)
    y = y - np.mean(y)
    n_samples = len(y)
    eigenvalues, eigenvectors = np.linalg.eigh(grm)
    transformed_y = np.dot(np.transpose(eigenvectors), y)

    def log_likelihood(h2):
        variance_scale = h2 * eigenvalues + (1.0 - h2)
        if np.any(variance_scale <= 0.0):
            return -np.inf
        residual_var = np.mean(np.square(transformed_y) / variance_scale)
        if residual_var <= 0.0:
            return -np.inf
        return -0.5 * (n_samples * np.log(2.0 * np.pi) + n_samples * np.log(residual_var) + np.sum(np.log(variance_scale)) + n_samples)

    null_log_likelihood = log_likelihood(0.0)
    opt = minimize_scalar(lambda h2: -log_likelihood(h2), bounds=(0.0, 0.999999), method='bounded')
    h2 = opt.x
    alt_log_likelihood = -opt.fun
    lrt_stat = np.maximum(2.0 * (alt_log_likelihood - null_log_likelihood), 0.0)
    lrt_pvalue = 0.5 * stats.chi2.sf(lrt_stat, df=1)
    return h2, lrt_pvalue


def estimate_cis_snp_heritability_with_he_regression(grm, expr_vec):
    # Haseman-Elston estimate and analytic standard error as estimate_cis_snp_heritability_with_he_regression of
    # gdl_uncertainty_paper/expression_correlations/personalized_expression_correlations_per_tissue.py (takes the cis GRM), plus a one-sided
    # p-value for h2 > 0: z = he_h2 / se_0 with se_0 the same analytic standard error evaluated at h2 = 0 (the null distribution of the
    # estimate, rather than at the fitted h2).
    # Regress y_i y_k on GRM_ik (with an intercept) over all pairs i != k of the mean-centered expression y; the slope divided by Var(y) is h2.
    # Standard error: the estimate is a ratio of quadratic forms in y, R = (y^T W y) / (y^T y / n), with W symmetric and zero-diagonal.
    # Var(R) follows from Var(y^T M y) = 2 tr(M S M S), Cov(y^T M y, y^T P y) = 2 tr(M S P S) and the delta method, with S the covariance of
    # the centered y under the fitted single-component model h2*GRM + (1-h2)*I (h2 clipped to [0, 1]).
    # Under h2 = 0, S = Var(y) C (C the centering matrix) and the delta-method variance reduces to Var(y^T W y) / Var(y)^2 = 2 tr(W C W C).
    y = np.asarray(expr_vec, dtype=float)
    y = y - np.mean(y)
    n = len(y)
    var_y = np.mean(np.square(y))
    if grm is None or n < 3 or var_y <= 0.0:
        return np.nan, np.nan, np.nan

    # W_ik = (GRM_ik - mean off-diagonal GRM) / sum_{i != k} (GRM_ik - mean)^2 for i != k, and 0 on the diagonal,
    # so that y^T W y is the OLS slope (with intercept) of y_i y_k on GRM_ik over pairs i != k
    off_diag = ~np.eye(n, dtype=bool)
    W = grm - np.mean(grm[off_diag])
    W[~off_diag] = 0.0
    sxx = np.sum(np.square(W))
    if sxx <= 0.0:
        return np.nan, np.nan, np.nan
    W = W / sxx
    he_h2 = float(np.dot(y, np.dot(W, y)) / var_y)

    # Analytic standard error at the fitted h2. Covariance of the centered y under the fitted model:
    # C (s2_g GRM + s2_e I) C = s2_g GRM + s2_e C, since the in-sample GRM is already double-centered (GRM 1 = 0)
    h2_fit = np.clip(he_h2, 0.0, 1.0)
    centering = np.eye(n) - 1.0 / n
    Sigma = var_y * (h2_fit * grm + (1.0 - h2_fit) * centering)
    W_Sigma = np.dot(W, Sigma)
    var_q1 = 2.0 * np.sum(W_Sigma * np.transpose(W_Sigma))   # Var(y^T W y)            = 2 tr(W S W S)
    cov_q1_q2 = 2.0 * np.sum(W_Sigma * Sigma) / n             # Cov(y^T W y, y^T y / n) = 2 tr(W S S) / n
    var_q2 = 2.0 * np.sum(np.square(Sigma)) / np.square(n)    # Var(y^T y / n)          = 2 tr(S S) / n^2
    var_h2 = (var_q1 - 2.0 * he_h2 * cov_q1_q2 + np.square(he_h2) * var_q2) / np.square(var_y)
    he_h2_se = float(np.sqrt(np.maximum(var_h2, 0.0)))

    # Standard error under h2 = 0 and one-sided p-value
    W_C = W - np.mean(W, axis=1)[:, None]
    se_0 = float(np.sqrt(np.maximum(2.0 * np.sum(W_C * np.transpose(W_C)), 0.0)))
    he_h2_pvalue = stats.norm.sf(he_h2 / se_0) if se_0 > 0.0 else np.nan
    return he_h2, he_h2_se, he_h2_pvalue


def compute_expected_quantities(U, sigma2_e_samples, y):
    # U: n_samples x n_test genetic expression samples; sigma2_e_samples: n_samples residual variances; y: held-out expression
    # Returns dictionary of the observed and expected quantities (see top of file)
    n_test = U.shape[1]
    out = {}
    y_hat = np.mean(U, axis=0)
    y_hat_c = y_hat - np.mean(y_hat)
    V = np.mean(np.square(y_hat_c))
    U_c = U - np.mean(U, axis=1)[:, None]
    var_u = np.mean(np.square(U_c), axis=1)
    out['expected_cis_h2'] = np.mean(var_u)
    out['predicted_expression_variance'] = V
    # Posterior predictive draws of held-out expression: u^(s) + eps^(s)
    W = U + np.random.normal(size=U.shape) * np.sqrt(sigma2_e_samples)[:, None]
    W_c = W - np.mean(W, axis=1)[:, None]
    var_w = np.mean(np.square(W_c), axis=1)
    if V <= 0.0 or np.std(y) == 0.0:
        for name in ['observed_r', 'expected_r_plugin', 'expected_r', 'expected_r_sd', 'expected_genetic_r', 'predicted_r_variance', 'genetic_FSR', 'expression_FSR']:
            out[name] = np.nan
        for level in coverage_levels:
            out['expected_r_lb_' + str(level)] = np.nan
            out['expected_r_ub_' + str(level)] = np.nan
        return out
    out['observed_r'] = np.corrcoef(y_hat, y)[0, 1]
    out['expected_r_plugin'] = np.sqrt(V)
    cov_u = np.dot(U_c, y_hat_c) / n_test
    cov_w = np.dot(W_c, y_hat_c) / n_test
    corr_w = cov_w / np.sqrt(V * var_w)
    corr_u = np.zeros(len(cov_u))
    varying = var_u > 0.0
    corr_u[varying] = cov_u[varying] / np.sqrt(V * var_u[varying])
    out['expected_r'] = np.mean(corr_w)
    out['expected_r_sd'] = np.std(corr_w)
    for level in coverage_levels:
        tail = (1.0 - level / 100.0) / 2.0
        out['expected_r_lb_' + str(level)] = np.quantile(corr_w, tail)
        out['expected_r_ub_' + str(level)] = np.quantile(corr_w, 1.0 - tail)
    out['expected_genetic_r'] = np.mean(corr_u)
    out['predicted_r_variance'] = np.var(cov_u / np.sqrt(V))
    out['genetic_FSR'] = np.mean(cov_u < 0.0)
    out['expression_FSR'] = np.mean(corr_w < 0.0)
    return out


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--gibbs_input_summary_file', type=str)  # summary file of prepare_gtex_tissue_data_for_gibbs.py
parser.add_argument('--fold_file', type=str)
parser.add_argument('--held_out_fold', type=int)
parser.add_argument('--gibbs_results_stem', type=str)  # --output_stem of run_eqtl_mixture_prior_gibbs.py
parser.add_argument('--output_file', type=str)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.seed)
samples = pbs.load_posterior_beta_samples(args.gibbs_results_stem + '_posterior_beta_samples.npz')
genotype_index_to_fold = load_in_fold_file(args.fold_file)
chrom_to_file = load_in_gibbs_input_summary(args.gibbs_input_summary_file)
gene_ids = samples['gene_ids'].astype(str)
gene_chroms = samples['gene_chroms']
train_individual_ids = samples['train_individual_ids'].astype(str)
print(str(len(gene_ids)) + ' genes, ' + str(samples['n_samples']) + ' retained samples', flush=True)

# Open output file handle
columns = ['gene_id', 'chrom', 'fold', 'n_train', 'n_test', 'n_cis_variants', 'n_cis_variants_found', 'sum_pip', 'max_pip', 'expected_n_slab1', 'expected_n_slab2',
           'observed_r', 'predicted_expression_variance', 'expected_r_plugin', 'expected_r', 'expected_r_sd']
for level in coverage_levels:
    columns += ['expected_r_lb_' + str(level), 'expected_r_ub_' + str(level)]
columns += ['expected_cis_h2', 'expected_genetic_r', 'predicted_r_variance', 'genetic_FSR', 'expression_FSR',
            'posterior_cis_h2_train', 'posterior_cis_h2_train_lb', 'posterior_cis_h2_train_ub']
for split in ['train', 'test']:
    columns += ['cis_snp_h2_' + split, 'cis_snp_h2_pvalue_' + split, 'cis_snp_h2_he_' + split, 'cis_snp_h2_he_se_' + split, 'cis_snp_h2_he_pvalue_' + split]
# Written to a node-local temporary file and copied to --output_file at the end: writing in place on the shared file system left a block
# of NUL bytes in the middle of one output file (at the same offset in two runs)
t = tempfile.NamedTemporaryFile(mode='w', suffix='_held_out_evaluation.txt', delete=False)
t.write('\t'.join(columns) + '\n')

ptr = samples['gene_pair_ptr']
for chrom_num in np.unique(gene_chroms):
    gibbs_input = np.load(chrom_to_file[chrom_num])
    folds = np.asarray([genotype_index_to_fold[genotype_index][1] for genotype_index in gibbs_input['genotype_indices']])
    train = folds != args.held_out_fold
    test = folds == args.held_out_fold
    chrom_train_individual_ids = np.asarray([genotype_index_to_fold[genotype_index][0] for genotype_index in gibbs_input['genotype_indices'][train]])
    if np.array_equal(chrom_train_individual_ids, train_individual_ids) == False:
        raise ValueError('training individuals of the gibbs input and of the fit differ')
    genotype = gibbs_input['genotype']
    chrom_variant_ids = gibbs_input['variant_ids'].astype(str)
    variant_id_to_row = {variant_id: row for row, variant_id in enumerate(chrom_variant_ids)}
    chrom_gene_id_to_index = {gene_id: index for index, gene_id in enumerate(gibbs_input['gene_ids'].astype(str))}
    expression = gibbs_input['expression']
    chrom_gene_indices = np.flatnonzero(gene_chroms == chrom_num)
    print('chr' + str(chrom_num) + ': ' + str(len(chrom_gene_indices)) + ' genes, ' + str(np.sum(train)) + ' training and ' + str(np.sum(test)) + ' held-out individuals', flush=True)
    for g in chrom_gene_indices:
        gene_id = gene_ids[g]
        variant_id_arr = pbs.gene_variant_ids(samples, g)
        # Dosages of the gene's variants, aligned through the variant ids (the route used for any other genotype data)
        rows = np.asarray([variant_id_to_row[variant_id] for variant_id in variant_id_arr if variant_id in variant_id_to_row], dtype=int)
        dosage = np.transpose(genotype[rows, :]).astype(float)
        aligned_dosage, found = pbs.align_dosage_to_gene_variants(variant_id_arr, chrom_variant_ids[rows], dosage)
        y_all = expression[chrom_gene_id_to_index[gene_id]]
        y_test = y_all[test]
        y_train = y_all[train]

        # Observed and expected quantities in the held-out individuals
        U_test = pbs.compute_genetic_value_samples(samples, g, aligned_dosage[test], found)
        out = compute_expected_quantities(U_test, samples['sigma2_e_samples'][:, g].astype(float), y_test)

        # Model-based training-fold cis-h2
        U_train = pbs.compute_genetic_value_samples(samples, g, aligned_dosage[train], found)
        var_u_train = np.var(U_train, axis=1)
        out['posterior_cis_h2_train'] = np.mean(var_u_train)
        out['posterior_cis_h2_train_lb'] = np.quantile(var_u_train, 0.025)
        out['posterior_cis_h2_train_ub'] = np.quantile(var_u_train, 0.975)

        # Model-free cis-h2 in the training and held-out individuals
        for split, mask, y_split in [('train', train, y_train), ('test', test, y_test)]:
            grm = compute_cis_grm(aligned_dosage[mask][:, found])
            out['cis_snp_h2_' + split], out['cis_snp_h2_pvalue_' + split] = estimate_cis_snp_heritability_with_lrt(grm, y_split)
            out['cis_snp_h2_he_' + split], out['cis_snp_h2_he_se_' + split], out['cis_snp_h2_he_pvalue_' + split] = estimate_cis_snp_heritability_with_he_regression(grm, y_split)

        pip = samples['pair_pip'][ptr[g]:ptr[g + 1]]
        out['gene_id'] = gene_id
        out['chrom'] = chrom_num
        out['fold'] = args.held_out_fold
        out['n_train'] = np.sum(train)
        out['n_test'] = np.sum(test)
        out['n_cis_variants'] = len(variant_id_arr)
        out['n_cis_variants_found'] = np.sum(found)
        out['sum_pip'] = np.sum(pip)
        out['max_pip'] = np.max(pip)
        out['expected_n_slab1'] = np.sum(samples['pair_p_slab1'][ptr[g]:ptr[g + 1]])
        out['expected_n_slab2'] = np.sum(samples['pair_p_slab2'][ptr[g]:ptr[g + 1]])
        t.write('\t'.join(str(out[column]) for column in columns) + '\n')
    t.flush()
    gibbs_input.close()
t.close()
shutil.copyfile(t.name, args.output_file)
os.remove(t.name)

# Check the copy: no NUL bytes, one row per gene, every row with all columns
with open(args.output_file, 'rb') as f:
    content = f.read()
if b'\0' in content:
    raise ValueError('output file contains NUL bytes: ' + args.output_file)
lines = content.decode().rstrip('\n').split('\n')
if len(lines) != len(gene_ids) + 1:
    raise ValueError('output file has ' + str(len(lines) - 1) + ' rows for ' + str(len(gene_ids)) + ' genes: ' + args.output_file)
for line_num, line in enumerate(lines):
    if len(line.split('\t')) != len(columns):
        raise ValueError('line ' + str(line_num + 1) + ' of ' + args.output_file + ' does not have ' + str(len(columns)) + ' columns')
print('wrote and checked ' + str(len(gene_ids)) + ' rows', flush=True)
