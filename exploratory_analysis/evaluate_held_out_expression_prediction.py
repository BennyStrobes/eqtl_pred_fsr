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
#   y_hat_mc_variance: Monte Carlo variance of y_hat (mean over individuals of the batch-means variance of y_hat, batches of --block_size
#       consecutive retained samples). When y_hat and the predictive draws below come from the same samples, the mean over s of
#       Cov_i(y_hat, u^(s)) is V, about y_hat_mc_variance more than for an independent draw, so expected_r is biased up (expression_FSR down)
#       by roughly y_hat_mc_variance / V in relative terms
# With --split_samples, y_hat (and observed_r) uses the even blocks of --block_size consecutive retained samples and every expected quantity
# uses the odd blocks, so the predictive draws are (up to autocorrelation across block boundaries) independent of the Monte Carlo error of y_hat
#   expected_r, expected_r_sd, expected_r_{lb,ub}_{80,90,95,99}: mean, standard deviation and quantile intervals over s of corr(y_hat, u^(s) + eps^(s)),
#       eps^(s) ~ N(0, sigma2_e^(s) I) (posterior predictive distribution of the observed correlation)
#   expected_cis_h2: mean_s Var_i(u^(s)); expected_genetic_r: mean_s corr(y_hat, u^(s)) (0 for samples with u^(s) constant)
#   predicted_r_variance: Var_s[Cov_i(y_hat, u^(s)) / sqrt(V)]
#   genetic_FSR: mean_s 1[Cov_i(y_hat, u^(s)) < 0]; expression_FSR: mean_s 1[corr(y_hat, u^(s) + eps^(s)) < 0]
#   posterior_cis_h2_train (_lb, _ub): mean (95% interval) over s of Var_i(X_train beta^(s)) in the training individuals
#   expected_cis_h2_independent, posterior_cis_h2_train_independent (_lb, _ub): expected_cis_h2 and posterior_cis_h2_train with the correlations
#       between variants set to 0, sum_j Var_i(x_j) (beta_j^(s))^2 in place of Var_i(X beta^(s)), over the same samples; the difference is the
#       part of the model's cis-h2 that comes from LD between the non-zero effects of a sample
#   cis_snp_h2 (ML, LRT p-value) and cis_snp_h2_he (Haseman-Elston, standard error, p-value) in the training (_train) and held-out (_test)
#       individuals, from genotypes and expression only (no model)
#   var_y (_train, _test): variance of the expression in those individuals; cis_snp_h2_he_var (_train, _test): cis_snp_h2_he times var_y, the HE
#       estimate of Var(genetic expression). cis_snp_h2_he is a proportion of var_y, while expected_cis_h2 and posterior_cis_h2_train are variances,
#       so compare these with cis_snp_h2_he_var
#   expected_cis_snp_h2_he_var_test: what the model predicts cis_snp_h2_he_var_test to be: mean_s of the expected HE slope given u^(s). HE is not an
#       estimate of Var(u) for a given set of effects: it weights an effect by the LD of its variant relative to the window (in the population, the
#       slope is m beta' R^2 beta / tr(R^2)). With y = u + iid noise, E[y' W y] = u' W u exactly (W, compute_he_weights, has a zero diagonal and sums
#       to 0, so neither the noise nor centering y adds to it), computed here with the W of the held-out genotypes, over the samples of expected_cis_h2.
#       If it matches cis_snp_h2_he_var_test where expected_cis_h2 does not, the difference between HE and expected_cis_h2 is how HE weights LD
# Genes whose prediction is constant in the held-out individuals (V = 0) get nan for the quantities that need V > 0.
#
# External test set (--test_expression_file, --test_genotype_sample_mapping_file, --plink_stem; README.md, section 7b): the same quantities with the
# held-out fold replaced by other individuals, e.g. the African-ancestry GTEx individuals of the tissue. The fit, its training individuals and the
# _train columns are unchanged. The test dosages are read from the plink files (the step 1 input holds only the tissue's EUR individuals), and
# standardized with the training (EUR) mean and standard deviation like any held-out dosage; a test missing call is set to the training mean.
# Their GRM, for the model-free _test cis-h2, uses the variants with no missing test call, standardized within the test individuals. The residual
# variance sigma2_e^(s) of the fit is carried over unchanged. Genes without test expression are left out (one row per gene with test expression).
# The fold column is still --held_out_fold (the fit used).

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


def load_in_test_expression(expression_file, n_test_individuals):
    # External test set: residualized, renormalized expression bed file (chrom, start, tss, gene_id, then one column per individual, in the order
    # of the test genotype sample mapping file). Returns dictionary gene_id (no version) -> expression vector
    gene_id_to_expression = {}
    f = open(expression_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            if len(data) - 4 != n_test_individuals:
                raise ValueError(expression_file + ' has ' + str(len(data) - 4) + ' individuals and the test genotype sample mapping ' + str(n_test_individuals))
            continue
        gene_id = data[3].split('.')[0]
        if gene_id in gene_id_to_expression:
            raise ValueError('repeated gene id in ' + expression_file + ': ' + gene_id)
        gene_id_to_expression[gene_id] = np.asarray(data[4:]).astype(float)
    f.close()
    return gene_id_to_expression


def load_in_plink_alt_dosage(plink_stem, chrom_num, variant_ids, genotype_indices_list, chunk_size=20000):
    # Alt-allele dosage (nan for a missing call) of the given GTEx variant ids (chr_pos_ref_alt_b38, the ids of the step 1 input) in the individuals
    # of each array of plink .fam rows in genotype_indices_list. Same convention as prepare_gtex_tissue_data_for_gibbs.py: GTEx bim a1 = ref and
    # pandas_plink G counts a1, so the alt-allele dosage is 2 - G.
    # Returns the ids found in the bim (bim order) and one n_individuals x n_found matrix per entry of genotype_indices_list
    from pandas_plink import read_plink
    (bim, fam, G) = read_plink(plink_stem + str(chrom_num), verbose=False)
    bim_snp_ids = np.asarray(bim['snp']).astype(str)
    rows = np.flatnonzero(np.isin(bim_snp_ids, np.asarray(variant_ids)))
    if len(np.unique(bim_snp_ids[rows])) != len(rows):
        raise ValueError('repeated variant ids in ' + plink_stem + str(chrom_num) + '.bim')
    dosage_chunks = [[] for indices in genotype_indices_list]
    for chunk_start in range(0, len(rows), chunk_size):
        geno_mat = G[rows[chunk_start:chunk_start + chunk_size], :].compute()
        for k, indices in enumerate(genotype_indices_list):
            dosage_chunks[k].append(2.0 - geno_mat[:, indices])
    dosages = [np.transpose(np.concatenate(chunks, axis=0)) if len(chunks) > 0 else np.zeros((len(indices), 0)) for chunks, indices in zip(dosage_chunks, genotype_indices_list)]
    return bim_snp_ids[rows], dosages


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


def compute_he_weights(grm):
    # W_ik = (GRM_ik - mean off-diagonal GRM) / sum_{i != k} (GRM_ik - mean)^2 for i != k, and 0 on the diagonal, so that y^T W y is the OLS slope
    # (with intercept) of y_i y_k on GRM_ik over pairs i != k. None if the GRM has no off-diagonal variation
    n = grm.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    W = grm - np.mean(grm[off_diag])
    W[~off_diag] = 0.0
    sxx = np.sum(np.square(W))
    if sxx <= 0.0:
        return None
    return W / sxx


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

    W = compute_he_weights(grm)
    if W is None:
        return np.nan, np.nan, np.nan
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


def y_hat_mc_variance(U, block):
    # Batch-means Monte Carlo variance of y_hat = mean of the rows of U, averaged over individuals; block: batch of each row of U
    block_ids = np.unique(block)
    if len(block_ids) < 2:
        return np.nan
    block_means = np.asarray([np.mean(U[block == b], axis=0) for b in block_ids])
    block_means = block_means - np.mean(block_means, axis=1)[:, None]  # constant shifts do not change a correlation
    return np.mean(np.var(block_means, axis=0, ddof=1)) / len(block_ids)


def compute_expected_quantities(U, sigma2_e_samples, y, block, split_samples):
    # U: n_samples x n_test genetic expression samples; sigma2_e_samples: n_samples residual variances; y: held-out expression; block: batch
    # of --block_size consecutive retained samples of each sample; split_samples: y_hat from the even blocks, expected quantities from the odd
    # Returns dictionary of the observed and expected quantities (see top of file)
    n_test = U.shape[1]
    out = {}
    if split_samples:
        fit = block % 2 == 0
        y_hat = np.mean(U[fit], axis=0)
        out['y_hat_mc_variance'] = y_hat_mc_variance(U[fit], block[fit])
        U = U[fit == False]
        sigma2_e_samples = sigma2_e_samples[fit == False]
    else:
        y_hat = np.mean(U, axis=0)
        out['y_hat_mc_variance'] = y_hat_mc_variance(U, block)
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
# Consecutive retained samples per batch (y_hat_mc_variance, --split_samples). Must be well above the autocorrelation time: on exact posteriors
# drawn as an AR(1) chain with correlation 0.95 between retained samples, --split_samples with blocks of 100 removed only half of the bias of
# expression_FSR, with blocks of 250 all of it
parser.add_argument('--block_size', type=int, default=250)
parser.add_argument('--split_samples', action='store_true')  # y_hat from the even blocks, expected quantities from the odd blocks
# External test set in place of the held-out fold (see top of file); all three or none
parser.add_argument('--test_expression_file', type=str, default=None)  # residualized, renormalized expression bed file of the test individuals
parser.add_argument('--test_genotype_sample_mapping_file', type=str, default=None)  # plink .fam row of each test expression column, in order
parser.add_argument('--plink_stem', type=str, default=None)  # chromosome-specific plink files: <plink_stem><chrom_num>.{bed,bim,fam}
args = parser.parse_args()
external_test = args.test_expression_file is not None
if external_test and (args.test_genotype_sample_mapping_file is None or args.plink_stem is None):
    raise ValueError('--test_expression_file needs --test_genotype_sample_mapping_file and --plink_stem')

np.random.seed(args.seed)
samples = pbs.load_posterior_beta_samples(args.gibbs_results_stem + '_posterior_beta_samples.npz')
sample_block = np.arange(samples['n_samples']) // args.block_size
if args.split_samples:
    print('split samples: y_hat from ' + str(np.sum(sample_block % 2 == 0)) + ' and expected quantities from ' + str(np.sum(sample_block % 2 == 1)) + ' retained samples (blocks of ' + str(args.block_size) + ')', flush=True)
genotype_index_to_fold = load_in_fold_file(args.fold_file)
chrom_to_file = load_in_gibbs_input_summary(args.gibbs_input_summary_file)
gene_ids = samples['gene_ids'].astype(str)
gene_chroms = samples['gene_chroms']
train_individual_ids = samples['train_individual_ids'].astype(str)
print(str(len(gene_ids)) + ' genes, ' + str(samples['n_samples']) + ' retained samples', flush=True)
if external_test:
    test_genotype_indices = np.atleast_1d(np.loadtxt(args.test_genotype_sample_mapping_file)).astype(int)
    test_gene_id_to_expression = load_in_test_expression(args.test_expression_file, len(test_genotype_indices))
    print('external test set: ' + str(len(test_genotype_indices)) + ' individuals, ' + str(len(test_gene_id_to_expression)) + ' genes with expression, ' +
          str(np.sum(np.isin(gene_ids, list(test_gene_id_to_expression.keys())))) + ' of the ' + str(len(gene_ids)) + ' fitted genes', flush=True)

# Open output file handle
columns = ['gene_id', 'chrom', 'fold', 'n_train', 'n_test', 'n_cis_variants', 'n_cis_variants_found', 'sum_pip', 'max_pip', 'expected_n_slab1', 'expected_n_slab2',
           'observed_r', 'predicted_expression_variance', 'y_hat_mc_variance', 'expected_r_plugin', 'expected_r', 'expected_r_sd']
for level in coverage_levels:
    columns += ['expected_r_lb_' + str(level), 'expected_r_ub_' + str(level)]
columns += ['expected_cis_h2', 'expected_genetic_r', 'predicted_r_variance', 'genetic_FSR', 'expression_FSR',
            'posterior_cis_h2_train', 'posterior_cis_h2_train_lb', 'posterior_cis_h2_train_ub',
            'expected_cis_h2_independent', 'posterior_cis_h2_train_independent', 'posterior_cis_h2_train_independent_lb', 'posterior_cis_h2_train_independent_ub']
for split in ['train', 'test']:
    columns += ['cis_snp_h2_' + split, 'cis_snp_h2_pvalue_' + split, 'cis_snp_h2_he_' + split, 'cis_snp_h2_he_se_' + split, 'cis_snp_h2_he_pvalue_' + split,
                'var_y_' + split, 'cis_snp_h2_he_var_' + split]
columns += ['expected_cis_snp_h2_he_var_test']
# Written to a node-local temporary file and copied to --output_file at the end: writing in place on the shared file system left a block
# of NUL bytes in the middle of one output file (at the same offset in two runs)
t = tempfile.NamedTemporaryFile(mode='w', suffix='_held_out_evaluation.txt', delete=False)
t.write('\t'.join(columns) + '\n')

ptr = samples['gene_pair_ptr']
n_genes_written = 0
n_genes_skipped = 0  # external test set: fitted genes without test expression
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
    if external_test:
        if len(np.intersect1d(test_genotype_indices, gibbs_input['genotype_indices'])) > 0:
            raise ValueError('test individuals overlap the individuals of the gibbs input')
        # Test dosages of the fit's variants on this chromosome. The same variants are also read for the gibbs input individuals and checked
        # against the gibbs input genotype, which checks the allele orientation and the plink rows
        chrom_model_variant_ids = np.unique(np.concatenate([pbs.gene_variant_ids(samples, g) for g in chrom_gene_indices]))
        test_variant_ids, (test_chrom_dosage, gibbs_input_chrom_dosage) = load_in_plink_alt_dosage(args.plink_stem, chrom_num, chrom_model_variant_ids, [test_genotype_indices, gibbs_input['genotype_indices']])
        gibbs_input_rows = np.asarray([variant_id_to_row[variant_id] for variant_id in test_variant_ids], dtype=int)
        if np.array_equal(gibbs_input_chrom_dosage, np.transpose(genotype[gibbs_input_rows, :]).astype(float)) == False:
            raise ValueError('chr' + str(chrom_num) + ': plink dosages of the gibbs input individuals differ from the gibbs input genotype')
        del gibbs_input_chrom_dosage
        test_variant_id_to_col = {variant_id: col for col, variant_id in enumerate(test_variant_ids)}
        print('chr' + str(chrom_num) + ': ' + str(len(test_variant_ids)) + ' of ' + str(len(chrom_model_variant_ids)) + ' fitted variants in the plink files, test missing call rate ' +
              '%.2g' % (np.mean(np.isnan(test_chrom_dosage)) if test_chrom_dosage.size > 0 else 0.0), flush=True)
    n_test_print = len(test_genotype_indices) if external_test else np.sum(test)
    print('chr' + str(chrom_num) + ': ' + str(len(chrom_gene_indices)) + ' genes, ' + str(np.sum(train)) + ' training and ' + str(n_test_print) + (' external test' if external_test else ' held-out') + ' individuals', flush=True)
    for g in chrom_gene_indices:
        gene_id = gene_ids[g]
        if external_test and gene_id not in test_gene_id_to_expression:
            n_genes_skipped += 1
            continue
        variant_id_arr = pbs.gene_variant_ids(samples, g)
        # Dosages of the gene's variants, aligned through the variant ids (the route used for any other genotype data)
        rows = np.asarray([variant_id_to_row[variant_id] for variant_id in variant_id_arr if variant_id in variant_id_to_row], dtype=int)
        dosage = np.transpose(genotype[rows, :]).astype(float)
        aligned_dosage, found = pbs.align_dosage_to_gene_variants(variant_id_arr, chrom_variant_ids[rows], dosage)
        y_all = expression[chrom_gene_id_to_index[gene_id]]
        y_train = y_all[train]
        train_dosage = aligned_dosage[train]
        if external_test:
            y_test = test_gene_id_to_expression[gene_id]
            cols = np.asarray([test_variant_id_to_col[variant_id] for variant_id in variant_id_arr if variant_id in test_variant_id_to_col], dtype=int)
            test_dosage, test_found = pbs.align_dosage_to_gene_variants(variant_id_arr, test_variant_ids[cols], test_chrom_dosage[:, cols])
        else:
            y_test = y_all[test]
            test_dosage, test_found = aligned_dosage[test], found
        n_test = len(y_test)
        # Variants of the test GRM: found, with no missing test call (all found variants for the held-out fold)
        test_complete = test_found & (np.sum(np.isnan(test_dosage), axis=0) == 0)

        # Observed and expected quantities in the held-out (or external test) individuals
        U_test = pbs.compute_genetic_value_samples(samples, g, test_dosage, test_found)
        out = compute_expected_quantities(U_test, samples['sigma2_e_samples'][:, g].astype(float), y_test, sample_block, args.split_samples)

        # Model-based training-fold cis-h2
        U_train = pbs.compute_genetic_value_samples(samples, g, train_dosage, found)
        var_u_train = np.var(U_train, axis=1)
        out['posterior_cis_h2_train'] = np.mean(var_u_train)
        out['posterior_cis_h2_train_lb'] = np.quantile(var_u_train, 0.025)
        out['posterior_cis_h2_train_ub'] = np.quantile(var_u_train, 0.975)

        # The same with the correlations between variants set to 0, over the samples of expected_cis_h2 and of posterior_cis_h2_train
        expected_samples = sample_block % 2 == 1 if args.split_samples else np.ones(samples['n_samples'], dtype=bool)
        out['expected_cis_h2_independent'] = np.mean(pbs.compute_independent_snp_h2_samples(samples, g, test_dosage, test_found)[expected_samples])
        independent_h2_train = pbs.compute_independent_snp_h2_samples(samples, g, train_dosage, found)
        out['posterior_cis_h2_train_independent'] = np.mean(independent_h2_train)
        out['posterior_cis_h2_train_independent_lb'] = np.quantile(independent_h2_train, 0.025)
        out['posterior_cis_h2_train_independent_ub'] = np.quantile(independent_h2_train, 0.975)

        # Model-free cis-h2 in the training and held-out (or external test) individuals
        for split, split_dosage, split_variants, y_split in [('train', train_dosage, found, y_train), ('test', test_dosage, test_complete, y_test)]:
            grm = compute_cis_grm(split_dosage[:, split_variants])
            out['cis_snp_h2_' + split], out['cis_snp_h2_pvalue_' + split] = estimate_cis_snp_heritability_with_lrt(grm, y_split)
            out['cis_snp_h2_he_' + split], out['cis_snp_h2_he_se_' + split], out['cis_snp_h2_he_pvalue_' + split] = estimate_cis_snp_heritability_with_he_regression(grm, y_split)
            out['var_y_' + split] = np.var(y_split)  # the var_y that the HE estimate divides by
            out['cis_snp_h2_he_var_' + split] = out['cis_snp_h2_he_' + split] * out['var_y_' + split]

        # What the model predicts the held-out HE slope to be (see top of file); grm is the held-out GRM
        he_weights = compute_he_weights(grm) if grm is not None and n_test >= 3 else None
        out['expected_cis_snp_h2_he_var_test'] = np.nan
        if he_weights is not None:
            U_expected = U_test[expected_samples] - np.mean(U_test[expected_samples], axis=1)[:, None]
            out['expected_cis_snp_h2_he_var_test'] = np.mean(np.sum(np.dot(U_expected, he_weights) * U_expected, axis=1))

        pip = samples['pair_pip'][ptr[g]:ptr[g + 1]]
        out['gene_id'] = gene_id
        out['chrom'] = chrom_num
        out['fold'] = args.held_out_fold
        out['n_train'] = np.sum(train)
        out['n_test'] = n_test
        out['n_cis_variants'] = len(variant_id_arr)
        out['n_cis_variants_found'] = np.sum(test_found)
        out['sum_pip'] = np.sum(pip)
        out['max_pip'] = np.max(pip)
        out['expected_n_slab1'] = np.sum(samples['pair_p_slab1'][ptr[g]:ptr[g + 1]])
        out['expected_n_slab2'] = np.sum(samples['pair_p_slab2'][ptr[g]:ptr[g + 1]])
        t.write('\t'.join(str(out[column]) for column in columns) + '\n')
        n_genes_written += 1
    t.flush()
    gibbs_input.close()
    if external_test:
        del test_chrom_dosage
t.close()
shutil.copyfile(t.name, args.output_file)
os.remove(t.name)

# Check the copy: no NUL bytes, one row per gene, every row with all columns
with open(args.output_file, 'rb') as f:
    content = f.read()
if b'\0' in content:
    raise ValueError('output file contains NUL bytes: ' + args.output_file)
lines = content.decode().rstrip('\n').split('\n')
if n_genes_written + n_genes_skipped != len(gene_ids) or (external_test == False and n_genes_skipped > 0):
    raise ValueError(str(n_genes_written) + ' genes written and ' + str(n_genes_skipped) + ' left out of ' + str(len(gene_ids)))
if len(lines) != n_genes_written + 1:
    raise ValueError('output file has ' + str(len(lines) - 1) + ' rows for ' + str(n_genes_written) + ' genes: ' + args.output_file)
for line_num, line in enumerate(lines):
    if len(line.split('\t')) != len(columns):
        raise ValueError('line ' + str(line_num + 1) + ' of ' + args.output_file + ' does not have ' + str(len(columns)) + ' columns')
print('wrote and checked ' + str(n_genes_written) + ' rows' + (' (' + str(n_genes_skipped) + ' fitted genes without test expression left out)' if external_test else ''), flush=True)
