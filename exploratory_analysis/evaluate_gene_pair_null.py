import sys
import os
import argparse
import shutil
import tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posterior_beta_samples as pbs

# Gene-pair null of the held-out correlation (README.md, section 5i), one line per gene. The prediction of gene A, y_hat_A (the mean over
# retained samples of u^(s), exactly as evaluate_held_out_expression_prediction.py), is correlated with the held-out expression of every gene B
# on another chromosome, whose cis-genetic and residual expression A's fit knows nothing about. If the held-out individuals were exchangeable
# (no structure shared by expression and genotypes, such as ancestry or a hidden factor), E[(n - 1) r_AB^2] = 1 exactly, for any y_hat_A and any
# distribution of y_B. The expected quantities of the evaluation assume exactly that of A's own residual; a dispersion above 1 means the noise
# part of A's own observed r is wider than the model allows, and expression FSR is over-confident by about that much.
#   null_dispersion: mean over partners B of (n_test - 1) r_AB^2
#   null_mean_r: mean over partners of r_AB (0 unless y_hat_A tracks a factor shared by many genes in one direction)
#   n_partners: genes B on other chromosomes with non-constant held-out expression
# Genes whose prediction is constant in the held-out individuals get nan.


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
    # Same as evaluate_held_out_expression_prediction.py: chromosome number -> gibbs input npz file
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


def standardize_rows(M):
    # Each row centered and scaled to mean square 1 (rows with no variance: nan)
    M = M - np.mean(M, axis=1)[:, None]
    sd = np.sqrt(np.mean(np.square(M), axis=1))
    out = np.full(M.shape, np.nan)
    out[sd > 0.0] = M[sd > 0.0] / sd[sd > 0.0][:, None]
    return out


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--gibbs_input_summary_file', type=str)  # summary file of prepare_gtex_tissue_data_for_gibbs.py
parser.add_argument('--fold_file', type=str)
parser.add_argument('--held_out_fold', type=int)
parser.add_argument('--gibbs_results_stem', type=str)  # --output_stem of run_eqtl_mixture_prior_gibbs.py or run_eqtl_grid_prior_gibbs.py
parser.add_argument('--output_file', type=str)
parser.add_argument('--chunk_size', type=int, default=1000)  # genes A per block of the correlation matrix
args = parser.parse_args()

samples = pbs.load_posterior_beta_samples(args.gibbs_results_stem + '_posterior_beta_samples.npz')
genotype_index_to_fold = load_in_fold_file(args.fold_file)
chrom_to_file = load_in_gibbs_input_summary(args.gibbs_input_summary_file)
gene_ids = samples['gene_ids'].astype(str)
gene_chroms = samples['gene_chroms'].astype(int)
train_individual_ids = samples['train_individual_ids'].astype(str)
ptr = samples['gene_pair_ptr']
n_genes = len(gene_ids)
# Mean over retained samples of every pair's effect (y_hat = X_test beta_mean)
beta_mean = np.bincount(samples['nz_pair'], weights=samples['nz_beta'].astype(np.float64), minlength=ptr[-1]) / samples['n_samples']
print(str(n_genes) + ' genes, ' + str(samples['n_samples']) + ' retained samples', flush=True)

# Held-out prediction and expression of every gene
y_hat = None
y_test = None
test_individual_ids = None
for chrom_num in np.unique(gene_chroms):
    gibbs_input = np.load(chrom_to_file[chrom_num])
    folds = np.asarray([genotype_index_to_fold[genotype_index][1] for genotype_index in gibbs_input['genotype_indices']])
    train = folds != args.held_out_fold
    test = folds == args.held_out_fold
    chrom_train_individual_ids = np.asarray([genotype_index_to_fold[genotype_index][0] for genotype_index in gibbs_input['genotype_indices'][train]])
    if np.array_equal(chrom_train_individual_ids, train_individual_ids) == False:
        raise ValueError('training individuals of the gibbs input and of the fit differ')
    chrom_test_individual_ids = np.asarray([genotype_index_to_fold[genotype_index][0] for genotype_index in gibbs_input['genotype_indices'][test]])
    if test_individual_ids is None:
        test_individual_ids = chrom_test_individual_ids
        y_hat = np.full((n_genes, len(test_individual_ids)), np.nan)
        y_test = np.full((n_genes, len(test_individual_ids)), np.nan)
    elif np.array_equal(chrom_test_individual_ids, test_individual_ids) == False:
        raise ValueError('held-out individuals differ between chromosomes')
    genotype = gibbs_input['genotype']
    chrom_variant_ids = gibbs_input['variant_ids'].astype(str)
    variant_id_to_row = {variant_id: row for row, variant_id in enumerate(chrom_variant_ids)}
    chrom_gene_id_to_index = {gene_id: index for index, gene_id in enumerate(gibbs_input['gene_ids'].astype(str))}
    expression = gibbs_input['expression']
    chrom_gene_indices = np.flatnonzero(gene_chroms == chrom_num)
    for g in chrom_gene_indices:
        variant_id_arr = pbs.gene_variant_ids(samples, g)
        # Dosages of the gene's variants, aligned through the variant ids (as evaluate_held_out_expression_prediction.py)
        rows = np.asarray([variant_id_to_row[variant_id] for variant_id in variant_id_arr if variant_id in variant_id_to_row], dtype=int)
        dosage = np.transpose(genotype[rows][:, test]).astype(float)
        aligned_dosage, found = pbs.align_dosage_to_gene_variants(variant_id_arr, chrom_variant_ids[rows], dosage)
        X = pbs.standardize_aligned_dosage(samples, g, aligned_dosage, found)
        y_hat[g] = np.dot(X, beta_mean[ptr[g]:ptr[g + 1]])
        y_test[g] = expression[chrom_gene_id_to_index[gene_ids[g]]][test]
    gibbs_input.close()
    print('chr' + str(chrom_num) + ': ' + str(len(chrom_gene_indices)) + ' genes, ' + str(len(test_individual_ids)) + ' held-out individuals', flush=True)

# Correlation of every prediction with the expression of every gene on another chromosome, in blocks of genes A
n_test = len(test_individual_ids)
Z_hat = standardize_rows(y_hat)
Z_y = standardize_rows(y_test)
valid_b = np.all(np.isfinite(Z_y), axis=1)
dispersion = np.full(n_genes, np.nan)
mean_r = np.full(n_genes, np.nan)
n_partners = np.zeros(n_genes, dtype=int)
for lo in range(0, n_genes, args.chunk_size):
    hi = min(lo + args.chunk_size, n_genes)
    r = np.dot(np.nan_to_num(Z_hat[lo:hi]), np.transpose(np.nan_to_num(Z_y))) / n_test
    partner = (gene_chroms[lo:hi][:, None] != gene_chroms[None, :]) & valid_b[None, :]
    n_partners[lo:hi] = np.sum(partner, axis=1)
    dispersion[lo:hi] = np.sum(np.where(partner, (n_test - 1.0) * np.square(r), 0.0), axis=1) / n_partners[lo:hi]
    mean_r[lo:hi] = np.sum(np.where(partner, r, 0.0), axis=1) / n_partners[lo:hi]
constant = np.any(np.isnan(Z_hat), axis=1)
dispersion[constant] = np.nan
mean_r[constant] = np.nan
print('mean null dispersion over genes: ' + str(np.nanmean(dispersion)), flush=True)

# Written to a node-local temporary file and copied to --output_file at the end (as evaluate_held_out_expression_prediction.py)
columns = ['gene_id', 'chrom', 'tss', 'fold', 'n_test', 'n_cis_variants', 'n_partners', 'null_dispersion', 'null_mean_r']
t = tempfile.NamedTemporaryFile(mode='w', suffix='_gene_pair_null.txt', delete=False)
t.write('\t'.join(columns) + '\n')
for g in range(n_genes):
    t.write('\t'.join(str(value) for value in [gene_ids[g], gene_chroms[g], samples['gene_tss'][g], args.held_out_fold, n_test, ptr[g + 1] - ptr[g], n_partners[g], dispersion[g], mean_r[g]]) + '\n')
t.close()
shutil.copyfile(t.name, args.output_file)
os.remove(t.name)
print('wrote ' + str(n_genes) + ' rows to ' + args.output_file, flush=True)
