import numpy as np


###############################################################################
# Sparse posterior samples of causal eQTL effects (written by run_eqtl_mixture_prior_gibbs.py), and their application to genotype data
# (README.md, section 6). One .npz file per tissue and fold, holding:
#   gene_ids, gene_chroms, gene_tss                       per gene
#   gene_pair_ptr                                         the variants of gene g are pairs gene_pair_ptr[g]:gene_pair_ptr[g + 1]
#   pair_variant_ids                                      GTEx variant id chr_pos_ref_alt_b38; effects are of the alt allele
#   pair_train_mean, pair_train_sd                        training-fold mean and standard deviation (1/n) of the alt-allele dosage
#   pair_pip, pair_p_slab1, pair_p_slab2, pair_posterior_mean
#   nz_pair, nz_sample, nz_beta                           non-zero effect samples sorted by pair then sample: the effect of pair nz_pair[i]
#                                                         in retained sample nz_sample[i] is nz_beta[i]; every other entry is exactly 0
#   sigma2_e_samples                                      residual variance of each gene in each retained sample (n_samples x n_genes)
#   n_samples, train_individual_ids
# Effects are per training standard deviation of the alt-allele dosage, on the scale of the tissue's residualized, renormalized expression;
# per-allele effects are beta / pair_train_sd.
###############################################################################


def save_posterior_beta_samples(output_file, arrays):
    # arrays: dictionary of the fields above
    np.savez(output_file, **arrays)


def load_posterior_beta_samples(samples_file):
    # Returns a dictionary of the fields above, plus nz_gene_ptr: the non-zero samples of gene g are nz_gene_ptr[g]:nz_gene_ptr[g + 1]
    loaded = np.load(samples_file)
    samples = {key: loaded[key] for key in loaded.files}
    loaded.close()
    samples['n_samples'] = int(samples['n_samples'])
    samples['nz_gene_ptr'] = np.searchsorted(samples['nz_pair'], samples['gene_pair_ptr'])
    return samples


def gene_variant_ids(samples, gene_index):
    ptr = samples['gene_pair_ptr']
    return samples['pair_variant_ids'][ptr[gene_index]:ptr[gene_index + 1]].astype(str)


def gene_beta_sample_matrix(samples, gene_index):
    # Dense effect samples of one gene: n_samples x p_g
    ptr = samples['gene_pair_ptr']
    p = ptr[gene_index + 1] - ptr[gene_index]
    beta_mat = np.zeros((samples['n_samples'], p))
    lo = samples['nz_gene_ptr'][gene_index]
    hi = samples['nz_gene_ptr'][gene_index + 1]
    beta_mat[samples['nz_sample'][lo:hi], samples['nz_pair'][lo:hi] - ptr[gene_index]] = samples['nz_beta'][lo:hi]
    return beta_mat


def variant_key(variant_id):
    # chr_pos_a1_a2 from a variant id chr_pos_a1_a2[_anything], with a 'chr' prefix whether or not the id has one
    info = variant_id.split('_')
    chrom = info[0]
    if chrom.startswith('chr') == False:
        chrom = 'chr' + chrom
    return chrom + '_' + info[1] + '_' + info[2] + '_' + info[3]


def align_dosage_to_gene_variants(gene_variant_id_arr, new_variant_ids, new_dosage):
    # new_dosage (n_individuals x len(new_variant_ids)): dosage of the last allele of each new variant id (chr_pos_a1_a2[_...])
    # Returns the alt-allele dosage of each of the gene's variants (n_individuals x p_g; nan for variants not found) and a boolean vector of
    # found variants. A new variant listed with the alleles swapped (chr_pos_alt_ref) is matched and its dosage flipped (2 - dosage).
    new_key_to_index = {}
    for index, new_variant_id in enumerate(new_variant_ids):
        new_key_to_index[variant_key(new_variant_id)] = index
    n_individuals = new_dosage.shape[0]
    aligned = np.full((n_individuals, len(gene_variant_id_arr)), np.nan)
    found = np.zeros(len(gene_variant_id_arr), dtype=bool)
    for jj, gene_variant_id in enumerate(gene_variant_id_arr):
        key = variant_key(gene_variant_id)
        info = key.split('_')
        swapped_key = info[0] + '_' + info[1] + '_' + info[3] + '_' + info[2]
        if key in new_key_to_index:
            aligned[:, jj] = new_dosage[:, new_key_to_index[key]]
            found[jj] = True
        elif swapped_key in new_key_to_index:
            aligned[:, jj] = 2.0 - new_dosage[:, new_key_to_index[swapped_key]]
            found[jj] = True
    return aligned, found


def standardize_aligned_dosage(samples, gene_index, aligned_dosage, found):
    # Alt-allele dosages standardized with the training mean and standard deviation; variants not found (or missing calls) are set to the
    # training mean (standardized value 0), so they contribute nothing
    ptr = samples['gene_pair_ptr']
    train_mean = samples['pair_train_mean'][ptr[gene_index]:ptr[gene_index + 1]]
    train_sd = samples['pair_train_sd'][ptr[gene_index]:ptr[gene_index + 1]]
    X = (aligned_dosage - train_mean[None, :]) / train_sd[None, :]
    X[:, found == False] = 0.0
    X[np.isnan(X)] = 0.0
    return X


def compute_genetic_value_samples(samples, gene_index, aligned_dosage, found):
    # Genetic expression of each individual under each retained effect sample: n_samples x n_individuals (u^(s) = X beta^(s))
    X = standardize_aligned_dosage(samples, gene_index, aligned_dosage, found)
    return np.dot(gene_beta_sample_matrix(samples, gene_index), np.transpose(X))
