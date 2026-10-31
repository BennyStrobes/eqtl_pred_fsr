import sys
import os
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eqtl_grid_prior_gibbs import EQTL_GRID_PRIOR_GIBBS, DEFAULT_GRID_VARIANCES
from eqtl_grid_prior_gwfm_tempered_gibbs import EQTL_GRID_PRIOR_GWFM_TEMPERED_GIBBS
from posterior_beta_samples import save_posterior_beta_samples

# Fit the fixed-grid mixture prior eQTL Gibbs sampler (eqtl_grid_prior_gibbs.py; README.md, section 5e) to one GTEx tissue, training on the
# individuals not in --held_out_fold. With --gwfm, the GWFM-style tempered Gibbs sampler of eqtl_grid_prior_gwfm_tempered_gibbs.py instead.
# Same inputs, training-fold variant filter and outputs as run_eqtl_mixture_prior_gibbs.py (whose data loaders are copied below), so that
# evaluate_held_out_expression_prediction.py and the organize / visualize scripts run unchanged: the slab 1 / slab 2 columns are the small /
# large groups of grid components, and the hyper-parameters pi0, pi1, pi2, sigma2_1, sigma2_2 the collapsed two-group mixture (see the top
# of eqtl_grid_prior_gibbs.py). Give each fit its own --output_stem. Additional outputs (<output_stem>_*):
#   grid_weight_trace.txt: weight of every grid component after every iteration (retained = 1 for retained samples)
#   grid_weight_summary.txt: per component, its variance, group, and posterior mean weight with 95% interval over the retained samples


def load_in_fold_file(fold_file):
    # genotype index (row of the plink .fam file) -> (individual id, fold)
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


def load_in_training_data(gibbs_input_summary_file, genotype_index_to_fold, held_out_fold, min_maf, chromosomes, max_genes):
    # Concatenates over chromosomes the training-fold inputs of EQTL_MIXTURE_PRIOR_GIBBS.fit and the per-gene / per-pair annotations
    G_list = []
    var_mean_list = []
    var_sd_list = []
    var_id_list = []
    gene_var_idx_list = []
    gene_n_vars = []
    Y_list = []
    gene_ids = []
    gene_chroms = []
    gene_tss = []
    n_variants_total = 0
    train_individual_ids = None
    f = open(gibbs_input_summary_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            continue
        chrom_num = int(data[0].replace('chr', ''))
        if chromosomes is not None and chrom_num not in chromosomes:
            continue
        if max_genes is not None and len(gene_ids) >= max_genes:
            break
        gibbs_input = np.load(data[1])
        folds = np.asarray([genotype_index_to_fold[genotype_index][1] for genotype_index in gibbs_input['genotype_indices']])
        train = folds != held_out_fold
        chrom_train_individual_ids = np.asarray([genotype_index_to_fold[genotype_index][0] for genotype_index in gibbs_input['genotype_indices'][train]])
        if train_individual_ids is None:
            train_individual_ids = chrom_train_individual_ids
        elif np.array_equal(train_individual_ids, chrom_train_individual_ids) == False:
            raise ValueError('training individuals differ between chromosomes')

        # Training-fold variant filter
        genotype_train = np.ascontiguousarray(gibbs_input['genotype'][:, train])
        af = np.mean(genotype_train, axis=1) / 2.0
        var_sd = np.std(genotype_train, axis=1)
        keep = (np.minimum(af, 1.0 - af) >= min_maf) & (var_sd > 0.0)
        new_index = np.full(len(keep), -1, dtype=np.int64)
        new_index[keep] = n_variants_total + np.arange(np.sum(keep))

        expression = gibbs_input['expression']
        chrom_gene_ids = gibbs_input['gene_ids'].astype(str)
        n_chrom_genes = 0
        for g in range(len(chrom_gene_ids)):
            rows = np.arange(gibbs_input['gene_variant_start'][g], gibbs_input['gene_variant_end'][g])
            rows = rows[keep[rows]]
            if len(rows) == 0:
                continue
            gene_var_idx_list.append(new_index[rows])
            gene_n_vars.append(len(rows))
            y = expression[g, train]
            Y_list.append(y - np.mean(y))
            gene_ids.append(chrom_gene_ids[g])
            gene_chroms.append(chrom_num)
            gene_tss.append(gibbs_input['gene_tss'][g])
            n_chrom_genes += 1
            if max_genes is not None and len(gene_ids) >= max_genes:
                break
        G_list.append(genotype_train[keep])
        var_mean_list.append(2.0 * af[keep])
        var_sd_list.append(var_sd[keep])
        var_id_list.append(gibbs_input['variant_ids'][keep])
        n_variants_total += int(np.sum(keep))
        print('chr' + str(chrom_num) + ': ' + str(n_chrom_genes) + ' genes, ' + str(np.sum(keep)) + ' of ' + str(len(keep)) + ' variants pass the training-fold filter, ' + str(np.sum(train)) + ' training individuals', flush=True)
        gibbs_input.close()
    f.close()
    gene_var_ptr = np.zeros(len(gene_ids) + 1, dtype=np.int64)
    gene_var_ptr[1:] = np.cumsum(gene_n_vars)
    data = {
        'G': np.ascontiguousarray(np.concatenate(G_list, axis=0)), 'var_mean': np.concatenate(var_mean_list), 'var_sd': np.concatenate(var_sd_list),
        'var_ids': np.concatenate(var_id_list), 'gene_var_ptr': gene_var_ptr, 'gene_var_idx': np.concatenate(gene_var_idx_list),
        'Y': np.ascontiguousarray(np.asarray(Y_list)), 'gene_ids': np.asarray(gene_ids), 'gene_chroms': np.asarray(gene_chroms), 'gene_tss': np.asarray(gene_tss),
        'train_individual_ids': train_individual_ids,
    }
    return data


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--gibbs_input_summary_file', type=str)  # summary file of prepare_gtex_tissue_data_for_gibbs.py
parser.add_argument('--fold_file', type=str)  # fold file of generate_cross_validation_folds.py
parser.add_argument('--held_out_fold', type=int)
parser.add_argument('--output_stem', type=str)
parser.add_argument('--n_iters', type=int, default=3000)
parser.add_argument('--burn_in', type=int, default=500)
parser.add_argument('--thin', type=int, default=5)
parser.add_argument('--grid_variances', type=float, nargs='+', default=list(DEFAULT_GRID_VARIANCES))  # fixed non-zero component variances
parser.add_argument('--small_large_cutoff', type=float, default=1e-3)  # components with variance below it are the "small" group
parser.add_argument('--dirichlet_alpha', type=float, default=0.1)  # Dirichlet(alpha, ..., alpha) prior on the component weights
parser.add_argument('--inverse_gamma_a', type=float, default=1e-10)  # Inverse-Gamma(a, b) prior on the residual variances
parser.add_argument('--inverse_gamma_b', type=float, default=1e-10)
parser.add_argument('--min_maf', type=float, default=0.01)  # training-fold minor allele frequency filter
parser.add_argument('--chromosomes', type=int, nargs='+', default=None)  # only these chromosomes (for testing)
parser.add_argument('--max_genes', type=int, default=None)  # only the first max_genes genes (for testing)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gwfm', action='store_true')  # fit with the GWFM-style tempered Gibbs sampler (eqtl_grid_prior_gwfm_tempered_gibbs.py)
parser.add_argument('--gwfm_interval', type=int, default=10)  # gwfm only: TGS blocks every this many iterations
parser.add_argument('--gwfm_r2_threshold', type=float, default=0.95)  # gwfm only: LD r^2 defining a variant's set
parser.add_argument('--gwfm_pip_threshold', type=float, default=0.05)  # gwfm only: running PIP a variant needs for its block
parser.add_argument('--gwfm_steps_per_snp', type=float, default=10.0)  # gwfm only: sub-steps per block, per member of the set
parser.add_argument('--gwfm_min_steps', type=int, default=20)  # gwfm only: minimum sub-steps per block
args = parser.parse_args()

print('settings: ' + ' '.join(key + '=' + str(value) for key, value in vars(args).items()), flush=True)

# Load in data
genotype_index_to_fold = load_in_fold_file(args.fold_file)
data = load_in_training_data(args.gibbs_input_summary_file, genotype_index_to_fold, args.held_out_fold, args.min_maf, args.chromosomes, args.max_genes)

# Fit
model_args = dict(n_iters=args.n_iters, burn_in=args.burn_in, thin=args.thin, dirichlet_alpha=args.dirichlet_alpha, inverse_gamma_a=args.inverse_gamma_a, inverse_gamma_b=args.inverse_gamma_b,
                  grid_variances=args.grid_variances, small_large_cutoff=args.small_large_cutoff, seed=args.seed)
if args.gwfm:
    model = EQTL_GRID_PRIOR_GWFM_TEMPERED_GIBBS(gwfm_interval=args.gwfm_interval, gwfm_r2_threshold=args.gwfm_r2_threshold, gwfm_pip_threshold=args.gwfm_pip_threshold, gwfm_steps_per_snp=args.gwfm_steps_per_snp, gwfm_min_steps=args.gwfm_min_steps, **model_args)
else:
    model = EQTL_GRID_PRIOR_GIBBS(**model_args)
model.fit(data['G'], data['var_mean'], data['var_sd'], data['gene_var_ptr'], data['gene_var_idx'], data['Y'])

# Sparse effect samples (one entry per variant-gene pair: a variant in several genes' windows appears once per gene)
gene_var_idx = data['gene_var_idx']
save_posterior_beta_samples(args.output_stem + '_posterior_beta_samples.npz', {
    'gene_ids': data['gene_ids'].astype('S'), 'gene_chroms': data['gene_chroms'], 'gene_tss': data['gene_tss'], 'gene_pair_ptr': data['gene_var_ptr'],
    'pair_variant_ids': data['var_ids'][gene_var_idx], 'pair_train_mean': data['var_mean'][gene_var_idx], 'pair_train_sd': data['var_sd'][gene_var_idx],
    'pair_pip': model.pip.astype(np.float32), 'pair_p_slab1': model.p_slab[:, 0].astype(np.float32), 'pair_p_slab2': model.p_slab[:, 1].astype(np.float32),
    'pair_posterior_mean': model.posterior_mean.astype(np.float32),
    'nz_pair': model.nz_pair, 'nz_sample': model.nz_sample, 'nz_beta': model.nz_beta,
    'sigma2_e_samples': model.sigma2_e_samples, 'n_samples': model.n_samples, 'train_individual_ids': data['train_individual_ids'].astype('S'),
})

# Per-gene summary
ptr = data['gene_var_ptr']
t = open(args.output_stem + '_gene_summary.txt', 'w')
t.write('gene_id\tchrom\ttss\tn_train\tn_cis_variants\tsum_pip\tmax_pip\texpected_n_slab1\texpected_n_slab2\tposterior_mean_sigma2_e\n')
for g, gene_id in enumerate(data['gene_ids']):
    pip = model.pip[ptr[g]:ptr[g + 1]]
    p_slab = model.p_slab[ptr[g]:ptr[g + 1]]
    t.write(gene_id + '\t' + str(data['gene_chroms'][g]) + '\t' + str(data['gene_tss'][g]) + '\t' + str(data['Y'].shape[1]) + '\t' + str(len(pip)) + '\t' + str(np.sum(pip)) + '\t' + str(np.max(pip)) + '\t' + str(np.sum(p_slab[:, 0])) + '\t' + str(np.sum(p_slab[:, 1])) + '\t' + str(np.mean(model.sigma2_e_samples[:, g])) + '\n')
t.close()

# Hyper-parameters: trace of every iteration, and posterior mean and 95% interval over the retained samples
trace = model.hyperparameter_trace
t = open(args.output_stem + '_hyperparameter_trace.txt', 'w')
t.write('iteration\tretained\t' + '\t'.join(model.hyperparameter_names) + '\n')
for itera in range(trace.shape[0]):
    t.write(str(itera) + '\t' + str(int(model.retained[itera])) + '\t' + '\t'.join(str(value) for value in trace[itera]) + '\n')
t.close()
retained_trace = trace[model.retained]
t = open(args.output_stem + '_hyperparameter_summary.txt', 'w')
t.write('hyperparameter\tposterior_mean\tposterior_95_lb\tposterior_95_ub\n')
for index, name in enumerate(model.hyperparameter_names):
    t.write(name + '\t' + str(np.mean(retained_trace[:, index])) + '\t' + str(np.quantile(retained_trace[:, index], 0.025)) + '\t' + str(np.quantile(retained_trace[:, index], 0.975)) + '\n')
t.close()
# Grid component weights: trace of every iteration, and posterior mean and 95% interval over the retained samples
trace = model.grid_weight_trace
t = open(args.output_stem + '_grid_weight_trace.txt', 'w')
t.write('iteration\tretained\t' + '\t'.join(model.grid_weight_names) + '\n')
for itera in range(trace.shape[0]):
    t.write(str(itera) + '\t' + str(int(model.retained[itera])) + '\t' + '\t'.join(str(value) for value in trace[itera]) + '\n')
t.close()
retained_trace = trace[model.retained]
t = open(args.output_stem + '_grid_weight_summary.txt', 'w')
t.write('component\tvariance\tgroup\tposterior_mean\tposterior_95_lb\tposterior_95_ub\n')
for index, name in enumerate(model.grid_weight_names):
    group = ['spike', 'small', 'large'][model.comp_group[index] + 1]
    t.write(name + '\t' + str(model.grid_var[index]) + '\t' + group + '\t' + str(np.mean(retained_trace[:, index])) + '\t' + str(np.quantile(retained_trace[:, index], 0.025)) + '\t' + str(np.quantile(retained_trace[:, index], 0.975)) + '\n')
t.close()
print(str(len(data['gene_ids'])) + ' genes written to ' + args.output_stem + '_*')
