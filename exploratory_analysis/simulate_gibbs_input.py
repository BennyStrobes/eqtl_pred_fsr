import argparse
import numpy as np
from scipy import stats

# Simulated input for the whole pipeline (README.md, section 5j): the same files as prepare_gtex_tissue_data_for_gibbs.py (one .npz per
# chromosome and a summary file) and generate_cross_validation_folds.py (fold file), plus the true effects, so that the fits, evaluations and
# checks run unchanged. Genotypes: per chromosome, variant positions with a density that varies between 2 Mb regions (so the number of cis
# variants varies between genes; 500 to 4,500 variants per Mb), MAF log-uniform on [--min_maf, 0.5], and haplotypes from a latent Gaussian AR(1) process along the variants
# (correlation --ld_rho between neighbours), thresholded at each variant's frequency; dosage = sum of two haplotypes.
# Effects, per variant-gene pair (on dosages standardized in all individuals; expression = X beta + N(0, 1)):
#   beta_gj = 0 with probability 1 - q_gj, otherwise N(0, sigma2) with sigma2 drawn from --effect_variances / --effect_weights
# --scenario sets q_gj (and a scale of sigma2):
#   shared:       q = --q
#   maf:          q by MAF bin (0.01-0.02, 0.02-0.05, 0.05-0.1, 0.1-0.2, 0.2-0.5) = --q x (0.3, 0.5, 1, 1.4, 1.8), and sigma2 x 0.5 below MAF 0.05
#   tss_distance: q = --q x (0.4 + 6 exp(-|position - TSS| / 5 kb))
#   gene_classes: genes drawn from three classes: 40% null (q = 0), 45% sparse (q = --q), 15% polygenic (q = 4 --q, sigma2 x 0.15)
# Outputs (in --output_dir): chr<c>.npz, gibbs_input_summary.txt, folds.txt (fold = individual index mod --n_folds + 1) and truth.npz
# (per gene: id, chromosome, class (gene_classes; else 0); beta_true of every pair, the gene's cis variants in order, concatenated over genes
# with gene_pair_ptr; the mean and sd of every variant's dosage in all individuals, by chromosome and row, used for the standardization).


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--scenario', type=str, default='shared', choices=['shared', 'maf', 'tss_distance', 'gene_classes'])
parser.add_argument('--n_individuals', type=int, default=550)
parser.add_argument('--n_folds', type=int, default=4)
parser.add_argument('--n_chromosomes', type=int, default=4)
parser.add_argument('--chromosome_length', type=float, default=2e7)
parser.add_argument('--gene_spacing', type=float, default=4e4)  # distance between consecutive TSSs
parser.add_argument('--cis_window', type=int, default=100000)
parser.add_argument('--min_maf', type=float, default=0.01)
parser.add_argument('--ld_rho', type=float, default=0.95)
parser.add_argument('--q', type=float, default=0.01)
parser.add_argument('--effect_variances', type=float, nargs='+', default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
parser.add_argument('--effect_weights', type=float, nargs='+', default=[0.3, 0.25, 0.2, 0.13, 0.08, 0.04])
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
n = args.n_individuals
effect_variances = np.asarray(args.effect_variances)
effect_weights = np.asarray(args.effect_weights) / np.sum(args.effect_weights)
maf_edges = np.asarray([0.02, 0.05, 0.1, 0.2])
maf_multiplier = np.asarray([0.3, 0.5, 1.0, 1.4, 1.8])

f = open(args.output_dir + 'folds.txt', 'w')
f.write('genotype_index\tindividual_id\tfold\n')
for i in range(n):
    f.write(str(i) + '\tIND' + str(i) + '\t' + str(i % args.n_folds + 1) + '\n')
f.close()

t = open(args.output_dir + 'gibbs_input_summary.txt', 'w')
t.write('chrom\tgibbs_input_file\tn_genes\tn_variants\tn_individuals\n')
truth = {'gene_ids': [], 'gene_chroms': [], 'gene_classes': [], 'beta_true': [], 'gene_n_pairs': []}
for chrom in range(1, args.n_chromosomes + 1):
    # Variant positions: density per 2 Mb region between 500 and 4,500 variants per Mb
    positions = []
    n_regions = int(np.ceil(args.chromosome_length / 2e6))
    for r in range(n_regions):
        density = rng.uniform(500, 4500)
        positions.append(rng.uniform(r * 2e6, (r + 1) * 2e6, int(density * 2)))
    positions = np.unique(np.concatenate(positions).astype(np.int64))
    p = len(positions)
    maf = np.exp(rng.uniform(np.log(args.min_maf), np.log(0.5), p))
    threshold = stats.norm.ppf(maf)
    genotype = np.zeros((p, n), dtype=np.int8)
    for haplotype in range(2):
        latent = rng.normal(size=n)
        for j in range(p):
            if j > 0:
                latent = args.ld_rho * latent + np.sqrt(1.0 - args.ld_rho ** 2) * rng.normal(size=n)
            genotype[j] += (latent < threshold[j]).astype(np.int8)
    keep = np.std(genotype, axis=1) > 0
    genotype, positions = genotype[keep], positions[keep]
    dosage_mean = np.mean(genotype, axis=1)
    dosage_sd = np.std(genotype, axis=1)
    af = dosage_mean / 2.0
    variant_maf = np.minimum(af, 1.0 - af)

    gene_tss = np.arange(args.cis_window, args.chromosome_length - args.cis_window, args.gene_spacing).astype(np.int64)
    gene_start = np.searchsorted(positions, gene_tss - args.cis_window, side='left')
    gene_end = np.searchsorted(positions, gene_tss + args.cis_window, side='right')
    expression = np.zeros((len(gene_tss), n))
    gene_ids = []
    for g in range(len(gene_tss)):
        rows = np.arange(gene_start[g], gene_end[g])
        scale = np.ones(len(rows))
        gene_class = 0
        if args.scenario == 'shared':
            q = np.full(len(rows), args.q)
        elif args.scenario == 'maf':
            maf_bin = np.searchsorted(maf_edges, variant_maf[rows], side='right')
            q = args.q * maf_multiplier[maf_bin]
            scale = np.where(variant_maf[rows] < 0.05, 0.5, 1.0)
        elif args.scenario == 'tss_distance':
            q = args.q * (0.4 + 6.0 * np.exp(-np.abs(positions[rows] - gene_tss[g]) / 5000.0))
        else:
            gene_class = rng.choice(3, p=[0.4, 0.45, 0.15])
            q = np.full(len(rows), [0.0, args.q, 4.0 * args.q][gene_class])
            scale = np.full(len(rows), [1.0, 1.0, 0.15][gene_class])
        nonzero = rng.uniform(size=len(rows)) < q
        sigma2 = effect_variances[rng.choice(len(effect_variances), size=len(rows), p=effect_weights)] * scale
        beta = np.where(nonzero, rng.normal(size=len(rows)) * np.sqrt(sigma2), 0.0)
        X = (genotype[rows].T - dosage_mean[rows]) / dosage_sd[rows]
        y = X @ beta + rng.normal(size=n)
        expression[g] = y
        gene_id = 'SIM' + str(chrom) + '_' + str(g)
        gene_ids.append(gene_id)
        truth['gene_ids'].append(gene_id)
        truth['gene_chroms'].append(chrom)
        truth['gene_classes'].append(gene_class)
        truth['beta_true'].append(beta)
        truth['gene_n_pairs'].append(len(rows))
    variant_ids = np.asarray(['chr' + str(chrom) + '_' + str(pos) + '_A_G_b38' for pos in positions])
    gibbs_input_file = args.output_dir + 'chr' + str(chrom) + '.npz'
    np.savez(gibbs_input_file, genotype=genotype, variant_ids=variant_ids.astype('S'), variant_positions=positions, gene_ids=np.asarray(gene_ids).astype('S'),
             gene_tss=gene_tss, gene_variant_start=gene_start, gene_variant_end=gene_end, expression=expression, genotype_indices=np.arange(n))
    t.write('chr' + str(chrom) + '\t' + gibbs_input_file + '\t' + str(len(gene_tss)) + '\t' + str(len(positions)) + '\t' + str(n) + '\n')
    truth['dosage_mean_chr' + str(chrom)] = dosage_mean
    truth['dosage_sd_chr' + str(chrom)] = dosage_sd
    print('chr' + str(chrom) + ': ' + str(len(positions)) + ' variants, ' + str(len(gene_tss)) + ' genes, cis variants per gene ' + str(np.min(gene_end - gene_start)) + ' to ' + str(np.max(gene_end - gene_start)), flush=True)
t.close()

gene_pair_ptr = np.concatenate([[0], np.cumsum(truth.pop('gene_n_pairs'))])
np.savez(args.output_dir + 'truth.npz', gene_ids=np.asarray(truth.pop('gene_ids')).astype('S'), gene_chroms=np.asarray(truth.pop('gene_chroms')),
         gene_classes=np.asarray(truth.pop('gene_classes')), beta_true=np.concatenate(truth.pop('beta_true')), gene_pair_ptr=gene_pair_ptr, **truth)
print('scenario ' + args.scenario + ': written to ' + args.output_dir, flush=True)
