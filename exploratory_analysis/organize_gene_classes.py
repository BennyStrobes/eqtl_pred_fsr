import argparse
import numpy as np

# Organize the latent gene classes of one fit (run_eqtl_grid_prior_gibbs.py --n_gene_classes, README.md, section 5j) into tables for
# visualize_gene_classes.R (README.md, section 8, plot 9). From each fold's <stem>_prior_rows.npz and <stem>_gene_summary.txt, and the
# <organized stem>_per_gene_fold.txt of organize_cross_validation_results.py:
#   <output_stem>_class_summary.txt: one row per fold and class: genes, inclusion probability, weight of the large group, prior E[beta^2]
#   <output_stem>_class_component_weights.txt: one row per fold, class and grid component: weight of the component among the non-zero effects
#   <output_stem>_class_trace.txt: every --trace_thin iterations, per fold and class: class weight rho and inclusion probability
#   <output_stem>_per_gene_fold.txt: the input per-gene file with the gene-level class quantities added (below)
#
# Class labels mean nothing between folds, and an empty class says nothing about the data:
#   - Within a fold, classes are ranked by the posterior mean weight of the large group (variance >= --small_large_cutoff, the part of the prior
#     that is identified, README.md, section 5e), so rank 1 is the class with the fewest identified effects per variant. Ranks of different folds
#     are only comparable as positions along that ordering. The ranking is of a label's whole chain, so it means little for a class that empties and
#     fills again with different content, which small classes do (plot 9c shows it). Ignore the class-level summaries of classes under about 1% of genes.
#   - A class holding no genes in a sample has grid weights drawn from the Dirichlet prior, so class summaries use only the samples in which the
#     class holds at least one gene. Classes with fewer than --min_class_genes genes on average have no rank (rank NA) and are not plotted.
# The gene-level quantities do not depend on any of this: in each retained sample a gene takes the weights of the class it is in, and the
# posterior mean over samples does not change if classes are relabelled:
#   class_inclusion, class_large_weight, class_beta2: posterior means of 1 - pi_{c_g, 0}, the large group's weight, and E[beta^2 | c_g]
#   prior_expected_n_large, prior_expected_h2: n_cis_variants times class_large_weight and class_beta2
# and, using the labels of the fold:
#   class_certainty: posterior probability of the gene's most probable class; class_rank: the rank of that class
# All of these come from the training individuals of the fold, so genes can be stratified on them in the held-out evaluation.


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--gibbs_results_stems', type=str, nargs='+')  # one per fold
parser.add_argument('--folds', type=int, nargs='+')  # fold of each stem
parser.add_argument('--per_gene_fold_file', type=str)  # <organized stem>_per_gene_fold.txt
parser.add_argument('--output_stem', type=str)
parser.add_argument('--min_class_genes', type=float, default=10.0)  # classes with fewer genes on average are not ranked
parser.add_argument('--small_large_cutoff', type=float, default=1e-3)  # as in run_eqtl_grid_prior_gibbs.py
parser.add_argument('--trace_thin', type=int, default=10)  # iterations between rows of the trace table
args = parser.parse_args()
if len(args.gibbs_results_stems) != len(args.folds):
    raise ValueError('one stem per fold')


def read_gene_summary(gibbs_results_stem):
    # Gene ids and numbers of cis variants, in the order of the genes in the fit
    f = open(gibbs_results_stem + '_gene_summary.txt')
    header = f.readline().rstrip('\n').split('\t')
    id_column = header.index('gene_id')
    variant_column = header.index('n_cis_variants')
    gene_ids = []
    n_variants = []
    for line in f:
        data = line.rstrip('\n').split('\t')
        gene_ids.append(data[id_column])
        n_variants.append(float(data[variant_column]))
    f.close()
    return gene_ids, np.asarray(n_variants)


def counts_by_row(classes, n_classes):
    # classes (n_rows x n_columns), values in 0, ..., n_classes - 1: number of entries of every row equal to each class (n_rows x n_classes)
    n_rows = classes.shape[0]
    offset = classes + n_classes * np.arange(n_rows, dtype=np.int64)[:, None]
    return np.bincount(offset.ravel(), minlength=n_rows * n_classes).reshape(n_rows, n_classes)


def summary_over_occupied(values, occupied):
    # Posterior mean and 95% interval of one class's values over the samples in which it holds genes (NaN if there are none)
    if not np.any(occupied):
        return np.nan, np.nan, np.nan
    return np.mean(values[occupied]), np.quantile(values[occupied], 0.025), np.quantile(values[occupied], 0.975)


def format_value(value):
    return 'NA' if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def organize_fold(gibbs_results_stem):
    # Class-level and gene-level quantities of one fit
    prior_rows = np.load(gibbs_results_stem + '_prior_rows.npz')
    if 'gene_class_samples' not in prior_rows.files:
        raise ValueError('not a gene-class fit (no gene_class_samples): ' + gibbs_results_stem)
    grid_var = prior_rows['grid_var'].astype(np.float64)
    retained = prior_rows['retained'].astype(bool)
    weight_trace = prior_rows['row_weight_trace'].astype(np.float64)  # iterations x classes x components (spike first)
    class_weight_trace = prior_rows['gene_class_weight_trace'].astype(np.float64)  # iterations x classes
    classes = prior_rows['gene_class_samples'].astype(np.int64)  # retained samples x genes
    gene_ids, n_variants = read_gene_summary(gibbs_results_stem)
    if classes.shape[1] != len(gene_ids):
        raise ValueError(gibbs_results_stem + ': ' + str(classes.shape[1]) + ' genes in the class samples, ' + str(len(gene_ids)) + ' in the gene summary')
    if classes.shape[0] != np.sum(retained):
        raise ValueError(gibbs_results_stem + ': class samples do not match the retained iterations')
    n_samples, n_genes = classes.shape
    n_classes = weight_trace.shape[1]
    large_components = grid_var >= args.small_large_cutoff

    # The same three quantities for every class: posterior samples (n_samples x n_classes) and traces
    weights = weight_trace[retained]
    inclusion = 1.0 - weights[:, :, 0]
    large_weight = np.sum(weights[:, :, large_components], axis=2)
    beta2 = np.sum(weights * grid_var[None, None, :], axis=2)
    genes_per_class = counts_by_row(classes, n_classes)
    occupied = genes_per_class > 0
    mean_genes = np.mean(genes_per_class, axis=0)
    pairs_per_class = np.bincount(classes.ravel(), weights=np.tile(n_variants, n_samples), minlength=n_classes) / n_samples

    # Rank: fewest identified effects first, among the classes with enough genes
    mean_large = np.asarray([summary_over_occupied(large_weight[:, l], occupied[:, l])[0] for l in range(n_classes)])
    ranked = np.flatnonzero((mean_genes >= args.min_class_genes) & ~np.isnan(mean_large))
    ranked = ranked[np.argsort(mean_large[ranked])]
    class_rank = np.full(n_classes, -1, dtype=np.int64)
    class_rank[ranked] = np.arange(1, len(ranked) + 1)

    class_rows = []
    for l in range(n_classes):
        row = {'class_label': l + 1, 'class_rank': class_rank[l] if class_rank[l] > 0 else None, 'n_genes': mean_genes[l], 'gene_fraction': mean_genes[l] / n_genes,
               'n_pairs': pairs_per_class[l], 'mean_n_cis_variants': pairs_per_class[l] / mean_genes[l] if mean_genes[l] > 0 else np.nan,
               'fraction_samples_occupied': np.mean(occupied[:, l])}
        for name, values in [('inclusion', inclusion), ('large_weight', large_weight), ('prior_beta2', beta2)]:
            row[name + '_mean'], row[name + '_95_lb'], row[name + '_95_ub'] = summary_over_occupied(values[:, l], occupied[:, l])
        class_rows.append(row)

    # Weight of each grid component among the non-zero effects, for the ranked classes
    component_rows = []
    for l in ranked:
        keep = occupied[:, l]
        conditional = weights[keep, l, 1:] / inclusion[keep, l, None]
        for k in range(conditional.shape[1]):
            component_rows.append({'class_rank': class_rank[l], 'class_label': l + 1, 'variance': grid_var[k + 1], 'weight_among_nonzero': np.mean(conditional[:, k])})

    # Traces of the ranked classes
    trace_iterations = np.arange(0, weight_trace.shape[0], args.trace_thin)
    trace_rows = []
    for l in ranked:
        for itera in trace_iterations:
            trace_rows.append({'iteration': itera, 'retained': int(retained[itera]), 'class_rank': class_rank[l], 'class_label': l + 1, 'class_weight': class_weight_trace[itera, l],
                               'inclusion': 1.0 - weight_trace[itera, l, 0]})

    # Gene level: the weights of the class each gene is in, sample by sample (does not depend on the labels)
    sample_index = np.arange(n_samples)[:, None]
    class_inclusion = np.mean(inclusion[sample_index, classes], axis=0)
    class_large_weight = np.mean(large_weight[sample_index, classes], axis=0)
    class_beta2 = np.mean(beta2[sample_index, classes], axis=0)
    samples_per_gene_class = counts_by_row(classes.T, n_classes)  # genes x classes
    # Ties (a gene split evenly between two classes) go to the lower rank, so that the result does not depend on the labels
    rank_order = np.argsort(np.where(class_rank > 0, class_rank, n_classes + 1), kind='stable')
    modal_class = rank_order[np.argmax(samples_per_gene_class[:, rank_order], axis=1)]
    gene_table = {}
    for g, gene_id in enumerate(gene_ids):
        gene_table[gene_id] = [class_inclusion[g], class_large_weight[g], class_beta2[g], n_variants[g] * class_large_weight[g], n_variants[g] * class_beta2[g],
                               samples_per_gene_class[g, modal_class[g]] / n_samples, class_rank[modal_class[g]] if class_rank[modal_class[g]] > 0 else None]
    print(gibbs_results_stem + ': ' + str(len(ranked)) + ' of ' + str(n_classes) + ' classes ranked, gene fractions ' + ' '.join('%.3f' % (mean_genes[l] / n_genes) for l in ranked), flush=True)
    return class_rows, component_rows, trace_rows, gene_table


fold_class_rows = []
fold_component_rows = []
fold_trace_rows = []
fold_gene_tables = {}
for fold, gibbs_results_stem in zip(args.folds, args.gibbs_results_stems):
    class_rows, component_rows, trace_rows, gene_table = organize_fold(gibbs_results_stem)
    fold_class_rows += [dict(row, fold=fold) for row in class_rows]
    fold_component_rows += [dict(row, fold=fold) for row in component_rows]
    fold_trace_rows += [dict(row, fold=fold) for row in trace_rows]
    fold_gene_tables[fold] = gene_table


def write_rows(output_file, columns, rows):
    t = open(output_file, 'w')
    t.write('\t'.join(columns) + '\n')
    for row in rows:
        t.write('\t'.join(format_value(row[column]) for column in columns) + '\n')
    t.close()


write_rows(args.output_stem + '_class_summary.txt', ['fold', 'class_label', 'class_rank', 'n_genes', 'gene_fraction', 'n_pairs', 'mean_n_cis_variants', 'fraction_samples_occupied',
                                                    'inclusion_mean', 'inclusion_95_lb', 'inclusion_95_ub', 'large_weight_mean', 'large_weight_95_lb', 'large_weight_95_ub',
                                                    'prior_beta2_mean', 'prior_beta2_95_lb', 'prior_beta2_95_ub'], fold_class_rows)
write_rows(args.output_stem + '_class_component_weights.txt', ['fold', 'class_rank', 'class_label', 'variance', 'weight_among_nonzero'], fold_component_rows)
write_rows(args.output_stem + '_class_trace.txt', ['fold', 'iteration', 'retained', 'class_rank', 'class_label', 'class_weight', 'inclusion'], fold_trace_rows)

# Per-gene file with the gene-level quantities. Rows are matched on (gene_id, fold), not on their order
new_columns = ['class_inclusion', 'class_large_weight', 'class_beta2', 'prior_expected_n_large', 'prior_expected_h2', 'class_certainty', 'class_rank']
f = open(args.per_gene_fold_file)
header = f.readline().rstrip('\n').split('\t')
id_column = header.index('gene_id')
fold_column = header.index('fold')
t = open(args.output_stem + '_per_gene_fold.txt', 'w')
t.write('\t'.join(header + new_columns) + '\n')
n_rows = 0
n_left_out = 0
n_seen = {fold: 0 for fold in args.folds}
for line in f:
    data = line.rstrip('\n').split('\t')
    fold = int(float(data[fold_column]))
    if fold not in fold_gene_tables:
        n_left_out += 1
        continue
    if data[id_column] not in fold_gene_tables[fold]:
        raise ValueError('gene ' + data[id_column] + ' of fold ' + str(fold) + ' is not in the fit')
    t.write('\t'.join(data + [format_value(value) for value in fold_gene_tables[fold][data[id_column]]]) + '\n')
    n_rows += 1
    n_seen[fold] += 1
f.close()
t.close()
for fold in args.folds:
    if n_seen[fold] != len(fold_gene_tables[fold]):
        raise ValueError('fold ' + str(fold) + ': ' + str(n_seen[fold]) + ' evaluated genes, ' + str(len(fold_gene_tables[fold])) + ' in the fit')
if n_left_out > 0:
    print('warning: ' + str(n_left_out) + ' rows of folds without a gene-class fit left out', flush=True)
print(str(n_rows) + ' gene-fold rows written to ' + args.output_stem + '_per_gene_fold.txt', flush=True)
