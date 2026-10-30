import argparse
import numpy as np

# Organize the cross-validation results of one tissue (one fit of run_eqtl_mixture_prior_gibbs.py and one evaluation of
# evaluate_held_out_expression_prediction.py per fold) into two tables for visualize_cross_validation_results.R:
#   <output_stem>_per_gene_fold.txt: the evaluation files of every fold, concatenated (one row per gene and fold; the fold column says which)
#   <output_stem>_hyperparameter_trace.txt: hyper-parameter traces of every fold, one row per fold, iteration and hyper-parameter


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_files', type=str, nargs='+')  # one per fold
parser.add_argument('--gibbs_results_stems', type=str, nargs='+')  # one per fold, in the order of --evaluation_files
parser.add_argument('--folds', type=int, nargs='+')  # fold of each file
parser.add_argument('--output_stem', type=str)
args = parser.parse_args()

# Per gene and fold
t = open(args.output_stem + '_per_gene_fold.txt', 'w')
header = None
n_rows = 0
for evaluation_file in args.evaluation_files:
    f = open(evaluation_file)
    head_count = 0
    for line in f:
        if head_count == 0:
            head_count += 1
            if header is None:
                header = line
                t.write(header)
            elif line != header:
                raise ValueError('evaluation files have different columns: ' + evaluation_file)
            continue
        t.write(line)
        n_rows += 1
    f.close()
t.close()
print(str(n_rows) + ' gene-fold rows', flush=True)

# Hyper-parameter traces
t = open(args.output_stem + '_hyperparameter_trace.txt', 'w')
t.write('fold\titeration\tretained\thyperparameter\tvalue\n')
for fold, gibbs_results_stem in zip(args.folds, args.gibbs_results_stems):
    f = open(gibbs_results_stem + '_hyperparameter_trace.txt')
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            names = data[2:]
            continue
        for name, value in zip(names, data[2:]):
            t.write(str(fold) + '\t' + data[0] + '\t' + data[1] + '\t' + name + '\t' + value + '\n')
    f.close()
t.close()
