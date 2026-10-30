import argparse
import numpy as np

# Assign every genotyped GTEx individual (rows of the plink .fam file) to one of n_folds cross-validation folds, at random with a fixed seed
# (README.md, section 3). Folds are numbered 1, ..., n_folds and nearly equal in size. Each tissue uses its own subset of individuals, so an
# individual is held out in the same fold in every tissue.
# Output: genotype_index (row of the .fam file, as in the genotype sample mapping files), individual_id, fold


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--fam_file', type=str)
parser.add_argument('--n_folds', type=int, default=4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--fold_file', type=str)
args = parser.parse_args()

individual_ids = []
f = open(args.fam_file)
for line in f:
    data = line.rstrip('\n').split()
    individual_ids.append(data[1])
f.close()
n_individuals = len(individual_ids)
if len(np.unique(individual_ids)) != n_individuals:
    raise ValueError('repeated individual ids in ' + args.fam_file)

np.random.seed(args.seed)
folds = np.zeros(n_individuals, dtype=int)
folds[np.random.permutation(n_individuals)] = np.arange(n_individuals) % args.n_folds + 1

t = open(args.fold_file, 'w')
t.write('genotype_index\tindividual_id\tfold\n')
for genotype_index in range(n_individuals):
    t.write(str(genotype_index) + '\t' + individual_ids[genotype_index] + '\t' + str(folds[genotype_index]) + '\n')
t.close()
print(str(n_individuals) + ' individuals assigned to ' + str(args.n_folds) + ' folds (sizes ' + ', '.join(str(np.sum(folds == k)) for k in range(1, args.n_folds + 1)) + ')')
