#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=30G                        # Memory total in MiB (for all cores): the sparse effect samples are held in memory


gibbs_results_stem=${1}
output_stem=${2}
# Optional: regions whose genes are left out of the comparison, chrom:start-end of the TSS (e.g. 6:28510120-33480577, the GRCh38 MHC)
exclude_regions=${3:-}

region_args=""
if [ -n "${exclude_regions}" ]; then
	region_args="--exclude_regions ${exclude_regions}"
fi

source ~/.bash_profile
conda activate plink_env

date

python check_prior_self_consistency.py --gibbs_results_stem ${gibbs_results_stem} --output_stem ${output_stem} ${region_args}

date
