#!/bin/bash
#SBATCH -c 1                               # Request one core (genes run in parallel through numba prange if given more)
#SBATCH -t 1-00:00                        # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=40G                        # Memory total in MiB (for all cores): int8 training genotypes and the sparse effect samples are held in memory


gibbs_input_summary_file=${1}
fold_file=${2}
held_out_fold=${3}
output_stem=${4}
n_iters=${5}
burn_in=${6}
thin=${7}

source ~/.bash_profile
conda activate plink_env
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

date

python run_eqtl_mixture_prior_gibbs.py --gibbs_input_summary_file ${gibbs_input_summary_file} --fold_file ${fold_file} --held_out_fold ${held_out_fold} --output_stem ${output_stem} --n_iters ${n_iters} --burn_in ${burn_in} --thin ${thin}

date
