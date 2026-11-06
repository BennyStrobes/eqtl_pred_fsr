#!/bin/bash
#SBATCH -c 1                               # Request one core (genes run in parallel through numba prange if given more)
#SBATCH -t 2-30:00                        # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=40G                        # Memory total in MiB (for all cores): int8 training genotypes and the sparse effect samples are held in memory


gibbs_input_summary_file=${1}
fold_file=${2}
held_out_fold=${3}
output_stem=${4}
n_iters=${5}
burn_in=${6}
thin=${7}
# Optional: "gwfm" fits the fixed-grid prior with the GWFM-style tempered Gibbs sampler (README.md, section 5e); give each its own
# output_stem. Default "gibbs"
sampler=${8:-gibbs}
# Optional: the prior's structure. Default: one shared prior. A number B: B variant-count bins, each with its own inclusion probability
# (README.md, section 5g). maf or tss_distance: grid weights per bin of the variant's MAF or distance to the TSS; gene_classes_L: L latent gene
# classes with grid weights per class (README.md, section 5j)
prior=${9:-}
# Optional: number of non-centred moves of the grid weights per iteration (shared prior only; README.md, section 5k). Default 0
ncp_moves=${10:-0}
# Optional: "hierarchical" to learn the Inverse-Gamma prior of the residual variances (README.md, section 5k). Default: fixed prior
sigma2_e_prior=${11:-fixed}

sampler_args=""
if [ "${sampler}" = "gwfm" ]; then
	sampler_args="--gwfm"
elif [ "${sampler}" != "gibbs" ]; then
	echo "unknown sampler: ${sampler}"
	exit 1
fi
if [ -z "${prior}" ]; then
	:
elif [[ "${prior}" =~ ^[0-9]+$ ]]; then
	sampler_args="${sampler_args} --n_variant_bins ${prior}"
elif [ "${prior}" = "maf" ] || [ "${prior}" = "tss_distance" ]; then
	sampler_args="${sampler_args} --variant_categories ${prior}"
elif [[ "${prior}" =~ ^gene_classes_[0-9]+$ ]]; then
	sampler_args="${sampler_args} --n_gene_classes ${prior#gene_classes_}"
else
	echo "unknown prior: ${prior}"
	exit 1
fi
if [ "${ncp_moves}" != "0" ]; then
	sampler_args="${sampler_args} --ncp_moves ${ncp_moves}"
fi
if [ "${sigma2_e_prior}" = "hierarchical" ]; then
	sampler_args="${sampler_args} --hierarchical_sigma2_e"
elif [ "${sigma2_e_prior}" != "fixed" ]; then
	echo "unknown sigma2_e prior: ${sigma2_e_prior}"
	exit 1
fi

source ~/.bash_profile
conda activate plink_env
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

date

python run_eqtl_grid_prior_gibbs.py --gibbs_input_summary_file ${gibbs_input_summary_file} --fold_file ${fold_file} --held_out_fold ${held_out_fold} --output_stem ${output_stem} --n_iters ${n_iters} --burn_in ${burn_in} --thin ${thin} ${sampler_args}

date
