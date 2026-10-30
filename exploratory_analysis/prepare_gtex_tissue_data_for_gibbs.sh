#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=30G                        # Memory total in MiB (for all cores)


tissue=${1}
expression_file=${2}
plink_stem=${3}
genotype_sample_mapping_file=${4}
pc_genes_gtf=${5}
cis_window=${6}
gibbs_input_dir=${7}

source ~/.bash_profile
conda activate plink_env
mkdir -p ${gibbs_input_dir}

date

python prepare_gtex_tissue_data_for_gibbs.py --tissue ${tissue} --expression_file ${expression_file} --plink_stem ${plink_stem} --genotype_sample_mapping_file ${genotype_sample_mapping_file} --pc_genes_gtf ${pc_genes_gtf} --cis_window ${cis_window} --output_dir ${gibbs_input_dir}

date
