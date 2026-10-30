import argparse
import pdb
import numpy as np
from pandas_plink import read_plink

# Prepare the input of run_eqtl_mixture_prior_gibbs.py for one GTEx tissue (README.md, sections 2 and 9). Fold-agnostic: the training-fold
# variant filter (MAF, variance) is applied by the run script.
# Per chromosome, one .npz with:
#   genotype (n_variants x n_individuals, int8): alt-allele dosage of every bi-allelic SNV within cis_window of a gene's TSS, with no missing
#       call and non-zero variance in the tissue's individuals; variants sorted by position
#   variant_ids (GTEx chr_pos_ref_alt_b38), variant_positions
#   gene_ids (no version), gene_tss, gene_variant_start, gene_variant_end: the cis variants of gene g are rows gene_variant_start[g]:gene_variant_end[g]
#   expression (n_genes x n_individuals): residualized, renormalized expression
#   genotype_indices (n_individuals): row of the plink .fam file of each individual (column), in expression file order
# and a summary file with one line per chromosome (chrom, npz file, number of genes, number of variants, number of individuals).


def extract_dictionary_list_of_protein_coding_genes(pc_genes_gtf):
    # Same as gdl_uncertainty_paper/gtex_eqtl_expression_processing/residualize_expression.py
    valid_chroms = {}
    for chrom_num in range(1, 23):
        valid_chroms['chr' + str(chrom_num)] = 1
    f = open(pc_genes_gtf)
    dicti = {}
    for line in f:
        data = line.rstrip('\n').split('\t')
        if data[0] not in valid_chroms:
            continue
        ens_id = data[8].split(';')[0].split('"')[1]
        if ens_id.startswith('ENSG') == False:
            print('assumption error: gene id ' + ens_id)
            pdb.set_trace()
        dicti[ens_id.split('.')[0]] = 1
    f.close()
    return dicti


def load_in_expression(expression_file, pc_genes):
    # Bed file: chrom, start, tss, gene_id, then one column per sample
    # Returns dictionary: chrom string -> list of (gene_id (no version), tss, expression vector), and the ordered sample names
    chrom_to_genes = {}
    f = open(expression_file)
    head_count = 0
    for line in f:
        data = line.rstrip('\n').split('\t')
        if head_count == 0:
            head_count += 1
            sample_names = np.asarray(data[4:])
            continue
        chrom = data[0]
        if chrom.startswith('chr') == False:
            chrom = 'chr' + chrom
        gene_id = data[3].split('.')[0]
        if gene_id not in pc_genes:
            continue
        if chrom not in chrom_to_genes:
            chrom_to_genes[chrom] = []
        chrom_to_genes[chrom].append((gene_id, int(data[2]), np.asarray(data[4:]).astype(float)))
    f.close()
    return chrom_to_genes, sample_names


def extract_cis_snv_rows(bim, gene_tss, cis_window):
    # Rows of the bim (sorted by position) of bi-allelic SNVs within cis_window of any gene's TSS
    # GTEx bim: snp id chr_pos_ref_alt_b38, a1 = ref, a0 = alt, and pandas_plink G counts a1 (as in simple_eqtl_analysis.py), so the
    # alt-allele dosage is 2 - G
    positions = np.asarray(bim['pos']).astype(int)
    if np.any(np.diff(positions) < 0):
        print('assumption error: bim not sorted by position')
        pdb.set_trace()
    a0s = np.asarray(bim['a0']).astype(str)
    a1s = np.asarray(bim['a1']).astype(str)
    snp_ids = np.asarray(bim['snp']).astype(str)
    is_snv = (np.char.str_len(a0s) == 1) & (np.char.str_len(a1s) == 1)
    in_cis = np.zeros(len(positions), dtype=bool)
    for tss in gene_tss:
        start = np.searchsorted(positions, tss - cis_window, side='left')
        end = np.searchsorted(positions, tss + cis_window, side='right')
        in_cis[start:end] = True
    rows = np.flatnonzero(is_snv & in_cis)
    for row in rows:
        snp_info = snp_ids[row].split('_')
        if snp_info[2] != a1s[row] or snp_info[3] != a0s[row]:
            print('assumption error: alleles of ' + snp_ids[row] + ' do not match bim a1 / a0')
            pdb.set_trace()
    return rows


def extract_alt_dosage(G, rows, ordered_genotype_indices, chunk_size=20000):
    # Alt-allele dosage (int8) of the given (sorted) bim rows in the tissue's individuals, dropping variants with any missing call or no variance
    # Returns dosage matrix (n_kept x n_individuals) and the kept rows
    dosage_chunks = []
    kept_rows = []
    for chunk_start in range(0, len(rows), chunk_size):
        chunk_rows = rows[chunk_start:chunk_start + chunk_size]
        geno_mat = (G[chunk_rows, :].compute())[:, ordered_genotype_indices]
        complete = np.sum(np.isnan(geno_mat), axis=1) == 0
        geno_mat = geno_mat[complete, :]
        chunk_rows = chunk_rows[complete]
        varying = np.std(geno_mat, axis=1) > 0.0
        dosage_chunks.append((2.0 - geno_mat[varying, :]).astype(np.int8))
        kept_rows.append(chunk_rows[varying])
    return np.concatenate(dosage_chunks, axis=0), np.concatenate(kept_rows)


##########################
# Command line arguments
##########################
parser = argparse.ArgumentParser()
parser.add_argument('--tissue', type=str)
parser.add_argument('--expression_file', type=str)  # residualized, renormalized expression bed file of the tissue
parser.add_argument('--plink_stem', type=str)  # chromosome-specific plink files: <plink_stem><chrom_num>.{bed,bim,fam}
parser.add_argument('--genotype_sample_mapping_file', type=str)  # plink .fam row of each expression sample (column), in order
parser.add_argument('--pc_genes_gtf', type=str)
parser.add_argument('--cis_window', type=int, default=100000)
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

pc_genes = extract_dictionary_list_of_protein_coding_genes(args.pc_genes_gtf)
chrom_to_genes, sample_names = load_in_expression(args.expression_file, pc_genes)
ordered_genotype_indices = (np.loadtxt(args.genotype_sample_mapping_file)).astype(int)
if len(ordered_genotype_indices) != len(sample_names):
    print('assumption error: number of expression samples does not match genotype sample mapping')
    pdb.set_trace()
print(args.tissue + ': ' + str(len(sample_names)) + ' individuals, ' + str(sum(len(genes) for genes in chrom_to_genes.values())) + ' protein-coding genes', flush=True)

summary_file = args.output_dir + args.tissue + '_gibbs_input_summary.txt'
t = open(summary_file, 'w')
t.write('chrom\tgibbs_input_file\tn_genes\tn_variants\tn_individuals\n')
for chrom_num in range(1, 23):
    chrom_string = 'chr' + str(chrom_num)
    if chrom_string not in chrom_to_genes:
        continue
    genes = sorted(chrom_to_genes[chrom_string], key=lambda gene: gene[1])
    gene_ids = np.asarray([gene[0] for gene in genes])
    gene_tss = np.asarray([gene[1] for gene in genes])
    expression = np.asarray([gene[2] for gene in genes])
    if len(np.unique(gene_ids)) != len(gene_ids):
        print('assumption error: repeated gene ids on ' + chrom_string)
        pdb.set_trace()

    (bim, fam, G) = read_plink(args.plink_stem + str(chrom_num), verbose=False)
    rows = extract_cis_snv_rows(bim, gene_tss, args.cis_window)
    genotype, kept_rows = extract_alt_dosage(G, rows, ordered_genotype_indices)
    variant_ids = np.asarray(bim['snp']).astype(str)[kept_rows]
    variant_positions = np.asarray(bim['pos']).astype(int)[kept_rows]
    if len(np.unique(variant_ids)) != len(variant_ids):
        print('assumption error: repeated variant ids on ' + chrom_string)
        pdb.set_trace()

    # Cis variants of each gene: a contiguous range of the position-sorted variants
    gene_variant_start = np.searchsorted(variant_positions, gene_tss - args.cis_window, side='left')
    gene_variant_end = np.searchsorted(variant_positions, gene_tss + args.cis_window, side='right')

    gibbs_input_file = args.output_dir + args.tissue + '_' + chrom_string + '_gibbs_input.npz'
    np.savez(gibbs_input_file, genotype=genotype, variant_ids=variant_ids.astype('S'), variant_positions=variant_positions, gene_ids=gene_ids.astype('S'),
             gene_tss=gene_tss, gene_variant_start=gene_variant_start, gene_variant_end=gene_variant_end, expression=expression, genotype_indices=ordered_genotype_indices)
    t.write(chrom_string + '\t' + gibbs_input_file + '\t' + str(len(gene_ids)) + '\t' + str(len(variant_ids)) + '\t' + str(len(ordered_genotype_indices)) + '\n')
    t.flush()
    print(chrom_string + ': ' + str(len(gene_ids)) + ' genes, ' + str(len(variant_ids)) + ' cis SNVs (of ' + str(len(rows)) + ' in cis windows), mean ' + '%.1f' % np.mean(gene_variant_end - gene_variant_start) + ' per gene', flush=True)
t.close()
