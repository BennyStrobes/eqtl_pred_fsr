# Cross-validated eQTL effect prediction with a genome-wide shared prior: proposed analysis

Ben Strober, 2026-10-29

## 1. Summary

Implemented (section 9 lists the files) and tested end to end on simulated data; not yet run on GTEx. For one GTEx tissue (Whole Blood first, any tissue by changing one variable), it fits a Gibbs sampler that learns a single prior on causal eQTL effects shared by every variant-gene pair in the tissue, using individual-level GTEx genotype and expression data. The prior is a three-part mixture: a spike at zero and two normal slabs, one small-effect and one large-effect. Hyper-parameters are sampled with everything else, so the posterior of each gene's effects carries the uncertainty of the shared prior.

The fit uses 3/4 of the individuals. On the held-out 1/4, for each gene:

1. **observed:** the correlation between the predicted expression (posterior mean effects) and the measured expression.
2. **expected:** the correlation and the false sign rate (FSR) that the posterior predicts for those same held-out individuals, computed from post-burn-in posterior samples. These are the quantities of the gdl_uncertainty paper (`gdl_uncertainty_paper/expression_correlations/predicted_expression_correlation_plots.md`), with the posterior of a fitted model in place of rescaled Borzoi effects.

The question is whether the expected quantities are calibrated: across genes binned by expected correlation (or FSR), does the mean observed correlation (or fraction of wrong signs) match? Held-out data give an honest answer that in-sample fits cannot.

What changes relative to CoFine (`ca_qtls/fine_mapping/cofine.py`):

- eQTLs only. No caQTL, peak-gene link or path structure.
- Individual-level data instead of summary statistics plus LD. The likelihood is exact, and the held-out individuals get their own genotypes.
- Standard random-scan Gibbs, not tempered Gibbs. This is prediction, not fine-mapping: when variants in high LD trade an effect between them, the prediction barely changes, so slow mixing between LD partners does not matter here.

## 2. Data

All inputs already exist from `gdl_uncertainty_paper/gtex_eqtl_expression_processing/`:

| Input | Path (under `/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/`) |
|---|---|
| expression | `residualized_expression/${tissue}.v8.residualized_expression_renormalized.bed` (EUR individuals, GTEx v8 covariates regressed out, protein-coding genes, renormalized to unit variance) |
| genotype | `plink_processed_genotype/gtex_v9_eqtl_chr${chrom}.{bed,bim,fam}` |
| sample mapping | `plink_processed_genotype/genotype_sample_mapping_to_${tissue}_expression_samples.txt` (genotype index of each expression column) |
| gene annotation | `/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gencode.v39.gtex.protein_coding.genes.gtf` (TSS and strand) |

**Genes.** Every protein-coding gene in the residualized expression file, on autosomes. Not only eGenes. Calibration has to hold across all genes, and selecting on significance would bias it.

**Variants per gene.** Bi-allelic SNVs within $\pm 100$ kb of the TSS (as in `simple_eqtl_analysis.py`), with no missing genotype calls in the tissue's individuals, and with MAF $\ge 0.01$ and non-zero variance **in the training fold** (so the variant set can differ slightly between folds).

## 3. Cross-validation design

- Fold labels are assigned once, over every genotyped GTEx individual (the plink `.fam`), at random with a fixed seed into $K = 4$ folds. Each tissue uses its own subset of individuals. An individual is therefore held out in the same fold in every tissue, which keeps later cross-tissue comparisons clean.
- For fold $k$: train on the individuals not in fold $k$, evaluate on fold $k$. Running all four folds gives every individual one out-of-fold prediction and four replicates of every calibration plot.
- Whole Blood EUR has $n \approx 560$, so $n_{tr} \approx 420$ and $n_{te} \approx 140$ per fold.
- Everything learned (variant filter, genotype means and standard deviations, effects, hyper-parameters, residual variances) uses training individuals only. Held-out genotypes are standardized with the **training** means and standard deviations.

## 4. Model

For gene $g$ with $p_g$ cis variants, on the training individuals:

$$
y_g = X_g \beta_g + \varepsilon_g, \qquad \varepsilon_g \sim N(0, \sigma^2_{e,g} I_{n_{tr}}),
$$

with $y_g$ the residualized expression centered on its training mean, and $X_g$ ($n_{tr} \times p_g$) the genotypes standardized in the training fold (mean 0, variance 1). Then $\sigma^2_c$ below is a variance explained per variant, in units of total expression variance.

**Prior on effects** (shared across all genes and variants of the tissue):

$$
\beta_{gj} \sim \pi_0\, \delta_0 + \pi_1\, N(0, \sigma^2_1) + \pi_2\, N(0, \sigma^2_2), \qquad \sigma^2_1 < \sigma^2_2 .
$$

The large slab ($\sigma^2_2$) captures the few strong eQTLs. The small slab ($\sigma^2_1$) can absorb weak, polygenic cis signal that a single slab would either drop or overfit.

**Hyper-priors:**

All weakly informative (command-line arguments, with these defaults):

- $(\pi_0, \pi_1, \pi_2) \sim \text{Dirichlet}(\alpha, \alpha, \alpha)$, $\alpha = 0.1$
- $\sigma^2_c \sim \text{Inverse-Gamma}(a, b)$, $a = b = 10^{-10}$, truncated to $\sigma^2_1 < \sigma^2_2$
- $\sigma^2_{e,g} \sim \text{Inverse-Gamma}(a, b)$, $a = b = 10^{-10}$, one per gene

Shared: $\pi$, $\sigma^2_1$, $\sigma^2_2$. Per gene: $\sigma^2_{e,g}$. With roughly $10^7$ variant-gene pairs for the shared parameters and $n_{tr} \approx 420$ individuals per residual variance, the data dominate.

**Numerical guards.** These priors are near-improper, so a parameter with almost no data behind it can be drawn at an extreme value:

- **Empty slab.** If a slab has no members ($n_c = 0$), its variance's conditional is the prior itself. With $a = b = 10^{-10}$ that draw is effectively $0$ or $\infty$, and the slab could never be repopulated. The variance is kept at its previous value until the slab has members again (as `cofine.py` does).
- **Tiny weights.** $\text{Dirichlet}(0.1 + n_c)$ with $n_c = 0$ can underflow $\pi_c$ to exactly 0, which gives $\log \pi_c = -\infty$. The Dirichlet is drawn through log-Gamma variates, and each $\pi_c$ is floored at $1 / N$, with $N$ the total number of variant-gene pairs (as `sample_beta_with_floor` in `cofine.py`).

Neither guard changes anything while every slab holds variants. If either one fires, it is printed in the iteration log.

## 5. Gibbs sampler

Latent class $z_{gj} \in \{0, 1, 2\}$ per variant ($0$ = spike). One iteration:

**1. Effects, every gene** (random-scan over variants within each gene). Keep the residual $r = y_g - X_g \beta_g$. For variant $j$, with $X_j^\top X_j = n_{tr}$:

$$
b_j = \frac{X_j^\top (r + X_j \beta_j)}{n_{tr}}, \qquad v_g = \frac{\sigma^2_{e,g}}{n_{tr}},
$$

$$
P(z_j = 0 \mid \cdot) \propto \pi_0\, N(b_j; 0, v_g), \qquad P(z_j = c \mid \cdot) \propto \pi_c\, N(b_j; 0, \sigma^2_c + v_g),
$$

$$
\beta_j \mid z_j = c \sim N\Big(\frac{\sigma^2_c}{\sigma^2_c + v_g}\, b_j,\ \frac{\sigma^2_c\, v_g}{\sigma^2_c + v_g}\Big), \qquad \beta_j = 0 \text{ if } z_j = 0,
$$

then $r \leftarrow r - X_j(\beta_j^{new} - \beta_j^{old})$. Cost: $O(n_{tr})$ per variant, $O(n_{tr}\, p_g)$ per gene, whatever the number of non-zero effects (this matters because the small slab may hold many variants).

**2. Residual variances, every gene:** $\sigma^2_{e,g} \sim \text{Inverse-Gamma}(a + n_{tr}/2,\ b + r^\top r / 2)$.

**3. Shared hyper-parameters**, from the counts $n_c = \#\{(g, j) : z_{gj} = c\}$ and sums of squares over all genes:

- $\pi \sim \text{Dirichlet}(\alpha + n_0, \alpha + n_1, \alpha + n_2)$
- $\sigma^2_c \sim \text{Inverse-Gamma}\big(a + n_c/2,\ b + \sum_{z_{gj} = c} \beta_{gj}^2 / 2\big)$ (kept at its previous value if $n_c = 0$)
- if $\sigma^2_1 > \sigma^2_2$ after the draw, swap the labels of the two slabs (their $\pi$, $\sigma^2$ and $z$) to prevent label switching

**Speed.** Given the hyper-parameters, genes are independent, so steps 1 and 2 can run in parallel over genes (numba `prange`; optional, one core is the default) and only step 3 synchronizes. The inner loops are numba-compiled, as in `cofine.py`. Genotypes are kept as `int8` alt-allele dosages of the training individuals (one row per variant passing the training filter, with per-gene index arrays) and standardized on the fly: $X_j^\top r = (G_j^\top r - m_j \sum_i r_i) / s_j$, with $G_j$ the raw dosages and $m_j$, $s_j$ the training mean and standard deviation. The residuals are recomputed from scratch every 100 iterations, to remove accumulated rounding error.

**Defaults.** 3,000 iterations, 500 burn-in, keep every 5th: $S = 500$ retained samples. Initialization: $\beta = 0$, $\pi_0 = 0.99$, $(\sigma^2_1, \sigma^2_2) = (10^{-4}, 10^{-2})$, $\sigma^2_{e,g} = \operatorname{Var}(y_g)$. Initial slab weights: $(\pi_1, \pi_2) = (0.009, 0.001)$. PIPs, slab probabilities and posterior means average every post burn-in iteration, not only the retained ones.

## 6. What the sampler saves

**The $S = 500$ retained samples of every causal effect $\beta_{gj}^{(s)}$**, so the fitted posterior can be applied to any other genotype data (another cohort, another tissue's individuals, simulated genotypes) without refitting.

**Sparse storage.** Under the spike, $\beta_{gj}^{(s)}$ is exactly 0, so only the non-zero entries are stored and nothing is lost. One `.npz` per tissue and fold, `<stem>_posterior_beta_samples.npz`, holds the non-zero entries as triples sorted by variant-gene pair and then sample: `nz_pair` (`int32`), `nz_sample` (`int16`) and `nz_beta` (`float32`), 10 bytes per non-zero entry. `posterior_beta_samples.py` documents every field. For about $10^7$ variant-gene pairs per tissue, dense float32 storage would be about 20 GB per fold. Sparse storage needs about 0.5 to 5 GB per fold, depending on how many variants the small slab holds (a few large-slab effects per gene, plus the small slab's members). While sampling, the non-zero entries accumulate in memory per gene and are written at the end.

**What is needed to apply the samples elsewhere.** The effects are on the training-fold standardized-genotype scale, so a per-gene variant file is saved with: variant id (`chr_pos_ref_alt_b38`), the counted allele, and the training-fold mean dosage $m_j$ and standard deviation $s_j$. For new dosages $G$ (counting the same allele), genetic expression under sample $s$ is

$$
u^{(s)} = \sum_j \frac{G_j - m_j}{s_j}\, \beta_j^{(s)}, \qquad \text{or per allele: } \beta_j^{(s)} / s_j ,
$$

on the scale of the residualized, renormalized GTEx expression of the tissue. One small module, `posterior_beta_samples.py`, writes and loads the samples and computes $u^{(s)}$ for a new genotype matrix after matching variants and alleles (flipping signs where the counted allele differs). The held-out evaluation (section 7) uses this same function, so the route to other data is exercised in every run.

Also saved:

- per gene, $S$ samples of $\sigma^2_{e,g}$ (for the posterior predictive in section 7)
- per gene, per variant: PIP, $P(z_j = c)$ for each slab (Rao-Blackwellized), posterior mean of $\beta_j$
- per gene: expected number of non-zero effects in each slab
- hyper-parameter trace for every iteration, and samples at the retained iterations

## 7. Held-out evaluation, per gene

Every quantity below, observed and expected, is computed **per gene and per fold**: the posterior fit without fold $k$ is evaluated on fold $k$'s held-out individuals. Each gene therefore has four values of each quantity. When folds are pooled in the plots, confidence intervals are clustered by gene (any two folds' training sets share half their individuals).

Notation for one gene in held-out fold $k$: $y$ is the measured expression of the $n_{te}$ held-out individuals, and $u^{(s)} = X^{te} \beta^{(s)}$ is their genetic expression under effect sample $s$. It is computed from the saved samples and the held-out dosages, standardized with the training means and standard deviations, by `posterior_beta_samples.py`. The prediction is the posterior mean

$$
\hat y = \frac{1}{S} \sum_s u^{(s)} = X^{te} \mu, \qquad \mu = E[\beta \mid \text{training data}] .
$$

All variances and covariances are across held-out individuals, so every expected quantity is conditional on the held-out genotypes: their allele frequencies and LD, with no reference panel. The held-out expression $y$ enters only the observed quantities.

Standardizing the held-out genotypes with the training mean and standard deviation:

- Scaling by the training standard deviation is required, because $\beta_j$ is an effect per training standard deviation (per allele, $\beta_j / s_j$).
- Centering with the training mean has no effect on the evaluation, because every quantity is a variance, covariance or correlation across held-out individuals.
- The held-out columns do not have exactly mean 0 and variance 1. The gdl closed forms relied on that (for example $R_{jj} = 1$ in $E[\operatorname{Var}(X\beta)] = V + \sum_j \sigma_j^2$). Here every expectation is computed directly from the samples ($\operatorname{Var}_i(u^{(s)})$ and so on), so nothing depends on it.
- The one exception is `expected_r_plugin`, which assumes $\operatorname{Var}(y) = 1$. That holds approximately because the expression is renormalized over all EUR individuals.

**Observed**

| Column | Definition |
|---|---|
| `observed_r` | $r_{obs} = \operatorname{corr}(\hat y, y)$ |

**Cis-SNP heritability, training and held-out folds**

These are estimated from genotypes and expression alone, without the fitted model, and separately in the training and held-out individuals of the fold. Each uses the gene's cis variants (the training-filtered set), standardized within that set of individuals for the GRM. The suffix is `_train` or `_test`.

| Column | Definition |
|---|---|
| `cis_snp_h2_{train,test}` | Maximum-likelihood $h^2$ of the single-component model $y \sim N(0, \sigma^2[h^2 K + (1 - h^2) I])$, with $K$ the cis GRM, bounded to $[0, 1]$ (gdl `estimate_cis_snp_heritability_with_lrt`) |
| `cis_snp_h2_pvalue_{train,test}` | Likelihood-ratio test of $h^2 = 0$, with a 50:50 mixture of $\chi^2_0$ and $\chi^2_1$ because $h^2 = 0$ is on the boundary (same function) |
| `cis_snp_h2_he_{train,test}` | Haseman-Elston $h^2$: unbiased and can be negative, so it can be averaged over low-$h^2$ genes (gdl `estimate_cis_snp_heritability_with_he_regression`) |
| `cis_snp_h2_he_se_{train,test}` | Its analytic standard error at the fitted $h^2$ (same function) |
| `cis_snp_h2_he_pvalue_{train,test}` | One-sided test of $h^2 = 0$: $1 - \Phi(\hat h^2_{HE} / \text{SE}_0)$, with $\text{SE}_0$ the same analytic standard error evaluated at $h^2 = 0$ (the null distribution, rather than the fitted $h^2$) |
| `posterior_cis_h2_train`, `_lb`, `_ub` | Model-based: posterior mean and 95% credible interval of $\operatorname{Var}_i(X^{tr} \beta^{(s)})$ over the training individuals |

Training-fold heritability, and its p-value, depend only on training data. Selecting genes on it (for example "heritable genes": `cis_snp_h2_pvalue_train` $< 0.05$) does not bias the held-out evaluation. The gdl heritable-gene set did, because it was selected on the same data it was evaluated on (winner's curse). Held-out heritability (`cis_snp_h2_he_test`) is used for the genetic-scale calibration plot, and to compare against `expected_cis_h2`.

**Expected (from the posterior)**

| Column | Definition |
|---|---|
| `predicted_expression_variance` | $V = \operatorname{Var}_i(\hat y)$ |
| `expected_r_plugin` | $\sqrt{V}$: the gdl plug-in, ratio of expected moments with $\operatorname{Var}(y) = 1$ (step 8 of the gdl doc) |
| `expected_r` | posterior predictive mean of the observed correlation, $\frac{1}{S}\sum_s \operatorname{corr}(\hat y,\ u^{(s)} + \varepsilon^{(s)})$, with $\varepsilon^{(s)} \sim N(0, \sigma^{2\,(s)}_{e} I_{n_{te}})$ drawn once per sample |
| `expected_r_sd` | standard deviation of the same draws (used for the "expected r $-$ 1 SD" ranking) |
| `expected_r_lb_{80,90,95,99}`, `_ub_` | quantiles of the same draws: posterior predictive intervals for $r_{obs}$ |
| `expected_cis_h2` | $\frac{1}{S}\sum_s \operatorname{Var}_i(u^{(s)})$ |
| `expected_genetic_r` | $\frac{1}{S}\sum_s \operatorname{corr}(\hat y, u^{(s)})$: correlation with true genetic expression |
| `predicted_r_variance` | $\operatorname{Var}_s\big[\operatorname{Cov}_i(\hat y, u^{(s)}) / \sqrt{V}\big]$: the gdl $\operatorname{Var}(\rho(\beta))$, now from samples and without assuming independent effects |
| `genetic_FSR` | $\frac{1}{S}\sum_s 1\big[\operatorname{Cov}_i(\hat y, u^{(s)}) < 0\big]$: the gdl directional FSR (prediction points the wrong way relative to true genetic expression) |
| `expression_FSR` | $\frac{1}{S}\sum_s 1\big[\operatorname{corr}(\hat y, u^{(s)} + \varepsilon^{(s)}) < 0\big]$: probability that the **observed** held-out correlation is negative |

Why two FSRs and two expected correlations:

- `genetic_FSR` is the quantity of interest but cannot be checked gene by gene, because true genetic expression is not observed. The fraction of genes with $r_{obs} < 0$ includes the sampling noise of $n_{te} \approx 140$ individuals, so comparing it with `genetic_FSR` (as the gdl FSR plot does) conflates the two. `expression_FSR` includes that noise, so the fraction of genes with $r_{obs} < 0$ should match it exactly if the model is right. Both are reported, and `expression_FSR` is the calibration target.
- Likewise, `expected_r` includes the noise of $n_{te}$ and the posterior of $\sigma^2_e$, and `expected_r_plugin` does not. They should be close, except for genes with a small $V$.

Also written per gene: `n_train`, `n_test`, `n_cis_variants`, `n_cis_variants_found` (always all of them here, but not necessarily when the samples are applied to other data), `sum_pip`, `max_pip`, and `expected_n_slab1` and `expected_n_slab2` (sums of the slab probabilities). These are training-fold quantities too, so selecting genes on them (for example "genes with a confident eQTL") does not bias the held-out evaluation either.

## 8. Plots

One set per tissue, with folds as replicates (separate panels, or pooled with fold as a stratum). Each plot is made for two gene sets: all genes, and heritable genes (`cis_snp_h2_pvalue_train` $< 0.05$, selected on training data only).

1. **Correlation calibration.** 10 equal-count bins of `expected_r`: mean `observed_r` with a 95% CI against mean `expected_r`, with the line $y = x$. Repeated with `expected_r_plugin`.
2. **Genetic-scale calibration.** As gdl Plot 2: bins of `expected_genetic_r` against bin mean $r_{obs}$ / $\sqrt{\text{bin mean } \hat h^2_{test}}$.
3. **FSR calibration.** Bins of `expression_FSR` (0 to 0.1, 0.1 to 0.2, 0.2 to 0.3, 0.3 to 0.4, above 0.4, as in gdl): the fraction of genes with $r_{obs} < 0$ (binomial CI) against mean `expression_FSR`. Repeated with `genetic_FSR` for comparison with gdl.
4. **Coverage.** Nominal level (80, 90, 95, 99%) against the fraction of genes whose $r_{obs}$ falls inside its posterior predictive interval.
5. **Top-N genes.** Mean $r_{obs}$ of the top $N$ genes ranked by `expected_r`, $1 -$ `expression_FSR`, `expected_r` $-$ 1 SD, `expected_cis_h2`, `max_pip`, `sum_pip` and, as model-free baselines, `cis_snp_h2_he_train` and $-\log_{10}$ `cis_snp_h2_pvalue_train`.
6. **Hyper-parameters.** Traces (convergence), and posterior mean with 95% interval per fold. The implied prior on per-gene cis-$h^2$ and on the number of causal variants per gene, split by slab.
7. **Heritability.** (a) Calibration: 10 equal-count bins of `expected_cis_h2`, with bin mean `cis_snp_h2_he_test` against bin mean `expected_cis_h2`. (b) Training against held-out: `posterior_cis_h2_train` and `cis_snp_h2_he_train` against `cis_snp_h2_he_test`, binned. This shows how much the in-sample estimates overstate held-out heritability.

## 9. Files

Same layout as `ca_qtls/fine_mapping` and `gdl_uncertainty_paper`: `driver_key.sh` holds the input paths and output directories, with each step in an `if false; then` block. Each step is an sbatch `.sh` wrapper around a Python script, and plots come from an R script.

| Step | Files | Runs per |
|---|---|---|
| 0. Fold assignment | `generate_cross_validation_folds.py` / `.sh` | once (all GTEx individuals) |
| 1. Prepare input | `prepare_gtex_tissue_data_for_gibbs.py` / `.sh`: per chromosome, expression of the tissue's individuals, `int8` alt-allele dosages for the union of cis variants (sorted by position, so each gene's variants are a contiguous range), and a summary file. Fold-agnostic. | tissue |
| 2. Fit | `eqtl_mixture_prior_gibbs.py` (numba kernels and the `EQTL_MIXTURE_PRIOR_GIBBS` class, as `cofine.py`); `run_eqtl_mixture_prior_gibbs.py` / `.sh` (loads input, applies the fold, fits, writes section 6) | tissue × fold |
| (shared) | `posterior_beta_samples.py`: write and load the sparse effect samples, and compute $u^{(s)}$ for any genotype matrix | — |
| 3. Evaluate | `evaluate_held_out_expression_prediction.py` / `.sh` (section 7): loads the effect samples and the fold's training and held-out dosages and expression from the step 1 input | tissue × fold |
| 4. Organize and plot | `organize_and_visualize_cross_validation_results.sh`, calling `organize_cross_validation_results.py` and `visualize_cross_validation_results.R` (section 8) | tissue |

For testing, `run_eqtl_mixture_prior_gibbs.py` takes `--max_genes` and `--chromosomes`.

Output root: `/lab-share/CHIP-Strober-e2/Public/ben/eqtl_pred_fsr/`, with `cv_folds/`, `gibbs_input/`, `gibbs_results/`, `held_out_evaluation/`, `visualization/`.

## 10. Compute (rough, Whole Blood)

- About 17k genes and several hundred variants per gene at $\pm 100$ kb. One iteration is about $\sum_g n_{tr}\, p_g \approx 420 \times 600 \times 17{,}000 \approx 4 \times 10^9$ multiply-adds, about 4 s on one core. 3,000 iterations take about 3 to 4 hours per tissue and fold.
- Cores: one. The 4 jobs per tissue (one per fold) already run in parallel as separate sbatch jobs. The gene loop is a numba `prange`, so a job given more cores runs faster without code changes (about 0.5 hour on 8 cores). That is worth it only for larger windows.
- Memory: genotypes take a few GB as `int8`, and the sparse effect samples up to a few GB (section 6). An sbatch request of `-c 1 --mem=40G` should suffice.
- Disk: effect samples of roughly 2 to 20 GB per tissue (4 folds), depending on how much the small slab holds.
- A $\pm 1$ Mb window multiplies the variants, and the time, by about 10.
- Evaluation: the cis-$h^2$ estimates need $n \times n$ GRM operations per gene. The costliest are the eigendecomposition for the likelihood-ratio test and the Haseman-Elston standard errors on the 420 training individuals, about $10^8$ operations per gene, or under an hour per tissue and fold on one core.

## 11. Known issues and open decisions

1. **Covariates were fit on all individuals.** The residualized expression used GTEx PEER factors computed from every sample, held-out ones included. This leaks no genotype information and is standard, but it is not a perfectly clean split. Remedy if needed: re-residualize within fold (the PEER factors would still come from all samples).
2. **Effect scale.** The prior is on standardized genotypes, as in gdl and S-LDMC, which implies larger per-allele effects for rarer variants. The alternative is a per-allele prior. The default is standardized.
3. **Window.** $\pm 100$ kb (matches gdl, cheap) or $\pm 1$ Mb (GTEx cis window). Effects beyond 100 kb are missed, which lowers $r_{obs}$, but calibration should still hold because the expected quantities are for the same window.
4. **Monte Carlo coupling.** $\hat y$ is the mean of the same $S$ samples it is compared with, which biases `genetic_FSR` down by $O(1/S)$. With $S = 500$ this is small. Check: compute $\hat y$ from the odd samples and the FSR from the even ones.
5. **Mixing.** Random-scan Gibbs mixes slowly between variants in near-perfect LD. This should not affect predictions or the expected quantities, because those variants carry the same signal in held-out individuals of the same population. Check: two chains with different seeds, comparing $\hat y$ and `expression_FSR` per gene.
6. **Two-slab identifiability.** If one slab holds only a handful of variants, its variance is poorly determined under the near-flat prior and its draws are noisy. The ordering constraint stops label switching, but a slab with near-zero weight means a single slab would have been enough. The hyper-parameter plots (plot 6) show this.

   This happened in the small simulation test (40k variant-gene pairs, with a small slab too weak to detect at $n \approx 420$). Under the near-flat Inverse-Gamma prior, the small slab's variance repeatedly collapsed toward $10^{-10}$, where it acts as a second spike. Once, the large slab emptied and kept an extreme variance ($\sim 36$) for a while before recovering. Predictions, FSRs and intervals stayed calibrated, so this mostly affects interpreting the hyper-parameters. Real data have about 250 times more pairs, so the slabs should be better identified. If the traces still wander, the remedy is a mildly informative Inverse-Gamma on the slab variances, or a floor on $\sigma^2_1$.
7. **One fold or four.** The default is all four. A single fold is a quarter of the compute, but each calibration plot then rests on one split.

## 12. Validation plan

1. **Small run.** One chromosome, `--max_genes 200`. Check hyper-parameter traces and per-gene runtime, and that fixing $\pi_0 = 1$ gives $\hat y = 0$.
2. **Simulation with the model true.** Simulate expression from the real Whole Blood genotypes with effects drawn from a known mixture prior, then run steps 2 to 4 unchanged. All calibration plots must lie on $y = x$ and the hyper-parameters must be recovered. Real-data deviations can then be attributed to model misspecification rather than to the code.
3. **Real data**, Whole Blood, four folds. Then the other GTEx tissues by looping over the tissue list in `driver_key.sh`, as in `gtex_eqtl_expression_processing/driver_key.sh`.
