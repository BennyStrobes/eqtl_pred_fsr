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
- Standard random-scan Gibbs by default. This is prediction, not fine-mapping: when variants in high LD trade an effect between them, the prediction barely changes, so slow mixing between LD partners should not matter much here. Two tempered Gibbs alternatives of the same model check this: a full tempered sampler (section 5b) and a GWFM-style sampler as in GCTB's genome-wide fine-mapping (section 5c). Their downstream analyses can be compared with the standard fit's. In simulations (section 5d) all three gave the same predictions and intervals, while the tempered samplers fixed standard Gibbs's over-confident PIPs among near-duplicate variants.

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

## 5b. Tempered Gibbs alternative

`run_eqtl_mixture_prior_gibbs.py --tempered` (`run_eqtl_mixture_prior_gibbs.sh` with `tempered` as its 8th argument; steps 2b to 4b of `driver_key.sh`) fits the **same model** with a tempered Gibbs sampler (TGS; Zanella & Roberts 2019), as in the colocalization sampler `eb_coloc/multi_causal_variant/sims/ld/mcvc_tempered.py`. The outputs have the same format, so evaluation, organization and plots run unchanged. Results are named `${tissue}_tempered_*` so that both fits can be compared. Code: `eqtl_mixture_prior_tempered_gibbs.py`.

**Class updates.** Per gene and sweep, $p_g$ sub-steps (`--tgs_sub_steps_per_snp`). Each variant has its full conditional $P_j(\cdot)$ (section 5, step 1) and a selection weight $p_j = g_j(z_j) / P_j(z_j)$, with $g_j = (1 - \epsilon) P_j + \epsilon / 3$ ($\epsilon$ = `--tgs_epsilon`, 0.5). A sub-step selects $j$ with probability $\propto p_j$, draws $z_j$ from $g_j$ and $\beta_j$ from its exact conditional given $z_j$, then updates the residual. A variant in an improbable class (for example, two LD partners both carrying one signal) is selected quickly. This lets an effect move between LD partners, which single-site Gibbs essentially never does once one partner holds a strong signal.

**Weights.** The chain targets $\pi(x) Z(x)$ with $Z = \operatorname{mean}_j p_j$, so posterior expectations weight each visited sub-state by $1/Z$:

- PIPs, slab probabilities and posterior means: $1/Z$-weighted Rao-Blackwellized averages over every post burn-in sub-state.
- **Retained samples:** for each gene, one sub-state drawn with probability $\propto 1/Z$ among the sub-states since the previous retained iteration (`thin` sweeps). The saved effect samples are therefore approximately equally weighted posterior draws. The matching $\sigma^2_{e,g}$ sample is drawn from $\text{Inverse-Gamma}(a + n_{tr}/2, b + r^\top r/2)$ given that sub-state's residual.
- **Hyper-parameters** ($\pi$, $\sigma^2_1$, $\sigma^2_2$, $\sigma^2_{e,g}$) are drawn from the same conditionals as in section 5, given importance-weighted *expected* class counts, sums of squares and residual sums of squares, pooled over the last 30 sweeps (`--tgs_window_sweeps`). This is an approximation, not an exact Gibbs step. The self-normalized weights bias a sweep's expected counts slightly upward, and the bias feeds back through $\pi$. In simulations (section 5d), pooling 1 sweep sent $\pi_1$ from 0.008 to 0.2, with $\sigma^2_1 \approx 5 \times 10^{-5}$ and PIP calibration errors of 0.2. Pooling 3 sweeps drifted slowly, 10 held $\pi_1$ but let $\sigma^2_1$ creep up, and 30 stayed close to the true values, with no drift over 10,000 iterations. The window costs only memory.

**Individual-level data.** The conditional of a variant depends on the data only through its single-variant regression coefficient $b_a$. The sweep keeps $b_a$ for every variant. After a move of $j$ by $d$, it updates $b_a \leftarrow b_a - d R_{aj}$ for the variants in training-fold LD $r^2 \ge 0.01$ with $j$ (`--tgs_r2_update_threshold`; 0 is exact TGS), and refreshes all of them exactly at the start of the next sweep. The selected variant's $b_j$ is recomputed exactly from the residual unless it is known to be exact already. The residual is updated lazily, before the next exact recomputation. The sparse per-gene LD (int16 index and quantized $R$, 4 bytes per neighbour pair) is built once per fit, and its size is printed (Whole Blood: about 330 neighbours per variant, 10 GB).

**Speed.** Each move changes the $b_a$ of a few hundred LD neighbours. A neighbour's conditional and selection weight are recomputed only once $b_a$ has moved enough that its selection weight may have changed by a relative `--tgs_selection_tolerance` (default $10^{-3}$; an exact bound, 0 recomputes every neighbour). The selected variant is drawn through per-block sums of the selection weights.

**Checks.** On a small exact test (7 variants including a pair at $r = 0.99$, fixed hyper-parameters, all $3^7$ configurations enumerated), PIPs and posterior means averaged over 8 chains matched the exact posterior within 0.003 for tolerances 0, $10^{-3}$ and $10^{-2}$. Standard Gibbs was off by 0.017 on the LD pair, with twice the chain-to-chain spread.

**Cost.** On simulated data resembling Whole Blood (640 variants per gene, 50% of them LD neighbours, $\sigma^2_1 = 10^{-4}$), one core takes about 4.5 times the standard sampler per iteration at the default tolerance and 3.8 times at $10^{-2}$. The first version took 7.6 times. Further cuts trade accuracy or mixing for speed: `--tgs_r2_update_threshold 0.05` (3.4 times), `--tgs_epsilon 0.25` (3.6 times), `--tgs_sub_steps_per_snp 0.5` (2.8 times). Genes run in parallel through numba `prange`: 4 cores ran 3.1 times faster than 1. Request cores for step 2b (for example `sbatch -c 8`).

## 5c. GWFM-style tempered Gibbs alternative

`run_eqtl_mixture_prior_gibbs.py --tempered --tgs_mode gwfm` (`run_eqtl_mixture_prior_gibbs.sh` with `tempered_gwfm` as its 8th argument; steps 2c to 4c of `driver_key.sh`, results `${tissue}_tempered_gwfm_*`) fits the same model as GCTB's genome-wide fine-mapping does (`--gwfm`, algorithm `TGS_thin` in `github.com/jianzeng/GCTB`, `scr/model.cpp`, `sampleFromTGS_eigen`). Code: `eqtl_mixture_prior_gwfm_tempered_gibbs.py`.

- **Every iteration:** the standard Gibbs sweep of section 5. Effect samples, hyper-parameters and residual variances all come from this chain, as in the standard sampler.
- **Every 10th iteration** (`--gwfm_interval`), tempered blocks on small sets. For each variant with running PIP $\ge 0.05$ (`--gwfm_pip_threshold`; mean over the previous iterations) that has an LD partner in the gene at training-fold $r^2 \ge 0.95$ (`--gwfm_r2_threshold`), the set is that variant plus its partners. A block is a fully tempered TGS, as in GCTB: a member is selected with probability $\propto 1/(3 P_a(z_a))$, its class is drawn uniformly from the two other classes, and its effect from its exact conditional. The members' $b_a$ are kept exact through their pairwise LD.
- **End of a block:** the block ends in one of its sub-states, drawn with probability $\propto 1/Z$, after $\max(20, 10 |S|)$ sub-steps (`--gwfm_min_steps`, `--gwfm_steps_per_snp`). The $1/Z$-weighted slab probabilities and effects over the block replace the members' contributions of that iteration to the PIPs and posterior means, as GCTB replaces the members' PIPs.

**Two departures from GCTB, both checked on the exact test of section 5b** (7 variants, an LD pair at $r^2 = 0.974$, fixed hyper-parameters, 16 chains):

1. GCTB runs a block only when the focal variant's current class is non-zero. Whether the block runs then depends on the state the block resamples, which biases the chain. The pair's PIPs were off by 0.014 (10 standard errors), and by 0.029 with longer blocks. Here a block runs on the running PIP alone, and the error is 0.001 (within 1 standard error). Standard Gibbs had a chain-to-chain standard deviation 4 times larger on that pair.
2. GCTB continues from the block's last sub-state, which is distributed as the tempered target, not the posterior. Applying GCTB's rule as written (block only when the focal class is non-zero, last sub-state, $|S|$ sub-steps) put the pair's PIPs off by 0.09 in this reimplementation. This is not GCTB itself: it has 3 classes, individual-level data and exact slab-to-slab updates.

**Cost:** close to the standard sampler, about 3% more time on simulated data with 54% of variants in a near-duplicate pair.

## 5d. Comparison of the three samplers in fine-mapping simulations

**Design.** 400 genes × 500 variants per scenario. Haplotypes are mosaics of a few founders, with noise and shared branches, so that 53% of variants have a training-fold partner at $r^2 > 0.95$. Individuals: 560 in 4 folds, 420 for training. Causal variants have MAF $\ge 0.02$. Effects follow the fitted prior, $\pi = (0.9895, 0.0085, 0.002)$ and $\sigma^2 = (10^{-4}, 0.04)$, except where noted.

Scenarios:

- **A1, A2:** replicate datasets.
- **B:** 75% of variants with a near-duplicate partner.
- **C:** every large effect gets a second large effect on a variant in $r^2$ 0.3 to 0.9 with it.
- **D:** misspecified, with 0.4% causal variants and Laplace effects.
- **E:** the dataset of the first comparison with 10,000 iterations.

A to D ran 3,000 iterations with 1,500 burn-in. Each sampler ran 2 chains, with all other settings at their defaults. Full TGS used the 30-sweep hyper-parameter window.

**Reading the table.** PIP calibration error is the bin-size-weighted mean |predicted − observed| over PIP bins $\ge 0.1$. Each cell lists standard / full TGS / GWFM TGS.

| scenario | PIP calibration error | same, variants with an $r^2 > 0.95$ partner | false PIP $\ge 0.95$ per chain | PIP $\ge 0.9$: expected : observed causal | between-chain PIP correlation, near-duplicates | held-out $r$, 90% interval coverage |
|---|---|---|---|---|---|---|
| A1 | 0.068 / 0.035 / 0.030 | 0.105 / 0.038 / 0.026 | 7.5 / 0 / 0.5 | 48:39 / 29:30 / 32:31 | 0.50 / 0.98 / 0.99 | 0.90 / 0.90 / 0.91 |
| A2 | 0.052 / 0.020 / 0.027 | 0.092 / 0.033 / 0.041 | 8.0 / 1.0 / 0.5 | 61:54 / 42:42 / 43:43 | 0.39 / 0.98 / 0.97 | 0.88 / 0.87 / 0.88 |
| B | 0.057 / 0.019 / 0.017 | 0.068 / 0.016 / 0.018 | 10.0 / 0 / 0 | 40:30 / 21:22 / 22:22 | 0.53 / 0.97 / 0.97 | 0.89 / 0.89 / 0.90 |
| C | 0.052 / 0.023 / 0.016 | 0.064 / 0.024 / 0.022 | 14.0 / 1.5 / 1.5 | 80:66 / 55:52 / 56:54 | 0.50 / 0.97 / 0.97 | 0.87 / 0.88 / 0.87 |
| D | 0.039 / 0.033 / 0.024 | 0.067 / 0.024 / 0.036 | 1.5 / 0 / 0 | 23:22 / 17:17 / 16:16 | 0.76 / 0.98 / 0.99 | 0.88 / 0.85 / 0.86 |
| E (10k) | 0.013 / 0.019 / 0.008 | 0.028 / 0.028 / 0.022 | 0 / 0 / 0 | 55:55 / 52:53 / 52:53 | 0.66 / 0.99 / 0.99 | 0.89 / 0.89 / 0.88 |

- **Fine-mapping.** At 3,000 iterations, standard Gibbs is over-confident. It puts a near-duplicate pair's signal on one member, and the member it picks differs between chains. Its PIP $\ge 0.95$ hits included 1.5 to 14 false positives per chain, and its PIP $\ge 0.9$ variants expected 5 to 33% more causal variants than were present. Both tempered samplers removed most of this and agreed between chains. GWFM matched full TGS everywhere.
- **Longer chains:** with 10,000 iterations (E), standard Gibbs pooled over chains was calibrated, but its chains still disagreed on near-duplicates (correlation 0.66).
- **Group level:** summing PIPs within a near-duplicate group gave similar calibration for all three samplers. Standard Gibbs gets a group's total right but puts it on the wrong member.
- **Predictions** (held-out $r$, its intervals, `expression_FSR`) were the same for the three samplers in every scenario, as section 11, issue 5 expects.
- **Hyper-parameters.** The standard and GWFM chains, whose hyper-parameter steps are exact, often emptied one slab: $\min(\pi_1, \pi_2) < 10^{-4}$ in part of the post-burn-in iterations of 8 of 12 standard chains and 7 of 12 GWFM chains. The emptied slab's variance then wandered (for example $\sigma^2_2 \approx 10^{2}$ to $10^{3}$ in E). Full TGS never did: its expected-statistics update (section 5b) keeps both slabs populated, and with the 30-sweep window it recovered $\pi_1 \approx 0.008$ to $0.01$ and $\sigma^2_1 \approx 1.2$ to $1.8 \times 10^{-4}$. That stability comes from the approximation, not from better mixing, so it does not show that the exact posterior of the hyper-parameters is that tight (section 11, issue 6).
- **Speed** (one core, nothing else running, the A1-sized data): standard 0.046 s per iteration, full TGS 0.183 (4.0 times), GWFM 0.048 (1.03 times). On 4 cores: 0.014, 0.060 and 0.015.

**Recommendation:** GWFM for fine-mapping at the standard sampler's cost. Full TGS when stable hyper-parameter estimates matter more than a fourfold cost.

## 5e. Fixed-grid mixture prior (alternative model)

`run_eqtl_grid_prior_gibbs.py` (`run_eqtl_grid_prior_gibbs.sh`; code `eqtl_grid_prior_gibbs.py`, and with `--gwfm` / `gwfm` as the 8th argument `eqtl_grid_prior_gwfm_tempered_gibbs.py`) fits a second model to the same data (steps 2d to 4e of `driver_key.sh`; results `${tissue}_grid_*` and `${tissue}_grid_gwfm_*`). The existing model's code is untouched.

**Model.** As section 4, but the effect prior has fixed component variances, as in BayesR (Erbe et al. 2012) and close to ash / mr.ash:

$$
\beta_{gj} \sim \pi_0\, \delta_0 + \sum_{k=1}^{K} \pi_k\, N(0, \sigma^2_k), \qquad \pi \sim \text{Dirichlet}(\alpha, \ldots, \alpha).
$$

- **Grid** (`--grid_variances`): half-decades from $10^{-4}$ to 1, so $K = 9$.
- **Dirichlet** $\alpha$ = 0.1 (`--dirichlet_alpha`).

Only $\pi$ is learned, so no variance can collapse and labels cannot switch.

**Samplers:**
- **Standard Gibbs:** section 5 with $K + 1$ classes. The per-variant cost is the same $O(n_{tr})$ dot product plus $K + 1$ exponentials.
- **GWFM:** section 5c with $K + 1$ classes, as GCTB does for its classes. A member is selected with probability $\propto 1/((K+1) P_a(z_a))$, and its new class is drawn uniformly from the $K$ others.
- **$\pi$ draws** are in log space and floored only at the smallest positive double. The two-slab model's floor of $1/n_{pairs}$ held the empty large-variance components (0.32 and 1) far above their posterior weight. In a 164k-pair simulation that inflated the prior's per-variant variance $E[\beta^2]$ by 25%.

**Outputs.** The same files as section 6, so evaluation, organization and plots run unchanged:
- **PIP:** the probability of any non-zero component.
- **"Slab 1" / "slab 2"** (`pair_p_slab1/2`, `expected_n_slab1/2`): the small group (variance $<$ `--small_large_cutoff`, $10^{-3}$) and the large group.
- **Hyper-parameters:** pi0, pi1, pi2, sigma2_1 and sigma2_2 are the collapsed two-group mixture. pi1 is the small group's total weight and sigma2_1 its weight-averaged variance, and likewise for the large group. The prior-implied cis-$h^2$ of plot 6 is therefore exact.
- **Additional files:** `grid_weight_trace.txt` and `grid_weight_summary.txt` (per component: variance, group, posterior mean and 95% interval).

**Checks.** The exact test of section 5b, with a spike and 3 components, enumerated all $4^7$ configurations. Both kernels matched the exact PIPs, group probabilities and posterior means. Grid GWFM was within 2 SE everywhere. Grid standard Gibbs mixed slowly on the $r = 0.987$ pair: single chains were off by up to 0.036, as with the two-slab model, and its chain-to-chain spread was 3.8 times GWFM's.

**Simulations** (section 5d design, 2 chains each):
- **G1:** effects drawn from a grid prior. $\pi$ = 0.006, 0.003, 0.0015 and 0.0005 on variances $10^{-4}$, $10^{-3}$, $10^{-2}$ and $10^{-1}$.
- **A1:** the two-slab truth, whose large variance (0.04) is off the grid.
- **D:** Laplace effects.

PIP calibration error (bins $\ge 0.1$), standard / GWFM:

| scenario | two-slab model | grid model |
|---|---|---|
| G1 | 0.086 / 0.045 | 0.075 / 0.038 |
| A1 | 0.068 / 0.030 | 0.067 / 0.041 |
| D | 0.039 / 0.024 | 0.043 / 0.027 |

- **PIPs.** Grid GWFM had no false PIP $\ge 0.95$ hits, and its chains agreed on near-duplicates (correlation 0.95 to 0.97). Grid standard Gibbs was over-confident, as standard Gibbs is under the two-slab prior. Predictions and their intervals were the same for all four samplers.
- **Prior recovery at $n_{tr} = 420$.** Fitted against realized $P(|\beta| > t)$ among the analysed pairs:
  - In G1 and A1 (off-grid truth), the grid model was within about 25% for $0.02 \le t \le 0.4$.
  - In D (Laplace), it was within 16% for $0.05 \le t \le 0.2$. At $t = 0.02$ grid Gibbs was 2 times too high (two-slab GWFM: 2.7 times). At $t = 0.4$, which covers about a dozen pairs, both grid samplers were 1.5 times too high (two-slab GWFM: 1.7 times).
  - Its $E[\beta^2]$ was within 1 to 14% everywhere.
- **What is not identified at $n_{tr} = 420$:** effects below about 0.01 (variance $\lesssim 10^{-4}$, under 1/20 of a variant's sampling variance) and the weights of individual half-decade components. Chains put the small-effect mass anywhere from 0.3 to 4 times the truth, and neighbouring components trade weight between chains. The two-slab model has the same limitation (section 11, issue 6).
- **Starting the grid at $10^{-5}$** ("grid from 1e-5", 11 components) left PIP calibration and the tails above 0.02 unchanged. It moved more of the unidentified small-effect mass into the near-spike components.

**Prior recovery with more data.** G1's prior was simulated again with 2,000 individuals ($n_{tr} = 1{,}500$):
- **Identified components:** both grid samplers recovered the weights of the $10^{-2}$ and $10^{-1}$ components (truth $1.40 \times 10^{-3}$ and $5.4 \times 10^{-4}$; the four chains gave $1.4$ to $1.7 \times 10^{-3}$ and $4.5$ to $5.1 \times 10^{-4}$), and left the empty neighbours (0.032, 0.32, 1) near zero.
- **Tails and variance:** $P(|\beta| > t)$ was within 6% for $t \ge 0.05$, and $E[\beta^2]$ within 5%.
- **Small end:** variances $10^{-4}$ to $10^{-3}$, whose effects have $z \approx 1$, were still placed differently by different chains.
- **Two-slab model on the same data:** right for the large tail, but $P(|\beta| > 0.01)$ was about half the truth. Two slabs cannot represent the four components.

The sampler therefore recovers what the data identify, and the remaining disagreement is identifiability, not a sampler error.
- **PIP calibration error (bins $\ge 0.1$):** grid GWFM 0.031, two-slab GWFM 0.026. Grid standard Gibbs (0.055) was over-confident, as standard Gibbs is.

**Cost** (same batch, so comparable): the grid samplers took 14% longer than the two-slab samplers at $n_{tr} = 420$, and 5% longer at $n_{tr} = 1{,}500$, where the dot products dominate the $K + 1$ exponentials.

**Recommendation:** grid GWFM, for the reasons section 5d gives for GWFM. Read the grid weights for components with variance below about $10^{-3}$ (at GTEx sample sizes) as not identified, and summarize the prior through its tails or $E[\beta^2]$ rather than individual weights.

## 5f. Self-consistency check of the shared prior

**Question.** Both priors (two-slab and grid) draw every variant's effect, in every gene, from one shared distribution. Real genes differ: some have several independent eQTLs, many have none. If the shared prior is too homogeneous, the posterior can be too confident for individual genes even when it is right on average.

**Identity.** If each gene's effects were drawn from the fitted prior, one posterior sample per gene, pooled over genes, would be distributed as draws from the prior (the identity behind simulation-based calibration). `check_prior_self_consistency.py` compares two per-gene statistics of one posterior sample with the same statistics simulated from the prior, for the gene's $M_g$ cis variants and the hyper-parameters of that sample:
- $K_g(t)$, the number of variants with $|\beta| > t$ ($t$ = 0.05, 0.1, 0.2). Under the prior $K_g \sim \text{Binomial}(M_g, q)$ with $q = \sum_c \pi_c \, 2\Phi(-t/\sigma_c)$.
- $S_g = \sum_j \beta_{gj}^2$, an LD-free cis-$h^2$.

Each of 100 replicates draws a new posterior sample per gene and a new prior simulation. The prior interval over replicates is the spread of each statistic for this many genes under the prior, the reference for the observed value.

**Reading it.**
- **Pearson dispersion** of $K_g$, $\sum_g (K_g - M_g q)^2 / \sum_g M_g q (1 - q)$, is 1 under the prior. Above 1 (more genes with no and with several large effects than the binomial allows) means genes are more heterogeneous than the shared prior.
- The implied beta-binomial $\text{Beta}(a, b)$ is a starting point for a gene-specific inclusion probability.
- **Slope of $K_g$ on $M_g q$** is 1 under the prior. Below 1 means counts rise more slowly with the number of cis variants than the prior assumes. The dispersion around the fitted line is the over-dispersion that the slope does not explain. (The simulations have nearly equal $M_g$, so they cannot test the slope; its prior interval there is $-0.6$ to $2.5$.)
- Genes with little information have posterior close to the prior and match it by construction, so the signal is muted: a lower bound on the heterogeneity.
- Use every gene. Restricting to genes chosen by their data (for example heritable genes) breaks the identity.

**Simulations** (400 genes, $n_{tr} = 420$; 200 replicates):

| scenario | fit | Pearson dispersion of $K_g(0.1)$: observed (prior 95% interval) | z | fraction $K_g(0.1) \ge 3$: observed (prior) |
|---|---|---|---|---|
| A1: effects from the two-slab prior | standard / GWFM | 0.96 / 0.96 (0.81 to 1.27) | −0.4 / −0.5 | 0.011 / 0.011 (0.015) |
| G1: effects from a grid prior | grid GWFM | 1.05 (0.80 to 1.24) | 0.4 | 0.006 (0.003) |
| H: gene-specific inclusion, $\pi_g = \pi f_g$, $f_g \sim \text{Gamma}(0.5, 2)$ | standard / GWFM | 1.39 / 1.40 (0.83 to 1.20) | 3.8 / 3.8 | 0.040 / 0.039 (0.019) |

- **Correctly specified fits pass.** Every $|z|$ was below 1 for the dispersion and $S_g$ statistics; rare-count cells (prior fraction near 0) have unreliable z and should be read from the interval.
- **Heterogeneity is detected at 400 genes.** The true dispersion of large-effect counts in H was about 2.8. The posterior draws show 1.4: about a fifth of the excess, the muting expected from weakly informative genes. The implied $a + b \approx 1{,}000$ is accordingly an upper bound (the simulated heterogeneity corresponds to about 250).
- **$S_g$ is less sensitive** (z about 1.7 for its 99th percentile in H).
- With about 14,000 genes the prior interval is about 6 times narrower, so much smaller heterogeneity is detectable.

**Run:** driver step 5 (fold 1 of each finished fit). Output `${tissue}${label}_fold_1_prior_check_{summary,count_histogram,sum_sq_histogram}.txt`. Step 5b repeats it with the MHC genes left out of the comparison (`--exclude_regions 6:28510120-33480577`): the 137 MHC genes of Whole Blood fold 1 average twice as many large-slab variants as other genes and are 8 of the 11 genes with 5 or more.

**Whole Blood, fold 1, all genes:** the Pearson dispersion of $K_g(0.1)$ was 1.089 (standard), 1.092 (GWFM) and 1.094 (grid), against a prior 95% interval of about 0.97 to 1.03 (z = 6.3 to 6.6). The mean, the fraction of genes with $K_g = 0$ and the fraction with $K_g \ge 3$ all matched the prior ($|z| < 0.7$), so the excess is not in the distribution of counts over genes. Leaving out the MHC changed nothing (1.091, 1.094, 1.096). The count histogram of $K_g(0.1)$ matched the prior in every bin, and the posterior expected number of large-slab variants per gene rose as about $M_g^{0.8}$ (ratio to $\pi_2 M_g$ from 1.24 in the lowest decile of $M_g$ to 0.95 in the highest): counts are right over genes but less tied to $M_g$ than the prior assumes. The slope of $K_g$ on $M_g q$ (standard fit) was 0.89, 0.83 and 0.74 at $t$ = 0.05, 0.1, 0.2, against prior intervals of about 0.91 to 1.13 ($z$ = −4.2 to −5.8), but the dispersion around the fitted line was still 1.064, 1.081 and 1.048: the weak dependence on $M_g$ explains under a tenth of the excess, and the rest is gene-level variation in the number of eQTLs that $M_g$ does not capture. $S_g$ matched for the two-slab fits ($|z| \le 1.2$).

## 5g. Inclusion probability by number of cis variants

**Why.** In the grid GWFM fit of Whole Blood, expression FSR was over-confident in the FSR 0.1 to 0.4 range (observed fraction with $r_{obs} < 0$ 0.022 above the mean FSR, $z = 6.8$), and Monte Carlo coupling was ruled out (section 11, issue 4: the split-sample evaluation changed the excess by at most 0.002). The excess rose with the number of cis variants $M_g$, from about 0 below 450 variants to +0.06 above 740 (`check_expression_fsr_excess.py`). Genes with few variants were calibrated at every level of training evidence, including LRT $p \ge 0.5$; genes with more than 600 variants were over-confident at every level, including $p < 0.05$ (+0.038). The shared prior gives every variant the same inclusion probability, so a gene's prior number of eQTLs, and its prior $h^2$, grow linearly with $M_g$. In a fixed $\pm 100$ kb window, $M_g$ mostly reflects local diversity rather than regulatory content (section 5f: posterior counts grew only as about $M_g^{0.8}$ under the linear prior).

**Model.** The genes are split into $B$ equal-count bins of $M_g$ (`--n_variant_bins`), each with its own inclusion probability:

$$
\beta_{gj} \sim (1 - q_{b(g)})\,\delta_0 + q_{b(g)} \sum_k w_k\, N(0, \sigma^2_k), \qquad q_b \sim \text{Beta}(\alpha, \alpha), \quad w \sim \text{Dirichlet}(\alpha, \ldots, \alpha)
$$

with the grid variances $\sigma^2_k$ and the component weights $w$ shared. No functional form is assumed for $q_b$ against $M_g$: the fitted $q_b$ show it. The prior on the classes factorizes as $\prod_b q_b^{k_b} (1 - q_b)^{M_b - k_b} \times \prod_k w_k^{n_k}$ ($k_b$, $M_b$: non-zero effects and variants of the genes in bin $b$), so both updates are exact and conjugate. Default $B = 20$: about 700 genes and $3.5 \times 10^5$ variant-gene pairs per bin, so each $q_b$ is precise, and the top 5% of $M_g$ (where the excess was largest) has its own bin. Only for the grid prior (standard and GWFM samplers).

**Reading $q_b$.** $q_b$ includes the near-spike grid components, whose weight is not identified (section 5e). `variant_bin_summary.txt` also gives each bin's posterior fraction of variants in the large group (variance $\ge 10^{-3}$), which is identified: plot both against bin mean $M_g$ on log scales. A straight line would justify a two-parameter power law $q_g = q (M_g / \bar M)^{-\gamma}$.

**Simulation check** (300 genes, $M_g$ from 54 to 2,000, $n = 300$, true $q_g = 0.03 (M_g / 300)^{-0.7}$, 5 bins, 900 iterations): posterior $q_b$ fell from 0.070 to 0.021 across the bins against a true 0.051 to 0.016 (ratio first to last bin 3.4 against 3.3), for both samplers. All bins were about 35% high, as was $1 - \pi_0$ of the unbinned fits on the same data, so the level reflects the small simulation, not the bins.

**Outputs** (in addition to section 6): `variant_bin_trace.txt` ($q_b$ after every iteration) and `variant_bin_summary.txt` (per bin: range and mean of $M_g$, genes, pairs, posterior mean and 95% interval of $q_b$, large-group fraction). `grid_weight_trace` and the collapsed hyper-parameters hold the prior averaged over all pairs, and the prior-implied histograms of plot 6 use that average, so they are approximate for these fits. `check_prior_self_consistency.py` uses each gene's bin.

**Run:** driver steps 7 (fit, label `_grid_gwfm_bins20`), 7b (evaluate), 7c (plots, then `check_expression_fsr_excess.py`) and 7d (prior check). **It worked if** the decile gradient of the excess is flat, the mid-range excess falls from +0.022 to about 0, and the section 5f slope of $K_g$ on $M_g q$ moves to about 1.

**Result (Whole Blood, 20 bins).** The prior part worked: $q_b$ fell from 0.027 (mean $M_g$ 190) to 0.012 (1,070), so the prior expected count still rises from about 5 to 13, and the section 5f slope of $K_g$ on its expected count moved from 0.89, 0.83, 0.74 (thresholds 0.05, 0.1, 0.2) to 0.96, 0.94, 0.94, inside the prior interval. The dispersion of $K_g$ did not change (1.065 to 1.09). The calibration did not improve: the mid-range excess was +0.027 ($z = 8.5$), the decile gradient ran from +0.022 to +0.054 (the $\log_{10} M_g$ coefficient halved, to 0.048, because the genes with few variants got worse), and the LRT effects were unchanged (0.027, 0.043). So the gradient of the excess with $M_g$ is not the prior's linear scaling; candidates are what $M_g$ stands for (the kind of variants: section 5h), mixing in dense LD (`compare_sampler_fits.py`), heterogeneity between genes, or residual structure in held-out expression.

## 5h. Prior by variant annotation (check)

**Why.** Section 5g fixed the prior's scaling with $M_g$ without moving the gradient of the expression FSR excess with $M_g$, so $M_g$ may stand for the kind of variants in the window rather than their number: windows with many variants may be rich in rare or high-LD variants, which a prior shared by every standardized variant may over-credit. Two checks on existing fits, before any model change.

**1. Enrichment by annotation** (`check_prior_by_variant_annotation.py` / `.sh`). By the identity of section 5f, applied per variant: in any category of variant-gene pairs defined without the expression data, the posterior rate of $|\beta| > t$ (and the mean $\beta^2$, per-variant cis-$h^2$) averaged over the category equals the prior rate. Categories: training-fold MAF (fixed bins 0.01, 0.02, 0.05, 0.1, 0.2, 0.5), distance to the TSS (0, 2, 5, 10, 20, 50, 100 kb), in-window LD score (sum over the gene's cis variants of the bias-corrected $r^2$ in the training individuals; 5 equal-count bins) and MAF × LD-score tercile. The level over all pairs need not be exactly 1 for a finite chain (the prior mean $\beta^2$ is dominated by large-variance components with small, noisy weights), so the test is of the relative enrichment (category over all pairs), with standard errors clustered by gene. Genes with little information pull every enrichment toward 1 (conservative). Output `<stem>_enrichment.txt` and `<stem>_gene_composition.txt` (per gene: $M_g$, fraction of variants with MAF < 0.05, mean MAF, mean LD score).

**2. Composition in the excess regression** (`check_expression_fsr_excess.py --gene_composition_file`): the regression of section 5g repeated with the fraction of low-MAF variants and $\log_{10}$ mean LD score of each gene's window (from fold 1; genotype-only, nearly the same in every fold).

**Reading.** If categories differ (relative enrichment away from 1, consistently across folds) and the composition covariates take over the $\log_{10} M_g$ coefficient, a prior with an inclusion probability per variant category (the binned code of section 5g, indexed by the pair's category instead of the gene's bin) is the natural model. If enrichment is flat, a stratified prior would change little; if it is not flat but the $M_g$ coefficient survives, a stratified prior may help fine-mapping and prediction without fixing calibration.

**Simulation check** (300 genes, $M_g$ 88 to 729, $n = 400$, AR(1) LD, true inclusion 4 times lower for MAF < 0.05, 200 retained samples): relative enrichment of $\beta^2$ 0.57 and 0.57 for the two low-MAF bins ($z = -3.6$, $-6.8$), 1.03 to 1.37 above; TSS distance flat (no effect planted); the lowest LD-score bin depleted only through its rare variants (the MAF × LD table). Under no MAF effect, mean $z^2$ over the marginal rows was 0.54 with no $|z| > 2$ (conservative). The composition regression recovered a planted coefficient of 1.5 on the low-MAF fraction (1.64, se 0.12).

**Run:** driver step 9 (all four folds of `_grid_gwfm`), then 9b.

## 5i. Gene-pair null of the held-out correlation

**Why.** Section 5h ruled out variant composition as the source of the gradient of the expression FSR excess with $M_g$ (the $\log_{10} M_g$ coefficient rose from 0.102 to 0.114 with the low-MAF fraction and mean LD score added), and `compare_sampler_fits.py` ruled out mixing (the two samplers of the grid prior agreed to within Monte Carlo noise in every decile of $M_g$, ratio 0.95 to 1.05). The expected quantities of section 7 assume the held-out residuals are exchangeable across individuals. Structure shared by expression and genotypes (ancestry, or a hidden factor not removed by residualization) would make the noise part of the observed $r$ wider than the model allows, most for diffuse predictions (weak genes), which track genome-wide structure best, and more in dense windows.

**Test** (`evaluate_gene_pair_null.py` / `.sh`, per fold; `check_gene_pair_null.py`, login node). The prediction $\hat y_A$ of every gene (mean over retained samples, as section 7) is correlated with the held-out expression $y_B$ of every gene on another chromosome. Under exchangeability, $E[(n_{te} - 1) r_{AB}^2] = 1$ exactly, for any $\hat y_A$ and any distribution of $y_B$. The null dispersion $D_A$ is the mean over partners. The summary reports $D$ by decile of $M_g$ and by $M_g$ tercile × training LRT p-value, with standard errors clustered by 5 Mb TSS block (genes with overlapping windows have similar predictions), and the expression FSR excess that $D$ predicts: the FSR of a normal approximation to the posterior predictive of $r$, with its noise variance $(1 - \bar r^2)^2 / (n_{te} - 1)$ inflated by $D - 1$ ($D$ of the row's decile × LRT stratum), against the observed excess.

**Scale.** For structure to explain the excess, $D$ must be about 1.2 overall (mid-range excess +0.022 at mean FSR 0.25) and about 1.7 in the densest decile (+0.06 at FSR 0.22).

**Simulation check** (300 genes, 2 folds, $n = 400$): with exchangeable individuals, $D = 1.005$ ($z = 0.8$), no trend in $M_g$, predicted excess 0.000. With 15% of individuals from a second population (allele frequencies shifted by $N(0, 0.05^2)$, expression shifted by 0.12 sd in every gene), $D = 1.024$ ($z = 4.8$).

**Reading.** $D$ near 1 everywhere rules structure out, leaving heterogeneity between genes as the lead (a latent class per gene). $D$ above 1, rising with $M_g$ and the predicted excess matching the observed, means the fix is in the expression residuals (more genotype PCs or PEER factors) or a calibrated inflation of the noise, not in the prior.

**Run:** driver steps 10 (one job per fold of `_grid_gwfm`) and 10b.

## 5j. Structured priors: variant categories and latent gene classes

`run_eqtl_grid_prior_gibbs.py --variant_categories maf | tss_distance` gives each bin of a pair's training-fold MAF (edges 0.02, 0.05, 0.1, 0.2) or distance to the TSS (2, 5, 10, 20, 50 kb) its own grid weights, spike included. `--n_gene_classes L` puts every gene in one of L latent classes with their own grid weights (non-centred class moves, section 5k). The 9th argument of `run_eqtl_grid_prior_gibbs.sh` selects them (`maf`, `tss_distance`, `gene_classes_10`). Implementation: the sweeps take a table of log prior weights, one row per prior, and each pair's row. Extra outputs `prior_rows.npz` and `prior_row_summary.txt`.

**Simulations** (`simulate_gibbs_input.py`: about 3,000 genes, $n = 550$, AR(1) LD; truths shared, MAF-dependent, TSS-dependent, 40% null genes). The MAF and TSS priors recovered the per-bin prior $E[\beta^2]$ closely (MAF $4.7 \times 10^{-5}$ to $5.0 \times 10^{-4}$ against true $4.0 \times 10^{-5}$ to $4.9 \times 10^{-4}$) and improved the true genetic $r$ when the truth matched (TSS 0.465 to 0.492, MAF 0.552 to 0.561), with no loss otherwise. None improved calibration by itself; see section 5k.

## 5k. What calibration needs (simulations)

Simulations with the model exactly right (`simulate_gibbs_input.py`, and an in-memory harness of independent variants) separated the causes of over-confidence:

1. **Evaluation:** exact with an exact posterior (conjugate ridge prior, no MCMC: 80% coverage 0.800, expression FSR excess +0.005).
2. **Sampler:** exact with the grid weights fixed at the truth, 1,000 retained samples and split samples (`--split_samples`): excess +0.001 and +0.005, genetic FSR excess +0.002 and +0.009, coverage 0.803.
3. **Monte Carlo:** 200 retained samples gave a spurious expression FSR excess of about +0.02 (noisy FSRs selected into a range). y_hat from the same samples as the expected quantities inflates the genetic FSR by about 0.02 (split samples remove it).
4. **Residual-variance prior:** each gene's $\sigma^2_{e,g}$ trades against near-null effects and the data barely separate them, so a fixed prior on $\sigma^2_e$, multiplied over thousands of genes, sets the weight of the smallest components: with the default (near $1 / \sigma^2$) the weight of variance $10^{-3}$ rose to 3 times the truth, with a flat prior it fell to 0; both left the genetic FSR over-confident by about 0.03. With $\sigma^2_e$ known the weights stayed at the truth and the genetic FSR was calibrated. A learned Inverse-Gamma prior (`--hierarchical_sigma2_e`) and non-centred weight moves (`--ncp_moves`) reduced this only partly (LD pipeline: genetic FSR excess +0.042 default, +0.029 hierarchical, +0.030 both; expression FSR +0.022, +0.021, +0.006). About 0.03 of genetic FSR over-confidence at a mean FSR of 0.12 remains with the shared model; the expression FSR is much less affected.
5. **Genes with no eQTL** (40% null genes) gave the real-data pattern: a mid-range expression FSR excess of +0.06 concentrated in weak genes, unchanged by the shared prior's variants (+0.057 to +0.071). **10 latent gene classes fixed it.** A factorial on three independent datasets (classes or not, plain Gibbs or non-centred class moves, fixed or hierarchical residual-variance prior) gave an expression FSR excess over all genes (se about 0.008) of:

   | Fit | Dataset 1 | Dataset 2 | Dataset 3 |
   |---|---|---|---|
   | shared prior | +0.033 | +0.038 | +0.031 |
   | shared prior, hierarchical $\sigma^2_e$ | +0.032 | +0.040 | |
   | classes, plain Gibbs | +0.008 | +0.014 | +0.001 |
   | classes, plain Gibbs, hierarchical | -0.001 | +0.021 | -0.003 |
   | classes, class moves | +0.001 | +0.020 | +0.005 |
   | classes, class moves, hierarchical | +0.005 | +0.016 | +0.001 |

   The classes account for the whole improvement (about -0.025, to the +0.01 the shared prior leaves when it is the correct model). The class moves and the hierarchical prior did not change the expression FSR on average. Plain Gibbs once let a class absorb many tiny effects (dataset 2: 824 pairs at PIP 0.5 to 0.7, of which 13% were causal). The true genetic $r$ was unchanged (0.534 against 0.535). With no classes in the truth the classes model matched the shared prior. The non-centred class move: a proposed class; each variant keeps its uniform position within its component and its standardized effect, takes the component of that position under the new class, and the move is accepted with the likelihood ratio (exact).
6. **Mixing:** the effective sample size of the global grid weights was 3 to 30 per 5,000 iterations in every fit, with or without the extra moves. The predictive quantities were calibrated regardless.
7. **PIPs:** $P(|\beta| > 0.1)$ was calibrated in every fit (for example 922 expected against 915 causal pairs with probability $\ge 0.5$). PIP $= P(\beta \ne 0)$ was calibrated above 0.5 (970 against 958; the top bin slightly optimistic, 0.90 to 0.93 observed at a mean of 0.956), but summed PIPs were 1.2 to 4 times the number of causal pairs: inclusion of near-null effects is not identified.
8. **cis-h2:** the posterior `posterior_cis_h2_train` had a correlation of 0.98 with the truth, slope 1.00 to 1.04, and 95% coverage 0.92 to 0.96. The exception was the shared prior with null genes (coverage 0.63 to 0.72), which gives genes with no eQTL about 0.013. ML and HE were about unbiased for $\operatorname{Var}(u) / \operatorname{Var}(y)$, with correlations of 0.88 to 0.93.

**Recommended setup** (driver step 13): `gene_classes_10`, the GWFM sampler, 1,000 or more retained samples, and the split-sample evaluation. The class moves (about one sweep's cost per iteration) and the hierarchical residual-variance prior are optional. They did not change calibration, but they cost little, and the class moves guard against the plain-Gibbs failure in item 5.

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
| `expected_cis_h2_independent`, `posterior_cis_h2_train_independent`, `_lb`, `_ub` | `expected_cis_h2` and `posterior_cis_h2_train` with the correlations between variants set to 0: $\sum_j \operatorname{Var}_i(x_j) (\beta_j^{(s)})^2$ in place of $\operatorname{Var}_i(X \beta^{(s)})$, over the same samples |
| `expected_cis_snp_h2_he_var_test` | What the model predicts `cis_snp_h2_he_var_test` to be: $\frac{1}{S} \sum_s u^{(s)\top} W u^{(s)}$ with $W$ the HE weights of the held-out GRM, the exact expected HE slope given $u^{(s)}$ under iid noise. HE weights an effect by the LD of its variant relative to the window (in the population, the slope is $m\, \beta^\top R^2 \beta / \operatorname{tr}(R^2)$), so it need not equal $\operatorname{Var}(u)$ |
| `cis_snp_h2_{train,test}` | Maximum-likelihood $h^2$ of the single-component model $y \sim N(0, \sigma^2[h^2 K + (1 - h^2) I])$, with $K$ the cis GRM, bounded to $[0, 1]$ (gdl `estimate_cis_snp_heritability_with_lrt`) |
| `cis_snp_h2_pvalue_{train,test}` | Likelihood-ratio test of $h^2 = 0$, with a 50:50 mixture of $\chi^2_0$ and $\chi^2_1$ because $h^2 = 0$ is on the boundary (same function) |
| `cis_snp_h2_he_{train,test}` | Haseman-Elston $h^2$: unbiased and can be negative, so it can be averaged over low-$h^2$ genes (gdl `estimate_cis_snp_heritability_with_he_regression`) |
| `var_y_{train,test}`, `cis_snp_h2_he_var_{train,test}` | Variance of the expression in those individuals, and `cis_snp_h2_he` times it: the HE estimate of $\operatorname{Var}(u)$. `cis_snp_h2_he` is a proportion of `var_y`, while `expected_cis_h2` and `posterior_cis_h2_train` are variances, so these are the columns to compare with them |
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

Also written per gene: `y_hat_mc_variance`, the Monte Carlo variance of $\hat y$ (batch means over blocks of 250 consecutive retained samples, averaged over individuals; section 11, issue 4), `n_train`, `n_test`, `n_cis_variants`, `n_cis_variants_found` (always all of them here, but not necessarily when the samples are applied to other data), `sum_pip`, `max_pip`, and `expected_n_slab1` and `expected_n_slab2` (sums of the slab probabilities). These are training-fold quantities too, so selecting genes on them (for example "genes with a confident eQTL") does not bias the held-out evaluation either.

## 8. Plots

One set per tissue, with folds as replicates (separate panels, or pooled with fold as a stratum). Each plot is made for two gene sets: all genes, and heritable genes (`cis_snp_h2_pvalue_train` $< 0.05$, selected on training data only).

1. **Correlation calibration.** 10 equal-count bins of `expected_r`: mean `observed_r` with a 95% CI against mean `expected_r`, with the line $y = x$. Repeated with `expected_r_plugin`.
2. **Genetic-scale calibration.** As gdl Plot 2: bins of `expected_genetic_r` against bin mean $r_{obs}$ / $\sqrt{\text{bin mean } \hat h^2_{test}}$.
   - (b) `_genetic_correlation_calibration_ld_corrected`: the same with HE corrected for the way it weights LD, which makes $\sqrt{\hat h^2}$ too large at high $h^2$ in plot 2 (plot 7e): per bin, $\hat h^2$ is bin mean `cis_snp_h2_he_test` times bin mean `expected_cis_h2` / bin mean `expected_cis_snp_h2_he_var_test`. That ratio is unchanged if every effect is rescaled, so the model sets only the LD correction and the size of $h^2$ comes from the held-out expression. It does depend on the model having effects on the right variants.
3. **FSR calibration.** Bins of `expression_FSR` (0 to 0.1, 0.1 to 0.2, 0.2 to 0.3, 0.3 to 0.4, above 0.4, as in gdl): the fraction of genes with $r_{obs} < 0$ (binomial CI) against mean `expression_FSR`. Repeated with `genetic_FSR` for comparison with gdl.
   - (b) `expression_FSR` and `genetic_FSR`, each alone in 10 equal-count bins, on its own scale (`_expression_fsr_calibration_deciles`, `_genetic_fsr_calibration_deciles`).
   - (c) Genes called at FSR $\le$ 0.001, 0.01, 0.05, 0.1, 0.2 and 0.3: the fraction with $r_{obs} < 0$ (with the number of genes called), the mean FSR of the called genes (expected) and the threshold, for both FSRs (`_fsr_threshold_calibration`).
   - (d) (c) for `expression_FSR`, pooled folds only, with a legend (`_expression_fsr_threshold_calibration_pooled`).
   - (e) All genes only: (b) for `expression_FSR` within strata of training LRT p-value (< 0.001, 0.001 to 0.05, 0.05 to 0.5, $\ge$ 0.5), terciles of `n_cis_variants`, and quartiles of `y_hat_mc_variance` / $V$ (`_expression_fsr_calibration_by_{training_pvalue,n_cis_variants,mc_fraction}`), with a table of the genes with `expression_FSR` in 0.1 to 0.4 per stratum (`_expression_fsr_mid_range_by_stratum_all_genes.txt`). Every stratifier is a training-fold quantity, so calibration should hold within each stratum. An excess concentrated in the weak-evidence strata points to a shared prior too generous to genes with little cis signal; one concentrated where the Monte Carlo fraction is large points to section 11, issue 4.
4. **Coverage.** Nominal level (80, 90, 95, 99%) against the fraction of genes whose $r_{obs}$ falls inside its posterior predictive interval.
5. **Top-N genes.** Mean $r_{obs}$ of the top $N$ genes ranked by `expected_r`, $1 -$ `expression_FSR`, `expected_r` $-$ 1 SD, `expected_cis_h2`, `max_pip`, `sum_pip` and, as model-free baselines, `cis_snp_h2_he_train` and $-\log_{10}$ `cis_snp_h2_pvalue_train`.
   - (b) The same rankings, with the fraction of the top $N$ genes with $r_{obs} < 0$ in place of the mean $r_{obs}$, plus the mean `expression_FSR` of the top $N$ under the FSR ranking (the expected value; `_top_n_genes_observed_fsr`).
   - (c) (b) for the expression-FSR ranking only, observed and expected with a legend: pooled folds on a log $N$ axis (`_top_n_genes_observed_fsr_expression_fsr_pooled`), and fold 1 alone on a linear $N$ axis (`_top_n_genes_observed_fsr_expression_fsr_fold_1_linear`).
6. **Hyper-parameters.** Traces (convergence), and posterior mean with 95% interval per fold. The implied prior on per-gene cis-$h^2$ and on the number of causal variants per gene, split by slab.
7. **Heritability.** (a) Calibration: 10 equal-count bins of `expected_cis_h2`, with bin mean `cis_snp_h2_he_test` against bin mean `expected_cis_h2`. (b) Training against held-out: `posterior_cis_h2_train` and `cis_snp_h2_he_train` against `cis_snp_h2_he_test`, binned. This shows how much the in-sample estimates overstate held-out heritability.
   - (c) `_heritability_calibration_variance_scale`: `cis_snp_h2_he_var_test` against `expected_cis_h2`, which puts both on the scale of $\operatorname{Var}(u)$, and `var_y_test` by bin of `expected_cis_h2`.
   - (d) `_heritability_calibration_independent_snps`: `cis_snp_h2_he_var_test` against `expected_cis_h2_independent`, and `expected_cis_h2_independent` against `expected_cis_h2` (what LD between the effects of a sample contributes).
   - (e) `_heritability_calibration_predicted_he`: `cis_snp_h2_he_var_test` against `expected_cis_snp_h2_he_var_test`, and `expected_cis_snp_h2_he_var_test` against `expected_cis_h2`.
8. **Genetic FSR distribution, fold 1.** Histograms of `genetic_FSR` (bins of 0.01) for (a) all genes and (b) heritable genes (training LRT $p < 0.05$), stacked on the same x axis (`_genetic_fsr_histogram_fold_1`).

9. **Latent gene classes** (fits with `--n_gene_classes`; `visualize_gene_classes.sh`, driver step 14c). Class labels mean nothing between folds, and a class with no genes has weights drawn from the prior, so classes are summarized only over samples in which they hold genes, classes with fewer than 10 genes on average are not ranked, and the rest are ranked within each fold by the weight of the large group (variance $\ge 10^{-3}$, the identified part of the prior). The gene-level quantities (`class_large_weight`, `prior_expected_h2`, and so on) average the weights of the class a gene is in at each sample, so they do not depend on the labels. All come from the training individuals, so genes can be stratified on them.
   - (a) `_gene_classes_profile`: per fold and class, the share of genes, inclusion probability and large-group weight (values below $10^{-6}$ shown at $10^{-6}$).
   - (b) `_gene_classes_effect_sizes`: the weight of each grid component among each class's non-zero effects. Only read it for the large components; the small ones are not identified (section 5e).
   - (c) `_gene_classes_traces`: class weights and inclusion probabilities. Two classes crossing would be a label switch, which the ranking assumes does not happen.
   - (d) `_gene_classes_per_gene`: assignment certainty, the prior number of large-group effects per gene, the class rate against the number of cis variants, and the class-implied prior $h^2$ against held-out HE.
   - (e) `_gene_classes_expression_fsr_calibration`: plot 3e within terciles of `class_large_weight`.
   - (f) `_gene_classes_expression_fsr_excess` (and `.txt`): the mid-range expression FSR excess by tercile of `n_cis_variants`, tercile of `class_large_weight` and training LRT $p$ above or below 0.05.
   - (g) `_gene_classes_heritability_calibration`: `expected_cis_h2` against held-out HE within terciles of `class_large_weight`.

## 9. Files

Same layout as `ca_qtls/fine_mapping` and `gdl_uncertainty_paper`: `driver_key.sh` holds the input paths and output directories, with each step in an `if false; then` block. Each step is an sbatch `.sh` wrapper around a Python script, and plots come from an R script.

| Step | Files | Runs per |
|---|---|---|
| 0. Fold assignment | `generate_cross_validation_folds.py` / `.sh` | once (all GTEx individuals) |
| 1. Prepare input | `prepare_gtex_tissue_data_for_gibbs.py` / `.sh`: per chromosome, expression of the tissue's individuals, `int8` alt-allele dosages for the union of cis variants (sorted by position, so each gene's variants are a contiguous range), and a summary file. Fold-agnostic. | tissue |
| 2. Fit | `eqtl_mixture_prior_gibbs.py` (numba kernels and the `EQTL_MIXTURE_PRIOR_GIBBS` class, as `cofine.py`); `run_eqtl_mixture_prior_gibbs.py` / `.sh` (loads input, applies the fold, fits, writes section 6); with `--tempered`, `eqtl_mixture_prior_tempered_gibbs.py` (section 5b), or with `--tempered --tgs_mode gwfm`, `eqtl_mixture_prior_gwfm_tempered_gibbs.py` (section 5c) | tissue × fold (× sampler) |
| 2'. Fit, fixed-grid prior | `eqtl_grid_prior_gibbs.py`, `eqtl_grid_prior_gwfm_tempered_gibbs.py`, `run_eqtl_grid_prior_gibbs.py` / `.sh` (section 5e); outputs as step 2, plus the grid weights; optional 9th argument of the `.sh`, the number of variant-count bins (section 5g) | tissue × fold (× sampler) |
| 2''. Prior check | `check_prior_self_consistency.py` / `.sh` (section 5f): one posterior sample per gene against the fitted prior | tissue × fit (fold 1) |
| 2'''. Prior by annotation | `check_prior_by_variant_annotation.py` / `.sh` (section 5h): posterior against prior rates of large effects and $\beta^2$ by MAF, TSS distance and in-window LD score; per-gene variant composition | tissue × fit × fold |
| 1'. Simulated input | `simulate_gibbs_input.py` (section 5j): genotypes with LD, expression from a chosen prior (shared, MAF, TSS distance, gene classes), in the format of step 1 plus `truth.npz` | scenario |
| (shared) | `posterior_beta_samples.py`: write and load the sparse effect samples, and compute $u^{(s)}$ for any genotype matrix | — |
| 3. Evaluate | `evaluate_held_out_expression_prediction.py` / `.sh` (section 7): loads the effect samples and the fold's training and held-out dosages and expression from the step 1 input; optional 6th argument `split_samples` for `--split_samples` (section 11, issue 4) | tissue × fold |
| 4. Organize and plot | `organize_and_visualize_cross_validation_results.sh`, calling `organize_cross_validation_results.py` and `visualize_cross_validation_results.R` (section 8); optional 6th argument, the suffix of the evaluation files (`_split_samples`), appended to the tissue in the output names | tissue |
| 4'. Expression FSR excess | `check_expression_fsr_excess.py` (login node, seconds): on the step 4 per-gene file, the excess $1[r_{obs} < 0] -$ `expression_FSR` of the genes with FSR in 0.1 to 0.4 by decile of `n_cis_variants`, by `n_cis_variants` tercile × training LRT p-value, and regressed on both and the FSR, with standard errors clustered by gene (separates the correlated stratifiers of plot 3e); with `--gene_composition_file`, also on the variant composition of each window (section 5h) | tissue |
| 4'''. Gene classes | `visualize_gene_classes.sh`, calling `organize_gene_classes.py` and `visualize_gene_classes.R` (section 8, plot 9), after step 4 | tissue × gene-class fit |
| 4''. Sampler agreement | `compare_sampler_fits.py` (login node): per-gene observed $r$ of two fits of the same model against the Monte Carlo noise expected from `y_hat_mc_variance`, by decile of `n_cis_variants` (mixing check) | tissue × pair of fits |
| 3'. Gene-pair null | `evaluate_gene_pair_null.py` / `.sh` (section 5i): each gene's held-out prediction against the held-out expression of every gene on another chromosome; `check_gene_pair_null.py` (login node) summarizes it by `n_cis_variants` and training LRT p-value and compares the FSR excess it predicts with the observed | tissue × fold |

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
4. **Monte Carlo coupling.** $\hat y$ is the mean of the same samples the expected quantities are computed from. Averaged over samples, $\operatorname{Cov}_i(\hat y, u^{(s)})$ is exactly $V$, which exceeds its value for an independent draw by the Monte Carlo variance of $\hat y$ (`y_hat_mc_variance`). So `expected_r` is biased up and both FSRs down, by about `y_hat_mc_variance` / $V$ in relative terms. The bias is $O(1/S_{\text{eff}})$ rather than $O(1/S)$, and largest for genes with small $V$, the mid-range of the FSR. On exact Gaussian posteriors (3 variance levels, 20 variants, $n_{tr} = 420$, $n_{te} = 140$, 12,000 genes) drawn as an AR(1) chain with correlation 0.95 between the 1,000 retained samples, `expression_FSR` averaged 0.201 against an observed 0.221 (independent draws: 0.210 against 0.212). In the FSR 0.1 to 0.4 range it was 0.251 against 0.278, the size of the Whole Blood excess. Check: `--split_samples` (driver step 6) takes $\hat y$ and `observed_r` from alternating blocks of `--block_size` consecutive retained samples and every expected quantity from the other blocks. Odd against even samples would not work, because neighbouring samples are correlated. In the same simulation, blocks of 100 removed half of the bias and blocks of 250 (the default) all of it (0.2226 against 0.2236).
5. **Mixing.** Random-scan Gibbs mixes slowly between variants in near-perfect LD. This should not affect predictions or the expected quantities, because those variants carry the same signal in held-out individuals of the same population. Check: two chains with different seeds, comparing $\hat y$ and `expression_FSR` per gene, or the tempered fit (section 5b).
6. **Two-slab identifiability.** If one slab holds only a handful of variants, its variance is poorly determined under the near-flat prior and its draws are noisy. The ordering constraint stops label switching, but a slab with near-zero weight means a single slab would have been enough. The hyper-parameter plots (plot 6) show this.

   This happened in the small simulation test (40k variant-gene pairs, with a small slab too weak to detect at $n \approx 420$). Under the near-flat Inverse-Gamma prior, the small slab's variance repeatedly collapsed toward $10^{-10}$, where it acts as a second spike. Once, the large slab emptied and kept an extreme variance ($\sim 36$) for a while before recovering. Predictions, FSRs and intervals stayed calibrated, so this mostly affects interpreting the hyper-parameters. Real data have about 250 times more pairs, so the slabs should be better identified. If the traces still wander, the remedy is a mildly informative Inverse-Gamma on the slab variances, or a floor on $\sigma^2_1$. The fine-mapping simulations of section 5d (200k pairs per scenario) showed the same thing in the standard and GWFM chains: one slab emptied in part of 8 of 12 and 7 of 12 chains. The full tempered sampler's expected-statistics hyper-parameter update did not do this.
7. **One fold or four.** The default is all four. A single fold is a quarter of the compute, but each calibration plot then rests on one split.

## 12. Validation plan

1. **Small run.** One chromosome, `--max_genes 200`. Check hyper-parameter traces and per-gene runtime, and that fixing $\pi_0 = 1$ gives $\hat y = 0$.
2. **Simulation with the model true.** Simulate expression from the real Whole Blood genotypes with effects drawn from a known mixture prior, then run steps 2 to 4 unchanged. All calibration plots must lie on $y = x$ and the hyper-parameters must be recovered. Real-data deviations can then be attributed to model misspecification rather than to the code.
3. **Real data**, Whole Blood, four folds. Then the other GTEx tissues by looping over the tissue list in `driver_key.sh`, as in `gtex_eqtl_expression_processing/driver_key.sh`.
