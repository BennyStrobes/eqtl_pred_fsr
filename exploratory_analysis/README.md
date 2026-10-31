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
   - (b) `expression_FSR` and `genetic_FSR`, each alone in 10 equal-count bins, on its own scale (`_expression_fsr_calibration_deciles`, `_genetic_fsr_calibration_deciles`).
   - (c) Genes called at FSR $\le$ 0.001, 0.01, 0.05, 0.1, 0.2 and 0.3: the fraction with $r_{obs} < 0$ (with the number of genes called), the mean FSR of the called genes (expected) and the threshold, for both FSRs (`_fsr_threshold_calibration`).
   - (d) (c) for `expression_FSR`, pooled folds only, with a legend (`_expression_fsr_threshold_calibration_pooled`).
4. **Coverage.** Nominal level (80, 90, 95, 99%) against the fraction of genes whose $r_{obs}$ falls inside its posterior predictive interval.
5. **Top-N genes.** Mean $r_{obs}$ of the top $N$ genes ranked by `expected_r`, $1 -$ `expression_FSR`, `expected_r` $-$ 1 SD, `expected_cis_h2`, `max_pip`, `sum_pip` and, as model-free baselines, `cis_snp_h2_he_train` and $-\log_{10}$ `cis_snp_h2_pvalue_train`.
   - (b) The same rankings, with the fraction of the top $N$ genes with $r_{obs} < 0$ in place of the mean $r_{obs}$, plus the mean `expression_FSR` of the top $N$ under the FSR ranking (the expected value; `_top_n_genes_observed_fsr`).
   - (c) (b) for the expression-FSR ranking only, observed and expected with a legend: pooled folds on a log $N$ axis (`_top_n_genes_observed_fsr_expression_fsr_pooled`), and fold 1 alone on a linear $N$ axis (`_top_n_genes_observed_fsr_expression_fsr_fold_1_linear`).
6. **Hyper-parameters.** Traces (convergence), and posterior mean with 95% interval per fold. The implied prior on per-gene cis-$h^2$ and on the number of causal variants per gene, split by slab.
7. **Heritability.** (a) Calibration: 10 equal-count bins of `expected_cis_h2`, with bin mean `cis_snp_h2_he_test` against bin mean `expected_cis_h2`. (b) Training against held-out: `posterior_cis_h2_train` and `cis_snp_h2_he_train` against `cis_snp_h2_he_test`, binned. This shows how much the in-sample estimates overstate held-out heritability.

## 9. Files

Same layout as `ca_qtls/fine_mapping` and `gdl_uncertainty_paper`: `driver_key.sh` holds the input paths and output directories, with each step in an `if false; then` block. Each step is an sbatch `.sh` wrapper around a Python script, and plots come from an R script.

| Step | Files | Runs per |
|---|---|---|
| 0. Fold assignment | `generate_cross_validation_folds.py` / `.sh` | once (all GTEx individuals) |
| 1. Prepare input | `prepare_gtex_tissue_data_for_gibbs.py` / `.sh`: per chromosome, expression of the tissue's individuals, `int8` alt-allele dosages for the union of cis variants (sorted by position, so each gene's variants are a contiguous range), and a summary file. Fold-agnostic. | tissue |
| 2. Fit | `eqtl_mixture_prior_gibbs.py` (numba kernels and the `EQTL_MIXTURE_PRIOR_GIBBS` class, as `cofine.py`); `run_eqtl_mixture_prior_gibbs.py` / `.sh` (loads input, applies the fold, fits, writes section 6); with `--tempered`, `eqtl_mixture_prior_tempered_gibbs.py` (section 5b), or with `--tempered --tgs_mode gwfm`, `eqtl_mixture_prior_gwfm_tempered_gibbs.py` (section 5c) | tissue × fold (× sampler) |
| 2'. Fit, fixed-grid prior | `eqtl_grid_prior_gibbs.py`, `eqtl_grid_prior_gwfm_tempered_gibbs.py`, `run_eqtl_grid_prior_gibbs.py` / `.sh` (section 5e); outputs as step 2, plus the grid weights | tissue × fold (× sampler) |
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
5. **Mixing.** Random-scan Gibbs mixes slowly between variants in near-perfect LD. This should not affect predictions or the expected quantities, because those variants carry the same signal in held-out individuals of the same population. Check: two chains with different seeds, comparing $\hat y$ and `expression_FSR` per gene, or the tempered fit (section 5b).
6. **Two-slab identifiability.** If one slab holds only a handful of variants, its variance is poorly determined under the near-flat prior and its draws are noisy. The ordering constraint stops label switching, but a slab with near-zero weight means a single slab would have been enough. The hyper-parameter plots (plot 6) show this.

   This happened in the small simulation test (40k variant-gene pairs, with a small slab too weak to detect at $n \approx 420$). Under the near-flat Inverse-Gamma prior, the small slab's variance repeatedly collapsed toward $10^{-10}$, where it acts as a second spike. Once, the large slab emptied and kept an extreme variance ($\sim 36$) for a while before recovering. Predictions, FSRs and intervals stayed calibrated, so this mostly affects interpreting the hyper-parameters. Real data have about 250 times more pairs, so the slabs should be better identified. If the traces still wander, the remedy is a mildly informative Inverse-Gamma on the slab variances, or a floor on $\sigma^2_1$. The fine-mapping simulations of section 5d (200k pairs per scenario) showed the same thing in the standard and GWFM chains: one slab emptied in part of 8 of 12 and 7 of 12 chains. The full tempered sampler's expected-statistics hyper-parameter update did not do this.
7. **One fold or four.** The default is all four. A single fold is a quarter of the compute, but each calibration plot then rests on one split.

## 12. Validation plan

1. **Small run.** One chromosome, `--max_genes 200`. Check hyper-parameter traces and per-gene runtime, and that fixing $\pi_0 = 1$ gives $\hat y = 0$.
2. **Simulation with the model true.** Simulate expression from the real Whole Blood genotypes with effects drawn from a known mixture prior, then run steps 2 to 4 unchanged. All calibration plots must lie on $y = x$ and the hyper-parameters must be recovered. Real-data deviations can then be attributed to model misspecification rather than to the code.
3. **Real data**, Whole Blood, four folds. Then the other GTEx tissues by looping over the tissue list in `driver_key.sh`, as in `gtex_eqtl_expression_processing/driver_key.sh`.
