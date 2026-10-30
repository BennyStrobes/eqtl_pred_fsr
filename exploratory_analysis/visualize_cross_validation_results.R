args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
options(warn=1)

# Plots of the cross-validated eQTL prior fits of one tissue (README.md, section 8), from organize_cross_validation_results.py
#   args[1]: organized results stem (the --output_stem of the organizer); args[2]: output directory for the pdfs; args[3]: tissue
# Every gene-level plot has one panel per fold and a "Pooled" panel (all gene-fold rows; standard errors clustered by gene, since a gene's
# rows in different folds share training individuals). Each is made for all genes (<tissue>_<plot>_all_genes.pdf) and for heritable genes
# (training-fold cis-h2 LRT p < heritability_pvalue_threshold, selected on training data only; <tissue>_<plot>_heritable_genes.pdf).
organized_stem <- args[1]
viz_dir <- args[2]
tissue <- args[3]
dir.create(viz_dir, showWarnings=FALSE, recursive=TRUE)

# Number of equal-count bins of the expected quantities (per panel)
n_bins <- 10
# Expression-FSR bins (as in gdl_uncertainty_paper)
fsr_bin_breaks <- c(0, 0.1, 0.2, 0.3, 0.4, Inf)
fsr_bin_labels <- c("[0,0.1]", "(0.1,0.2]", "(0.2,0.3]", "(0.3,0.4]", ">0.4")
# Nominal levels of the posterior predictive intervals of the observed correlation
coverage_levels <- c(80, 90, 95, 99)
# Heritable genes: training-fold cis-h2 LRT p-value below this threshold
heritability_pvalue_threshold <- 0.05
# Smallest N shown in the top-N-genes plot
top_n_min_genes <- 10

figure_theme <- function() {
	return(
		theme_minimal(base_size=11) +
		theme(
			plot.title=element_text(face="bold", size=12, hjust=.5),
			axis.title=element_text(size=11, color="#25313B"),
			axis.text=element_text(size=10, color="#33424F"),
			strip.text=element_text(size=10, face="bold"),
			panel.grid.minor=element_blank(),
			axis.line.x=element_line(colour="#7E8A97", linewidth=.35),
			axis.line.y=element_line(colour="#7E8A97", linewidth=.35),
			legend.position="bottom",
			legend.title=element_blank()
		)
	)
}

gene_fold_df <- read.table(paste0(organized_stem, "_per_gene_fold.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))
trace_df <- read.table(paste0(organized_stem, "_hyperparameter_trace.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)

folds <- sort(unique(gene_fold_df$fold))
fold_levels <- c(paste0("Fold ", folds), "Pooled")
fold_colors <- setNames(c(hcl.colors(length(folds), "Dark 3"), "#3E4A56"), fold_levels)


#####################
# Helpers
#####################
with_pooled_panel <- function(df) {
	# Each row once in its fold's panel and once in the pooled panel
	per_fold <- df
	per_fold$panel <- paste0("Fold ", df$fold)
	pooled <- df
	pooled$panel <- "Pooled"
	out <- rbind(per_fold, pooled)
	out$panel <- factor(out$panel, levels=fold_levels)
	return(out)
}

cluster_cov_of_means <- function(a, b, cluster) {
	# Covariance of the means of a and b, clustered by gene: sum over genes of (sum of the gene's a deviations)(sum of its b deviations) / n^2.
	# With one row per gene this is the usual covariance / n (1/n convention).
	n <- length(a)
	sum_a <- tapply(a - mean(a), cluster, sum)
	sum_b <- tapply(b - mean(b), cluster, sum)
	return(sum(sum_a * sum_b) / n^2)
}

cluster_se_of_mean <- function(x, cluster) {
	return(sqrt(cluster_cov_of_means(x, x, cluster)))
}

equal_count_bins <- function(x, n_bins) {
	return(ceiling(n_bins * rank(x, ties.method="first") / length(x)))
}

binned_calibration_df <- function(df, x_col, y_col) {
	# One row per panel and equal-count bin of x_col: mean x and mean y with clustered standard errors
	rows <- list()
	for (panel in levels(df$panel)) {
		sub_df <- df[df$panel == panel & !is.na(df[[x_col]]) & !is.na(df[[y_col]]), ]
		if (nrow(sub_df) < n_bins) {
			next
		}
		bins <- equal_count_bins(sub_df[[x_col]], n_bins)
		for (bin in 1:n_bins) {
			bin_df <- sub_df[bins == bin, ]
			rows[[length(rows) + 1]] <- data.frame(panel=panel, bin=bin, n=nrow(bin_df),
				x=mean(bin_df[[x_col]]), x_se=cluster_se_of_mean(bin_df[[x_col]], bin_df$gene_id),
				y=mean(bin_df[[y_col]]), y_se=cluster_se_of_mean(bin_df[[y_col]], bin_df$gene_id))
		}
	}
	if (length(rows) == 0) {
		return(NULL)
	}
	out <- do.call(rbind, rows)
	out$panel <- factor(out$panel, levels=fold_levels)
	return(out)
}

genetic_calibration_df <- function(df) {
	# As gdl Plot 2: per equal-count bin of expected_genetic_r, y = mean(observed_r) / sqrt(mean(cis_snp_h2_he_test)), with a delta-method
	# standard error for g(a, c) = a / sqrt(c) from the clustered (co)variances of the two bin means. Bins with mean h2 <= 0 get no point.
	rows <- list()
	for (panel in levels(df$panel)) {
		sub_df <- df[df$panel == panel & !is.na(df$expected_genetic_r) & !is.na(df$observed_r) & !is.na(df$cis_snp_h2_he_test), ]
		if (nrow(sub_df) < n_bins) {
			next
		}
		bins <- equal_count_bins(sub_df$expected_genetic_r, n_bins)
		for (bin in 1:n_bins) {
			bin_df <- sub_df[bins == bin, ]
			a <- mean(bin_df$observed_r)
			c <- mean(bin_df$cis_snp_h2_he_test)
			if (c <= 0) {
				next
			}
			var_a <- cluster_cov_of_means(bin_df$observed_r, bin_df$observed_r, bin_df$gene_id)
			var_c <- cluster_cov_of_means(bin_df$cis_snp_h2_he_test, bin_df$cis_snp_h2_he_test, bin_df$gene_id)
			cov_ac <- cluster_cov_of_means(bin_df$observed_r, bin_df$cis_snp_h2_he_test, bin_df$gene_id)
			var_y <- var_a / c - a * cov_ac / c^2 + a^2 * var_c / (4 * c^3)
			rows[[length(rows) + 1]] <- data.frame(panel=panel, bin=bin, n=nrow(bin_df),
				x=mean(bin_df$expected_genetic_r), x_se=cluster_se_of_mean(bin_df$expected_genetic_r, bin_df$gene_id),
				y=a / sqrt(c), y_se=sqrt(max(var_y, 0)))
		}
	}
	if (length(rows) == 0) {
		return(NULL)
	}
	out <- do.call(rbind, rows)
	out$panel <- factor(out$panel, levels=fold_levels)
	return(out)
}

fsr_calibration_df <- function(df, fsr_col) {
	# One row per panel and FSR bin: mean FSR (expected) and fraction of genes with a negative observed correlation (observed)
	rows <- list()
	for (panel in levels(df$panel)) {
		sub_df <- df[df$panel == panel & !is.na(df[[fsr_col]]) & !is.na(df$observed_r), ]
		bins <- cut(sub_df[[fsr_col]], breaks=fsr_bin_breaks, labels=fsr_bin_labels, include.lowest=TRUE, right=TRUE)
		for (bin in fsr_bin_labels) {
			bin_df <- sub_df[!is.na(bins) & bins == bin, ]
			if (nrow(bin_df) == 0) {
				next
			}
			false_sign <- as.numeric(bin_df$observed_r < 0)
			rows[[length(rows) + 1]] <- data.frame(panel=panel, bin=bin, n=nrow(bin_df),
				x=mean(bin_df[[fsr_col]]), x_se=cluster_se_of_mean(bin_df[[fsr_col]], bin_df$gene_id),
				y=mean(false_sign), y_se=cluster_se_of_mean(false_sign, bin_df$gene_id))
		}
	}
	if (length(rows) == 0) {
		return(NULL)
	}
	out <- do.call(rbind, rows)
	out$panel <- factor(out$panel, levels=fold_levels)
	return(out)
}

calibration_plot <- function(summary_df, xlab, ylab, row_var=NULL) {
	# Observed against expected per bin (95% intervals in both directions), dashed y = x, one panel per fold (and per row_var, if given)
	pp <- ggplot(summary_df, aes(x=x, y=y, color=panel)) +
		geom_abline(slope=1, intercept=0, linetype="dashed", color="#7E8A97") +
		geom_errorbar(aes(ymin=y - 1.96 * y_se, ymax=y + 1.96 * y_se), width=0) +
		geom_errorbarh(aes(xmin=x - 1.96 * x_se, xmax=x + 1.96 * x_se), height=0) +
		geom_line(linewidth=.35) +
		geom_point(size=1.6) +
		scale_color_manual(values=fold_colors, guide="none") +
		labs(x=xlab, y=ylab) +
		figure_theme()
	if (is.null(row_var)) {
		pp <- pp + facet_wrap(~panel, nrow=1)
	} else {
		pp <- pp + facet_grid(as.formula(paste0(row_var, " ~ panel")), scales="free")
	}
	return(pp)
}

save_plot <- function(pp, plot_name, gene_set_suffix, width, height) {
	ggsave(paste0(viz_dir, tissue, "_", plot_name, gene_set_suffix, ".pdf"), pp, width=width, height=height, units="in", limitsize=FALSE)
}


#####################
# Gene-level plots (plots 1 to 5 and 7 of README.md, section 8)
#####################
make_gene_level_plots <- function(df, gene_set_suffix, gene_set_label) {
	df <- with_pooled_panel(df)
	n_panels <- length(fold_levels)
	panel_width <- 2.6 * n_panels

	# 1. Correlation calibration: observed correlation against the posterior predictive expected correlation and the plug-in sqrt(V)
	summary_df <- rbind(
		cbind(binned_calibration_df(df, "expected_r", "observed_r"), predictor="Posterior predictive"),
		cbind(binned_calibration_df(df, "expected_r_plugin", "observed_r"), predictor="Plug-in sqrt(V)"))
	summary_df$predictor <- factor(summary_df$predictor, levels=c("Posterior predictive", "Plug-in sqrt(V)"))
	pp <- calibration_plot(summary_df, "Expected correlation", "Observed held-out correlation", row_var="predictor") + ggtitle(paste0(tissue, ", ", gene_set_label))
	save_plot(pp, "correlation_calibration", gene_set_suffix, panel_width, 6)

	# 2. Genetic-scale calibration
	summary_df <- genetic_calibration_df(df)
	if (!is.null(summary_df)) {
		pp <- calibration_plot(summary_df, "Expected correlation with genetic expression", "Mean observed r / sqrt(mean held-out HE h2)") + ggtitle(paste0(tissue, ", ", gene_set_label))
		save_plot(pp, "genetic_correlation_calibration", gene_set_suffix, panel_width, 3.4)
	}

	# 3. FSR calibration: fraction of genes with a negative observed correlation against the expression FSR (the calibration target) and the genetic FSR
	summary_df <- rbind(
		cbind(fsr_calibration_df(df, "expression_FSR"), fsr_type="Expression FSR"),
		cbind(fsr_calibration_df(df, "genetic_FSR"), fsr_type="Genetic FSR"))
	summary_df$fsr_type <- factor(summary_df$fsr_type, levels=c("Expression FSR", "Genetic FSR"))
	pp <- calibration_plot(summary_df, "Expected FSR (bin mean)", "Fraction of genes with observed r < 0", row_var="fsr_type") + ggtitle(paste0(tissue, ", ", gene_set_label))
	save_plot(pp, "fsr_calibration", gene_set_suffix, panel_width, 6)

	# 4. Coverage of the posterior predictive intervals of the observed correlation
	rows <- list()
	for (panel in fold_levels) {
		sub_df <- df[df$panel == panel & !is.na(df$observed_r) & !is.na(df$expected_r), ]
		for (level in coverage_levels) {
			inside <- as.numeric(sub_df$observed_r >= sub_df[[paste0("expected_r_lb_", level)]] & sub_df$observed_r <= sub_df[[paste0("expected_r_ub_", level)]])
			rows[[length(rows) + 1]] <- data.frame(panel=panel, x=level / 100, x_se=0, y=mean(inside), y_se=cluster_se_of_mean(inside, sub_df$gene_id))
		}
	}
	summary_df <- do.call(rbind, rows)
	summary_df$panel <- factor(summary_df$panel, levels=fold_levels)
	pp <- calibration_plot(summary_df, "Nominal level", "Empirical coverage") + ggtitle(paste0(tissue, ", ", gene_set_label))
	save_plot(pp, "predictive_interval_coverage", gene_set_suffix, panel_width, 3.4)

	# 5. Mean observed correlation of the top N genes under each ranking
	rankings <- list(
		"Expected r"=function(d) d$expected_r,
		"Expected r - 1 SD"=function(d) d$expected_r - d$expected_r_sd,
		"1 - expression FSR"=function(d) 1 - d$expression_FSR,
		"Expected cis-h2"=function(d) d$expected_cis_h2,
		"Max PIP"=function(d) d$max_pip,
		"Sum PIP"=function(d) d$sum_pip,
		"Training HE cis-h2"=function(d) d$cis_snp_h2_he_train,
		"Training cis-h2 -log10 p"=function(d) -log10(d$cis_snp_h2_pvalue_train))
	rows <- list()
	for (panel in fold_levels) {
		sub_df <- df[df$panel == panel & !is.na(df$observed_r), ]
		for (ranking in names(rankings)) {
			score <- rankings[[ranking]](sub_df)
			valid <- !is.na(score)
			if (sum(valid) < top_n_min_genes) {
				next
			}
			ordered_r <- sub_df$observed_r[valid][order(-score[valid])]
			running_mean <- cumsum(ordered_r) / seq_along(ordered_r)
			top_n <- unique(round(10^seq(log10(top_n_min_genes), log10(length(ordered_r)), length.out=200)))
			rows[[length(rows) + 1]] <- data.frame(panel=panel, ranking=ranking, top_n=top_n, mean_observed_r=running_mean[top_n])
		}
	}
	top_n_df <- do.call(rbind, rows)
	top_n_df$panel <- factor(top_n_df$panel, levels=fold_levels)
	top_n_df$ranking <- factor(top_n_df$ranking, levels=names(rankings))
	pp <- ggplot(top_n_df, aes(x=top_n, y=mean_observed_r, color=ranking)) +
		geom_line(linewidth=.5) +
		scale_x_log10() +
		facet_wrap(~panel, nrow=1) +
		labs(x="Top N genes", y="Mean observed held-out correlation") +
		ggtitle(paste0(tissue, ", ", gene_set_label)) +
		figure_theme()
	save_plot(pp, "top_n_genes_mean_observed_r", gene_set_suffix, panel_width, 4)

	# 7. Heritability: (a) held-out HE h2 against expected cis-h2; (b) held-out HE h2 against the training-fold estimates
	summary_df <- binned_calibration_df(df, "expected_cis_h2", "cis_snp_h2_he_test")
	pp_a <- calibration_plot(summary_df, "Expected cis-h2 (held-out individuals)", "Held-out HE cis-h2") + ggtitle(paste0(tissue, ", ", gene_set_label))
	summary_df <- rbind(
		cbind(binned_calibration_df(df, "posterior_cis_h2_train", "cis_snp_h2_he_test"), training_estimate="Posterior, training"),
		cbind(binned_calibration_df(df, "cis_snp_h2_he_train", "cis_snp_h2_he_test"), training_estimate="HE, training"))
	summary_df$training_estimate <- factor(summary_df$training_estimate, levels=c("Posterior, training", "HE, training"))
	pp_b <- calibration_plot(summary_df, "Training-fold cis-h2", "Held-out HE cis-h2", row_var="training_estimate")
	save_plot(plot_grid(pp_a, pp_b, ncol=1, rel_heights=c(1, 1.7)), "heritability_calibration", gene_set_suffix, panel_width, 9)
}

gene_sets <- list(
	list(df=gene_fold_df, suffix="_all_genes", label="all genes"),
	list(df=gene_fold_df[!is.na(gene_fold_df$cis_snp_h2_pvalue_train) & gene_fold_df$cis_snp_h2_pvalue_train < heritability_pvalue_threshold, ], suffix="_heritable_genes", label=paste0("heritable genes (training LRT p < ", heritability_pvalue_threshold, ")")))
for (gene_set in gene_sets) {
	print(paste0(gene_set$label, ": ", nrow(gene_set$df), " gene-fold rows"))
	make_gene_level_plots(gene_set$df, gene_set$suffix, gene_set$label)
}


#####################
# 6. Hyper-parameters
#####################
# (a) Traces, with the fraction of non-zero effects 1 - pi0 in place of pi0; dotted line at the first retained iteration
trace_df$fold_label <- factor(paste0("Fold ", trace_df$fold), levels=fold_levels)
trace_df$value[trace_df$hyperparameter == "pi0"] <- 1 - trace_df$value[trace_df$hyperparameter == "pi0"]
trace_df$hyperparameter[trace_df$hyperparameter == "pi0"] <- "1 - pi0"
hyperparameter_levels <- c("1 - pi0", "pi1", "pi2", "sigma2_1", "sigma2_2")
trace_df$hyperparameter <- factor(trace_df$hyperparameter, levels=hyperparameter_levels)
first_retained <- min(trace_df$iteration[trace_df$retained == 1])
pp_trace <- ggplot(trace_df, aes(x=iteration, y=value, color=fold_label)) +
	geom_line(linewidth=.3) +
	geom_vline(xintercept=first_retained, linetype="dotted", color="#7E8A97") +
	scale_y_log10() +
	scale_color_manual(values=fold_colors) +
	facet_wrap(~hyperparameter, nrow=1, scales="free_y") +
	labs(x="Iteration", y="Value") +
	ggtitle(paste0(tissue, ": hyper-parameter traces")) +
	figure_theme()

# (b) Posterior mean and 95% interval over the retained samples, per fold
retained_df <- trace_df[trace_df$retained == 1, ]
posterior_df <- aggregate(value ~ fold_label + hyperparameter, data=retained_df, FUN=mean)
posterior_df$lb <- aggregate(value ~ fold_label + hyperparameter, data=retained_df, FUN=function(x) quantile(x, 0.025))$value
posterior_df$ub <- aggregate(value ~ fold_label + hyperparameter, data=retained_df, FUN=function(x) quantile(x, 0.975))$value
pp_posterior <- ggplot(posterior_df, aes(x=fold_label, y=value, color=fold_label)) +
	geom_errorbar(aes(ymin=lb, ymax=ub), width=0) +
	geom_point(size=1.8) +
	scale_y_log10() +
	scale_color_manual(values=fold_colors, guide="none") +
	facet_wrap(~hyperparameter, nrow=1, scales="free_y") +
	labs(x="", y="Posterior mean (95% interval)") +
	figure_theme() +
	theme(axis.text.x=element_text(angle=45, hjust=1))

# (c) Per gene (all gene-fold rows): number of variants in each slab and cis-h2 implied by the prior (posterior mean hyper-parameters of the
# fold, times the gene's number of variants) against their posterior expectations
posterior_wide <- reshape(posterior_df[, c("fold_label", "hyperparameter", "value")], idvar="fold_label", timevar="hyperparameter", direction="wide")
gene_hyper_df <- merge(transform(gene_fold_df, fold_label=paste0("Fold ", fold)), posterior_wide, by="fold_label")
implied_rows <- list(
	data.frame(quantity="Variants in slab 1", source="Prior-implied", value=gene_hyper_df$n_cis_variants * gene_hyper_df[["value.pi1"]]),
	data.frame(quantity="Variants in slab 1", source="Posterior", value=gene_hyper_df$expected_n_slab1),
	data.frame(quantity="Variants in slab 2", source="Prior-implied", value=gene_hyper_df$n_cis_variants * gene_hyper_df[["value.pi2"]]),
	data.frame(quantity="Variants in slab 2", source="Posterior", value=gene_hyper_df$expected_n_slab2),
	data.frame(quantity="Training cis-h2", source="Prior-implied", value=gene_hyper_df$n_cis_variants * (gene_hyper_df[["value.pi1"]] * gene_hyper_df[["value.sigma2_1"]] + gene_hyper_df[["value.pi2"]] * gene_hyper_df[["value.sigma2_2"]])),
	data.frame(quantity="Training cis-h2", source="Posterior", value=gene_hyper_df$posterior_cis_h2_train))
implied_df <- do.call(rbind, implied_rows)
implied_df$quantity <- factor(implied_df$quantity, levels=c("Variants in slab 1", "Variants in slab 2", "Training cis-h2"))
implied_df <- implied_df[!is.na(implied_df$value) & implied_df$value > 0, ]
pp_implied <- ggplot(implied_df, aes(x=value, fill=source)) +
	geom_histogram(bins=50, alpha=.6, position="identity") +
	scale_x_log10() +
	scale_fill_manual(values=c("Prior-implied"="#A3ACB6", "Posterior"="#E0457B")) +
	facet_wrap(~quantity, nrow=1, scales="free") +
	labs(x="Per gene (log scale)", y="Gene-fold rows") +
	figure_theme()

ggsave(paste0(viz_dir, tissue, "_hyperparameters.pdf"), plot_grid(pp_trace, pp_posterior, pp_implied, ncol=1, rel_heights=c(1.1, 1, 1)), width=14, height=10, units="in")
