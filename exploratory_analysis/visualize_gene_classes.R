args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
options(warn=1)

# Plots of the latent gene classes of one fit (README.md, section 8, plot 9), from organize_gene_classes.py
#   args[1]: organized stem (the --output_stem of organize_gene_classes.py); args[2]: output directory for the pdfs; args[3]: tissue
# Class ranks are within a fold: rank 1 has the fewest identified effects per variant. Classes are not matched between folds, so the plots of
# classes have one panel per fold and no pooled panel. The gene-level plots use quantities that do not depend on the labels, and pool as usual.
organized_stem <- args[1]
viz_dir <- args[2]
tissue <- args[3]
dir.create(viz_dir, showWarnings=FALSE, recursive=TRUE)

# Number of equal-count bins of the expected quantities (per panel)
n_bins <- 10
# Strata of the class-level rate of identified effects, and of the number of cis variants
n_class_strata <- 3
n_variant_strata <- 3
# The expression FSR range of the mid-range excess, as in visualize_cross_validation_results.R
mid_range_fsr <- c(0.1, 0.4)
# Training-fold evidence strata of the excess plot
heritability_pvalue_threshold <- 0.05
# Log axes: a near-null class can have a large-group weight near 1e-100, which would flatten everything else, so values are shown at these floors
weight_floor <- 1e-6
n_large_floor <- 1e-3

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

class_df <- read.table(paste0(organized_stem, "_class_summary.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))
component_df <- read.table(paste0(organized_stem, "_class_component_weights.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
trace_df <- read.table(paste0(organized_stem, "_class_trace.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE)
gene_fold_df <- read.table(paste0(organized_stem, "_per_gene_fold.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"), quote="", comment.char="")

folds <- sort(unique(gene_fold_df$fold))
fold_panels <- paste0("Fold ", folds)
fold_levels <- c(fold_panels, "Pooled")
fold_colors <- setNames(c(hcl.colors(length(folds), "Dark 3"), "#3E4A56"), fold_levels)
class_df <- class_df[!is.na(class_df$class_rank), ]
class_df$panel <- factor(paste0("Fold ", class_df$fold), levels=fold_panels)
component_df$panel <- factor(paste0("Fold ", component_df$fold), levels=fold_panels)
trace_df$panel <- factor(paste0("Fold ", trace_df$fold), levels=fold_panels)
n_ranks <- max(class_df$class_rank)
rank_levels <- as.character(1:n_ranks)
rank_colors <- setNames(hcl.colors(n_ranks, "Viridis"), rank_levels)


#####################
# Helpers (as in visualize_cross_validation_results.R)
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
	# Covariance of the means of a and b, clustered by gene
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

quantile_strata <- function(x, n, label) {
	# n equal-count strata of x (fewer if quantiles tie, as when most genes sit in one class)
	breaks <- unique(quantile(x, probs=seq(0, 1, length.out=n + 1), na.rm=TRUE))
	if (length(breaks) < 2) {
		return(factor(ifelse(is.na(x), NA, paste0(label, " ", signif(breaks[1], 2)))))
	}
	labels <- make.unique(paste0(label, " ", signif(breaks[-length(breaks)], 2), " to ", signif(breaks[-1], 2)))
	return(cut(x, breaks=breaks, labels=labels, include.lowest=TRUE))
}

save_plot <- function(pp, plot_name, width, height) {
	ggsave(paste0(viz_dir, tissue, "_", plot_name, ".pdf"), pp, width=width, height=height, units="in", limitsize=FALSE)
}

# The strata used below: the class rate of identified effects does not grow with the number of variants, so it can be crossed with it
gene_fold_df$class_stratum <- quantile_strata(gene_fold_df$class_large_weight, n_class_strata, "Class large weight")
gene_fold_df$variant_stratum <- quantile_strata(gene_fold_df$n_cis_variants, n_variant_strata, "Variants")
gene_fold_df$false_sign <- as.numeric(gene_fold_df$observed_r < 0)
class_stratum_levels <- levels(gene_fold_df$class_stratum)
class_stratum_colors <- setNames(hcl.colors(length(class_stratum_levels), "Viridis", rev=TRUE), class_stratum_levels)


#####################
# 9a. What the classes are, per fold: share of the genes, inclusion probability and weight of the large group, with 95% intervals
#####################
profile_rows <- list()
for (metric in list(list(column="gene_fraction", label="Share of genes", lb=NA, ub=NA),
                    list(column="inclusion_mean", label="Inclusion probability", lb="inclusion_95_lb", ub="inclusion_95_ub"),
                    list(column="large_weight_mean", label="Large-group weight", lb="large_weight_95_lb", ub="large_weight_95_ub"))) {
	profile_rows[[length(profile_rows) + 1]] <- data.frame(panel=class_df$panel, class_rank=class_df$class_rank, metric=metric$label, value=class_df[[metric$column]],
		lb=if (is.na(metric$lb)) class_df[[metric$column]] else class_df[[metric$lb]], ub=if (is.na(metric$ub)) class_df[[metric$column]] else class_df[[metric$ub]])
}
profile_df <- do.call(rbind, profile_rows)
profile_df$metric <- factor(profile_df$metric, levels=c("Share of genes", "Inclusion probability", "Large-group weight"))
profile_df$value <- pmax(profile_df$value, weight_floor)
profile_df$lb <- pmax(profile_df$lb, weight_floor)
profile_df$ub <- pmax(profile_df$ub, weight_floor)
pp <- ggplot(profile_df, aes(x=class_rank, y=value, color=panel)) +
	geom_errorbar(aes(ymin=lb, ymax=ub), width=0) +
	geom_line(linewidth=.35) +
	geom_point(size=1.8) +
	scale_x_continuous(breaks=1:n_ranks) +
	scale_y_log10() +
	scale_color_manual(values=fold_colors, guide="none") +
	facet_grid(metric ~ panel, scales="free_y") +
	labs(x="Class rank within fold (fewest identified effects first)", y="Posterior mean (95% interval)") +
	ggtitle(paste0(tissue, ": latent gene classes")) +
	figure_theme()
save_plot(pp, "gene_classes_profile", 2.6 * length(fold_panels), 7)


#####################
# 9b. Effect sizes of each class: weight of each grid component among the class's non-zero effects
#####################
component_df$variance_label <- factor(sprintf("%.1e", component_df$variance), levels=sprintf("%.1e", sort(unique(component_df$variance))))
pp <- ggplot(component_df, aes(x=variance_label, y=factor(class_rank), fill=weight_among_nonzero)) +
	geom_tile(color="white", linewidth=.3) +
	scale_fill_gradient(low="#F4F6F8", high="#1F4E79", limits=c(0, 1), name="Weight among non-zero effects") +
	facet_wrap(~panel, nrow=1) +
	labs(x="Grid component variance", y="Class rank within fold") +
	ggtitle(paste0(tissue, ": effect sizes of the gene classes")) +
	figure_theme() +
	theme(axis.text.x=element_text(angle=90, vjust=.5, hjust=1), legend.title=element_text(size=10), panel.grid=element_blank())
save_plot(pp, "gene_classes_effect_sizes", 2.8 * length(fold_panels), 4.5)


#####################
# 9c. Traces of the ranked classes. Two classes crossing would be a label switch, which the ranks assume does not happen. Dotted: end of burn-in
#####################
first_retained <- min(trace_df$iteration[trace_df$retained == 1])
trace_long_df <- rbind(
	data.frame(panel=trace_df$panel, iteration=trace_df$iteration, class_rank=trace_df$class_rank, metric="Class weight", value=trace_df$class_weight),
	data.frame(panel=trace_df$panel, iteration=trace_df$iteration, class_rank=trace_df$class_rank, metric="Inclusion probability", value=trace_df$inclusion))
trace_long_df$metric <- factor(trace_long_df$metric, levels=c("Class weight", "Inclusion probability"))
trace_long_df$class_rank <- factor(trace_long_df$class_rank, levels=rank_levels)
trace_long_df$value <- pmax(trace_long_df$value, weight_floor)
pp <- ggplot(trace_long_df, aes(x=iteration, y=value, color=class_rank, group=class_rank)) +
	geom_vline(xintercept=first_retained, linetype="dotted", color="#7E8A97") +
	geom_line(linewidth=.3) +
	scale_y_log10() +
	scale_color_manual(values=rank_colors, name="Class rank") +
	facet_grid(metric ~ panel, scales="free_y") +
	labs(x="Iteration", y="Value") +
	ggtitle(paste0(tissue, ": traces of the gene classes")) +
	figure_theme() +
	theme(legend.title=element_text(size=10))
save_plot(pp, "gene_classes_traces", 2.8 * length(fold_panels), 6)


#####################
# 9d. How the classes divide the genes (pooled gene-fold rows): how sure each gene's class is, the prior number of identified effects its class
# implies, whether the class rate tracks the size of the window, and whether the class-implied h2 separates genes in the held-out individuals
#####################
pp_certainty <- ggplot(gene_fold_df[!is.na(gene_fold_df$class_certainty), ], aes(x=class_certainty)) +
	geom_histogram(bins=40, boundary=0, fill="#3E4A56", color=NA) +
	labs(x="Posterior probability of the gene's most probable class", y="Gene-fold rows") +
	ggtitle("Assignment certainty") +
	figure_theme()
pp_large <- ggplot(gene_fold_df[!is.na(gene_fold_df$prior_expected_n_large), ], aes(x=pmax(prior_expected_n_large, n_large_floor))) +
	geom_histogram(bins=40, fill="#3E4A56", color=NA) +
	scale_x_log10() +
	labs(x="Prior expected large-group effects per gene", y="Gene-fold rows") +
	ggtitle("Identified effects implied by the class") +
	figure_theme()
pooled_df <- gene_fold_df
pooled_df$panel <- factor("Pooled", levels=fold_levels)
binned_point_plot <- function(summary_df, xlab, ylab, title) {
	pp <- ggplot(summary_df, aes(x=x, y=y)) +
		geom_errorbar(aes(ymin=y - 1.96 * y_se, ymax=y + 1.96 * y_se), width=0, color="#3E4A56") +
		geom_line(linewidth=.35, color="#3E4A56") +
		geom_point(size=1.6, color="#3E4A56") +
		labs(x=xlab, y=ylab) +
		ggtitle(title) +
		figure_theme()
	return(pp)
}
pp_variants <- binned_point_plot(binned_calibration_df(pooled_df, "n_cis_variants", "class_large_weight"), "Number of cis variants (decile mean)", "Mean class large-group weight", "Class rate by window size") +
	scale_x_log10()
pp_h2 <- binned_point_plot(binned_calibration_df(pooled_df, "prior_expected_h2", "cis_snp_h2_he_test"), "Class-implied prior cis-h2 (decile mean)", "Mean held-out HE cis-h2", "Class-implied h2 in held-out individuals") +
	geom_abline(slope=1, intercept=0, linetype="dashed", color="#7E8A97")
save_plot(plot_grid(pp_certainty, pp_large, pp_variants, pp_h2, ncol=2), "gene_classes_per_gene", 11, 8)


#####################
# 9e. Expression FSR calibration within strata of the gene's class. Every stratifier here comes from the training individuals of the fold, so
# calibration should hold within each stratum (README.md, section 8, plot 3e)
#####################
fsr_df <- with_pooled_panel(gene_fold_df[!is.na(gene_fold_df$class_stratum), ])
summary_rows <- list()
for (stratum in class_stratum_levels) {
	summary_df <- binned_calibration_df(fsr_df[fsr_df$class_stratum == stratum, ], "expression_FSR", "false_sign")
	if (!is.null(summary_df)) {
		summary_rows[[length(summary_rows) + 1]] <- cbind(summary_df, class_stratum=stratum)
	}
}
summary_df <- do.call(rbind, summary_rows)
summary_df$class_stratum <- factor(summary_df$class_stratum, levels=class_stratum_levels)
pp <- calibration_plot(summary_df, "Expected expression FSR (decile mean)", "Fraction of genes with observed r < 0", row_var="class_stratum") +
	ggtitle(paste0(tissue, ", all genes, by the gene's class"))
save_plot(pp, "gene_classes_expression_fsr_calibration", 2.6 * length(fold_levels), 2.4 * length(class_stratum_levels) + 1)


#####################
# 9f. Mid-range expression FSR excess by the number of cis variants, the gene's class and the training evidence. What is left of the gradient with
# the number of variants? And is it in genes with strong evidence, which genes with no eQTL cannot explain?
# <tissue>_gene_classes_expression_fsr_excess.txt: the same numbers
#####################
excess_df <- gene_fold_df[!is.na(gene_fold_df$class_stratum) & !is.na(gene_fold_df$variant_stratum) & !is.na(gene_fold_df$expression_FSR) & !is.na(gene_fold_df$observed_r) &
                          !is.na(gene_fold_df$cis_snp_h2_pvalue_train) & gene_fold_df$expression_FSR > mid_range_fsr[1] & gene_fold_df$expression_FSR <= mid_range_fsr[2], ]
evidence_levels <- c(paste0("Training LRT p < ", heritability_pvalue_threshold), paste0("Training LRT p >= ", heritability_pvalue_threshold))
excess_df$evidence <- factor(ifelse(excess_df$cis_snp_h2_pvalue_train < heritability_pvalue_threshold, evidence_levels[1], evidence_levels[2]), levels=evidence_levels)
excess_df$excess <- excess_df$false_sign - excess_df$expression_FSR
excess_df <- with_pooled_panel(excess_df)
excess_rows <- list()
for (panel in fold_levels) {
	for (evidence in evidence_levels) {
		for (variant_stratum in levels(gene_fold_df$variant_stratum)) {
			for (class_stratum in class_stratum_levels) {
				cell_df <- excess_df[excess_df$panel == panel & excess_df$evidence == evidence & excess_df$variant_stratum == variant_stratum & excess_df$class_stratum == class_stratum, ]
				if (nrow(cell_df) < 10) {
					next
				}
				excess_rows[[length(excess_rows) + 1]] <- data.frame(panel=panel, evidence=evidence, variant_stratum=variant_stratum, class_stratum=class_stratum, n=nrow(cell_df),
					expected=mean(cell_df$expression_FSR), observed=mean(cell_df$false_sign), excess=mean(cell_df$excess), excess_se=cluster_se_of_mean(cell_df$excess, cell_df$gene_id))
			}
		}
	}
}
excess_summary_df <- do.call(rbind, excess_rows)
write.table(excess_summary_df, paste0(viz_dir, tissue, "_gene_classes_expression_fsr_excess.txt"), sep="\t", quote=FALSE, row.names=FALSE)
print(excess_summary_df[excess_summary_df$panel == "Pooled", ])
excess_summary_df$panel <- factor(excess_summary_df$panel, levels=fold_levels)
excess_summary_df$evidence <- factor(excess_summary_df$evidence, levels=evidence_levels)
excess_summary_df$variant_stratum <- factor(excess_summary_df$variant_stratum, levels=levels(gene_fold_df$variant_stratum))
excess_summary_df$class_stratum <- factor(excess_summary_df$class_stratum, levels=class_stratum_levels)
pp <- ggplot(excess_summary_df, aes(x=variant_stratum, y=excess, color=class_stratum)) +
	geom_hline(yintercept=0, linetype="dashed", color="#7E8A97") +
	geom_errorbar(aes(ymin=excess - 1.96 * excess_se, ymax=excess + 1.96 * excess_se), width=0, position=position_dodge(width=.6)) +
	geom_point(size=1.8, position=position_dodge(width=.6)) +
	scale_color_manual(values=class_stratum_colors) +
	facet_grid(evidence ~ panel, scales="free_y") +
	labs(x="Number of cis variants", y=paste0("Observed - expected, expression FSR in (", mid_range_fsr[1], ", ", mid_range_fsr[2], "]")) +
	ggtitle(paste0(tissue, ": expression FSR excess by the gene's class")) +
	figure_theme() +
	theme(axis.text.x=element_text(angle=30, hjust=1))
save_plot(pp, "gene_classes_expression_fsr_excess", 2.8 * length(fold_levels), 6.5)


#####################
# 9g. Expected cis-h2 against held-out HE within strata of the gene's class: is the gap at high h2 (expected below HE) in one kind of gene?
#####################
h2_df <- with_pooled_panel(gene_fold_df[!is.na(gene_fold_df$class_stratum), ])
summary_rows <- list()
for (stratum in class_stratum_levels) {
	summary_df <- binned_calibration_df(h2_df[h2_df$class_stratum == stratum, ], "expected_cis_h2", "cis_snp_h2_he_test")
	if (!is.null(summary_df)) {
		summary_rows[[length(summary_rows) + 1]] <- cbind(summary_df, class_stratum=stratum)
	}
}
summary_df <- do.call(rbind, summary_rows)
summary_df$class_stratum <- factor(summary_df$class_stratum, levels=class_stratum_levels)
pp <- calibration_plot(summary_df, "Expected cis-h2 (held-out individuals, decile mean)", "Held-out HE cis-h2", row_var="class_stratum") +
	ggtitle(paste0(tissue, ", all genes: cis-h2 by the gene's class"))
save_plot(pp, "gene_classes_heritability_calibration", 2.6 * length(fold_levels), 2.4 * length(class_stratum_levels) + 1)
