args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
library(RColorBrewer)
library(scales)
options(warn=1)

# Plots of the cross-validated eQTL prior fits of one tissue, from organize_cross_validation_results.py
#   args[1]: organized results stem (the --output_stem of the organizer); args[2]: output directory for the pdfs; args[3]: tissue
# The earlier exploratory plots are kept in visualize_cross_validation_results_exploratory.R

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}




compute_decile_summary_df <- function(df, x_col, y_col, n_bins) {
	# Split genes into n_bins equal-count bins (deciles when n_bins=10) by x_col (the expected quantity). Each row gives the bin's mean x and
	# the mean of y_col (the observed quantity) with a 95% CI (mean +/- 1.96 * empirical SE across genes in the bin).
	df = df[!is.na(df[[x_col]]) & !is.na(df[[y_col]]), ]
	bin_id = ceiling(rank(df[[x_col]], ties.method="first") / nrow(df) * n_bins)

	bin_arr = c()
	n_gene_arr = c()
	mean_x_arr = c()
	mean_y_arr = c()
	mean_y_lb_arr = c()
	mean_y_ub_arr = c()
	for (bin_iter in 1:n_bins) {
		bin_df = df[bin_id == bin_iter, ]
		mean_y = mean(bin_df[[y_col]])
		se_y = sd(bin_df[[y_col]]) / sqrt(nrow(bin_df))

		bin_arr = c(bin_arr, bin_iter)
		n_gene_arr = c(n_gene_arr, nrow(bin_df))
		mean_x_arr = c(mean_x_arr, mean(bin_df[[x_col]]))
		mean_y_arr = c(mean_y_arr, mean_y)
		mean_y_lb_arr = c(mean_y_lb_arr, mean_y - 1.96*se_y)
		mean_y_ub_arr = c(mean_y_ub_arr, mean_y + 1.96*se_y)
	}

	summary_df = data.frame(bin=bin_arr, n_genes=n_gene_arr, mean_x=mean_x_arr, mean_y=mean_y_arr, mean_y_lb=mean_y_lb_arr, mean_y_ub=mean_y_ub_arr)
	return(summary_df)
}


compute_decile_summary_df_by_fold <- function(df, x_col, y_col, n_bins, folds) {
	# compute_decile_summary_df within each fold (deciles are formed within the fold), stacked, with a "Fold <k>" label column
	summary_df_list = list()
	for (fold in folds) {
		fold_summary_df = compute_decile_summary_df(df[df$fold == fold, ], x_col, y_col, n_bins)
		fold_summary_df$fold = paste0("Fold ", fold)
		summary_df_list[[length(summary_df_list) + 1]] = fold_summary_df
	}
	summary_df = do.call(rbind, summary_df_list)
	summary_df$fold = factor(summary_df$fold, levels=paste0("Fold ", folds))
	return(summary_df)
}


compute_genetic_correlation_decile_summary_df <- function(df, n_bins, h2_source="ld_corrected_he") {
	# Split genes into n_bins equal-count bins by expected_genetic_r (expected correlation of the prediction with genetic expression). In each
	# bin, observed r is about r(y_hat, u) sqrt(h2), so y = mean(observed_r) / sqrt(h2) estimates the genetic correlation. h2 is set by h2_source:
	#   "ld_corrected_he": held-out HE corrected for the way HE weights LD: mean(cis_snp_h2_he_test) x mean(expected_cis_h2) /
	#       mean(expected_cis_snp_h2_he_var_test). That ratio is computed from bin means and treated as fixed
	#   "model": the model's own expected cis-h2, mean(expected_cis_h2) (Var(X beta) in the held-out individuals). y / x is then about observed r
	#       over the model's expected r, so this cannot separate an error in the genetic correlation from one in h2
	# The 95% CI of y is from a delta-method SE for a / sqrt(c), with a and c the bin means of observed_r and of the per-gene h2. Bins whose
	# predicted HE or h2 is <= 0 get no point.
	if (h2_source == "ld_corrected_he") {
		df = df[!is.na(df$expected_genetic_r) & !is.na(df$observed_r) & !is.na(df$cis_snp_h2_he_test) & !is.na(df$expected_cis_h2) & !is.na(df$expected_cis_snp_h2_he_var_test), ]
	} else {
		df = df[!is.na(df$expected_genetic_r) & !is.na(df$observed_r) & !is.na(df$expected_cis_h2), ]
	}
	bin_id = ceiling(rank(df$expected_genetic_r, ties.method="first") / nrow(df) * n_bins)

	bin_arr = c()
	n_gene_arr = c()
	mean_x_arr = c()
	mean_y_arr = c()
	mean_y_lb_arr = c()
	mean_y_ub_arr = c()
	for (bin_iter in 1:n_bins) {
		bin_df = df[bin_id == bin_iter, ]
		n_genes = nrow(bin_df)
		if (h2_source == "ld_corrected_he") {
			predicted_he = mean(bin_df$expected_cis_snp_h2_he_var_test)
			if (predicted_he <= 0) {
				next
			}
			h2 = bin_df$cis_snp_h2_he_test * mean(bin_df$expected_cis_h2) / predicted_he
		} else {
			h2 = bin_df$expected_cis_h2
		}
		a = mean(bin_df$observed_r)
		c = mean(h2)
		if (c <= 0) {
			next
		}
		var_a = var(bin_df$observed_r) / n_genes
		var_c = var(h2) / n_genes
		cov_ac = cov(bin_df$observed_r, h2) / n_genes
		var_y = var_a / c - a * cov_ac / c^2 + a^2 * var_c / (4 * c^3)
		mean_y = a / sqrt(c)
		se_y = sqrt(max(var_y, 0))

		bin_arr = c(bin_arr, bin_iter)
		n_gene_arr = c(n_gene_arr, n_genes)
		mean_x_arr = c(mean_x_arr, mean(bin_df$expected_genetic_r))
		mean_y_arr = c(mean_y_arr, mean_y)
		mean_y_lb_arr = c(mean_y_lb_arr, mean_y - 1.96*se_y)
		mean_y_ub_arr = c(mean_y_ub_arr, mean_y + 1.96*se_y)
	}

	summary_df = data.frame(bin=bin_arr, n_genes=n_gene_arr, mean_x=mean_x_arr, mean_y=mean_y_arr, mean_y_lb=mean_y_lb_arr, mean_y_ub=mean_y_ub_arr)
	return(summary_df)
}


compute_genetic_correlation_decile_summary_df_by_fold <- function(df, n_bins, folds, h2_source="ld_corrected_he") {
	# compute_genetic_correlation_decile_summary_df within each fold, stacked, with a "Fold <k>" label column
	summary_df_list = list()
	for (fold in folds) {
		fold_summary_df = compute_genetic_correlation_decile_summary_df(df[df$fold == fold, ], n_bins, h2_source)
		fold_summary_df$fold = paste0("Fold ", fold)
		summary_df_list[[length(summary_df_list) + 1]] = fold_summary_df
	}
	summary_df = do.call(rbind, summary_df_list)
	summary_df$fold = factor(summary_df$fold, levels=paste0("Fold ", folds))
	return(summary_df)
}


compute_coverage_summary_df <- function(df, coverage_levels) {
	# For each nominal level L (in %), the fraction of genes whose observed held-out correlation lies inside their L% posterior predictive interval
	# (expected_r_lb_L, expected_r_ub_L: quantiles over posterior samples of corr(y_hat, u + eps)), with a 95% binomial (normal approximation) CI.
	# Same columns as compute_decile_summary_df (mean_x is the nominal level as a fraction), so make_calibration_plot draws it.
	df = df[!is.na(df$observed_r), ]

	level_arr = c()
	n_gene_arr = c()
	coverage_arr = c()
	coverage_lb_arr = c()
	coverage_ub_arr = c()
	for (level in coverage_levels) {
		lb = df[[paste0("expected_r_lb_", level)]]
		ub = df[[paste0("expected_r_ub_", level)]]
		keep = !is.na(lb) & !is.na(ub)
		covered = (df$observed_r[keep] >= lb[keep]) & (df$observed_r[keep] <= ub[keep])
		coverage = mean(covered)
		coverage_se = sqrt(coverage*(1.0 - coverage) / sum(keep))

		level_arr = c(level_arr, level / 100)
		n_gene_arr = c(n_gene_arr, sum(keep))
		coverage_arr = c(coverage_arr, coverage)
		coverage_lb_arr = c(coverage_lb_arr, coverage - 1.96*coverage_se)
		coverage_ub_arr = c(coverage_ub_arr, coverage + 1.96*coverage_se)
	}

	summary_df = data.frame(n_genes=n_gene_arr, mean_x=level_arr, mean_y=coverage_arr, mean_y_lb=coverage_lb_arr, mean_y_ub=coverage_ub_arr)
	return(summary_df)
}


compute_coverage_summary_df_by_fold <- function(df, coverage_levels, folds) {
	# compute_coverage_summary_df within each fold, stacked, with a "Fold <k>" label column
	summary_df_list = list()
	for (fold in folds) {
		fold_summary_df = compute_coverage_summary_df(df[df$fold == fold, ], coverage_levels)
		fold_summary_df$fold = paste0("Fold ", fold)
		summary_df_list[[length(summary_df_list) + 1]] = fold_summary_df
	}
	summary_df = do.call(rbind, summary_df_list)
	summary_df$fold = factor(summary_df$fold, levels=paste0("Fold ", folds))
	return(summary_df)
}


make_calibration_plot <- function(summary_df, xlab, ylab, point_color=single_color) {
	# Calibration plot: mean observed (points with 95% CI, joined by a line) against mean expected in each decile of the expected quantity.
	# The dashed line is y = x. If summary_df has a fold column, one panel per fold, colored by fold (fold_colors); otherwise everything is
	# drawn in point_color.
	reference_color = "#3F3F46"

	if ("fold" %in% colnames(summary_df)) {
		pp = ggplot(summary_df, aes(x=mean_x, y=mean_y, color=fold)) +
			geom_abline(slope=1, intercept=0, color=reference_color, linetype="dashed", linewidth=0.6) +
			geom_line(linewidth=0.7) +
			geom_pointrange(aes(ymin=mean_y_lb, ymax=mean_y_ub), linewidth=0.7) +
			geom_point(size=2.4) +
			scale_color_manual(values=fold_colors, guide="none") +
			facet_wrap(~fold, nrow=1) +
			figure_theme() +
			theme(strip.background=element_blank(), strip.text=element_text(size=11), panel.spacing=unit(1.2, "lines")) +
			labs(x=xlab, y=ylab)
	} else {
		pp = ggplot(summary_df, aes(x=mean_x, y=mean_y)) +
			geom_abline(slope=1, intercept=0, color=reference_color, linetype="dashed", linewidth=0.6) +
			geom_line(color=point_color, linewidth=0.7) +
			geom_pointrange(aes(ymin=mean_y_lb, ymax=mean_y_ub), color=point_color, linewidth=0.7) +
			geom_point(color=point_color, size=2.4) +
			figure_theme() +
			labs(x=xlab, y=ylab)
	}
	return(pp)
}


compute_fsr_threshold_summary_df <- function(df, fsr_col, fsr_thresholds) {
	# For each FSR threshold, the genes called (FSR <= threshold): their number, the fraction with a negative held-out correlation (observed)
	# with a 95% binomial (normal approximation) CI, and their mean FSR (expected fraction with the wrong sign).
	df = df[!is.na(df[[fsr_col]]) & !is.na(df$false_sign), ]

	threshold_arr = c()
	n_gene_arr = c()
	observed_arr = c()
	observed_lb_arr = c()
	observed_ub_arr = c()
	expected_arr = c()
	for (threshold in fsr_thresholds) {
		called_df = df[df[[fsr_col]] <= threshold, ]
		if (nrow(called_df) == 0) {
			next
		}
		observed = mean(called_df$false_sign)
		observed_se = sqrt(observed*(1.0 - observed) / nrow(called_df))

		threshold_arr = c(threshold_arr, threshold)
		n_gene_arr = c(n_gene_arr, nrow(called_df))
		observed_arr = c(observed_arr, observed)
		observed_lb_arr = c(observed_lb_arr, observed - 1.96*observed_se)
		observed_ub_arr = c(observed_ub_arr, observed + 1.96*observed_se)
		expected_arr = c(expected_arr, mean(called_df[[fsr_col]]))
	}

	summary_df = data.frame(threshold=threshold_arr, n_genes=n_gene_arr, observed=observed_arr, observed_lb=observed_lb_arr, observed_ub=observed_ub_arr, expected=expected_arr)
	summary_df$threshold_label = factor(summary_df$threshold, levels=fsr_thresholds)
	return(summary_df)
}


compute_fsr_threshold_summary_df_by_fold <- function(df, fsr_col, fsr_thresholds, folds) {
	# compute_fsr_threshold_summary_df within each fold, stacked, with a "Fold <k>" label column
	summary_df_list = list()
	for (fold in folds) {
		fold_summary_df = compute_fsr_threshold_summary_df(df[df$fold == fold, ], fsr_col, fsr_thresholds)
		fold_summary_df$fold = paste0("Fold ", fold)
		summary_df_list[[length(summary_df_list) + 1]] = fold_summary_df
	}
	summary_df = do.call(rbind, summary_df_list)
	summary_df$fold = factor(summary_df$fold, levels=paste0("Fold ", folds))
	return(summary_df)
}


make_fsr_threshold_plot <- function(summary_df, xlab, ylab, point_color=single_color, label_size=2.8) {
	# At each FSR threshold: observed fraction of called genes with r < 0 (filled points with 95% CI, solid line; number of genes called above),
	# and the mean FSR of the called genes (expected; open points, dotted line). If summary_df has a fold column, one panel per fold, colored by
	# fold (fold_colors); otherwise everything is drawn in point_color. A legend (black keys, so it serves every fold color) for observed and
	# expected sits inside the top left of the (first) panel.
	series_levels = c("Observed", "Expected")
	by_fold = "fold" %in% colnames(summary_df)
	if (by_fold) {
		pp = ggplot(summary_df, aes(x=threshold_label, group=fold, color=fold))
	} else {
		summary_df$series = "single"
		pp = ggplot(summary_df, aes(x=threshold_label, group=series, color=series))
	}
	pp = pp +
		geom_line(aes(y=expected, linetype=series_levels[2]), linewidth=0.5) +
		geom_point(aes(y=expected, shape=series_levels[2]), size=1.8) +
		geom_line(aes(y=observed, linetype=series_levels[1]), linewidth=0.7) +
		geom_pointrange(aes(y=observed, ymin=observed_lb, ymax=observed_ub), linewidth=0.7, size=0.4, show.legend=FALSE) +
		geom_point(aes(y=observed, shape=series_levels[1]), size=1.6) +
		geom_text(aes(y=observed_ub, label=n_genes), vjust=-0.7, size=label_size, show.legend=FALSE) +
		scale_linetype_manual(name=NULL, values=setNames(c("solid", "dotted"), series_levels), breaks=series_levels) +
		scale_shape_manual(name=NULL, values=setNames(c(16, 1), series_levels), breaks=series_levels) +
		scale_y_continuous(expand=expansion(mult=c(0.05, 0.12))) +
		figure_theme() +
		theme(legend.position="inside", legend.position.inside=c(0.01, 0.99), legend.justification=c(0, 1), legend.background=element_blank(),
		      legend.key=element_blank(), legend.key.width=unit(1.6, "lines")) +
		labs(x=xlab, y=ylab)
	if (by_fold) {
		pp = pp + scale_color_manual(values=fold_colors, guide="none") + facet_wrap(~fold, nrow=1) +
			theme(strip.background=element_blank(), strip.text=element_text(size=11), panel.spacing=unit(1.2, "lines"), axis.text.x=element_text(angle=45, hjust=1),
			      legend.text=element_text(size=9), legend.key.height=unit(0.9, "lines"))
	} else {
		pp = pp + scale_color_manual(values=c(single=point_color), guide="none")
	}
	return(pp)
}


assign_quantile_bins <- function(x, n_bins, format_value, display_x=x) {
	# Split x into n_bins equal-count bins (ties broken by row order, as for the FSR deciles), bin 1 holding the smallest x. Returns a factor
	# whose labels give the bin and the range of display_x (by default x) in it, formatted with format_value.
	bin_id = ceiling(rank(x, ties.method="first") / length(x) * n_bins)
	bin_labels = c()
	for (bin_iter in 1:n_bins) {
		bin_range = range(display_x[bin_id == bin_iter])
		bin_labels = c(bin_labels, paste0("Q", bin_iter, "\n[", format_value(bin_range[1]), ", ", format_value(bin_range[2]), "]"))
	}
	return(factor(bin_labels[bin_id], levels=bin_labels))
}


make_violin_plot <- function(df, bin_col, value_col, xlab, ylab, fill_color=single_color) {
	# Violin of value_col within each bin of bin_col (each violin scaled to the same width), with a boxplot (median and quartiles) inside
	pp = ggplot(df, aes(x=.data[[bin_col]], y=.data[[value_col]])) +
		geom_violin(fill=fill_color, color=fill_color, alpha=0.55, scale="width", linewidth=0.4) +
		geom_boxplot(width=0.12, outlier.shape=NA, fill="white", color="black", linewidth=0.4) +
		figure_theme() +
		labs(x=xlab, y=ylab)
	return(pp)
}



compute_coverage_by_decile_summary_df <- function(df, x_col, coverage_levels, n_bins) {
	# Split genes into n_bins equal-count bins by x_col. In each bin and at each nominal level L (in %), the fraction of genes whose observed
	# held-out correlation lies inside their L% posterior predictive interval, with a 95% binomial (normal approximation) CI.
	df = df[!is.na(df[[x_col]]) & !is.na(df$observed_r), ]
	bin_id = ceiling(rank(df[[x_col]], ties.method="first") / nrow(df) * n_bins)

	rows = list()
	for (bin_iter in 1:n_bins) {
		bin_df = df[bin_id == bin_iter, ]
		for (level in coverage_levels) {
			lb = bin_df[[paste0("expected_r_lb_", level)]]
			ub = bin_df[[paste0("expected_r_ub_", level)]]
			keep = !is.na(lb) & !is.na(ub)
			covered = (bin_df$observed_r[keep] >= lb[keep]) & (bin_df$observed_r[keep] <= ub[keep])
			coverage = mean(covered)
			coverage_se = sqrt(coverage*(1.0 - coverage) / sum(keep))
			rows[[length(rows) + 1]] = data.frame(bin=bin_iter, level=level, n_genes=sum(keep), mean_x=mean(bin_df[[x_col]]), coverage=coverage,
				coverage_lb=coverage - 1.96*coverage_se, coverage_ub=coverage + 1.96*coverage_se)
		}
	}
	summary_df = do.call(rbind, rows)
	summary_df$level_label = factor(paste0(summary_df$level, "%"), levels=paste0(coverage_levels, "%"))
	return(summary_df)
}


make_coverage_by_decile_plot <- function(summary_df, xlab, ylab) {
	# Coverage (points with 95% CI, joined by a line) against the decile mean of the binning quantity, one color per nominal level, with a dashed
	# horizontal line at each nominal level in its color
	level_colors = setNames(hcl.colors(nlevels(summary_df$level_label) + 1, "Plasma")[1:nlevels(summary_df$level_label)], levels(summary_df$level_label))
	reference_df = unique(summary_df[, c("level", "level_label")])

	pp = ggplot(summary_df, aes(x=mean_x, y=coverage, color=level_label)) +
		geom_hline(data=reference_df, aes(yintercept=level / 100, color=level_label), linetype="dashed", linewidth=0.5, show.legend=FALSE) +
		geom_line(linewidth=0.7) +
		geom_pointrange(aes(ymin=coverage_lb, ymax=coverage_ub), linewidth=0.6, size=0.35) +
		scale_color_manual(values=level_colors, name="Nominal level") +
		figure_theme() +
		labs(x=xlab, y=ylab)
	return(pp)
}


compute_top_n_summary_df <- function(df, rankings, top_n_min_genes, value_col="observed_r") {
	# For each ranking (a named list of functions of the gene table, higher = better), rank the genes and give the mean of value_col over the top
	# N genes (observed_r: mean observed held-out correlation; true_sign: 1 - observed FSR, the fraction with r >= 0), at about 200 values of N spaced
	# evenly on a log scale from top_n_min_genes to all genes, plus about 200 spaced evenly on a linear scale (for a linear x axis in
	# make_top_n_plot). A ranking function returns either a score or a list of scores: the first orders
	# the genes and the later ones break its ties, in turn (remaining ties keep the order of the rows). Genes with any score missing are left out.
	df = df[!is.na(df[[value_col]]), ]
	rows = list()
	for (ranking in names(rankings)) {
		scores = rankings[[ranking]](df)
		if (!is.list(scores)) {
			scores = list(scores)
		}
		valid = Reduce(`&`, lapply(scores, function(score) !is.na(score)))
		if (sum(valid) < top_n_min_genes) {
			next
		}
		ordering = do.call(order, lapply(scores, function(score) -score[valid]))
		ordered_value = df[[value_col]][valid][ordering]
		running_mean = cumsum(ordered_value) / seq_along(ordered_value)
		n_genes = length(ordered_value)
		top_n = sort(unique(round(c(10^seq(log10(top_n_min_genes), log10(n_genes), length.out=200), seq(top_n_min_genes, n_genes, length.out=200)))))
		rows[[length(rows) + 1]] = data.frame(ranking=ranking, top_n=top_n, mean_value=running_mean[top_n])
	}
	summary_df = do.call(rbind, rows)
	summary_df$ranking = factor(summary_df$ranking, levels=names(rankings))
	return(summary_df)
}


compute_top_n_summary_df_by_fold <- function(df, rankings, top_n_min_genes, folds, value_col="observed_r") {
	# compute_top_n_summary_df within each fold, stacked, with a "Fold <k>" label column
	summary_df_list = list()
	for (fold in folds) {
		fold_summary_df = compute_top_n_summary_df(df[df$fold == fold, ], rankings, top_n_min_genes, value_col)
		fold_summary_df$fold = paste0("Fold ", fold)
		summary_df_list[[length(summary_df_list) + 1]] = fold_summary_df
	}
	summary_df = do.call(rbind, summary_df_list)
	summary_df$fold = factor(summary_df$fold, levels=paste0("Fold ", folds))
	return(summary_df)
}


make_top_n_plot <- function(summary_df, xlab, ylab, log_x=TRUE, legend_bottom=FALSE) {
	# Mean value over the top N genes (compute_top_n_summary_df) against N (log scale, or linear with log_x=FALSE), one line per ranking
	# (ranking_colors). If summary_df has a fold column, one panel per fold with the legend below; otherwise the legend is on the right (below,
	# in three rows, with legend_bottom).
	if (log_x) {
		x_scale = scale_x_log10()
	} else {
		x_scale = scale_x_continuous()
	}
	pp = ggplot(summary_df, aes(x=top_n, y=mean_value, color=ranking)) +
		geom_line(linewidth=0.7) +
		x_scale +
		scale_color_manual(values=ranking_colors, name="Ranking") +
		figure_theme() +
		labs(x=xlab, y=ylab)
	by_fold = "fold" %in% colnames(summary_df)
	if (by_fold) {
		pp = pp + facet_wrap(~fold, nrow=1) +
			theme(strip.background=element_blank(), strip.text=element_text(size=11), panel.spacing=unit(1.2, "lines"))
	}
	if (by_fold | legend_bottom) {
		pp = pp + guides(color=guide_legend(nrow=3, byrow=TRUE)) + theme(legend.position="bottom", legend.title=element_blank(), legend.key=element_blank())
	} else {
		pp = pp + theme(legend.key=element_blank())
	}
	return(pp)
}


make_histogram <- function(df, value_col, bin_width, xlab, ylab, fill_color=single_color) {
	# Histogram of value_col in bins of width bin_width, starting at 0
	df = df[!is.na(df[[value_col]]), ]
	pp = ggplot(df, aes(x=.data[[value_col]])) +
		geom_histogram(binwidth=bin_width, boundary=0, fill=fill_color, color="white", linewidth=0.3) +
		scale_y_continuous(expand=expansion(mult=c(0, 0.05))) +
		figure_theme() +
		labs(x=xlab, y=ylab)
	return(pp)
}


make_scatter_plot <- function(df, x_col, y_col, xlab, ylab, point_color=single_color) {
	# Scatter of y_col against x_col, one point per gene (small and semi-transparent, since there are thousands of genes)
	pp = ggplot(df, aes(x=.data[[x_col]], y=.data[[y_col]])) +
		geom_point(color=point_color, size=0.6, alpha=0.35, stroke=0) +
		figure_theme() +
		labs(x=xlab, y=ylab)
	return(pp)
}


#####################
# Command line args
#####################
organized_stem = args[1]
viz_dir = args[2]
tissue = args[3]
dir.create(viz_dir, showWarnings=FALSE, recursive=TRUE)

# Number of equal-count bins of the expected quantity in the calibration plots
n_bins = 10
# Fold shown on its own in Figures 1, 3, 4, 6, 8, 9, 11, 13, 17, 19, 20 and 22
single_fold = 3
# Number of equal-count bins of training cis-h2 (and of its p-value) in Figure 3
n_h2_bins = 5
# FSR thresholds in Figures 6 and 7 (genes called at FSR <= threshold), and the training-fold cis-h2 LRT p-value threshold of the heritable
# genes they use (training data only, so selecting on it does not bias the held-out evaluation)
fsr_thresholds = c(0.001, 0.01, 0.05, 0.1, 0.2, 0.3)
fsr_threshold_heritable_pvalue_threshold = 0.05
# Nominal levels (%) of the posterior predictive intervals of the observed correlation in Figures 17 and 18 (the levels in the evaluation files)
coverage_levels = c(80, 90, 95, 99)
# Rankings of genes in Figures 20 to 23 (score: higher = better; a list of scores breaks ties in the first by the next, see compute_top_n_summary_df).
# Thousands of genes have an FSR of exactly 0, so the FSR rankings break ties by the matching expected correlation. The smallest N shown
top_n_rankings = list(
	"Expected r"=function(d) d$expected_r,
	"1 - expression FSR"=function(d) list(1 - d$expression_FSR, d$expected_r),
	"Expected genetic correlation"=function(d) d$expected_genetic_r,
	"1 - genetic FSR"=function(d) list(1 - d$genetic_FSR, d$expected_genetic_r),
	"Expected cis-h2"=function(d) d$expected_cis_h2,
	"Training HE cis-h2"=function(d) d$cis_snp_h2_he_train,
	"Training cis-h2 -log10 p"=function(d) -log10(d$cis_snp_h2_pvalue_train))
top_n_min_genes = 10


#####################
# Load in cross-validation results (one row per gene and held-out fold)
#####################
gene_fold_df = read.table(paste0(organized_stem, "_per_gene_fold.txt"), header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))
folds = sort(unique(gene_fold_df$fold))
# Indicator of a negative held-out correlation (the observed counterpart of the FSR)
gene_fold_df$false_sign = as.numeric(gene_fold_df$observed_r < 0)
# Its complement (1 - observed FSR when averaged over genes)
gene_fold_df$true_sign = 1 - gene_fold_df$false_sign

# Colors: one per fold (figures with a panel per fold), and one for figures without folds
fold_colors = setNames(hcl.colors(length(folds), "Dark 3"), paste0("Fold ", folds))
single_color = "#E8590C"
# One color per gene ranking (Figures 20 to 23)
ranking_colors = setNames(brewer.pal(8, "Dark2")[1:length(top_n_rankings)], names(top_n_rankings))


#####################
# Figure 1: Expression FSR calibration (deciles of expression FSR), single fold
#####################
fold_summary_df = compute_decile_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], "expression_FSR", "false_sign", n_bins)
fold_plot = make_calibration_plot(fold_summary_df, "Expected expression FSR", "Fraction of genes with observed r < 0")
output_file = paste0(viz_dir, tissue, "_expression_fsr_calibration_deciles_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 2: Expression FSR calibration (deciles of expression FSR), one panel per fold
#####################
by_fold_summary_df = compute_decile_summary_df_by_fold(gene_fold_df, "expression_FSR", "false_sign", n_bins, folds)
by_fold_plot = make_calibration_plot(by_fold_summary_df, "Expected expression FSR", "Fraction of genes with observed r < 0")
output_file = paste0(viz_dir, tissue, "_expression_fsr_calibration_deciles_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.0, units="in")



#####################
# Figure 3: Distribution of genetic FSR within quintiles of training-fold cis-h2 (ML estimate; top) and of its LRT p-value (bottom), single fold.
# Both are model-free and use training individuals only. The p-value is binned on -p, so in both panels Q1 is the weakest evidence of cis
# heritability and Q5 the strongest
#####################
h2_df = gene_fold_df[gene_fold_df$fold == single_fold & !is.na(gene_fold_df$genetic_FSR) & !is.na(gene_fold_df$cis_snp_h2_train) & !is.na(gene_fold_df$cis_snp_h2_pvalue_train), ]
h2_df$h2_bin = assign_quantile_bins(h2_df$cis_snp_h2_train, n_h2_bins, function(v) sprintf("%.3f", v))
h2_df$h2_pvalue_bin = assign_quantile_bins(-h2_df$cis_snp_h2_pvalue_train, n_h2_bins, function(v) formatC(v, digits=2, format="g"), display_x=h2_df$cis_snp_h2_pvalue_train)
h2_violin = make_violin_plot(h2_df, "h2_bin", "genetic_FSR", "Training cis-h2 (quintile)", "Genetic FSR")
h2_pvalue_violin = make_violin_plot(h2_df, "h2_pvalue_bin", "genetic_FSR", "Training cis-h2 LRT p-value (quintile)", "Genetic FSR")
output_file = paste0(viz_dir, tissue, "_genetic_fsr_by_cis_h2_quintile_fold_", single_fold, ".pdf")
ggsave(output_file, plot_grid(h2_violin, h2_pvalue_violin, ncol=1, labels=c("a", "b")), width=7.2, height=6.5, units="in")



#####################
# Figure 4: Correlation calibration (deciles of the plug-in expected correlation sqrt(V), V = variance of the prediction in the held-out
# individuals), single fold
#####################
fold_summary_df = compute_decile_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], "expected_r_plugin", "observed_r", n_bins)
fold_plot = make_calibration_plot(fold_summary_df, "Expected correlation", "Observed held-out correlation")
output_file = paste0(viz_dir, tissue, "_correlation_calibration_plugin_deciles_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 5: Correlation calibration (deciles of the plug-in expected correlation), one panel per fold
#####################
by_fold_summary_df = compute_decile_summary_df_by_fold(gene_fold_df, "expected_r_plugin", "observed_r", n_bins, folds)
by_fold_plot = make_calibration_plot(by_fold_summary_df, "Expected correlation", "Observed held-out correlation")
output_file = paste0(viz_dir, tissue, "_correlation_calibration_plugin_deciles_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.0, units="in")



#####################
# Figure 6: Expression FSR calibration at FSR thresholds in heritable genes (training cis-h2 LRT p < fsr_threshold_heritable_pvalue_threshold),
# single fold. Filled points: observed fraction of called genes with r < 0 (95% CI; number of genes called above). Open points: mean expression
# FSR of the called genes (expected)
#####################
fsr_threshold_df = gene_fold_df[!is.na(gene_fold_df$cis_snp_h2_pvalue_train) & gene_fold_df$cis_snp_h2_pvalue_train < fsr_threshold_heritable_pvalue_threshold, ]
fold_summary_df = compute_fsr_threshold_summary_df(fsr_threshold_df[fsr_threshold_df$fold == single_fold, ], "expression_FSR", fsr_thresholds)
fold_plot = make_fsr_threshold_plot(fold_summary_df, "Expression FSR threshold (genes with FSR <= threshold)", "Fraction of called genes with observed r < 0")
output_file = paste0(viz_dir, tissue, "_expression_fsr_threshold_calibration_heritable_p_", fsr_threshold_heritable_pvalue_threshold, "_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 7: As Figure 6, one panel per fold
#####################
by_fold_summary_df = compute_fsr_threshold_summary_df_by_fold(fsr_threshold_df, "expression_FSR", fsr_thresholds, folds)
by_fold_plot = make_fsr_threshold_plot(by_fold_summary_df, "Expression FSR threshold (genes with FSR <= threshold)", "Fraction of called genes\nwith observed r < 0", label_size=2.2)
output_file = paste0(viz_dir, tissue, "_expression_fsr_threshold_calibration_heritable_p_", fsr_threshold_heritable_pvalue_threshold, "_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.2, units="in")



#####################
# Figure 8: Histogram of genetic FSR (bins of 0.01), all genes, single fold
#####################
genetic_fsr_histogram = make_histogram(gene_fold_df[gene_fold_df$fold == single_fold, ], "genetic_FSR", 0.01, "Genetic FSR", "Number of genes")
output_file = paste0(viz_dir, tissue, "_genetic_fsr_histogram_fold_", single_fold, ".pdf")
ggsave(output_file, genetic_fsr_histogram, width=7.2, height=4.0, units="in")


#####################
# Figure 9: Heritability calibration (deciles of expected cis-h2, mean over posterior samples of the variance of genetic expression in the
# held-out individuals) against held-out HE cis-h2, all genes, single fold
#####################
fold_summary_df = compute_decile_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], "expected_cis_h2", "cis_snp_h2_he_test", n_bins)
fold_plot = make_calibration_plot(fold_summary_df, "Expected cis-h2 (held-out individuals)", "Held-out HE cis-h2")
output_file = paste0(viz_dir, tissue, "_heritability_calibration_deciles_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 10: As Figure 9, one panel per fold
#####################
by_fold_summary_df = compute_decile_summary_df_by_fold(gene_fold_df, "expected_cis_h2", "cis_snp_h2_he_test", n_bins, folds)
by_fold_plot = make_calibration_plot(by_fold_summary_df, "Expected cis-h2 (held-out individuals)", "Held-out HE cis-h2")
output_file = paste0(viz_dir, tissue, "_heritability_calibration_deciles_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.0, units="in")



#####################
# Figure 11: Held-out HE cis-h2 against what the model predicts it to be (deciles of expected_cis_snp_h2_he_var_test, mean over posterior samples
# of the expected HE slope u' W u in the held-out individuals, which weights effects by LD the way HE does), all genes, single fold. Neither axis
# is multiplied by Var(y) (held-out Var(y) is about 1)
#####################
fold_summary_df = compute_decile_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], "expected_cis_snp_h2_he_var_test", "cis_snp_h2_he_test", n_bins)
fold_plot = make_calibration_plot(fold_summary_df, "Model-predicted held-out HE cis-h2", "Held-out HE cis-h2")
output_file = paste0(viz_dir, tissue, "_predicted_he_calibration_deciles_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 12: As Figure 11, one panel per fold
#####################
by_fold_summary_df = compute_decile_summary_df_by_fold(gene_fold_df, "expected_cis_snp_h2_he_var_test", "cis_snp_h2_he_test", n_bins, folds)
by_fold_plot = make_calibration_plot(by_fold_summary_df, "Model-predicted held-out HE cis-h2", "Held-out HE cis-h2")
output_file = paste0(viz_dir, tissue, "_predicted_he_calibration_deciles_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.0, units="in")



#####################
# Figure 13: Genetic correlation calibration with LD-corrected held-out HE (deciles of expected_genetic_r; see
# compute_genetic_correlation_decile_summary_df), all genes, single fold
#####################
fold_summary_df = compute_genetic_correlation_decile_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], n_bins)
fold_plot = make_calibration_plot(fold_summary_df, "Expected correlation with genetic expression", "Mean observed r /\nsqrt(LD-corrected HE h2)")
output_file = paste0(viz_dir, tissue, "_genetic_correlation_calibration_ld_corrected_deciles_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 14: As Figure 13, one panel per fold
#####################
by_fold_summary_df = compute_genetic_correlation_decile_summary_df_by_fold(gene_fold_df, n_bins, folds)
by_fold_plot = make_calibration_plot(by_fold_summary_df, "Expected correlation with genetic expression", "Mean observed r /\nsqrt(LD-corrected HE h2)")
output_file = paste0(viz_dir, tissue, "_genetic_correlation_calibration_ld_corrected_deciles_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.0, units="in")



#####################
# Figure 17: Coverage of the posterior predictive intervals of the observed held-out correlation at each nominal level, all genes, single fold
#####################
fold_summary_df = compute_coverage_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], coverage_levels)
fold_plot = make_calibration_plot(fold_summary_df, "Nominal level", "Empirical coverage") + scale_x_continuous(breaks=coverage_levels / 100)
output_file = paste0(viz_dir, tissue, "_predictive_interval_coverage_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 18: As Figure 17, one panel per fold
#####################
by_fold_summary_df = compute_coverage_summary_df_by_fold(gene_fold_df, coverage_levels, folds)
by_fold_plot = make_calibration_plot(by_fold_summary_df, "Nominal level", "Empirical coverage") + scale_x_continuous(breaks=coverage_levels / 100)
output_file = paste0(viz_dir, tissue, "_predictive_interval_coverage_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=3.0, units="in")



#####################
# Figure 19: Coverage of the posterior predictive intervals of the observed held-out correlation (Figure 17) within deciles of the posterior
# predictive expected correlation, at each nominal level, all genes, single fold. Shows whether the shortfall is spread over genes or sits in the
# weakly or strongly predicted ones
#####################
decile_coverage_df = compute_coverage_by_decile_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], "expected_r", coverage_levels, n_bins)
decile_coverage_plot = make_coverage_by_decile_plot(decile_coverage_df, "Expected correlation (decile mean)", "Empirical coverage")
output_file = paste0(viz_dir, tissue, "_predictive_interval_coverage_by_expected_r_decile_fold_", single_fold, ".pdf")
ggsave(output_file, decile_coverage_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 20: Mean observed held-out correlation of the top N genes under each ranking (top_n_rankings), all genes, single fold. Made with a linear
# x axis (<tissue>_top_n_genes_mean_observed_r_fold_<k>.pdf) and a log x axis (..._log_x.pdf), which shows the top few hundred genes
#####################
fold_summary_df = compute_top_n_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], top_n_rankings, top_n_min_genes)
for (log_x in c(FALSE, TRUE)) {
	fold_plot = make_top_n_plot(fold_summary_df, "Top N genes", "Mean observed held-out correlation", log_x=log_x, legend_bottom=TRUE)
	output_file = paste0(viz_dir, tissue, "_top_n_genes_mean_observed_r_fold_", single_fold, ifelse(log_x, "_log_x", ""), ".pdf")
	ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")
}



#####################
# Figure 21: As Figure 20, one panel per fold (linear and log x axis)
#####################
by_fold_summary_df = compute_top_n_summary_df_by_fold(gene_fold_df, top_n_rankings, top_n_min_genes, folds)
for (log_x in c(FALSE, TRUE)) {
	by_fold_plot = make_top_n_plot(by_fold_summary_df, "Top N genes", "Mean observed\nheld-out correlation", log_x=log_x)
	output_file = paste0(viz_dir, tissue, "_top_n_genes_mean_observed_r_by_fold", ifelse(log_x, "_log_x", ""), ".pdf")
	ggsave(output_file, by_fold_plot, width=7.2, height=4.0, units="in")
}



#####################
# Figure 22: 1 - observed FSR (fraction of genes with observed r >= 0) of the top N genes under each ranking (top_n_rankings), all genes, single fold
# The x axis is linear
#####################
fold_summary_df = compute_top_n_summary_df(gene_fold_df[gene_fold_df$fold == single_fold, ], top_n_rankings, top_n_min_genes, value_col="true_sign")
fold_plot = make_top_n_plot(fold_summary_df, "Top N genes", "1 - observed FSR of the top N genes", log_x=FALSE, legend_bottom=TRUE)
output_file = paste0(viz_dir, tissue, "_top_n_genes_one_minus_observed_fsr_fold_", single_fold, ".pdf")
ggsave(output_file, fold_plot, width=7.2, height=4.5, units="in")



#####################
# Figure 23: As Figure 22, one panel per fold
#####################
by_fold_summary_df = compute_top_n_summary_df_by_fold(gene_fold_df, top_n_rankings, top_n_min_genes, folds, value_col="true_sign")
by_fold_plot = make_top_n_plot(by_fold_summary_df, "Top N genes", "1 - observed FSR\nof the top N genes", log_x=FALSE)
output_file = paste0(viz_dir, tissue, "_top_n_genes_one_minus_observed_fsr_by_fold.pdf")
ggsave(output_file, by_fold_plot, width=7.2, height=4.0, units="in")