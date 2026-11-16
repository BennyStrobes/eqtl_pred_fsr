args = commandArgs(trailingOnly=TRUE)
library(cowplot)
library(ggplot2)
options(warn=1)

# EUR held-out against African-ancestry (AFR) evaluation of the fits of one fold (README.md, section 8, plot 10). Driver step 6 runs it for one model at
# a time; given several, it draws them side by side (one panel or point group per model)
#   args[1]: EUR held-out evaluation directory (driver step 3); args[2]: AFR evaluation directory (driver step 5); args[3]: output directory
#   args[4]: fold of the fits; args[5]: suffix of the evaluation files (_split_samples, or none); args[6]: output name (the pdf and table prefix)
#   args[7...]: tissue and model label of each fit, e.g. Whole_Blood_grid_gibbs (evaluation files <it>_fold_<k>_held_out_evaluation<suffix>.txt and
#   <it>_fold_<k>_afr_evaluation<suffix>.txt)
# EUR: the fit's held-out fold (EUR expression, .v8.). AFR: every African-ancestry individual of the tissue (.v8_afr.), same fit, genotypes
# standardized with the EUR training mean and standard deviation. Per model, only genes evaluated in both. Both evaluations share the training
# individuals, so the heritable genes (training cis-h2 LRT p < heritable_pvalue_threshold) are the same genes in both.
# Outputs <output name>_*.pdf and two tables (_summary.txt, _portability.txt)

figure_theme <- function() {
	return(theme(plot.title = element_text(face="plain",size=11), text = element_text(size=11),axis.text=element_text(size=11), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(), axis.line = element_line(colour = "black"), legend.text = element_text(size=11), legend.title = element_text(size=11)))
}


model_display_name <- function(run_name) {
	# Display name of the model of a run (tissue and model label): the known model label the run name ends with (the longest, if several)
	display_names = c("_spike_slab_gibbs"="Spike-slab, Gibbs", "_spike_slab_gwfm"="Spike-slab, GWFM", "_grid_gibbs"="Grid, Gibbs", "_grid_gwfm"="Grid, GWFM",
		"_grid_classes_gibbs"="Grid + classes, Gibbs", "_grid_classes_gwfm"="Grid + classes, GWFM")
	matches = names(display_names)[endsWith(run_name, names(display_names))]
	if (length(matches) == 0) {
		return(run_name)
	}
	return(unname(display_names[matches[which.max(nchar(matches))]]))
}


compute_decile_summary_df <- function(df, x_col, y_col, n_bins) {
	# Same as visualize_cross_validation_results.R: n_bins equal-count bins of x_col; per bin, mean x and mean y_col with a 95% CI
	df = df[!is.na(df[[x_col]]) & !is.na(df[[y_col]]), ]
	bin_id = ceiling(rank(df[[x_col]], ties.method="first") / nrow(df) * n_bins)
	rows = list()
	for (bin_iter in 1:n_bins) {
		bin_df = df[bin_id == bin_iter, ]
		mean_y = mean(bin_df[[y_col]])
		se_y = sd(bin_df[[y_col]]) / sqrt(nrow(bin_df))
		rows[[length(rows) + 1]] = data.frame(bin=bin_iter, n_genes=nrow(bin_df), mean_x=mean(bin_df[[x_col]]), mean_y=mean_y, mean_y_lb=mean_y - 1.96*se_y, mean_y_ub=mean_y + 1.96*se_y)
	}
	return(do.call(rbind, rows))
}


compute_coverage_summary_df <- function(df, coverage_levels) {
	# Same as visualize_cross_validation_results.R: per nominal level, the fraction of genes whose observed r lies in its posterior predictive interval
	df = df[!is.na(df$observed_r), ]
	rows = list()
	for (level in coverage_levels) {
		lb = df[[paste0("expected_r_lb_", level)]]
		ub = df[[paste0("expected_r_ub_", level)]]
		keep = !is.na(lb) & !is.na(ub)
		covered = (df$observed_r[keep] >= lb[keep]) & (df$observed_r[keep] <= ub[keep])
		coverage = mean(covered)
		coverage_se = sqrt(coverage*(1.0 - coverage) / sum(keep))
		rows[[length(rows) + 1]] = data.frame(n_genes=sum(keep), mean_x=level / 100, mean_y=coverage, mean_y_lb=coverage - 1.96*coverage_se, mean_y_ub=coverage + 1.96*coverage_se)
	}
	return(do.call(rbind, rows))
}


summarize_by_model_and_ancestry <- function(df, summary_function) {
	# summary_function(subset of df) within each model and ancestry, stacked, with model and ancestry columns (factors in the levels of df)
	rows = list()
	for (model in levels(df$model)) {
		for (ancestry in levels(df$ancestry)) {
			sub_df = df[df$model == model & df$ancestry == ancestry, ]
			if (nrow(sub_df) == 0) {
				next
			}
			summary_df = summary_function(sub_df)
			summary_df$model = model
			summary_df$ancestry = ancestry
			rows[[length(rows) + 1]] = summary_df
		}
	}
	summary_df = do.call(rbind, rows)
	summary_df$model = factor(summary_df$model, levels=levels(df$model))
	summary_df$ancestry = factor(summary_df$ancestry, levels=levels(df$ancestry))
	return(summary_df)
}


compute_mean_r_summary_df <- function(df) {
	# Mean observed r and mean expected r of the genes (95% CI across genes)
	df = df[!is.na(df$observed_r) & !is.na(df$expected_r), ]
	rows = list()
	for (quantity in c("Observed", "Expected")) {
		values = if (quantity == "Observed") df$observed_r else df$expected_r
		mean_value = mean(values)
		se_value = sd(values) / sqrt(length(values))
		rows[[length(rows) + 1]] = data.frame(quantity=quantity, n_genes=length(values), mean=mean_value, lb=mean_value - 1.96*se_value, ub=mean_value + 1.96*se_value)
	}
	return(do.call(rbind, rows))
}


compute_mean_fsr_summary_df <- function(df) {
	# Observed: fraction of genes with observed r < 0; expected: mean expression FSR of the genes (95% CI across genes)
	df = df[!is.na(df$observed_r) & !is.na(df$expression_FSR), ]
	rows = list()
	for (quantity in c("Observed", "Expected")) {
		values = if (quantity == "Observed") as.numeric(df$observed_r < 0) else df$expression_FSR
		mean_value = mean(values)
		se_value = sd(values) / sqrt(length(values))
		rows[[length(rows) + 1]] = data.frame(quantity=quantity, n_genes=length(values), mean=mean_value, lb=mean_value - 1.96*se_value, ub=mean_value + 1.96*se_value)
	}
	return(do.call(rbind, rows))
}


make_mean_r_plot <- function(summary_df, ylab) {
	# Per model: mean observed (filled) and expected (open) r in each ancestry, with 95% CIs across genes
	summary_df$quantity = factor(summary_df$quantity, levels=c("Observed", "Expected"))
	dodge = position_dodge(width=0.7)
	pp = ggplot(summary_df, aes(x=model, y=mean, color=ancestry, shape=quantity, group=interaction(ancestry, quantity))) +
		geom_hline(yintercept=0, color="#3F3F46", linewidth=0.4) +
		geom_pointrange(aes(ymin=lb, ymax=ub), position=dodge, linewidth=0.6, size=0.45) +
		scale_color_manual(values=ancestry_colors, name=NULL) +
		scale_shape_manual(values=c("Observed"=16, "Expected"=1), name=NULL) +
		figure_theme() +
		theme(axis.text.x=element_text(angle=30, hjust=1), legend.position="right") +
		labs(x=NULL, y=ylab)
	return(pp)
}


make_ancestry_calibration_plot <- function(summary_df, xlab, ylab) {
	# Calibration (mean observed with 95% CI against mean expected), one panel per model, one color per ancestry; dashed line y = x
	pp = ggplot(summary_df, aes(x=mean_x, y=mean_y, color=ancestry)) +
		geom_abline(slope=1, intercept=0, color="#3F3F46", linetype="dashed", linewidth=0.5) +
		geom_line(linewidth=0.6) +
		geom_pointrange(aes(ymin=mean_y_lb, ymax=mean_y_ub), linewidth=0.5, size=0.25) +
		scale_color_manual(values=ancestry_colors, name=NULL) +
		facet_wrap(~model, ncol=facet_ncol) +
		figure_theme() +
		theme(strip.background=element_blank(), strip.text=element_text(size=10), panel.spacing=unit(1.0, "lines"), legend.position="bottom") +
		labs(x=xlab, y=ylab)
	return(pp)
}


make_ancestry_scatter_plot <- function(wide_df, x_col, y_col, xlab, ylab) {
	# Per gene: AFR against EUR, one panel per model, with the line y = x
	pp = ggplot(wide_df, aes(x=.data[[x_col]], y=.data[[y_col]])) +
		geom_abline(slope=1, intercept=0, color="#3F3F46", linetype="dashed", linewidth=0.5) +
		geom_point(color=scatter_color, size=0.5, alpha=0.3, stroke=0) +
		facet_wrap(~model, ncol=facet_ncol) +
		figure_theme() +
		theme(strip.background=element_blank(), strip.text=element_text(size=10), panel.spacing=unit(1.0, "lines")) +
		labs(x=xlab, y=ylab)
	return(pp)
}


make_ancestry_histogram <- function(df, value_col, bin_width, xlab) {
	# Histogram of value_col (bins of bin_width, starting at 0), EUR held-out fold on the top row and AFR on the bottom row (one column per model).
	# Same genes in both rows and the same y axis, so the bar heights are comparable
	df = df[!is.na(df[[value_col]]), ]
	pp = ggplot(df, aes(x=.data[[value_col]], fill=ancestry)) +
		geom_histogram(binwidth=bin_width, boundary=0, color="white", linewidth=0.2) +
		scale_fill_manual(values=ancestry_colors, guide="none") +
		scale_y_continuous(expand=expansion(mult=c(0, 0.05))) +
		facet_grid(ancestry ~ model) +
		figure_theme() +
		theme(strip.background=element_blank(), strip.text=element_text(size=10), panel.spacing=unit(1.0, "lines")) +
		labs(x=xlab, y="Number of genes")
	return(pp)
}


#####################
# Command line args
#####################
eur_evaluation_dir = args[1]
afr_evaluation_dir = args[2]
viz_dir = args[3]
fold = args[4]
evaluation_label = args[5]
if (evaluation_label == "none") {
	evaluation_label = ""
}
output_name = args[6]
run_names = args[7:length(args)]
dir.create(viz_dir, showWarnings=FALSE, recursive=TRUE)
output_stem = paste0(viz_dir, output_name)

n_bins = 10
coverage_levels = c(80, 90, 95, 99)
heritable_pvalue_threshold = 0.05
fsr_call_threshold = 0.05
# Significance of the held-out association (two-sided test of observed r), Bonferroni across the heritable genes of each model and ancestry
replication_alpha = 0.05
ancestry_levels = c("EUR (held-out fold)", "AFR")
ancestry_colors = setNames(c("#1971C2", "#E8590C"), ancestry_levels)
scatter_color = "#495057"


#####################
# Load the EUR held-out and AFR evaluations of every model (genes evaluated in both)
#####################
evaluation_columns = c("gene_id", "n_test", "observed_r", "expected_r", paste0("expected_r_lb_", coverage_levels), paste0("expected_r_ub_", coverage_levels),
	"expected_genetic_r", "expression_FSR", "genetic_FSR", "cis_snp_h2_pvalue_train")
long_list = list()
wide_list = list()
for (run_name in run_names) {
	eur_file = paste0(eur_evaluation_dir, run_name, "_fold_", fold, "_held_out_evaluation", evaluation_label, ".txt")
	afr_file = paste0(afr_evaluation_dir, run_name, "_fold_", fold, "_afr_evaluation", evaluation_label, ".txt")
	if (!file.exists(eur_file) | !file.exists(afr_file)) {
		print(paste0("warning: ", run_name, " left out (missing ", eur_file, " or ", afr_file, ")"))
		next
	}
	eur_df = read.table(eur_file, header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))[, evaluation_columns]
	afr_df = read.table(afr_file, header=TRUE, sep="\t", stringsAsFactors=FALSE, na.strings=c("NA", "nan", "NaN"))[, evaluation_columns]
	shared_genes = intersect(eur_df$gene_id, afr_df$gene_id)
	print(paste0(run_name, ": ", nrow(eur_df), " EUR genes, ", nrow(afr_df), " AFR genes, ", length(shared_genes), " in both; ",
		eur_df$n_test[1], " EUR held-out and ", afr_df$n_test[1], " AFR individuals"))
	eur_df = eur_df[match(shared_genes, eur_df$gene_id), ]
	afr_df = afr_df[match(shared_genes, afr_df$gene_id), ]
	model_name = model_display_name(run_name)
	eur_df$ancestry = ancestry_levels[1]
	afr_df$ancestry = ancestry_levels[2]
	eur_df$model = model_name
	afr_df$model = model_name
	long_list[[length(long_list) + 1]] = rbind(eur_df, afr_df)
	wide_list[[length(wide_list) + 1]] = data.frame(model=model_name, gene_id=shared_genes, eur_observed_r=eur_df$observed_r, afr_observed_r=afr_df$observed_r,
		eur_expected_r=eur_df$expected_r, afr_expected_r=afr_df$expected_r, eur_expected_genetic_r=eur_df$expected_genetic_r, afr_expected_genetic_r=afr_df$expected_genetic_r,
		eur_expression_FSR=eur_df$expression_FSR, afr_expression_FSR=afr_df$expression_FSR, eur_genetic_FSR=eur_df$genetic_FSR, afr_genetic_FSR=afr_df$genetic_FSR)
}
if (length(long_list) == 0) {
	stop("no model has both a EUR held-out and an AFR evaluation")
}
long_df = do.call(rbind, long_list)
model_levels = unique(long_df$model)
long_df$model = factor(long_df$model, levels=model_levels)
long_df$ancestry = factor(long_df$ancestry, levels=ancestry_levels)
long_df$false_sign = as.numeric(long_df$observed_r < 0)
long_df$heritable = !is.na(long_df$cis_snp_h2_pvalue_train) & long_df$cis_snp_h2_pvalue_train < heritable_pvalue_threshold
# Two-sided p-value of the observed r in each ancestry's own test data (t = r sqrt((n - 2) / (1 - r^2)), n - 2 df)
clipped_r = pmin(abs(long_df$observed_r), 1 - 1e-12)
long_df$observed_r_pvalue = 2 * pt(clipped_r * sqrt((long_df$n_test - 2) / (1 - clipped_r^2)), df=long_df$n_test - 2, lower.tail=FALSE)
# Replicating genes: heritable in training and observed r significant after Bonferroni across the heritable genes (with an observed r) of the model and ancestry
long_df$replicating = FALSE
for (model in levels(long_df$model)) {
	for (ancestry in levels(long_df$ancestry)) {
		tested = long_df$model == model & long_df$ancestry == ancestry & long_df$heritable & !is.na(long_df$observed_r_pvalue)
		n_tested = sum(tested)
		if (n_tested == 0) {
			next
		}
		long_df$replicating[tested] = long_df$observed_r_pvalue[tested] < replication_alpha / n_tested
		print(paste0(model, ", ", ancestry, ": ", sum(long_df$replicating[tested]), " of ", n_tested, " heritable genes with observed r significant (Bonferroni p < ",
			signif(replication_alpha / n_tested, 3), ")"))
	}
}
wide_df = do.call(rbind, wide_list)
wide_df$model = factor(wide_df$model, levels=model_levels)
n_models = length(model_levels)
# Panels per row in the per-model figures; a single model gets one wider panel
facet_ncol = min(n_models, 3)
facet_height = if (n_models == 1) 4.5 else 2.4 * ceiling(n_models / 3) + 0.8


#####################
# Figure 1: Mean observed and expected r per model in each ancestry, all genes (a), heritable genes (b), and heritable genes whose observed r is
# significant in that ancestry (c; selected on the observed r, so its observed r is biased up relative to the expected r)
#####################
replicating_label = paste0("(training p < ", heritable_pvalue_threshold, " and\nheld-out r significant)")
all_summary_df = summarize_by_model_and_ancestry(long_df, compute_mean_r_summary_df)
heritable_summary_df = summarize_by_model_and_ancestry(long_df[long_df$heritable, ], compute_mean_r_summary_df)
replicating_summary_df = summarize_by_model_and_ancestry(long_df[long_df$replicating, ], compute_mean_r_summary_df)
all_plot = make_mean_r_plot(all_summary_df, "Mean correlation\n(all genes)")
heritable_plot = make_mean_r_plot(heritable_summary_df, paste0("Mean correlation\n(training p < ", heritable_pvalue_threshold, ")"))
replicating_plot = make_mean_r_plot(replicating_summary_df, paste0("Mean correlation\n", replicating_label))
ggsave(paste0(output_stem, "_mean_r_by_model.pdf"), plot_grid(all_plot, heritable_plot, replicating_plot, ncol=1, labels=c("a", "b", "c")), width=7.2, height=10.5, units="in")
# Figure 1b: Panel (a) alone (all genes)
ggsave(paste0(output_stem, "_mean_r_by_model_all_genes.pdf"), all_plot, width=7.2, height=3.5, units="in")
# Figure 1c: As Figure 1b, for the expression FSR (observed: fraction of genes with observed r < 0)
fsr_summary_df = summarize_by_model_and_ancestry(long_df, compute_mean_fsr_summary_df)
fsr_plot = make_mean_r_plot(fsr_summary_df, "Expression FSR\n(all genes)")
ggsave(paste0(output_stem, "_mean_expression_fsr_by_model_all_genes.pdf"), fsr_plot, width=7.2, height=3.5, units="in")
# Figure 1d: As Figure 1, for the expression FSR: all genes (a), heritable genes (b), and heritable genes whose observed r is significant (c)
heritable_fsr_summary_df = summarize_by_model_and_ancestry(long_df[long_df$heritable, ], compute_mean_fsr_summary_df)
replicating_fsr_summary_df = summarize_by_model_and_ancestry(long_df[long_df$replicating, ], compute_mean_fsr_summary_df)
heritable_fsr_plot = make_mean_r_plot(heritable_fsr_summary_df, paste0("Expression FSR\n(training p < ", heritable_pvalue_threshold, ")"))
replicating_fsr_plot = make_mean_r_plot(replicating_fsr_summary_df, paste0("Expression FSR\n", replicating_label))
ggsave(paste0(output_stem, "_mean_expression_fsr_by_model.pdf"), plot_grid(fsr_plot, heritable_fsr_plot, replicating_fsr_plot, ncol=1, labels=c("a", "b", "c")), width=7.2, height=10.5, units="in")


#####################
# Figure 2: Per-gene observed r, AFR against EUR held-out, one panel per model
#####################
scatter_plot = make_ancestry_scatter_plot(wide_df, "eur_observed_r", "afr_observed_r", "Observed r, EUR held-out fold", "Observed r, AFR")
ggsave(paste0(output_stem, "_observed_r_scatter.pdf"), scatter_plot, width=7.2, height=facet_height, units="in")


#####################
# Figure 3: Per-gene expected r, AFR against EUR held-out, one panel per model (how much of the drop the model predicts, gene by gene)
#####################
scatter_plot = make_ancestry_scatter_plot(wide_df, "eur_expected_r", "afr_expected_r", "Expected r, EUR held-out fold", "Expected r, AFR")
ggsave(paste0(output_stem, "_expected_r_scatter.pdf"), scatter_plot, width=7.2, height=facet_height, units="in")


#####################
# Figures 3b-d: As Figure 3, for the expected correlation with genetic expression, the expression FSR and the genetic FSR
#####################
scatter_plot = make_ancestry_scatter_plot(wide_df, "eur_expected_genetic_r", "afr_expected_genetic_r", "Expected genetic correlation, EUR held-out fold", "Expected genetic correlation, AFR")
ggsave(paste0(output_stem, "_expected_genetic_r_scatter.pdf"), scatter_plot, width=7.2, height=facet_height, units="in")

scatter_plot = make_ancestry_scatter_plot(wide_df, "eur_expression_FSR", "afr_expression_FSR", "Expression FSR, EUR held-out fold", "Expression FSR, AFR")
ggsave(paste0(output_stem, "_expression_fsr_scatter.pdf"), scatter_plot, width=7.2, height=facet_height, units="in")

scatter_plot = make_ancestry_scatter_plot(wide_df, "eur_genetic_FSR", "afr_genetic_FSR", "Genetic FSR, EUR held-out fold", "Genetic FSR, AFR")
ggsave(paste0(output_stem, "_genetic_fsr_scatter.pdf"), scatter_plot, width=7.2, height=facet_height, units="in")


#####################
# Figures 3e-h: Histograms of the expected r, expected genetic correlation, expression FSR and genetic FSR, EUR held-out fold (top) and AFR (bottom)
#####################
histogram_height = 5.0
histogram_plot = make_ancestry_histogram(long_df, "expected_r", 0.02, "Expected r")
ggsave(paste0(output_stem, "_expected_r_histogram.pdf"), histogram_plot, width=7.2, height=histogram_height, units="in")

histogram_plot = make_ancestry_histogram(long_df, "expected_genetic_r", 0.02, "Expected genetic correlation")
ggsave(paste0(output_stem, "_expected_genetic_r_histogram.pdf"), histogram_plot, width=7.2, height=histogram_height, units="in")

histogram_plot = make_ancestry_histogram(long_df, "expression_FSR", 0.01, "Expression FSR")
ggsave(paste0(output_stem, "_expression_fsr_histogram.pdf"), histogram_plot, width=7.2, height=histogram_height, units="in")

histogram_plot = make_ancestry_histogram(long_df, "genetic_FSR", 0.01, "Genetic FSR")
ggsave(paste0(output_stem, "_genetic_fsr_histogram.pdf"), histogram_plot, width=7.2, height=histogram_height, units="in")


#####################
# Figure 4: Correlation calibration (deciles of expected r) in each ancestry, one panel per model
#####################
summary_df = summarize_by_model_and_ancestry(long_df, function(d) compute_decile_summary_df(d, "expected_r", "observed_r", n_bins))
calibration_plot = make_ancestry_calibration_plot(summary_df, "Expected correlation", "Observed correlation")
ggsave(paste0(output_stem, "_correlation_calibration_deciles.pdf"), calibration_plot, width=7.2, height=facet_height + 0.4, units="in")


#####################
# Figure 5: Expression FSR calibration (deciles of expression FSR) in each ancestry, one panel per model
#####################
summary_df = summarize_by_model_and_ancestry(long_df, function(d) compute_decile_summary_df(d, "expression_FSR", "false_sign", n_bins))
calibration_plot = make_ancestry_calibration_plot(summary_df, "Expected expression FSR", "Fraction of genes with observed r < 0")
ggsave(paste0(output_stem, "_expression_fsr_calibration_deciles.pdf"), calibration_plot, width=7.2, height=facet_height + 0.4, units="in")


#####################
# Figure 6: Coverage of the posterior predictive intervals of the observed r in each ancestry, one panel per model
#####################
summary_df = summarize_by_model_and_ancestry(long_df, function(d) compute_coverage_summary_df(d, coverage_levels))
coverage_plot = make_ancestry_calibration_plot(summary_df, "Nominal level", "Empirical coverage") + scale_x_continuous(breaks=coverage_levels / 100)
ggsave(paste0(output_stem, "_predictive_interval_coverage.pdf"), coverage_plot, width=7.2, height=facet_height + 0.4, units="in")


#####################
# Tables: per model and ancestry summary; per model portability (AFR / EUR ratio of mean observed and of mean expected r)
#####################
summary_rows = list()
for (model in model_levels) {
	for (ancestry in ancestry_levels) {
		d = long_df[long_df$model == model & long_df$ancestry == ancestry & !is.na(long_df$observed_r) & !is.na(long_df$expected_r), ]
		h = d[d$heritable, ]
		called = d[!is.na(d$expression_FSR) & d$expression_FSR <= fsr_call_threshold, ]
		covered_90 = (d$observed_r >= d$expected_r_lb_90) & (d$observed_r <= d$expected_r_ub_90)
		summary_rows[[length(summary_rows) + 1]] = data.frame(model=model, ancestry=ancestry, n_genes=nrow(d), n_individuals=d$n_test[1],
			mean_observed_r=mean(d$observed_r), mean_expected_r=mean(d$expected_r),
			n_heritable_genes=nrow(h), mean_observed_r_heritable=mean(h$observed_r), mean_expected_r_heritable=mean(h$expected_r),
			coverage_90=mean(covered_90, na.rm=TRUE), fraction_observed_r_negative=mean(d$false_sign), mean_expression_FSR=mean(d$expression_FSR, na.rm=TRUE),
			n_called_fsr_0.05=nrow(called), observed_fsr_called=mean(called$false_sign), expected_fsr_called=mean(called$expression_FSR))
	}
}
summary_table = do.call(rbind, summary_rows)
write.table(summary_table, paste0(output_stem, "_summary.txt"), sep="\t", quote=FALSE, row.names=FALSE)

portability_rows = list()
for (model in model_levels) {
	eur_row = summary_table[summary_table$model == model & summary_table$ancestry == ancestry_levels[1], ]
	afr_row = summary_table[summary_table$model == model & summary_table$ancestry == ancestry_levels[2], ]
	portability_rows[[length(portability_rows) + 1]] = data.frame(model=model,
		observed_r_ratio=afr_row$mean_observed_r / eur_row$mean_observed_r, expected_r_ratio=afr_row$mean_expected_r / eur_row$mean_expected_r,
		observed_r_ratio_heritable=afr_row$mean_observed_r_heritable / eur_row$mean_observed_r_heritable,
		expected_r_ratio_heritable=afr_row$mean_expected_r_heritable / eur_row$mean_expected_r_heritable)
}
write.table(do.call(rbind, portability_rows), paste0(output_stem, "_portability.txt"), sep="\t", quote=FALSE, row.names=FALSE)
print(summary_table)
