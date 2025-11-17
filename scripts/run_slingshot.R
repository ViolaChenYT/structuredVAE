#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(scater)
  library(slingshot)
  library(ggplot2)
  library(scuttle)
})

# -------- CLI Options --------
opt <- OptionParser() |>
  add_option("--h5ad", type="character", help="Input .h5ad file") |>
  add_option("--outdir", type="character", default="sl_out",
             help="Output directory") |>
  add_option("--use_obsm", type="character", default="X_umap",
             help="Reduced dimension in .obsm (e.g., X_umap, X_pca)") |>
  add_option("--start_cluster", type="character", default=NULL,
             help="Optional start cluster for trajectory rooting") |>
  add_option("--default_k", type="integer", default=5,
             help="Fallback k for k-means if no lineage info") |>
  add_option("--hist_bins", type="integer", default=40) |>
  add_option("--hist_binwidth", type="double",default=NA_real_,help="if set, overrides --hist_bins") |>
  parse_args()

stopifnot(!is.null(opt$h5ad))
dir.create(opt$outdir, showWarnings=FALSE, recursive=TRUE)

message("🔹 Loading ", opt$h5ad)
sce <- readH5AD(opt$h5ad)

# -------- Choose reduced dimension --------
use_dim <- opt$use_obsm
if (!(use_dim %in% names(reducedDims(sce)))) {
  # message("⚠️  ", use_dim, " not found in obsm; computing PCA(2D)")
  alt <- c("X_scVI", "X_pca", "X_umap")
  found <- alt[alt %in% names(reducedDims(sce))]
  if (length(found)) {
    use_dim <- found[1]
  } else {
    # no embedding present -> build PCA(2D) from assays
    a <- names(assays(sce))
    if (!"logcounts" %in% a) {
      if ("counts" %in% a) {
        sce <- logNormCounts(sce)  # creates 'logcounts'
      } else if ("X" %in% a) {
        assay(sce, "counts") <- assay(sce, "X")
        sce <- logNormCounts(sce)
      } else {
        stop("No reducedDims and no assays among {'logcounts','counts','X'}; cannot compute PCA.")
      }
    }
    sce <- runPCA(sce, ncomponents=2, exprs_values="logcounts")
    reducedDims(sce)[["X_pca_2d"]] <- reducedDim(sce, "PCA")[,1:2, drop=FALSE]
    use_dim <- "X_pca_2d"
    message("using 2D PCA")
  }
}

# -------- Determine number of clusters --------
set.seed(1)
rd <- reducedDims(sce)[[use_dim]]
d <- ncol(rd)
if ("lineage" %in% colnames(colData(sce))) {
  lineage_vals <- unique(na.omit(colData(sce)$lineage))
  k <- length(lineage_vals)
  message("🔸 Using k = ", k, " (from unique lineage count)")
} else {
  k <- opt$default_k
  message("🔸 Using default k = ", k)
}
k <- max(2, min(k, nrow(rd)))  # guardrails

# -------- Cluster cells (unsupervised) --------
clus <- as.factor(kmeans(rd, centers=k, nstart=10)$cluster)
if (!is.null(opt$start_cluster) && !(opt$start_cluster %in% levels(clus))) {
  warning("start_cluster not found among clusters; ignoring.")
  opt$start_cluster <- NULL
}

# -------- Run Slingshot --------
message("🚀 Running Slingshot...")
sce <- slingshot(
  sce,
  clusterLabels = clus,
  reducedDim    = use_dim,
  start.clus    = opt$start_cluster
)

# -------- Save results --------
pt <- slingPseudotime(sce)
pt_df <- as.data.frame(pt)
pt_df$cell <- colnames(sce)
write.csv(pt_df, file.path(opt$outdir, "pseudotime.csv"), row.names=FALSE)
cl_df <- data.frame(cell=colnames(sce), cluster=clus)
write.csv(cl_df, file.path(opt$outdir, "clusters.csv"), row.names=FALSE)
# ---- plotting ----
save_plot <- function(plot, file, width=11, height=7, dpi=150) {
  ggplot2::ggsave(filename=file, plot=plot, width=width, height=height, dpi=dpi)
}

rd_df <- as.data.frame(rd)
d <- ncol(rd_df)

# lineage labels (optional)
use_lineage_labels <- "lineage" %in% colnames(colData(sce))
if (use_lineage_labels) {
  rd_df$lineage <- as.character(colData(sce)$lineage)
  lineage_unique <- unique(rd_df$lineage)
  sort_keys <- sapply(lineage_unique, function(x) {
    first_part <- strsplit(as.character(x), "/", fixed = TRUE)[[1]][1]
    nchar(first_part)
  })
  sorted_lineages <- lineage_unique[order(sort_keys)]
  
  rd_df$lineage_ordered <- factor(rd_df$lineage, levels = sorted_lineages)
}

# scatter if ≥2D (clusters + optional lineage)
if (d >= 2) {
  names(rd_df)[1:2] <- c("d1","d2")
  rd_df$cluster <- clus

  p_clusters <- ggplot(rd_df, aes(x = d1, y = d2, color = cluster)) +
    geom_point(size=1, alpha=0.8) + theme_bw() +
    ggtitle(sprintf("Slingshot (unsup, k=%d) on %s", k, use_dim))
  save_plot(p_clusters, file.path(opt$outdir, "slingshot_scatter.png"))

  if (use_lineage_labels) {
    p_lineage <- ggplot(rd_df, aes(x = d1, y = d2, color = lineage_ordered)) +
      geom_point(size=1, alpha=0.8) + theme_bw() +
      labs(color="Lineage",
           title = sprintf("Colored by Lineage on %s", use_dim))
    save_plot(p_lineage, file.path(opt$outdir, "slingshot_scatter_lineage.png"))
  }
}

# 1D histogram (stacked) — lineage if available, else clusters
xcol <- names(rd_df)[1]
rd_df$cluster <- clus  # always available

if (use_lineage_labels) {
  p_hist <- ggplot(rd_df, aes(x = .data[[xcol]], fill = lineage_ordered)) +
    { if (!is.na(opt$hist_binwidth))
        geom_histogram(aes(y = after_stat(density)), binwidth = opt$hist_binwidth, position="identity", alpha=0.5)
      else
        geom_histogram(aes(y = after_stat(density)), bins = opt$hist_bins, position="identity", alpha=0.5) } +
    theme_bw() + xlab(xcol) + ylab("Frequency") +
    labs(fill = "Lineage",
         title = "Frequency Distributions by Label (normalized)")
} else {
  p_hist <- ggplot(rd_df, aes(x = .data[[xcol]], fill = cluster)) +
    { if (!is.na(opt$hist_binwidth))
        geom_histogram(aes(y = after_stat(density)), binwidth = opt$hist_binwidth, position="identity", alpha=0.5)
      else
        geom_histogram(aes(y = after_stat(density)), bins = opt$hist_bins, position="identity", alpha=0.5) } +
    theme_bw() + xlab(xcol) + ylab("Frequency") +
    ggtitle("Frequency Distributions by Cluster (normalized)")
}

save_plot(p_hist, file.path(opt$outdir, "slingshot_hist.png"))

saveRDS(sce, file.path(opt$outdir, "sce_slingshot.rds"))
message("✅ Done! Results written to ", opt$outdir)
