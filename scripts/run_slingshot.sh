#!/usr/bin/bash

zcat /n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_small.json.gz \
  | jq -r 'keys[]' \
  | xargs -n1 -P4 -I{} Rscript run_slingshot.R \
      --h5ad /n/fs/ragr-data/users/viola/structuredVAE/data/scvi_path_{}/trained.h5ad \
      --outdir /n/fs/ragr-data/users/viola/structuredVAE/results/slingshot/{}/ \
      --use_obsm X_umap

# zcat /n/fs/ragr-data/users/viola/mouse_dev/scripts/new_c_elegans/paths_dict_tree_small.json.gz \
#   | jq -r 'keys[]' \
#   | while read path; do
#       outdir="/n/fs/ragr-data/users/viola/structuredVAE/results/slingshot/${path}/"
#       if [ -f "${outdir}/pseudotime.csv" ]; then
#         echo "Skipping ${path} (already exists)"
#       else
#         echo "Processing ${path}"
#         Rscript run_slingshot.R \
#           --h5ad /n/fs/ragr-data/users/viola/structuredVAE/data/scvi_path_${path}/trained.h5ad \
#           --outdir ${outdir} \
#           --use_obsm X_umap
#       fi
#     done