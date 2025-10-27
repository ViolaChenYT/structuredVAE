# Snakemake workflow for parallel scVAE training
# This workflow parallelizes the training of scVAE models for all lineage paths
# Based on the functionality in scvae_run.py

# Import configuration
configfile: "config.yaml"

import json
import gzip
import shutil
with gzip.open(config["paths_dict_file"], "rt") as f:
    PATH_NAMES = list(json.load(f).keys())

rule all:
    input:
        expand("results/scvae_{path_name}/latent_values-z.tsv.gz", path_name=PATH_NAMES)

# This rule now handles the unpredictable script output.
rule run_scvae_train:
    input:
        loom="data/{path_name}.loom",
        path_dict=config["paths_dict_file"]
    output:
        # These are the FINAL, PREDICTABLE paths Snakemake needs to know about.
        y_file="results/scvae_{path_name}/latent_values-y.tsv.gz",
        z_file="results/scvae_{path_name}/latent_values-z.tsv.gz"
    params:
        # We'll tell the script to dump its messy output into this temp directory.
        temp_dir=directory("results/scvae_{path_name}/temp_output")
    log:
        "logs/{path_name}_scvae_training.log"
    conda:
        "scvae.yml"
    run:
        # A 'run' block lets us use Python to add more logic.
        from snakemake.shell import shell
        import glob
        import shutil

        # Step 1: Run your script.
        # Direct its output to the temporary directory.
        shell(
            "python scripts/train_single_path.py "
            "--loom {input.loom} "
            "--path_dict {input.path_dict} "
            "--path_name {wildcards.path_name} "
            "--output_folder {params.temp_dir} "
            "> {log} 2>&1"
        )

        # Step 2: Find the exact paths of the files the script created.
        # The '**' searches recursively through all the nested subdirectories
        # created by your script inside the temp folder.
        z_files_found = glob.glob(f"{params.temp_dir}/**/latent_values-z.tsv.gz", recursive=True)
        y_files_found = glob.glob(f"{params.temp_dir}/**/latent_values-y.tsv.gz", recursive=True)

        # It's good practice to check that the files were actually found.
        if not z_files_found:
            raise FileNotFoundError(f"Could not find 'latent_values-z.tsv.gz' in {params.temp_dir}")
        if not y_files_found:
            raise FileNotFoundError(f"Could not find 'latent_values-y.tsv.gz' in {params.temp_dir}")

        # Step 3: Move the files from their unpredictable location to the
        # final, permanent path that Snakemake expects from the 'output' block.
        shutil.move(z_files_found[0], output.z_file)
        shutil.move(y_files_found[0], output.y_file)

        # Step 4 (Optional but recommended): Clean up the temporary directory.
        shutil.rmtree(params.temp_dir)


# # Checkpoint to discover path names from the path dictionary
# checkpoint discover_paths:
#     output:
#         "data/path_names.txt"
#     run:
#         import json
#         import gzip
        
#         # Load path dictionary to get all path names
#         
        
#         path_names = list(path_dict.keys())
        
#         # Write path names to file
#         with open(output[0], 'w') as f:
#             for path_name in path_names:
#                 f.write(f"{path_name}\n")
        
#         print(f"Discovered {len(path_names)} paths: {path_names}")

# # Main rule - create completion marker
# rule all:
#     input:
#         "results/training_complete.txt"

# # Rule to create completion marker
# rule create_completion_marker:
#     input:
#         "data/path_names.txt"
#     output:
#         "results/training_complete.txt"
#     run:
#         # Read path names from file
#         with open(input[0], 'r') as f:
#             path_names = [line.strip() for line in f if line.strip()]
        
#         # Create completion markers for all paths
#         completion_markers = [f"data/scvae_path_{path_name}/{path_name}/scvae_training_complete.txt" for path_name in path_names]
        
#         # Check if all completion markers exist
#         import os
#         missing_markers = [marker for marker in completion_markers if not os.path.exists(marker)]
        
#         if missing_markers:
#             raise Exception(f"Missing completion markers: {missing_markers}")
        
#         # Create summary
#         with open(output[0], 'w') as f:
#             f.write(f"scVAE training completed for all {len(path_names)} paths\n")
#             f.write("Completed paths:\n")
#             for path_name in path_names:
#                 f.write(f"  - {path_name}\n")

# # Rule to create necessary directories
# rule create_directories:
#     output:
#         directory("results"),
#         directory("data"),
#         directory("logs")
#     shell:
#         "mkdir -p results data logs"

# # Rule to train scVAE for a single path
# rule train_scvae_single_path:
#     input:
#         # Input data files
#         adata_file = config["adata_file"],
#         paths_dict = config["paths_dict_file"],
#         node_abbrev = config["node_abbrev_file"],
#         fuzzy_mapping = config["fuzzy_mapping_file"]
#     output:
#         # scVAE training outputs
#         loom_file = "data/{path_name}.loom",
#         completion_marker = "data/scvae_path_{path_name}/{path_name}/scvae_training_complete.txt"
#     params:
#         path_name = "{path_name}",
#         base_path = config["base_path"]
#     log:
#         "logs/{path_name}_scvae_training.log"
#     threads: config["scvae_training"]["threads"]
#     resources:
#         mem_mb = config["scvae_training"]["memory_mb"],
#         scvae_jobs = 1
#     shell:
#         """
#         # Activate scvae conda environment and run the training script
#         conda activate scvae && python scripts/train_single_path.py \
#             --path_name {params.path_name} \
#             --base_path {params.base_path} \
#             --paths_dict {input.paths_dict} \
#             --node_abbrev {input.node_abbrev} \
#             --fuzzy_mapping {input.fuzzy_mapping} \
#             --adata_file {input.adata_file} \
#             > {log} 2>&1
#         """

# # Rule to generate summary report
# rule generate_summary:
#     input:
#         "results/training_complete.txt"
#     output:
#         "results/summary_report.txt"
#     shell:
#         """
#         echo "Parallel scVAE Training Summary" > {output}
#         echo "=================================" >> {output}
#         echo "" >> {output}
#         echo "Workflow completed successfully!" >> {output}
#         echo "Trained scVAE models for all available paths." >> {output}
#         echo "" >> {output}
#         echo "Results are stored in:" >> {output}
#         echo "  - data/scvae_path_*/" >> {output}
#         echo "  - data/*.loom (filtered data files)" >> {output}
#         echo "" >> {output}
#         echo "Logs are available in:" >> {output}
#         echo "  - logs/*_scvae_training.log" >> {output}
#         """

# # COMMENTED OUT: Plotting and analysis rules (to be uncommented later)
# # Rule to run plotting for a single path (depends on scVAE training completion)
# rule generate_plots:
#     input:
#         # scVAE training outputs (completion marker ensures training is done)
#         completion_marker = "data/scvae_path_{path_name}/{path_name}/scvae_training_complete.txt",
#         # Original loom file (input to plotting)
#         loom_file = "data/{path_name}.loom"
#     output:
#         plot1 = "results/plots/{path_name}_1dlatent_frequency_distribution.png",
#         plot2 = "results/plots/{path_name}_1dlatent_frequency_distribution_by_lineage.png"
#     params:
#         input_path = "data/scvae_path_{path_name}/{path_name}/no_split/no_preprocessing/GMVAE/gaussian_mixture-c_6/zero_inflated_negative_binomial-l_1-h_64_64-mc_1-iw_1-bn-wu_200/e_500-mc_1-iw_1/full/",
#         loom_file = "data/{path_name}.loom"
#     log:
#         "logs/{path_name}_scvae_plots.log"
#     resources:
#         mem_mb = config["plotting"]["memory_mb"],
#         plotting_jobs = 1
#     shell:
#         """
#         # Activate existing scvae conda environment and run the plotting script
#         conda activate scvae && python scripts/scvae-plots.py \
#             --input_path {params.input_path} \
#             --loom_file {params.loom_file} \
#             > {log} 2>&1
#         
#         # Move the generated plots to the results directory
#         mv {params.input_path}1dlatent_frequency_distribution.png {output.plot1}
#         mv {params.input_path}1dlatent_frequency_distribution_by_lineage.png {output.plot2}
#         """