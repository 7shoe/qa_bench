#!/bin/bash

# Activate the conda environment
source activate R

# Run the R script
Rscript sslasso.r ./tmp/X.txt ./tmp/y.txt

# Deactivate the conda environment
conda deactivate