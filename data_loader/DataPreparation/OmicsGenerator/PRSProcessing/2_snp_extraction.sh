#!/bin/bash

# Ref:
# - https://github.com/pjgreer/ukb-rap-tools/tree/main/prs-calc
# - https://2cjenn.github.io/PRS_Pipeline/#Introduction

# This script searches for SNPS by rsid and chom:position from given input files rsidlist.txt and chrposlist.txt
# over each od the autosomal files in the imputed UKB dataset and writes out a smaller subset file in .bgen 
# format for all found SNPs.

# Inputs:
# Note that you can adjust the output directory by setting the scorefiles_dir variable
# - /prs/*/textfiles/rsidlist.txt - made previously
# - /prs/*/textfiles/chrposlist.txt - made previously
# - /prs/*/textfiles/prsfile.txt - made previously
# - .bgen, .sample, and .bgi (index) for each chromosome

# Outputs (for each chromosome):
# - /prs/*/scorefiles/chr_*.bgen  

# Steps:
# for each chromosome 1-22:
# 	- filter by RSID and CHR:Start:Stop
#	- write out bgen file with only those markers that match


##################################### dx login #####################################

# Set this to the genomics directory that you want (should contain PLINK formatted files)
imp_file_dir="/Bulk/Imputation/UKB imputation from genotype"
# Set this to the imputed data field for your release
data_field="ukb22828"

# All outcomes
OUTCOMES=("cad" "stroke" "hf" "af" "va" "pad" "aaa" "vte")

for outcome in "${OUTCOMES[@]}"; do
    # Set paths for required data
    results_dir="/prs/${outcome}/results/"
    textfiles_dir="/prs/${outcome}/textfiles/"
    rsidlist="rsidlist.txt" 
    chrposlist="chrposlist.txt"
    prsfile="prsfile.txt"

    # Loop over each autosomal chromosome 1-22 to extract required SNPs
    for i in {1..22}; do
        run_snps="bgenix -g ${data_field}_c${i}_b0_v3.bgen -incl-rsids ${rsidlist} -incl-range ${chrposlist} > chr_${i}.bgen"

        dx run swiss-army-knife -iin="${imp_file_dir}/${data_field}_c${i}_b0_v3.bgen" \
         -iin="${imp_file_dir}/${data_field}_c${i}_b0_v3.sample" \
         -iin="${imp_file_dir}/${data_field}_c${i}_b0_v3.bgen.bgi" \
         -iin="${textfiles_dir}/${rsidlist}" -iin="${textfiles_dir}/${chrposlist}" \
         -icmd="${run_snps}" --tag="ExtractSNPs_${outcome}" --instance-type "mem2_ssd2_v2_x16" \
         --destination="${project}:${results_dir}" --brief --yes
    done
done

##################################### dx logout #####################################