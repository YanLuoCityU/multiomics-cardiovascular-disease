#!/bin/bash

# Ref:
# - https://github.com/pjgreer/ukb-rap-tools/tree/main/prs-calc
# - https://2cjenn.github.io/PRS_Pipeline/#Introduction

# This script collects the output from 01-pull-snps-imp37.sh and combines it into a single .bgen file, indexes it, 
# then creates plink pgen format cohort file filtered on MAF>0.01. It then calulates the PRS score using the plink 
# --score command and twites out a .raw file with the genotype dosages.

# Inputs:
# Note that you can adjust the output directory by setting the data_file_dir variable
# - /prs/*/textfiles/prsfile.txt - made previously
# - /prs/*/textfiles/scorefile.txt - made previously
# - .bgen, for each chromosome from step 01
# - .sample file from ukb bulk data folder 

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
  prsfile="prsfile.txt"
  scorefile="scorefile.txt"

  # Set path for outputs
  prsout="${outcome}"

  # Please modify the path "/mnt/project/prs/cad/results/chr_${i}.bgen"
  calc_prs='
    # build list of bgen files
    cmd=""; for i in {1..22}; do cmd=$cmd"/mnt/project/prs/'"${outcome}"'/results/chr_${i}.bgen "; done
    # Combine the .bgen files for each chromosome into one
    cat-bgen -g $cmd -og initial_chr.bgen -clobber
    # Write index file .bgen.bgi
    bgenix -g initial_chr.bgen -index -clobber
    # convert to plinkformat
    plink2 --bgen initial_chr.bgen ref-first --sample ukb22828_c1_b0_v3.sample --freq --maf 0.01 --make-pgen --sort-vars --out ukb-select-all
    # calculate prs score
    plink2 --pfile ukb-select-all --score '"${scorefile}"' no-mean-imputation list-variants cols=maybefid,nallele,denom,dosagesum,scoreavgs,scoresums --out '"${prsout}"'
  '
  dx run swiss-army-knife \
    -iin="${imp_file_dir}/${data_field}_c1_b0_v3.sample" \
    -iin="${textfiles_dir}/${scorefile}" \
    -iin="${textfiles_dir}/${prsfile}" \
    -icmd="${calc_prs}" --tag="CalcPRS_${outcome}" --instance-type "mem2_ssd2_v2_x16" \
    --destination="${project}:${results_dir}" --brief --yes

done


# Stroke
## 1.Generate initial_chr.bgen and initial_chr.bgen.bgi
## 2.Exclude duplicated SNPs (Filtered.csv)
## 3.Generate single_allelic.bgen and single_allelic.bgen.bgi with Filtered.csv
## 4.Calculate PRS score using single_allelic.bgen
OUTCOMES=("stroke")

for outcome in "${OUTCOMES[@]}"; do
  # Set paths for required data
  results_dir="/prs/${outcome}/results/"
  textfiles_dir="/prs/${outcome}/textfiles/"
  prsfile="prsfile.txt"
  scorefile="scorefile.txt"

  # Set path for outputs
  prsout="${outcome}"

  # Please modify the path "/mnt/project/prs/cad/results/chr_${i}.bgen"
  calc_prs='
    # build list of bgen files
    cmd=""; for i in {1..22}; do cmd=$cmd"/mnt/project/prs/'"${outcome}"'/results/chr_${i}.bgen "; done
    # Combine the .bgen files for each chromosome into one
    cat-bgen -g $cmd -og initial_chr.bgen -clobber
    # Write index file .bgen.bgi
    bgenix -g initial_chr.bgen -index -clobber
    # convert to plinkformat
    plink2 --bgen initial_chr.bgen ref-first --sample ukb22828_c1_b0_v3.sample --freq --maf 0.01 --make-pgen --sort-vars --out ukb-select-all
  '
  dx run swiss-army-knife \
    -iin="${imp_file_dir}/${data_field}_c1_b0_v3.sample" \
    -iin="${textfiles_dir}/${scorefile}" \
    -iin="${textfiles_dir}/${prsfile}" \
    -icmd="${calc_prs}" --tag="CalcPRS_${outcome}" --instance-type "mem2_ssd2_v2_x16" \
    --destination="${project}:${results_dir}" --brief --yes

done

# Download initial_chr.bgen.bgi from the UKB RAP
dx download "file-id" -o /your path/multiomics-cardiovascular-disease/data/processed/prs/stroke/results
dx download "file-id" -o /your path/multiomics-cardiovascular-disease/data/processed/prs/stroke/results

# Exclude duplicated SNPs
# /your path/multiomics-cardiovascular-disease/data_loader/DataPreparation/OmicsGenerator/PRSProcessing/prs_data_preparation.ipynb ("Stroke" Section)
dx upload "/your path/multiomics-cardiovascular-disease/data/processed/prs/stroke/results/Filtered.csv" --path prs/stroke/results/ --brief

# Generate single_allelic.bgen and single_allelic.bgen.bgi with Filtered.csv
bgenix -g initial_chr.bgen -table Filtered > single_allelic.bgen
bgenix -g single_allelic.bgen -index

dx upload "/your path/multiomics-cardiovascular-disease/data/processed/prs/stroke/results/single_allelic.bgen" --path prs/stroke/results/ --brief
dx upload "/your path/multiomics-cardiovascular-disease/data/processed/prs/stroke/results/single_allelic.bgen.bgi" --path prs/stroke/results/ --brief

# Calculate PRS score using single_allelic.bgen
for outcome in "${OUTCOMES[@]}"; do
  # Set paths for required data
  results_dir="/prs/${outcome}/results/"
  textfiles_dir="/prs/${outcome}/textfiles/"
  prsfile="prsfile.txt"
  scorefile="scorefile.txt"

  # Set path for outputs
  prsout="${outcome}"

  # Please modify the path "/mnt/project/prs/cad/results/chr_${i}.bgen"
  calc_prs='
    # Convert filtered bgen to plink format
    plink2 --bgen single_allelic.bgen ref-first --sample ukb22828_c1_b0_v3.sample --make-pgen --out ukb-select-all-filtered

    # Calculate PRS score using filtered data
    plink2 --pfile ukb-select-all-filtered --score '"${scorefile}"' no-mean-imputation list-variants cols=maybefid,nallele,denom,dosagesum,scoreavgs,scoresums --out '"${prsout}"'
  '
  dx run swiss-army-knife \
    -iin="${imp_file_dir}/${data_field}_c1_b0_v3.sample" \
    -iin="${project}${results_dir}single_allelic.bgen" \
    -iin="${project}${results_dir}single_allelic.bgen.bgi" \
    -iin="${textfiles_dir}/${scorefile}" \
    -iin="${textfiles_dir}/${prsfile}" \
    -icmd="${calc_prs}" --tag="CalcPRS_${outcome}" --instance-type "mem2_ssd2_v2_x16" \
    --destination="${project}:${results_dir}" --brief --yes

done