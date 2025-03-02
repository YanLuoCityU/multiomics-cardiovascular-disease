#!bin/bash

# Input and output directories
input_dir="/your path/multiomics-cardiovascular-disease/data/processed/prs"
output_dir="/your path/multiomics-cardiovascular-disease/data/processed/prs"

##################################### CAD #####################################
outcome="cad"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for CAD"

##################################### Stroke #####################################
outcome="stroke"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for Stroke"

##################################### HF #####################################
outcome="hf"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for HF"

##################################### AF #####################################
outcome="af"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for AF"

##################################### VA #####################################
outcome="va"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for VA"

##################################### PAD #####################################
outcome="pad"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for PAD"

##################################### AAA #####################################
outcome="aaa"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for AAA"

##################################### VTE #####################################
outcome="vte"

awk -F'\t' '{ if (NR>1) { print $1 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/rsidlist.txt"

awk -F'\t' '{ if (NR>1) { print sprintf("%02d", $2)":"$3"-"$3 }}' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/chrposlist.txt"

awk '{ print $1, $4, $6 }' "$input_dir/$outcome/textfiles/prsfile.txt" > "$output_dir/$outcome/textfiles/scorefile.txt"

echo "Processed data has been saved into $output_dir/$output_file for VTE"

##################################### Upload to UKB RAP #####################################