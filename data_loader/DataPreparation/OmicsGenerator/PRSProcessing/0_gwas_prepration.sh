#!bin/bash

# Input and output directories
input_dir="/your path/multiomics-cardiovascular-disease/data/gwas_summary_statistics"
output_dir="/your path/multiomics-cardiovascular-disease/data/processed/prs"

# Prepare GWAS summary statistics locally using GRCh37_20241214.Rmd

##################################### CAD #####################################
# Input and output file names
input_file="PMID36576811_cad_GRCh37.txt"
outcome="cad"

awk 'NR == 1 {
    print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight"
}
NR > 1 {
    split($2, markers, ":");
    split(markers[2], positions, "_");
    chr_name = markers[1];
    chr_position = $9;
    gsub(/[\r\n]+/, "", chr_position);  # Remove any newline or carriage return characters
    effect_allele = $3;
    noneffect_allele = $4;
    effect_weight = $5;

    print $1, chr_name, chr_position, effect_allele, noneffect_allele, effect_weight;
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"

echo "Processed data has been saved into $output_dir for CAD"


##################################### Stroke #####################################
# Input and output file names
input_file="PMID38134266_stroke_GRCh37.txt"
outcome="stroke"

awk 'NR == 1 {
    print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight";
}
NR > 1 {
    rsid = $1;                  # SNP ID (rsID)
    chr_name = $3;              # Chromosome
    chr_position = $6;          # chr_position
    gsub(/[\r\n]+/, "", chr_position);  # Remove any newline or carriage return characters
    effect_allele = $2;         # Risk allele
    noneffect_allele = "X";   # Placeholder for the non-effect allele
    effect_weight = $4;         # Beta

    print rsid, chr_name, chr_position, effect_allele, noneffect_allele, effect_weight;
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"

echo "Processed data saved to $output_dir/$outcome/textfiles/prsfile.txt"

##################################### HF #####################################
# Input and output file names
input_file="PGS003969_hf.txt"
outcome="hf"

awk '{
    if (!/^#/) {
        gsub(/^rsID/, "rsid");
        gsub(/other_allele/, "noneffect_allele");
        gsub(/allelefrequency_effect/, "eaf");

        if (NR == 1) {
            print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight";
        } else {
            print $1, $2, $3, $4, $5, $6;
        }
    }
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"

echo "Processed data has been saved into $output_dir for HF"

##################################### AF #####################################
# Input and output file names
input_file="PMID37523535_af.txt"
outcome="af"

awk 'NR == 1 {
    print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight";
}
NR > 1 {
    split($2, pos, ":");     # Split Position into chr_name and chr_position
    split($3, alleles, "/"); # Split Risk/reference into effect_allele and noneffect_allele

    rsid = $1;                # Rs_ID
    chr_name = substr(pos[1], 4);  # Remove "chr" prefix to get chromosome number
    chr_position = pos[2];    # Chromosome position
    effect_allele = alleles[1];  # Effect allele
    noneffect_allele = alleles[2];  # Non-effect allele
    effect_weight = $4;       # Effect (beta)
    
    print rsid, chr_name, chr_position, effect_allele, noneffect_allele, effect_weight;
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"

echo "Processed data has been saved into $output_dir for AF"

##################################### VA #####################################
# Input and output file names
input_file="PMID39657596_va.txt"
outcome="va"

awk 'NR == 1 {
    print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight";
}
NR > 1 {
    rsid = $3;  # SNP
    chr_name = $1;  # Chromosome
    chr_position = $2;  # Position
    effect_allele = $4;  # Effect allele
    noneffect_allele = $5;  # Other allele
    effect_weight = $6;  # Ors 

    print rsid, chr_name, chr_position, effect_allele, noneffect_allele, effect_weight;
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"

echo "Processed data has been saved into $output_dir for VA"

##################################### PAD #####################################
# Input and output file names
input_file="PMID31285632_pad.txt"
outcome="pad"

awk 'NR == 1 {
    print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight"
}
NR > 1 {
    split($1, pos, ":");
    chr_name = pos[1];
    chr_position = pos[2];
    effect_allele = $3;
    noneffect_allele = $4;
    effect_weight = $6;

    print $2, chr_name, chr_position, effect_allele, noneffect_allele, effect_weight;
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"


echo "Processed data has been saved into $output_dir for PAD"

##################################### AAA #####################################
# Input and output file names
input_file="PMID38241289_aaa.txt"
outcome="aaa"

awk 'NR == 1 {
    print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight"
}
NR > 1 {
    rsid = $2;
    chr_name = $1;
    chr_position = $3;
    effect_allele = $6;
    noneffect_allele = "X";
    effect_weight = $7;

    print rsid, chr_name, chr_position, effect_allele, noneffect_allele, effect_weight;
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"


echo "Processed data has been saved into $output_dir for AAA"

##################################### VTE #####################################
# Input and output file names
input_file="PGS000043_vte.txt"
outcome="vte"

awk '{
    if (!/^#/) {
        gsub(/^rsID/, "rsid");
        gsub(/other_allele/, "noneffect_allele");
        
        if (NR == 1) {
            print "rsid\tchr_name\tchr_position\teffect_allele\tnoneffect_allele\teffect_weight";
        } else {
            print $1, $2, $3, $4, $5, $6;
        }
    }
}' OFS="\t" "$input_dir/$input_file" > "$output_dir/$outcome/textfiles/prsfile.txt"

echo "Processed data has been saved into $output_dir for VTE"