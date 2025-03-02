# Run with 'r-4.2' anaconda environment

library(ukbnmr)
library(data.table)

# Read the raw NMR data
decoded <- fread(file.path('/home/ukb/data/phenotype_data/metabolomics.csv')) # file saved by ukbconv tool

# Decode the raw NMR data using the ukbnmr package
nmr <- extract_biomarkers(decoded)
biomarker_qc_flags <- extract_biomarker_qc_flags(decoded)
sample_qc_flags <- extract_sample_qc_flags(decoded)

nmr_ins0 <- nmr[nmr$visit_index==0,]

fwrite(nmr_ins0, file=file.path('/your path/multiomics-cardiovascular-disease/data/processed/omics/metabolomics_ins0.csv'))
fwrite(biomarker_qc_flags, file=file.path('/your path/multiomics-cardiovascular-disease/data/processed/omics/nmr_biomarker_qc_flags.csv'))
fwrite(sample_qc_flags, file=file.path('/your path/multiomics-cardiovascular-disease/data/processed/omics/nmr_sample_qc_flags.csv'))


#################################################################################################
# Read the NMR information data provided by the ukbnmr package (https://github.com/sritchie73/ukbnmr/blob/main/data/nmr_info.rda) 
# or from the [Supplementary Table 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41597-023-01949-y/MediaObjects/41597_2023_1949_MOESM2_ESM.xlsx)  of the paper (Ritchie, S.C., Surendran, P., Karthikeyan, S. et al. Quality control and removal of technical variation of NMR metabolic biomarker data in ~120,000 UK Biobank participants. Sci Data 10, 64 (2023). https://doi.org/10.1038/s41597-023-01949-y).
# Hard to run in the Linux server. Please run it in the Windows environment.

# load('nmr_info.rda')
# write.csv(nmr_info, '/your path/multiomics-cardiovascular-disease/data/processed/omics/nmr_info.csv')
#################################################################################################