import os
import time
import logging
import pandas as pd
import numpy as np

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/covariates/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'Biofluids.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
blood_count_df = pd.read_csv(data_dir + 'blood_count.csv', low_memory=False)
blood_biochem_df = pd.read_csv(data_dir + 'blood_biochem.csv', low_memory=False)

target_FieldID_bc = [
    '30160-0.0', # Basophill count
    # '30220-0.0', # Basophill percentage
    '30150-0.0', # Eosinophill count
    # '30210-0.0', # Eosinophill percentage
    '30030-0.0', # Haematocrit percentage
    '30020-0.0', # Haemoglobin concentration
    # '30300-0.0', # High light scatter reticulocyte count
    # '30290-0.0', # High light scatter reticulocyte percentage
    # '30280-0.0', # Immature reticulocyte fraction
    '30120-0.0', # Lymphocyte count
    # '30180-0.0', # Lymphocyte percentage
    # '30050-0.0', # Mean corpuscular haemoglobin
    # '30060-0.0', # Mean corpuscular haemoglobin concentration
    # '30040-0.0', # Mean corpuscular volume
    # '30100-0.0', # Mean platelet (thrombocyte) volume
    # '30260-0.0', # Mean reticulocyte volume
    # '30270-0.0', # Mean sphered cell volume
    '30130-0.0', # Monocyte count
    # '30190-0.0', # Monocyte percentage
    '30140-0.0', # Neutrophill count
    # '30200-0.0', # Neutrophill percentage
    # '30170-0.0', # Nucleated red blood cell count
    # '30230-0.0', # Nucleated red blood cell percentage
    '30080-0.0', # Platelet count
    # '30090-0.0', # Platelet crit
    # '30110-0.0', # Platelet distribution width
    # '30010-0.0', # Red blood cell (erythrocyte) count
    # '30070-0.0', # Red blood cell (erythrocyte) distribution width
    # '30250-0.0', # Reticulocyte count
    # '30240-0.0', # Reticulocyte percentage
    '30000-0.0' # White blood cell (leukocyte) count
    ]
target_FieldID_bc = ['eid'] + target_FieldID_bc
bc_df = blood_count_df[target_FieldID_bc]

target_FieldID_bb = [
    # Lipids
    '30630-0.0', # Apolipoprotein A
    '30640-0.0', # Apolipoprotein B
    '30690-0.0', # Total cholesterol
    '30780-0.0', # LDL cholesterol
    '30760-0.0', # HDL cholesterol
    '30790-0.0', # Lipoprotein A
    '30870-0.0', # Triglycerides
    # Glucose metabolism
    '30740-0.0', # Glucose
    '30750-0.0', # Glycated haemoglobin (HbA1c)
    # Kidney
    '30700-0.0', # Creatinine
    '30720-0.0', # Cystatin C
    '30880-0.0', # Urate
    '30670-0.0', # Urea
    # Liver
    '30600-0.0', # Albumin
    '30860-0.0', # Total protein
    '30620-0.0', # Alanine aminotransferase
    '30650-0.0', # Aspartate aminotransferase
    '30730-0.0', # Gamma glutamyltransferase
    '30610-0.0', # Alkaline phosphatase
    '30660-0.0', # Direct bilirubin
    '30840-0.0', # Total bilirubin
    
    # Inflammation/Immune system
    '30710-0.0', # C-reactive protein

    # Musculoskeletal system
    '30810-0.0', # Phosphate
    '30680-0.0', # Calcium
    '30890-0.0', # Vitamin D
    
    # Endocrine
    '30770-0.0', # IGF-1
    '30830-0.0', # Sex hormone-binding globulin
    # '30800-0.0', # Oestradiol
    # '30820-0.0', # Rheumatoid factor
    '30850-0.0' # Testosterone
    ]
target_FieldID_bb = ['eid'] + target_FieldID_bb
bb_df = blood_biochem_df[target_FieldID_bb]

feature_df = pd.merge(bc_df, bb_df, how='inner', on=['eid'])

# Rename columns
feature_df.columns = [
    'eid', 
    'baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc', # Blood count
    'apoA', 'apoB', 'total_cl', 'ldl_cl', 'hdl_cl', 'lpa', 'tg', 'glucose', 'hba1c', 'crt', 'cysc', 'urate', 'urea',
    'alb', 'tp', 'alt', 'ast', 'ggt', 'alp', 'dbil', 'tbil', 'crp', 'pho', 'ca', 'vd', 'igf1', 'shbg', 'trt' # Blood biochemistry
    ]
feature_df.to_csv(output_dir + 'Biofluids.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')