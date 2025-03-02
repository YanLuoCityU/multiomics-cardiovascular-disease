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
log_filename = os.path.join(log_dir, 'MedicationHistory.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
medication_df = pd.read_csv(data_dir + 'medication.csv', low_memory=False)
atc_code_df = pd.read_csv(resources_dir + 'UKB_drug_atc_code.csv', low_memory=False)

drug_df = medication_df[['eid'] + ['20003-0.'+str(i) for i in range(48)]]
drug_df = drug_df.replace(to_replace=99999, value=np.nan)

# Replace UKB Treatment/medication code with ATC code
drug_df = drug_df.replace(to_replace=atc_code_df['ukb_coding'].values, value=atc_code_df['atc_coding'].values)

# Create a dictionary with the eid as the key and the ATC code as the value
dict_drug = drug_df.set_index('eid').T.to_dict('list')
dict_drug = {k: [x for x in v if str(x) != 'nan'] for k, v in dict_drug.items()}
dict_drug = {k: [item for sublist in v for item in str(sublist).split(' |')] for k, v in dict_drug.items()}

# Find the eid who takes lipid lowering therapy, antihypertensive drugs, aspirin
lipidlower_eid = [k for k, v in dict_drug.items() if any(str(x).startswith('C10A') or str(x).startswith('C10B') for x in v)]
antihypt_eid = [k for k, v in dict_drug.items() if any(str(x).startswith('C02') for x in v)]
antidiab_eid = [k for k, v in dict_drug.items() if any(str(x).startswith('A10') for x in v)]
aspirin_eid = [k for k, v in dict_drug.items() if any(str(x).startswith('B01') for x in v)]
aap_eid = [k for k, v in dict_drug.items() if any(str(x).startswith('N05') for x in v)]
gcs_eid = [k for k, v in dict_drug.items() if any(str(x).startswith('H02') for x in v)]

# Create a new dataframe with the eid and the binary variables for lipid-lowering therapy, antihypertensive drugs, aspirin, atypical antipsychotics, and glucocorticoids
drug_df = pd.merge(drug_df, medication_df[['eid', '6153-0.0', '6177-0.0']], how='left', on='eid')
drug_df['lipidlower'] = np.where((drug_df['eid'].isin(lipidlower_eid)) | (drug_df['6153-0.0'] == 1) | (drug_df['6177-0.0'] == 1), 1, 0)
drug_df['antihypt'] = np.where((drug_df['eid'].isin(antihypt_eid)) | (drug_df['6153-0.0'] == 2) | (drug_df['6177-0.0'] == 2), 1, 0)
drug_df['antidiab'] = np.where((drug_df['eid'].isin(antidiab_eid)) | (drug_df['6153-0.0'] == 3) | (drug_df['6177-0.0'] == 3), 1, 0)
drug_df['aspirin'] = np.where(drug_df['eid'].isin(aspirin_eid), 1, 0)
drug_df['aap'] = np.where(drug_df['eid'].isin(aap_eid), 1, 0)
drug_df['gcs'] = np.where(drug_df['eid'].isin(gcs_eid), 1, 0)

# Print the summary statistics
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# logger.info(drug_df[['lipidlower', 'antihypt', 'antidiab', 'aspirin', 'aap', 'gcs']].describe())
logger.info(drug_df[['lipidlower', 'antihypt']].describe())
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Save the dataframe to a csv file
# feature_df = drug_df[['eid', 'lipidlower', 'antihypt', 'antidiab', 'aspirin', 'aap', 'gcs']]
feature_df = drug_df[['eid', 'lipidlower', 'antihypt']]
feature_df.to_csv(output_dir + 'MedicationHistory.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')
    
'''

Medication: 20003-0.0 to 20003-0.47, 6153-0.0, 6177-0.0
    Lipid Lowering Therapy: ATC C10A/C10B + 6153-0.0==1 + 6177-0.0==1
    Antihypertensive Treatment: ATC C02 + 6153-0.0==2 + 6177-0.0==2
    Diabetes Treatment: ATC A10 + 6153-0.0==3 + 6177-0.0==3
    Aspirin: ATC B01
    Atypical Antipsychotics: ATC N05
    Glucocorticoids: ATC H02

'''