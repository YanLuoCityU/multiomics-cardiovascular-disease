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
log_filename = os.path.join(log_dir, 'PhysicalMeasurements.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
phy_measure_df = pd.read_csv(data_dir + 'phy_measure.csv', low_memory=False)

target_FieldID = [
    '93-0.0', '93-0.1', # Systolic blood pressure, manual reading
    '94-0.0', '94-0.1', # Diastolic blood pressure, manual reading
    '4079-0.0', '4079-0.1', # Diastolic blood pressure, automated reading
    '4080-0.0', '4080-0.1', # Systolic blood pressure, automated reading
    '48-0.0', # Waist circumference
    '49-0.0', # Hip circumference
    '50-0.0', # Standing height
    '21002-0.0', # Weight
    '21001-0.0', # Body mass index (BMI)
    # '23105-0.0', # Basal metabolic rate
    # '102-0.0', '102-0.1', # Pulse rate, automated reading
    # '3064-0.0', '3064-0.1', '3064-0.2', # Peak expiratory flow (PEF)
    # '20150-0.0', # Forced expiratory volume in 1-second (FEV1), Best measure
    # '20151-0.0', # Forced vital capacity (FVC), Best measure
    # '20258-0.0', # FEV1/ FVC ratio Z-score
    # '21021-0.0', # Pulse wave Arterial Stiffness index
    # '23099-0.0', # Body fat percentage
    # '23127-0.0' # Trunk fat percentage
    ]
target_FieldID = ['eid'] + target_FieldID
feature_df = phy_measure_df[target_FieldID]

############## Blood pressure #############################
# Fill missing values on blood pressure (automated reading) with blood pressure (manual reading)
feature_df['4079-0.0'].fillna(feature_df['94-0.0'], inplace=True)
feature_df['4079-0.1'].fillna(feature_df['94-0.1'], inplace=True)
feature_df['4080-0.0'].fillna(feature_df['93-0.0'], inplace=True)
feature_df['4080-0.1'].fillna(feature_df['93-0.1'], inplace=True)

# Calculate mean blood pressure
feature_df['mean_dbp'] = (feature_df['4079-0.0'] + feature_df['4079-0.1']) / 2
feature_df['mean_sbp'] = (feature_df['4080-0.0'] + feature_df['4080-0.1']) / 2

############## Waist/hip ratio #############################
feature_df['waist_hip_ratio'] = feature_df['48-0.0'] / feature_df['49-0.0']

# Rename columns
feature_df.rename(columns = {'mean_sbp': 'sbp', 'mean_dbp': 'dbp', '50-0.0': 'height', '21002-0.0': 'weight',
                             '48-0.0': 'waist_cir', '21001-0.0': 'bmi'}, inplace = True)
feature_df = feature_df[['eid', 'sbp', 'dbp', 'height', 'weight', 'waist_cir', 'waist_hip_ratio', 'bmi']]
feature_df.to_csv(output_dir + 'PhysicalMeasurements.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')