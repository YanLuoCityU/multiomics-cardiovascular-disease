'''

This file excludes the proteins with more than 50% missingness across all individuals.

Contributors: Yan Luo

'''

import os
import time
import logging
import pandas as pd
import numpy as np

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/omics/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'ProteomicsQC.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
proteomics_df = pd.read_csv(output_dir + 'proteomics_ins0.csv', low_memory=False)

# Exclude proteins with more than 50% missingness
proteomics_df.isnull().mean().sort_values(ascending=False) # Only three proteins: GLIPR1, NPM1, and PCOLCE have more than 50% missingness
proteomics_df = proteomics_df.loc[:, proteomics_df.isnull().mean() < 0.5]

proteomics_df.to_csv(output_dir + 'Proteomics.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')