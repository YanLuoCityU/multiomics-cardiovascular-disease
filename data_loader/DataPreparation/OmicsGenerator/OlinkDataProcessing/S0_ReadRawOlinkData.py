'''

This file convert original downloaded Olink_data to prepared format of row*columns = eid*pro_ids.

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
log_filename = os.path.join(log_dir, 'ReadRawOlinkData.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
## Read the Showcase Field data and transform from wide format to long format
proteomics_main_df = pd.read_csv(data_dir + 'proteomics_main.csv')
proteomics_main_df_long = pd.melt(proteomics_main_df, id_vars=['eid'], var_name='variable', value_name='value')
proteomics_main_df_long[['variable', 'ins_index']] = proteomics_main_df_long['variable'].str.split('-', expand=True)[[0, 1]]
proteomics_main_df_long['ins_index'] = proteomics_main_df_long['ins_index'].astype(float)
proteomics_main_df = proteomics_main_df_long.groupby(['eid', 'ins_index', 'variable'])['value'].first().unstack().reset_index()
proteomics_main_df = proteomics_main_df.rename(columns={
    '30900': 'n_proteins',
    '30901': 'PlateID',
    '30902': 'WellID',
    '30903': 'UKB_PPP',
    'value': 'n_duplicates'
})
proteomics_main_df = proteomics_main_df.fillna(np.nan)

## Read the Encoding Index data
coding_df = pd.read_csv(resources_dir + 'Olink/coding143.tsv', sep="\t")
coding_df['assay'] = coding_df['meaning'].str.split(';').str[0]
coding_df['name'] = coding_df['meaning'].str.split(';').str[1]
coding_df.drop(columns=['meaning'], inplace=True)

## Read the olink_data (i.e., NPX data) and rename according to the Encoding Index data
olink_data_df = pd.read_csv(data_dir + 'olink_data.txt', sep="\t")
olink_data_df = pd.merge(olink_data_df, coding_df, left_on='protein_id', right_on='coding', how='left')
olink_data_df.drop(columns=['coding'], inplace=True)

## Read the Resource Datasets
batch_df = pd.read_csv(resources_dir + 'Olink/olink_batch_number.dat', sep="\t") # olink batch number
batch_df['PlateID'] = batch_df['PlateID'].astype('float')
lod_df = pd.read_csv(resources_dir + 'Olink/olink_limit_of_detection.dat', sep="\t") # olink limit of detection

# Merge data (reference: https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=4654)
## Find all unique eid
eid_lst = olink_data_df['eid'].unique()

## Process data in batches and save the results
step = 1000
num_batches = (len(eid_lst) + step - 1) // step
logger.info(num_batches)

for i in range(num_batches):
    start_idx = i * step
    end_idx = min((i + 1) * step, len(eid_lst))
    eid_batch = eid_lst[start_idx:end_idx]
    
    # Extract data for the current batch
    tmpdf = olink_data_df[olink_data_df['eid'].isin(eid_batch)]
    
    # Merge data
    proteomics_df = (
        tmpdf
        .merge(proteomics_main_df, on=['eid', 'ins_index'], how='left')
        .merge(batch_df, on='PlateID', how='left')
        .merge(lod_df, left_on=['PlateID', 'ins_index', 'assay'], right_on=['PlateID', 'Instance', 'Assay'], how='left')
        .drop(columns=['Assay', 'Instance'])
    )
    
    # Filter rows where instance is 0
    # instance 0 indicates "Initial assessment visit (2006-2010) at which participants were recruited and consent given"
    # https://biobank.ndph.ox.ac.uk/showcase/instance.cgi?id=2
    # There are 53,014 participants in total (https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=30900)
    proteomics_ins0_df = proteomics_df[proteomics_df['ins_index']==0]
    
    # Replace values below the limit of detection (LOD) with LOD/sqrt(2)
    proteomics_ins0_df.loc[proteomics_ins0_df['result'] < proteomics_ins0_df['LOD'], 'result'] = proteomics_ins0_df['LOD'] / np.sqrt(2)
    
    # Use pivot table to convert long-format data to wide-format
    proteomics_ins0_df = proteomics_ins0_df[['eid', 'ins_index', 'assay', 'result']]
    pivot_df = proteomics_ins0_df.pivot(index='assay', columns='eid', values='result')
    
    # Fill missing values with empty strings
    pivot_df.fillna(np.nan, inplace=True)
    
    # Transpose
    trans_df = pivot_df.transpose()
    trans_df.columns.name = None
    
    logger.info(f'Current batch is {i}')
    if i == num_batches - 1:
        logger.info("Proteomics data extraction is completed.")
    
    # Save the results to a file
    if i == 0:
        trans_df.to_csv(output_dir + 'proteomics_ins0.csv', index=True)
    else:
        with open(output_dir + 'proteomics_ins0.csv', 'a') as f:
            trans_df.to_csv(f, header=False, index=True) 

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')