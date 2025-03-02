import os
import time
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/omics/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'ReadRawPRSData.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
merged_scores = None
outcomes = ['cad', 'stroke', 'hf', 'af', 'pad', 'aaa', 'vte']
for outcome in outcomes:
    score = pd.read_csv(f'/your path/multiomics-cardiovascular-disease/data/processed/prs/{outcome}/results/{outcome}.sscore', sep='\t')
    score = score[score['IID'] >= 0]
    score = score[['IID', 'SCORE1_AVG']].rename(columns={'IID': 'eid', 'SCORE1_AVG': f'prs_{outcome}'})

    if merged_scores is None:
        merged_scores = score
    else:
        merged_scores = pd.merge(merged_scores, score, on='eid', how='outer')

# Standardize the PRS
prs_columns = merged_scores.columns[1:]
scaler = StandardScaler()
merged_scores[prs_columns] = scaler.fit_transform(merged_scores[prs_columns])

# Exclude participants withdrawn from the UK Biobank
withdrawal_list = pd.read_csv('/home/ukb/data/withdrawals/withdraw79146_204_20240527.txt', sep="\t", header=None)  
withdrawal_list = withdrawal_list[0].tolist()
merged_scores = merged_scores[~merged_scores['eid'].isin(withdrawal_list)]
logger.info(merged_scores.head())
merged_scores.to_csv(output_dir + 'PolygenicScores.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')