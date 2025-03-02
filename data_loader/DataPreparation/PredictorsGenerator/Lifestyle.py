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
log_filename = os.path.join(log_dir, 'Lifestyle.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)

target_FieldID = [
    '20116-0.0', # Smoking status
    # '1239-0.0', # Current tobacco smoking
    '3456-0.0', # Number of cigarettes currently smoked daily (current cigarette smokers)
    '1558-0.0', # Alcohol intake frequency
    '1160-0.0', # Sleep duration
    '22036-0.0' # At or above moderate/vigorous/walking recommendation
]

target_FieldID = ['eid'] + target_FieldID
feature_df = population_char_df[target_FieldID]

###################################################################################
##### Smoking          ############################################################
###################################################################################
'''
Smoke status is coded as:
    coding                meaning
    1                  non-smoker
    2                   ex-smoker
    3 light smoker (less than 10)
    4  moderate smoker (10 to 19)
    5   heavy smoker (20 or over)
'''

'''
# Smoking status
conditions = [
    (feature_df['20116-0.0'] == 0),
    (feature_df['20116-0.0'] == 1),
    (feature_df['20116-0.0'] == 2) & (feature_df['3456-0.0'] >= 20),
    (feature_df['20116-0.0'] == 2) & (feature_df['3456-0.0'] > 9) & (feature_df['3456-0.0'] <= 19),
    (feature_df['20116-0.0'] == 2) & (feature_df['3456-0.0'].isin([-10, -1, -3, np.nan]))
]
values = ['non-smoker', 'ex-smoker', 'heavy smoker', 'moderate smoker', 'light smoker']
feature_df['smoking_status'] = np.select(conditions, values, default=np.nan)
'''

# Current smoking
feature_df['current_smoking'] = np.where(feature_df['20116-0.0'].isin([0, 1]), 0, np.where(feature_df['20116-0.0'] == 2, 1, np.nan))


###################################################################################
##### Alcohol intake          #####################################################
###################################################################################
# coding_100402_dict = {
#     1: 'Daily or almost daily',
#     2: 'Three or four times a week',
#     3: 'Once or twice a week',
#     4: 'One to three times a month',
#     5: 'Special occasions only',
#     6: 'Never',
#     -3: np.nan,
#     np.nan: np.nan
# }
# feature_df['1558-0.0'].replace(coding_100402_dict, inplace=True)
# Daily drinking
feature_df['daily_drinking'] = np.where(feature_df['1558-0.0'] == 1, 1, 
                                        np.where(feature_df['1558-0.0'].notna(), 0, np.nan))

'''
# Weekly drinking
feature_df['weekly_drinking'] = np.where(feature_df['1558-0.0'].isin([1, 2, 3]), 1,
                                        np.where(feature_df['1558-0.0'].notna(), 0, np.nan))
'''

###################################################################################
##### Sleep duration          #####################################################
###################################################################################
feature_df['healthy_sleep'] = np.where(
    (feature_df['1160-0.0'].isna()), np.nan,
    np.where(
        (feature_df['1160-0.0'] < 7) | (feature_df['1160-0.0'] > 9), 0, 1
    )
)

###################################################################################
##### Physical activity       #####################################################
###################################################################################
'''
At or above moderate/vigorous/walking recommendation: https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=22036
'''
feature_df.rename(columns = {'22036-0.0': 'physical_act'}, inplace=True)

###################################################################################
##### Healthy diet & Social connection  ###########################################
###################################################################################
diet_df = pd.read_csv(output_dir + 'Lifestyle_Diet.csv', usecols = ['eid', 'healthy_diet'])
social_df = pd.read_csv(output_dir + 'Lifestyle_Social.csv', usecols = ['eid', 'social_active'])


# Pool all lifestyle features
feature_df = feature_df[['eid', 'current_smoking', 'daily_drinking', 'healthy_sleep', 'physical_act']]
feature_df = pd.merge(feature_df, diet_df, how = 'left', on = ['eid'])
feature_df = pd.merge(feature_df, social_df, how = 'left', on = ['eid'])

feature_df.to_csv(output_dir + 'Lifestyle.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')