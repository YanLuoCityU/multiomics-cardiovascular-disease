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
log_filename = os.path.join(log_dir, 'LifestyleSocial.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)

target_FieldID = ['709-0.0', '1031-0.0', '6160-0.0']
target_FieldID = ['eid'] + target_FieldID
feature_df = population_char_df[target_FieldID]


######################################################################################################
########################### Definition of Healthy Social connection  #################################
######################################################################################################

'''

The social isolation index was assessed using three questions: 
(1) “Including yourself, how many people are living together in your household? 
Include those who usually live in the house such as students living away from home during term time, partners in the armed forces or professions such as pilots” 
    (1 point for living alone); 
(2) “How often do you visit friends or family or have them visit you?”
    (1 point for friends and family visit less than once a month); 
(3) “Which of the following [leisure/social activities] do you engage in once a week or more often? You may select more than one” 
    (1 point for no participation in social activities at least weekly).

Individuals could score a total of 0-3.
An individual was defined as socially isolated if he or she scored 2 or 3; those who scored 0 or 1 were classified as not isolated.

Reference: https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(17)30075-0/fulltext

'''

###################################################################################
##### Living alone           ######################################################
###################################################################################
feature_df['709-0.0'] = feature_df['709-0.0'].replace([-1, -3, np.nan], np.nan)
feature_df['live_alone'] = np.where(feature_df['709-0.0'] <= 1, 1, 
                                     np.where(feature_df['709-0.0'] > 1, 0, np.nan))

logger.info("Number of rows where 709 is missing:", feature_df['709-0.0'].isna().sum())
logger.info("Number of rows where live_alone is missing:", feature_df['live_alone'].isna().sum())
logger.info(feature_df['live_alone'].value_counts())
logger.info("\n")


###################################################################################
##### Friends and family visit less than once a month           ###################
###################################################################################
feature_df['1031-0.0'] = feature_df['1031-0.0'].replace([-1, -3, np.nan], np.nan)
feature_df['infreq_visit'] = np.where(feature_df['1031-0.0'] >= 5, 1, 
                                     np.where(feature_df['1031-0.0'] < 5, 0, np.nan))

logger.info("Number of rows where 1031 is missing:", feature_df['1031-0.0'].isna().sum())
logger.info("Number of rows where infreq_visit is missing:", feature_df['infreq_visit'].isna().sum())
logger.info(feature_df['infreq_visit'].value_counts())
logger.info("\n")


###################################################################################
##### No participation in social activities at least weekly     ###################
###################################################################################
feature_df['6160-0.0'] = feature_df['6160-0.0'].replace([-3, np.nan], np.nan)
feature_df['no_activity'] = np.where(feature_df['6160-0.0'] == -7, 1, 
                                     np.where(feature_df['6160-0.0'] <= 5, 0, np.nan))

logger.info("Number of rows where 6160 is missing:", feature_df['6160-0.0'].isna().sum())
logger.info("Number of rows where no_activity is missing:", feature_df['no_activity'].isna().sum())
logger.info(feature_df['no_activity'].value_counts())
logger.info("\n")

###################################################################################
##### Get final score           ###################################################
###################################################################################
SocialIsolation = feature_df[['live_alone','infreq_visit','no_activity']].sum(axis=1, skipna=True)
mask = feature_df[['live_alone','infreq_visit','no_activity']].isna().all(axis=1)
SocialIsolation[mask] = np.nan
logger.info(SocialIsolation.value_counts())

feature_df['social_active'] = np.where(SocialIsolation <=1, 1, np.where(SocialIsolation >1, 0, np.nan))
logger.info(feature_df['social_active'].value_counts()) 
logger.info("\n")

'''

The percentage of social isolation is nearly 9%, similar to the reference: https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(17)30075-0/fulltext

'''

feature_df = feature_df[['eid', 'social_active']]
feature_df.to_csv(output_dir + '/Lifestyle_Social.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')