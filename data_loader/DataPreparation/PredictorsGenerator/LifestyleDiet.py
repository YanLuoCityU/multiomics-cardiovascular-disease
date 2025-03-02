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
log_filename = os.path.join(log_dir, 'LifestyleDiet.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)

target_FieldID = [
    '1289-0.0', '1299-0.0', '1309-0.0', '1319-0.0', '1329-0.0', 
    '1339-0.0', '1349-0.0', '1369-0.0', '1379-0.0', '1389-0.0', 
    '1438-0.0', '1448-0.0', '1458-0.0', '1468-0.0'
    ]
target_FieldID = ['eid'] + target_FieldID
feature_df = population_char_df[target_FieldID]

######################################################################################################
########################### Definition of Healthy Diet  ##############################################
######################################################################################################

'''

At least 4 of the following 7 food groups:
1. Vegetables: ≥ 3 servings/day (1289, 1299)
2. Fruits: ≥ 3 servings/day (1309, 1319)
3. Fish: ≥2 servings/week (1329, 1339)
4. Processed meats: ≤ 1 serving/week (1349)
5. Unprocessed red meats: ≤ 1.5 servings/week (1369, 1379, 1389)
6. Whole grains: ≥ 3 servings/day (1438, 1448)
7. Refined grains: ≤1.5 servings/day (1458, 1468)

For Field IDs 1329, 1339, 1369, 1379, and 1389, we used the following coding: 
'never' = 0, 'less than once per week' = 0·5, 'once per week' = 1, '2-4 times per week' = 3, '5-6 times per week' = 5·5, 'once or more daily' = 7.
We then summed the frequencies of fish and unprocessed red meats.

Finally, we summed all food groups to generate the healthy diet score (treated missing items as the non-healthy, i.e., 0).

Reference: 
https://www.nature.com/articles/s41598-021-91259-3
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5799609/
https://jamanetwork.com/journals/jama/fullarticle/2738355

'''

replace_dict = {
    0: 0, 1: 0.5, 2: 1, 3: 3, 4: 5.5, 5: 7, 
    -1: np.nan, -3: np.nan
}

####################################################################################################
##### Vegetables: ≥ 3 servings/day                 #################################################
##### Cooked vegetable intake (Field ID 1289)      #################################################
##### Salad / raw vegetable intake (Field ID 1299) #################################################
####################################################################################################
feature_df[['1289-0.0', '1299-0.0']] = feature_df[['1289-0.0', '1299-0.0']].replace([-10, -1, -3, np.nan], np.nan)
Veg_intake = feature_df[['1289-0.0', '1299-0.0']].sum(axis=1, skipna=True)
mask = feature_df[['1289-0.0', '1299-0.0']].isna().all(axis=1)
Veg_intake[mask] = np.nan
feature_df['veg_intake'] = np.where(Veg_intake >= 3, 1, np.where(Veg_intake < 3, 0, np.nan))

missing_both = feature_df[['1289-0.0', '1299-0.0']].isna().all(axis=1)
num_missing_both = missing_both.sum()
logger.info("Number of rows where both 1289 and 1299 are missing:", num_missing_both)
logger.info("Number of rows where veg_intake is missing:", feature_df['veg_intake'].isna().sum())
logger.info(feature_df['veg_intake'].value_counts())
logger.info("\n")

####################################################################################################
##### Fruits: ≥ 3 servings/day                     #################################################
##### Fresh fruit intake (Field ID 1309)           #################################################
##### Dired fruit intake (Field ID 1319)           #################################################
####################################################################################################
feature_df[['1309-0.0', '1319-0.0']] = feature_df[['1309-0.0', '1319-0.0']].replace([-10, -1, -3, np.nan], np.nan)
Fruit_intake = feature_df[['1309-0.0', '1319-0.0']].sum(axis=1, skipna=True)
mask = feature_df[['1309-0.0', '1319-0.0']].isna().all(axis=1)
Fruit_intake[mask] = np.nan
feature_df['fruit_intake'] = np.where(Fruit_intake >= 3, 1, np.where(Fruit_intake < 3, 0, np.nan))

missing_both = feature_df[['1309-0.0', '1319-0.0']].isna().all(axis=1)
num_missing_both = missing_both.sum()
logger.info("Number of rows where both 1309 and 1319 are missing:", num_missing_both)
logger.info("Number of rows where fruit_intake is missing:", feature_df['fruit_intake'].isna().sum())
logger.info(feature_df['fruit_intake'].value_counts())
logger.info("\n")

####################################################################################################
##### Fish: ≥2 servings/week                     ###################################################
##### Oily fish intake (Field ID 1329)           ###################################################
##### Non-oily fish intake (Field ID 1339)       ###################################################
####################################################################################################
for col in ['1329-0.0', '1339-0.0']:
    feature_df[col] = feature_df[col].replace(replace_dict)
Fish_intake = feature_df[['1329-0.0', '1339-0.0']].sum(axis=1, skipna=True)
mask = feature_df[['1329-0.0', '1339-0.0']].isna().all(axis=1)
Fish_intake[mask] = np.nan
feature_df['fish_intake'] = np.where(Fish_intake >= 2, 1, np.where(Fish_intake < 2, 0, np.nan))

missing_both = feature_df[['1329-0.0', '1339-0.0']].isna().all(axis=1)
num_missing_both = missing_both.sum()
logger.info("Number of rows where both 1329 and 1339 are missing:", num_missing_both)
logger.info("Number of rows where fish_intake is missing:", feature_df['fish_intake'].isna().sum())
logger.info(feature_df['fish_intake'].value_counts())
logger.info("\n")

####################################################################################################
##### Processed meats: ≤ 1 serving/week          ###################################################
##### Processed meat intake (Field ID 1349)      ###################################################
####################################################################################################
feature_df['1349-0.0'] = feature_df['1349-0.0'].replace([-10, -1, -3, np.nan], np.nan)
feature_df['pro_meat'] = np.where(feature_df['1349-0.0'] <= 2, 1, np.where(feature_df['1349-0.0'] > 2, 0, np.nan))

logger.info("Number of rows where 1349 is missing:", feature_df['1349-0.0'].isna().sum())
logger.info("Number of rows where proc_meat is missing:", feature_df['pro_meat'].isna().sum())
logger.info(feature_df['pro_meat'].value_counts())
logger.info("\n")

####################################################################################################
##### Unprocessed red meats: ≤ 1.5 servings/week ###################################################
##### Beef intake (Field ID 1369)                ###################################################
##### Lamb/mutton intake (Field ID 1379)         ###################################################
##### Pork intake (Field ID 1389)                ###################################################
####################################################################################################
for col in ['1369-0.0', '1379-0.0', '1389-0.0']:
    feature_df[col] = feature_df[col].replace(replace_dict)
unPro_RedMeat_intake = feature_df[['1369-0.0', '1379-0.0', '1389-0.0']].sum(axis=1, skipna=True)
mask = feature_df[['1369-0.0', '1379-0.0', '1389-0.0']].isna().all(axis=1)
unPro_RedMeat_intake[mask] = np.nan
feature_df['unpro_redmeat'] = np.where(unPro_RedMeat_intake <= 1.5, 1, np.where(unPro_RedMeat_intake > 1.5, 0, np.nan))

missing_both = feature_df[['1369-0.0', '1379-0.0', '1389-0.0']].isna().all(axis=1)
num_missing_both = missing_both.sum()
logger.info("Number of rows where 1369, 1379, and 1389 are missing:", num_missing_both)
logger.info("Number of rows where unpro_redmeat is missing:", feature_df['unpro_redmeat'].isna().sum())
logger.info(feature_df['unpro_redmeat'].value_counts())
logger.info("\n")

####################################################################################################
##### Whole grains: ≥ 3 servings/day (1438, 1448) ##################################################
##### Bread intake (Field ID 1438)               ###################################################
##### Bread type  (Field ID 1448)                ###################################################
####################################################################################################
feature_df['1438-0.0'] = feature_df['1438-0.0'].replace([-10, -1, -3, np.nan], np.nan)
feature_df['1448-0.0'] = feature_df['1448-0.0'].replace([-1, -3, np.nan], np.nan)
feature_df['whole_grain'] = np.where(feature_df['1438-0.0'] >= 3,
                                      np.where(feature_df['1448-0.0'] == 3, 1,
                                               np.where(feature_df['1448-0.0'].isna(), np.nan, 0)),
                                      np.where(feature_df['1438-0.0'].isna(), np.nan, 0))

missing_both = feature_df[['1438-0.0', '1448-0.0']].isna().all(axis=1)
num_missing_both = missing_both.sum()
logger.info("Number of rows where both 1438 and 1448 are missing:", num_missing_both)
logger.info("Number of rows where whole_grain is missing:", feature_df['whole_grain'].isna().sum())
logger.info(feature_df['whole_grain'].value_counts())
logger.info("\n")

####################################################################################################
##### Refined grains: ≤1.5 servings/day           ##################################################
##### Cereal intake (Field ID 1458)              ###################################################
##### Cereal type  (Field ID 1468)               ###################################################
####################################################################################################
feature_df['1458-0.0'] = feature_df['1458-0.0'].replace([-10, -1, -3, np.nan], np.nan)
feature_df['refined_grain'] = np.where(feature_df['1458-0.0'] <= 1.5, 1, np.where(feature_df['1458-0.0'] > 1.5, 0, np.nan))

logger.info("Number of rows where 1458 is missing:", feature_df['1458-0.0'].isna().sum())
logger.info("Number of rows where refined_grain is missing:", feature_df['refined_grain'].isna().sum())
logger.info(feature_df['refined_grain'].value_counts())
logger.info("\n")

###################################################################################
##### Get final score           ###################################################
###################################################################################
Healthy_diet_raw = feature_df[['veg_intake','fruit_intake','fish_intake','pro_meat','unpro_redmeat','whole_grain','refined_grain']].sum(axis=1, skipna=True)
mask = feature_df[['veg_intake','fruit_intake','fish_intake','pro_meat','unpro_redmeat','whole_grain','refined_grain']].isna().all(axis=1)
Healthy_diet_raw[mask] = np.nan
logger.info(Healthy_diet_raw.value_counts())
feature_df['healthy_diet'] = np.where(Healthy_diet_raw >=4, 1, np.where(Healthy_diet_raw <4, 0, np.nan))
logger.info(feature_df['healthy_diet'].value_counts())
logger.info("\n")

feature_df = feature_df[['eid', 'healthy_diet']]
feature_df.to_csv(output_dir + '/Lifestyle_Diet.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')