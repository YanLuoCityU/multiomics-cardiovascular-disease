'''

Contributors: Yan Luo

'''

import os
import time
import logging
import pandas as pd
import numpy as np

# get_date_interval: calculate the interval between two dates
def get_date_interval(start_date_var, end_date_var, df, unit):
    start_date = pd.to_datetime(df[start_date_var], format='%Y-%m-%d')
    end_date = pd.to_datetime(df[end_date_var], format='%Y-%m-%d')
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    if unit == 'day':
        return pd.DataFrame(days)
    elif unit == 'year':
        years = [ele/365 for ele in days]
        return pd.DataFrame(years)

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/outcomes/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'OutcomesInfo.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)
health_outcomes_df = pd.read_csv(data_dir + 'health_outcomes.csv', low_memory=False)

pc_df = population_char_df[['eid', '21022-0.0', '53-0.0', '54-0.0']]
death_df = health_outcomes_df[['eid', '40000-0.0']]
feature_df = pd.merge(pc_df, death_df, on='eid', how='left')
feature_df.rename(columns = {'53-0.0': 'bl_date', '54-0.0': 'center', '40000-0.0': 'death_date'}, inplace=True)

# Recode the assessment center into the region
feature_df['region'] = feature_df['center'].copy()
feature_df['region'].replace([10003, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
                             11011, 11012, 11013, 11014, 11016, 11017, 11018, 11020, 11021, 11022, 11023],
                            [2,     2,     7,     1,     9,     9,     5,     7,     2,     3,     4,
                             8,     0,     6,     4,     2,     3,     0,     0,     5,     1,     1], inplace = True)

# Get the days between baseline date and end date
'''

The end of follow-up date is 2023-11-30 for England & Wales and 2023-12-31 for Scotland.

https://biobank.ndph.ox.ac.uk/ukb/exinfo.cgi?src=Data_providers_and_dates

'''
feature_df['end_date'] = pd.to_datetime('2023-11-30')
feature_df.loc[feature_df['region'] == 9, 'end_date'] = pd.to_datetime('2023-12-31')
feature_df['bl2end_yrs'] = get_date_interval('bl_date', 'end_date', feature_df, unit='year')
feature_df['bl2death_yrs'] = get_date_interval('bl_date', 'death_date', feature_df, unit='year')

feature_df = feature_df[['eid', 'bl_date', 'end_date', 'death_date', 'bl2end_yrs', 'bl2death_yrs']]
feature_df.to_csv(output_dir + 'OutcomesBasicInfo.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')
    
'''

'London' = 0
'Wales' = 1
'North-West' = 2
'North-East' = 3
'Yorkshire and Humber' = 4
'West Midlands' = 5
'East Midlands' = 6
'South-East' = 7
'South-West' = 8
'Scotland' = 9

'''

'''
Original region code:
https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=10

coding	meaning
11012	Barts
11021	Birmingham
11011	Bristol
11008	Bury
11003	Cardiff
11024	Cheadle (revisit)
11020	Croydon
11005	Edinburgh
11004	Glasgow
11018	Hounslow
11010	Leeds
11016	Liverpool
11001	Manchester
11017	Middlesborough
11009	Newcastle
11013	Nottingham
11002	Oxford
11007	Reading
11014	Sheffield
10003	Stockport (pilot)
11006	Stoke
11022	Swansea
11023	Wrexham
11025	Cheadle (imaging)
11026	Reading (imaging)
11027	Newcastle (imaging)
11028	Bristol (imaging)

'''