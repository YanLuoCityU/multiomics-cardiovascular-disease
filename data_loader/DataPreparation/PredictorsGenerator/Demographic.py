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
log_filename = os.path.join(log_dir, 'Demographic.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)

target_FieldID = [
    '54-0.0', # assessment center
    '21022-0.0', # age at recruitment
    '31-0.0', # sex
    '21000-0.0', # ethnicity
    '22189-0.0', # Townsend index
    # '53-0.0', # date of baseline assessment
]
target_FieldID = ['eid'] + target_FieldID
feature_df = population_char_df[target_FieldID]

########################### Ethnic background mapping  ###############################################
coding_1001_df = pd.read_csv(resources_dir + 'coding1001.tsv', sep='\t', usecols=['coding', 'meaning']) # https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=1001
coding_1001_dict = coding_1001_df.set_index('coding')['meaning'].to_dict()
feature_df['21000-0.0'] = feature_df['21000-0.0'].replace(coding_1001_dict)

ethn_bg_def = {
    0: ["White", "British", "Irish", "Any other white background"],
    1: ["Mixed", "White and Black Caribbean", "White and Black African", "White and Asian", "Any other mixed background"],  # Too few participants
    2: ["Asian or Asian British", "Indian", "Pakistani", "Bangladeshi", "Any other Asian background"], 
    3: ["Black or Black British", "Caribbean", "African", "Any other Black background"],
    4: ["Chinese", "Other ethnic group", "Do not know", "Prefer not to answer"]
    }

ethn_bg_dict = {}
for key, values in ethn_bg_def.items(): 
    for value in values:
        ethn_bg_dict[value]=key
        
feature_df['21000-0.0'] = feature_df['21000-0.0'].replace(ethn_bg_dict)

########################### Recode the assessment center into the region  ###############################################
feature_df['region'] = feature_df['54-0.0'].copy()
feature_df['region'].replace([10003, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
                             11011, 11012, 11013, 11014, 11016, 11017, 11018, 11020, 11021, 11022, 11023],
                            [2,     2,     7,     1,     9,     9,     5,     7,     2,     3,     4,
                             8,     0,     6,     4,     2,     3,     0,     0,     5,     1,     1], inplace = True)


# Rename columns
feature_df.rename(columns = {'21022-0.0': 'age', '31-0.0': 'male', '21000-0.0': 'ethnicity', '22189-0.0': 'townsend', '54-0.0': 'center'}, inplace=True)
feature_df = feature_df.drop(columns=['center'])
feature_df.to_csv(output_dir + 'DemographicInfo.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')

'''

ethnicity background mapping:
'White': ["White", "British", "Irish", "Any other white background"],
'Mixed': ["Mixed", "White and Black Caribbean", "White and Black African", "White and Asian", "Any other mixed background"],  
'Asian': ["Asian or Asian British", "Indian", "Pakistani", "Bangladeshi", "Any other Asian background"], 
'Black': ["Black or Black British", "Caribbean", "African", "Any other Black background"],
'Others: ["Chinese", "Other ethnic group", "Do not know", "Prefer not to answer"]

region mapping:
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