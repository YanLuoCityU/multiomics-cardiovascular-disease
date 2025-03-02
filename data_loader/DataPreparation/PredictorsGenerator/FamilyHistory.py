import os
import time
import logging
import pandas as pd
import numpy as np

def have_heart_disease(row):
    return 1 if row.drop('eid').isin([1]).any() else 0

def have_stroke(row):
    return 1 if row.drop('eid').isin([2]).any() else 0

def have_hypertension(row):
    return 1 if row.drop('eid').isin([8]).any() else 0

def have_diabetes(row):
    return 1 if row.drop('eid').isin([9]).any() else 0

# Define the path
data_dir = '/home/ukb/data/phenotype_data/'
resources_dir = '/home/ukb/data/resources/'
output_dir = '/your path/multiomics-cardiovascular-disease/data/processed/covariates/'
log_dir = '/your path/multiomics-cardiovascular-disease/saved/log/DataPreparation/'

# Set up logger
log_filename = os.path.join(log_dir, 'FamilyHistory.log')
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

start_time = time.time()

# Read data
population_char_df = pd.read_csv(data_dir + 'population_char.csv', low_memory=False)

############## Family history of heart disease ############################# 
# Illnesses of father
target_FieldID = [f"20107-0.{i}" for i in range(10)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_heart_disease, axis=1)
father_df = pd.DataFrame({'eid': temp_df['eid'], 'father_hist': result})

# Illnesses of mother 
target_FieldID = [f"20110-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_heart_disease, axis=1)
mother_df = pd.DataFrame({'eid': temp_df['eid'], 'mother_hist': result})

# Illnesses of siblings 
target_FieldID = [f"20111-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_heart_disease, axis=1)
siblings_df = pd.DataFrame({'eid': temp_df['eid'], 'siblings_hist': result})

feature_df = pd.merge(father_df, mother_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, siblings_df, how = 'left', on=['eid'])
feature_df['family_heart_hist'] = (feature_df['father_hist'] == 1) | (feature_df['mother_hist'] == 1) | (feature_df['siblings_hist'] == 1)
feature_df['family_heart_hist'] = feature_df['family_heart_hist'].astype(int)
feature_df = feature_df[['eid', 'family_heart_hist']]

############## Family history of stroke ############################# 
# Illnesses of father
target_FieldID = [f"20107-0.{i}" for i in range(10)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_stroke, axis=1)
father_df = pd.DataFrame({'eid': temp_df['eid'], 'father_hist': result})

# Illnesses of mother 
target_FieldID = [f"20110-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_stroke, axis=1)
mother_df = pd.DataFrame({'eid': temp_df['eid'], 'mother_hist': result})

# Illnesses of siblings 
target_FieldID = [f"20111-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_stroke, axis=1)
siblings_df = pd.DataFrame({'eid': temp_df['eid'], 'siblings_hist': result})

feature_df = pd.merge(feature_df, father_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, mother_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, siblings_df, how = 'left', on=['eid'])
feature_df['family_stroke_hist'] = (feature_df['father_hist'] == 1) | (feature_df['mother_hist'] == 1) | (feature_df['siblings_hist'] == 1)
feature_df['family_stroke_hist'] = feature_df['family_stroke_hist'].astype(int)
feature_df = feature_df[['eid', 'family_heart_hist', 'family_stroke_hist']]

############## Family history of hypertension ############################# 
# Illnesses of father
target_FieldID = [f"20107-0.{i}" for i in range(10)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_hypertension, axis=1)
father_df = pd.DataFrame({'eid': temp_df['eid'], 'father_hist': result})

# Illnesses of mother 
target_FieldID = [f"20110-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_hypertension, axis=1)
mother_df = pd.DataFrame({'eid': temp_df['eid'], 'mother_hist': result})

# Illnesses of siblings 
target_FieldID = [f"20111-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_hypertension, axis=1)
siblings_df = pd.DataFrame({'eid': temp_df['eid'], 'siblings_hist': result})

feature_df = pd.merge(feature_df, father_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, mother_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, siblings_df, how = 'left', on=['eid'])
feature_df['family_hypt_hist'] = (feature_df['father_hist'] == 1) | (feature_df['mother_hist'] == 1) | (feature_df['siblings_hist'] == 1)
feature_df['family_hypt_hist'] = feature_df['family_hypt_hist'].astype(int)
feature_df = feature_df[['eid', 'family_heart_hist', 'family_stroke_hist', 'family_hypt_hist']]

############## Family history of diabetes ############################# 
# Illnesses of father
target_FieldID = [f"20107-0.{i}" for i in range(10)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_diabetes, axis=1)
father_df = pd.DataFrame({'eid': temp_df['eid'], 'father_hist': result})

# Illnesses of mother 
target_FieldID = [f"20110-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_diabetes, axis=1)
mother_df = pd.DataFrame({'eid': temp_df['eid'], 'mother_hist': result})

# Illnesses of siblings 
target_FieldID = [f"20111-0.{i}" for i in range(11)]
target_FieldID = ['eid'] + target_FieldID
temp_df = population_char_df[target_FieldID]
result = temp_df.apply(have_diabetes, axis=1)
siblings_df = pd.DataFrame({'eid': temp_df['eid'], 'siblings_hist': result})

feature_df = pd.merge(feature_df, father_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, mother_df, how = 'left', on=['eid'])
feature_df = pd.merge(feature_df, siblings_df, how = 'left', on=['eid'])
feature_df['family_diab_hist'] = (feature_df['father_hist'] == 1) | (feature_df['mother_hist'] == 1) | (feature_df['siblings_hist'] == 1)
feature_df['family_diab_hist'] = feature_df['family_diab_hist'].astype(int)
feature_df = feature_df[['eid', 'family_heart_hist', 'family_stroke_hist', 'family_hypt_hist', 'family_diab_hist']]

# Save data
feature_df.to_csv(output_dir + 'FamilyHistory.csv', index=False)

# Record end time
end_time = time.time()
total_time = end_time - start_time
logger.info(f'Total time: {total_time:.3f} seconds')

'''
/home/ukb/data/resources/coding1010.tsv
https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=1010

coding	meaning
-27	None of the above (group 2)
-23	Prefer not to answer (group 2)
-21	Do not know (group 2)
-17	None of the above (group 1)
-13	Prefer not to answer (group 1)
-11	Do not know (group 1)
1	Heart disease <--------------------
2	Stroke
3	Lung cancer
4	Bowel cancer
5	Breast cancer
6	Chronic bronchitis/emphysema
8	High blood pressure
9	Diabetes
10	Alzheimer's disease/dementia
11	Parkinson's disease
12	Severe depression
13	Prostate cancer
14	Hip fracture
'''