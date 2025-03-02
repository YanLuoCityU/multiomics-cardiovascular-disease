import os
import time
import feather
import json
import pickle
import numpy as np
import pandas as pd
from os.path import join, exists
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

# Prepare UK Biobank Data for model training and evaluation
class UKBiobankData():
    def __init__(self, 
                 data_dir,
                 split_times=None,
                 val_size=0.1,
                 shuffle=True,
                 seed=1,
                 logger=None):
        self.data_dir = data_dir
        self.split_times = split_times
        self.val_size = val_size # Validation set size within each cross-validation split
        self.shuffle = shuffle
        self.seed = seed
        self.logger = logger
        
        # Load data and define predictors, outcomes, and mapping dictionaries
        self.load_data()
        self.dict_predictor = self.get_predictors_dict()
        self.outcomes_list = ['cad', 'stroke', 'hf', 'af', 'va', 'pad', 'aaa', 'vt', 'cvd_death', 'ar', 'mace', 'cvd', 'cved']
        self.outcomes_mapping = {
            'cad': 'Coronary artery disease',
            'stroke': 'Stroke',
            'hf': 'Heart failure',
            'af': 'Atrial fibrillation',
            'va': 'Ventricular arrhythmias',
            'pad': 'Peripheral artery diseases',
            'aaa': 'Abdominal aortic aneurysm',
            'vt': 'Vein thrombosis',
            'cvd_death': 'Cardiovascular death',
            'ar': 'Arrhythmias',
            'mace': 'Major adverse cardiovascular event',
            'cvd': 'Cardiovascular diseases',
            'cved': 'All cardiovascular endpoints'
        }
        self.predictor_set_mapping = {
            'ASCVD': self.dict_predictor['ASCVD'],
            'SCORE2': self.dict_predictor['SCORE2'],
            'AgeSex': ['age', 'male'],
            'Clinical': self.dict_predictor['clinical'],
            'PANEL': self.dict_predictor['PANEL'],
            'PANELBlood': self.dict_predictor['PANELBlood'],
            'Genomics': self.dict_predictor['genomics'],
            'Metabolomics': self.dict_predictor['metabolomics'],
            'Proteomics': self.dict_predictor['proteomics']
        }
        self.clinical_continuous_list = ['age', 'townsend', 'sbp', 'dbp', 'height', 'weight', 'waist_cir', 'waist_hip_ratio', 'bmi']
        self.blood_count_continuous_list = ['baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc']
        self.blood_biochem_continuous_list = ['apoA', 'apoB', 'total_cl', 'ldl_cl', 'hdl_cl', 'lpa', 'tg', 'glucose', 'hba1c', 'crt', 'cysc', 'urate', 'urea',
                                              'alb', 'tp', 'alt', 'ast', 'ggt', 'alp', 'dbil', 'tbil', 'crp', 'pho', 'ca', 'vd', 'igf1', 'shbg', 'trt']
        self.clinical_categorical_list = ['male', 'ethnicity', 'current_smoking', 'daily_drinking', 'healthy_sleep', 'physical_act', 'healthy_diet', 'social_active',
                                          'family_heart_hist', 'family_stroke_hist', 'family_hypt_hist', 'family_diab_hist',
                                          'diab_hist', 'hypt_hist', 'lipidlower', 'antihypt']
        self.continuous_predictor_mapping = {
            'ASCVD': ['age', 'sbp', 'total_cl', 'hdl_cl'],
            'SCORE2': ['age', 'sbp', 'total_cl', 'hdl_cl'],
            'AgeSex': ['age'],
            'Clinical': self.clinical_continuous_list,
            'PANEL': self.clinical_continuous_list + self.blood_count_continuous_list,
            'PANELBlood': self.clinical_continuous_list + self.blood_count_continuous_list + self.blood_biochem_continuous_list,
            'Genomics': self.dict_predictor['genomics'],
            'Metabolomics': self.dict_predictor['metabolomics'],
            'Proteomics': self.dict_predictor['proteomics']            
        }
        self.categorical_predictor_mapping = {
            'ASCVD': ['male', 'ethnicity', 'current_smoking', 'diab_hist', 'antihypt'],
            'SCORE2': ['male', 'current_smoking', 'diab_hist'],
            'AgeSex': ['male'],
            'Clinical': self.clinical_categorical_list,
            'PANEL': self.clinical_categorical_list,
            'PANELBlood': self.clinical_categorical_list,
            'Genomics': [],
            'Metabolomics': [],
            'Proteomics': []           
        }
        
        # Merge data and exclude participants
        self.merge_predictors()
        self.exclusion()
        self.combine_outcomes(outcomes_list=self.outcomes_list, outcomes_mapping=self.outcomes_mapping)
        
        # Split data
        self.train_valid_test_split(predictor_set_mapping=self.predictor_set_mapping, outcomes_list=self.outcomes_list, 
                                    val_size=self.val_size, shuffle=self.shuffle, seed=self.seed, split_times=self.split_times)
        
        # Process and save data
        self.split_seed_filename = f"split_seed-{self.seed}"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)

        if not exists(self.split_seed_dir):
            os.makedirs(self.split_seed_dir)
            
        if exists(join(self.split_seed_dir, f'X_test_AgeSex_K0.feather')) and \
           exists(join(self.split_seed_dir, f'y_test_K0.feather')) and \
           exists(join(self.split_seed_dir, f'e_test_K0.feather')):
            raise FileExistsError(f"Split datasets have existed in {self.split_seed_dir}. Program will be terminated.")
        else:
            self.logger.info(f'Saving split datasets into {self.split_seed_dir}.')
            self.process_save_data(split_seed_dir=self.split_seed_dir,continuous_predictor_mapping=self.continuous_predictor_mapping, categorical_predictor_mapping=self.categorical_predictor_mapping)
        
    ''' Load and Merge data '''
    # Load the processed predictors and cardiovascular outcomes
    def load_data(self):
        processed_data_dir = join(self.data_dir, 'processed/')
        self.logger.info(f'Loading processed predictors and cardiovascular outcomes from {processed_data_dir}.\n')
        
        self.demographic_df = pd.read_csv(processed_data_dir + 'covariates/DemographicInfo.csv', low_memory=False)
        self.lifestyle_df = pd.read_csv(processed_data_dir + 'covariates/Lifestyle.csv', low_memory=False)
        self.familyhist_df = pd.read_csv(processed_data_dir + 'covariates/FamilyHistory.csv', low_memory=False)
        self.phymeasure_df = pd.read_csv(processed_data_dir + 'covariates/PhysicalMeasurements.csv', low_memory=False)
        self.biofluids_df = pd.read_csv(processed_data_dir + 'covariates/Biofluids.csv', low_memory=False)
        self.medication_df = pd.read_csv(processed_data_dir + 'covariates/MedicationHistory.csv', low_memory=False)
        self.diseases_df = pd.read_csv(processed_data_dir + 'covariates/DiseaseHistory.csv', low_memory=False, usecols=['eid', 'hypt_hist', 'diab_hist', 'cvd_hist'])
        self.genomics_df = pd.read_csv(processed_data_dir + 'omics/PolygenicScores.csv', low_memory=False)
        self.metabolomics_df = pd.read_csv(processed_data_dir + 'omics/Metabolomics.csv', low_memory=False)
        self.proteomics_df = pd.read_csv(processed_data_dir + 'omics/Proteomics.csv', low_memory=False)
        self.outcomes_df = pd.read_csv(processed_data_dir + 'outcomes/Outcomes.csv', low_memory=False)
    
    # Get the predictors
    def get_predictors_dict(self):
        dict_predictor = {}
        dict_predictor['demographic'] = self.demographic_df.columns.tolist()
        dict_predictor['lifestyle'] = self.lifestyle_df.columns.tolist()
        dict_predictor['familyhist'] = self.familyhist_df.columns.tolist()
        dict_predictor['phymeasure'] = self.phymeasure_df.columns.tolist()
        dict_predictor['biofluids'] = self.biofluids_df.columns.tolist()
        dict_predictor['blood_count'] = ['baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc']
        dict_predictor['blood_biochem'] = ['baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc']
        dict_predictor['medication'] = self.medication_df.columns.tolist()
        dict_predictor['diseases'] = self.diseases_df.columns.tolist()
        dict_predictor['genomics'] = self.genomics_df.columns.tolist()
        dict_predictor['metabolomics'] = self.metabolomics_df.columns.tolist()
        dict_predictor['proteomics'] = self.proteomics_df.columns.tolist()
        for key in dict_predictor.keys():
            dict_predictor[key] = [ele for ele in dict_predictor[key] if ele not in ['eid', 'region', 'cvd_hist']]
        dict_predictor['clinical'] = dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + \
                                            dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases']
        dict_predictor['PANEL'] = dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + \
                                            dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases'] + dict_predictor['blood_count']
        dict_predictor['PANELBlood'] = dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + \
                                            dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases'] + dict_predictor['biofluids']
        dict_predictor['ASCVD'] = ['male', 'ethnicity', 'current_smoking', 'diab_hist', 'antihypt', 'age', 'sbp', 'total_cl', 'hdl_cl']
        dict_predictor['SCORE2'] = ['male', 'current_smoking', 'diab_hist', 'age', 'sbp', 'total_cl', 'hdl_cl']
        
        return dict_predictor
    
    # Merge the predictors
    def merge_predictors(self):
        self.logger.info('Merging the predictors.')
        # Participants withdrawn from the UK Biobank have been excluded in all datasets. 
        self.merge_df = pd.merge(self.demographic_df, self.lifestyle_df, on='eid', how='inner')
        self.logger.info(f'Merge demographic information and lifestyles: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.familyhist_df, on='eid', how='inner')
        self.logger.info(f'Merge with family history: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.phymeasure_df, on='eid', how='inner')
        self.logger.info(f'Merge with physical measurements: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.biofluids_df, on='eid', how='inner')
        self.logger.info(f'Merge with blood count and blood biochemistry: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.medication_df, on='eid', how='inner')
        self.logger.info(f'Merge with medications: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.diseases_df, on='eid', how='inner')
        self.logger.info(f'Merge with disease history: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.genomics_df, on='eid', how='inner') # Include 52 participants who have been withdrawn from the UK Biobank
        self.logger.info(f'Merge with genomics data: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.metabolomics_df, on='eid', how='inner')
        self.logger.info(f'Merge with metabolomics data: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.proteomics_df, on='eid', how='inner')
        self.logger.info(f'Merge with proteomics data: {len(self.merge_df["eid"].unique())}.\n')
        
        # Delete the original dataframes to free up memory
        del self.demographic_df
        del self.lifestyle_df
        del self.familyhist_df
        del self.phymeasure_df
        del self.biofluids_df
        del self.medication_df
        del self.diseases_df
        del self.genomics_df
        del self.metabolomics_df
        del self.proteomics_df       
    
    # Exclude participants with pre-existing cardiovascular diseases at baseline and missing >50% proteomics data
    def exclusion(self):
        self.merge_excluded_df = self.merge_df[self.merge_df['cvd_hist'] == 0]
        self.logger.info(f'Exclude participants with pre-existing cardiovascular diseases at baseline: {len(self.merge_excluded_df["eid"].unique())}.')
        missing_percentage = self.merge_excluded_df[self.dict_predictor['proteomics']].isnull().mean(axis=1) * 100
        self.merge_excluded_df = self.merge_excluded_df[missing_percentage <= 50]
        self.logger.info(f'Exclude participants missing <=50% proteomics data: {len(self.merge_excluded_df["eid"].unique())}.\n')
    
    # Combine the outcomes with the predictors
    def combine_outcomes(self, outcomes_list, outcomes_mapping):
        self.logger.info('Combining the outcomes with the predictors.')
        self.merge_excluded_df = self._combine_outcomes(self.merge_excluded_df, self.outcomes_df, outcomes_list, outcomes_mapping)
    
    # Combine the outcomes with the predictors
    def _combine_outcomes(self, predictors_df, outcomes_df, outcomes_list, mapping):
        selected_columns = ['eid']
        
        for outcome in outcomes_list:
            selected_columns.append(f'bl2{outcome}_yrs')
            selected_columns.append(outcome)
        
        outcomes_df_selected = outcomes_df[selected_columns]
        combined_df = pd.merge(predictors_df, outcomes_df_selected, on='eid', how='left')
        
        for outcome in outcomes_list:
            incident_cases = combined_df[outcome].sum()
            if outcome == 'cved':
                self.logger.info(f"The incident cases of {mapping[outcome]} is {incident_cases}.\n")
            else:
                self.logger.info(f"The incident cases of {mapping[outcome]} is {incident_cases}.")
        
        return combined_df
    
    # Get the merged data
    def get_merged_data(self, excluded=False):
        if excluded:
            return(self.merge_excluded_df)
        else:
            return(self.merge_df)
    

    ''' Nested cross-validation split '''
    ## Outer loop: 
    #       - split the dataset into K-fold training set and testing sets.
    #       - split the dataset by the geographic region (10 folds). One fold is used as a test set while the remaining folds are used as a training set each time (i.e., repeated by 10 times).
    ## Inner loop: split the training set into training and validation sets.
    def train_valid_test_split(self, predictor_set_mapping, outcomes_list, val_size, shuffle, seed, split_times=None):
        if split_times is not None:
            self.logger.info(f'Spliting the dataset into K-fold training and testing sets.\n')
            split_data = self._split_data(shuffle, seed)
            self.len_split_data = len(split_data)
        else:
            self.logger.info(f'Spliting the dataset into {len(self.merge_excluded_df["region"].unique())} folds by the geographic region.\n')
            split_data = self._split_data_by_region()
            self.len_split_data = len(split_data)
        
        self.X = {}
        self.y = {}
        self.e = {}
        
        self.logger.info(f'Spliting the training set into training and validation ({val_size*100}%) sets.\n')
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_split, i, data_split, outcomes_list, predictor_set_mapping, val_size, shuffle, seed) 
                    for i, data_split in split_data.items()]
            for future in as_completed(futures):
                i, X, y, e = future.result()
                self.X[i] = X
                self.y[i] = y
                self.e[i] = e

    def _split_data(self, shuffle, seed):
        split_data = {}
        kf = KFold(n_splits=self.split_times, shuffle=shuffle, random_state=seed)
        
        for i, (train_index, test_index) in enumerate(kf.split(self.merge_excluded_df)):
            train_data = self.merge_excluded_df.iloc[train_index]
            test_data = self.merge_excluded_df.iloc[test_index]
            split_data[i] = {'train': train_data, 'test': test_data}
            
        return split_data
    
    def _split_data_by_region(self):
        split_data = {}
        for region in range(10):
            train_data = self.merge_excluded_df[self.merge_excluded_df['region'] != region]
            test_data = self.merge_excluded_df[self.merge_excluded_df['region'] == region]
            split_data[region] = {'train': train_data, 'test': test_data}
        
        return split_data
            
    def _process_split(self, i, split_data, outcomes_list, predictor_set_mapping, val_size, shuffle, seed):
        X = {}
        y = {}
        e = {}
        
        # Outer loop test set
        X_test = split_data['test']
        y_test = split_data['test'][['eid'] + [f'bl2{outcome}_yrs' for outcome in outcomes_list]]
        y_test.rename(columns={col: col.replace('bl2', '').replace('_yrs', '') for col in y_test.columns if 'bl2' in col}, inplace=True)
        e_test = split_data['test'][['eid'] + [f'{outcome}' for outcome in outcomes_list]]
        
        # Inner loop training and validation sets
        X_train_val = split_data['train']
        X_train, X_val = train_test_split(X_train_val, test_size=val_size, shuffle=shuffle, random_state=seed)
        
        y_train = X_train[['eid'] + [f'bl2{outcome}_yrs' for outcome in outcomes_list]]
        y_train.rename(columns={col: col.replace('bl2', '').replace('_yrs', '') for col in y_train.columns if 'bl2' in col}, inplace=True)
        e_train = X_train[['eid'] + [f'{outcome}' for outcome in outcomes_list]]
        
        y_val = X_val[['eid'] + [f'bl2{outcome}_yrs' for outcome in outcomes_list]]
        y_val.rename(columns={col: col.replace('bl2', '').replace('_yrs', '') for col in y_val.columns if 'bl2' in col}, inplace=True)
        e_val = X_val[['eid'] + [f'{outcome}' for outcome in outcomes_list]]
        
        X =  {
            'ASCVD': {
                'X_test': X_test[['eid'] + predictor_set_mapping['ASCVD']],
                'X_train': X_train[['eid'] + predictor_set_mapping['ASCVD']],
                'X_val': X_val[['eid'] + predictor_set_mapping['ASCVD']]                    
            },
            'SCORE2': {
                'X_test': X_test[['eid'] + predictor_set_mapping['SCORE2']],
                'X_train': X_train[['eid'] + predictor_set_mapping['SCORE2']],
                'X_val': X_val[['eid'] + predictor_set_mapping['SCORE2']]                    
            },
            'AgeSex': {
                'X_test': X_test[['eid'] + predictor_set_mapping['AgeSex']],
                'X_train': X_train[['eid'] + predictor_set_mapping['AgeSex']],
                'X_val': X_val[['eid'] + predictor_set_mapping['AgeSex']]                    
            },
            'Clinical': {
                'X_test': X_test[['eid'] + predictor_set_mapping['Clinical']],
                'X_train': X_train[['eid'] + predictor_set_mapping['Clinical']],
                'X_val': X_val[['eid'] + predictor_set_mapping['Clinical']]                    
            },
            'PANEL': {
                'X_test': X_test[['eid'] + predictor_set_mapping['PANEL']],
                'X_train': X_train[['eid'] + predictor_set_mapping['PANEL']],
                'X_val': X_val[['eid'] + predictor_set_mapping['PANEL']]                    
            },
            'PANELBlood': {
                'X_test': X_test[['eid'] + predictor_set_mapping['PANELBlood']],
                'X_train': X_train[['eid'] + predictor_set_mapping['PANELBlood']],
                'X_val': X_val[['eid'] + predictor_set_mapping['PANELBlood']]                    
            },
            'Genomics': {
                'X_test': X_test[['eid'] + predictor_set_mapping['Genomics']],
                'X_train': X_train[['eid'] + predictor_set_mapping['Genomics']],
                'X_val': X_val[['eid'] + predictor_set_mapping['Genomics']]                    
            },
            'Metabolomics': {
                'X_test': X_test[['eid'] + predictor_set_mapping['Metabolomics']],
                'X_train': X_train[['eid'] + predictor_set_mapping['Metabolomics']],
                'X_val': X_val[['eid'] + predictor_set_mapping['Metabolomics']]                    
            },
            'Proteomics': {
                'X_test': X_test[['eid'] + predictor_set_mapping['Proteomics']],
                'X_train': X_train[['eid'] + predictor_set_mapping['Proteomics']],
                'X_val': X_val[['eid'] + predictor_set_mapping['Proteomics']]                    
            }
        }
        y = {
            'y_test': y_test,
            'y_train': y_train,
            'y_val': y_val
        }
        e = {
            'e_test': e_test,
            'e_train': e_train,
            'e_val': e_val
        }
        
        return i, X, y, e
    
    
    ''' Process and Save data '''
    def process_save_data(self, continuous_predictor_mapping, categorical_predictor_mapping, split_seed_dir):
        start_time = time.time()
        self.logger.info('Processing (imputing missing values and transforming predictors) and saving data.')

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._impute_transform_save, i, self.X[i], continuous_predictor_mapping, categorical_predictor_mapping, split_seed_dir) 
                    for i in range(self.len_split_data)]
            for future in as_completed(futures):
                future.result()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Data processing and saving completed in {elapsed_time:.2f} seconds.\n')

    def _impute_transform_save(self, i, X, continuous_predictor_mapping, categorical_predictor_mapping, split_seed_dir):
        X_processed = {}

        for predictor_set, datasets in X.items():
            X_train = datasets['X_train']
            X_test = datasets['X_test']
            X_val = datasets['X_val']

            # Imputation
            imputer = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5))]), continuous_predictor_mapping[predictor_set]),
                    # ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), continuous_predictor_mapping[predictor_set]),
                    ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))]), categorical_predictor_mapping[predictor_set])
                ],
                remainder='passthrough'
            )

            X_train_imputed = imputer.fit_transform(X_train)
            X_val_imputed = imputer.transform(X_val)
            X_test_imputed = imputer.transform(X_test)

            feature_names = continuous_predictor_mapping[predictor_set] + categorical_predictor_mapping[predictor_set] + ['eid']
            X_train_imputed = pd.DataFrame(X_train_imputed, columns=feature_names)
            X_val_imputed = pd.DataFrame(X_val_imputed, columns=feature_names)
            X_test_imputed = pd.DataFrame(X_test_imputed, columns=feature_names)
            

            # Transformation
            if predictor_set == 'ASCVD': 
                X_train_processed = self._ascvd_variable_transform(X_train_imputed)
                X_val_processed = self._ascvd_variable_transform(X_val_imputed)
                X_test_processed = self._ascvd_variable_transform(X_test_imputed)
            elif predictor_set == 'SCORE2':
                X_train_processed = self._score2_variable_transform(X_train_imputed)
                X_val_processed = self._score2_variable_transform(X_val_imputed)
                X_test_processed = self._score2_variable_transform(X_test_imputed)
            else:
                if predictor_set in ['Genomics', 'Metabolomics', 'Proteomics']:
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=[('scaler', StandardScaler())]), continuous_predictor_mapping[predictor_set])
                        ],
                        remainder='passthrough'
                    )
                else:
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline(steps=[('scaler', StandardScaler())]), continuous_predictor_mapping[predictor_set]),
                            ('cat', Pipeline(steps=[('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))]), categorical_predictor_mapping[predictor_set])
                        ],
                        remainder='passthrough'
                    )
                
                X_train_processed = preprocessor.fit_transform(pd.DataFrame(X_train_imputed))
                X_val_processed = preprocessor.transform(pd.DataFrame(X_val_imputed))
                X_test_processed = preprocessor.transform(pd.DataFrame(X_test_imputed))
                    
                num_feature_names = continuous_predictor_mapping[predictor_set]
                cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_predictor_mapping[predictor_set]) if 'cat' in preprocessor.named_transformers_ else []
                feature_names = list(num_feature_names) + list(cat_feature_names) + ['eid']

                if predictor_set in ['PANEL', 'PANELBlood', 'Genomics', 'Metabolomics', 'Proteomics']:
                    # Extract mean and variance for the continuous predictors
                    scaler = preprocessor.named_transformers_['num']['scaler']
                    mean_ = scaler.mean_
                    var_ = scaler.var_

                    mean_var_dict = {
                        'variables': num_feature_names,
                        'mean': mean_,
                        'var': var_
                    }
    
                    # Save the mean and variance to files (or pickled objects)
                    with open(f"{split_seed_dir}/mean_var_{predictor_set}_K{i}.pkl", 'wb') as f:
                        pickle.dump(mean_var_dict, f)
                        
                X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
                X_val_processed = pd.DataFrame(X_val_processed, columns=feature_names)
                X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
                
                X_train_processed['eid'] = X_train_processed['eid'].astype(int)
                X_val_processed['eid'] = X_val_processed['eid'].astype(int)
                X_test_processed['eid'] = X_test_processed['eid'].astype(int)

            X_processed[predictor_set] = {
                'X_train': X_train_processed,
                'X_val': X_val_processed,
                'X_test': X_test_processed
            }

        # Save the processed data
        self.logger.info(f'Saving split datasets {i + 1}/{self.len_split_data}.')
        self.__save_processed_data(i, X_processed, split_seed_dir)
    
    def __save_processed_data(self, i, X_processed, split_seed_dir):
        # Save the processed predictors
        for predictor_set, datasets in X_processed.items():
            X_test_filename = f"X_test_{predictor_set}_K{i}.feather"
            X_train_filename = f"X_train_{predictor_set}_K{i}.feather"
            X_val_filename = f"X_val_{predictor_set}_K{i}.feather"
            
            feather.write_dataframe(datasets['X_test'], join(split_seed_dir, X_test_filename))
            feather.write_dataframe(datasets['X_train'], join(split_seed_dir, X_train_filename))
            feather.write_dataframe(datasets['X_val'], join(split_seed_dir, X_val_filename))
        
        # Save the event indicator for all outcomes
        y_train_filename = f"y_train_K{i}.feather"
        y_val_filename = f"y_val_K{i}.feather"
        y_test_filename = f"y_test_K{i}.feather"
            
        feather.write_dataframe(self.y[i]['y_train'], join(split_seed_dir, y_train_filename))
        feather.write_dataframe(self.y[i]['y_val'], join(split_seed_dir, y_val_filename))
        feather.write_dataframe(self.y[i]['y_test'], join(split_seed_dir, y_test_filename))
        
        # Save the survival time for all outcomes
        e_train_filename = f"e_train_K{i}.feather"
        e_val_filename = f"e_val_K{i}.feather"
        e_test_filename = f"e_test_K{i}.feather"
        
        feather.write_dataframe(self.e[i]['e_train'], join(split_seed_dir, e_train_filename))
        feather.write_dataframe(self.e[i]['e_val'], join(split_seed_dir, e_val_filename))
        feather.write_dataframe(self.e[i]['e_test'], join(split_seed_dir, e_test_filename))

    # Transform the predictors for ASCVD
    def _ascvd_variable_transform(self, df):
        df = df.copy()

        # Redefine variables
        df['black'] = df['ethnicity'].apply(lambda x: 1 if x == 3 else 0)
        df['totalcl_hdlcl'] = df['total_cl'] / df['hdl_cl']
        df.drop(columns=['ethnicity', 'total_cl', 'hdl_cl'], inplace=True)
        
        df['sbp2'] = df['sbp'] ** 2
        df['sbp_antihypt'] = df['sbp'] * df['antihypt']
        
        # Interaction terms with 'black'
        for column in ['age', 'sbp', 'antihypt', 'diab_hist', 'current_smoking', 'totalcl_hdlcl']:
            if column not in ['black']:
                df[f'black_{column}'] = df['black'] * df[column]
        df['black_sbp_antihypt'] = df['black'] * df['sbp_antihypt']
        df['black_age_sbp'] = df['black'] * df['age'] * df['sbp']

        # Interaction terms with 'age'
        df['age_sbp'] = df['age'] * df['sbp']
        
        return df
    
    # Transform the predictors for SCORE2
    def _score2_variable_transform(self, df):
        df = df.copy()
        df['age'] = (df['age'] - 60)/5
        df['sbp'] = (df['sbp'] - 120)/20
        df['total_cl'] = (df['total_cl'] - 6)/1
        df['hdl_cl'] = (df['hdl_cl'] - 1.3)/0.5
        for column in df.columns:
            if column not in ['age']:
                df[f'age_{column}'] = df['age'] * df[column]
            
        return df


# Prepare UK Biobank Data for description statistics  
class UKBiobankDataMerge():
    def __init__(self, data_dir, logger=None):
        self.data_dir = data_dir
        self.logger = logger
        
        # Load data and define predictors, outcomes, and mapping dictionaries
        self.load_data()
        self.dict_predictor = self.get_predictors_dict()
        self.outcomes_list = ['cad', 'stroke', 'hf', 'af', 'va', 'pad', 'aaa', 'vt', 'cvd_death', 'ar', 'mace', 'cvd', 'cved']
        self.outcomes_mapping = {
            'cad': 'Coronary artery disease',
            'stroke': 'Stroke',
            'hf': 'Heart failure',
            'af': 'Atrial fibrillation',
            'va': 'Ventricular arrhythmias',
            'pad': 'Peripheral artery diseases',
            'aaa': 'Abdominal aortic aneurysm',
            'vt': 'Vein thrombosis',
            'cvd_death': 'Cardiovascular death',
            'ar': 'Arrhythmias',
            'mace': 'Major adverse cardiovascular event',
            'cvd': 'Cardiovascular diseases',
            'cved': 'All cardiovascular endpoints'
        }

        # Merge data and exclude participants
        self.merge_predictors()
        self.exclusion()
        self.combine_outcomes(outcomes_list=self.outcomes_list, outcomes_mapping=self.outcomes_mapping)
        
        
    ''' Load and Merge data '''
    # Load the processed predictors and cardiovascular outcomes
    def load_data(self):
        processed_data_dir = join(self.data_dir, 'processed/')
        self.logger.info(f'Loading processed predictors and cardiovascular outcomes from {processed_data_dir}.\n')

        self.demographic_df = pd.read_csv(processed_data_dir + 'covariates/DemographicInfo.csv', low_memory=False)
        self.lifestyle_df = pd.read_csv(processed_data_dir + 'covariates/Lifestyle.csv', low_memory=False)
        self.familyhist_df = pd.read_csv(processed_data_dir + 'covariates/FamilyHistory.csv', low_memory=False)
        self.phymeasure_df = pd.read_csv(processed_data_dir + 'covariates/PhysicalMeasurements.csv', low_memory=False)
        self.biofluids_df = pd.read_csv(processed_data_dir + 'covariates/Biofluids.csv', low_memory=False)
        self.medication_df = pd.read_csv(processed_data_dir + 'covariates/MedicationHistory.csv', low_memory=False)
        self.diseases_df = pd.read_csv(processed_data_dir + 'covariates/DiseaseHistory.csv', low_memory=False)
        self.genomics_df = pd.read_csv(processed_data_dir + 'omics/PolygenicScores.csv', low_memory=False)
        self.metabolomics_df = pd.read_csv(processed_data_dir + 'omics/Metabolomics.csv', low_memory=False)
        self.proteomics_df = pd.read_csv(processed_data_dir + 'omics/Proteomics.csv', low_memory=False)
        self.outcomes_df = pd.read_csv(processed_data_dir + 'outcomes/Outcomes.csv', low_memory=False)
                                          
    # Get the predictors
    def get_predictors_dict(self):
        dict_predictor = {}
        dict_predictor['demographic'] = self.demographic_df.columns.tolist()
        dict_predictor['lifestyle'] = self.lifestyle_df.columns.tolist()
        dict_predictor['familyhist'] = self.familyhist_df.columns.tolist()
        dict_predictor['phymeasure'] = self.phymeasure_df.columns.tolist()
        dict_predictor['biofluids'] = self.biofluids_df.columns.tolist()
        dict_predictor['blood_count'] = ['baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc']
        dict_predictor['blood_biochem'] = ['baso', 'eos', 'hct', 'hb', 'lc', 'mc', 'nc', 'plt', 'wbc']
        dict_predictor['medication'] = self.medication_df.columns.tolist()
        # dict_predictor['diseases'] = self.diseases_df.columns.tolist()
        dict_predictor['diseases'] = ['hypt_hist', 'diab_hist']
        dict_predictor['genomics'] = self.genomics_df.columns.tolist()
        dict_predictor['metabolomics'] = self.metabolomics_df.columns.tolist()
        dict_predictor['proteomics'] = self.proteomics_df.columns.tolist()
        for key in dict_predictor.keys():
            dict_predictor[key] = [ele for ele in dict_predictor[key] if ele not in ['eid', 'region', 'cvd_hist']]
        dict_predictor['clinical'] = dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + \
                                            dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases']
        dict_predictor['PANEL'] = dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + \
                                            dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases'] + dict_predictor['blood_count']
        dict_predictor['PANELBlood'] = dict_predictor['demographic'] + dict_predictor['lifestyle'] + dict_predictor['familyhist'] + \
                                            dict_predictor['phymeasure'] + dict_predictor['medication'] + dict_predictor['diseases'] + dict_predictor['biofluids']
        dict_predictor['ASCVD'] = ['male', 'ethnicity', 'current_smoking', 'diab_hist', 'antihypt', 'age', 'sbp', 'total_cl', 'hdl_cl']
        dict_predictor['SCORE2'] = ['male', 'current_smoking', 'diab_hist', 'age', 'sbp', 'total_cl', 'hdl_cl']
        
        return dict_predictor
    
    # Merge the predictors
    def merge_predictors(self):
        self.logger.info('Merging the predictors.')
        self.merge_df = pd.merge(self.demographic_df, self.lifestyle_df, on='eid', how='inner')
        self.logger.info(f'Merge demographic information and lifestyles: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.familyhist_df, on='eid', how='inner')
        self.logger.info(f'Merge with family history: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.phymeasure_df, on='eid', how='inner')
        self.logger.info(f'Merge with physical measurements: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.biofluids_df, on='eid', how='inner')
        self.logger.info(f'Merge with blood count and blood biochemistry: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.medication_df, on='eid', how='inner')
        self.logger.info(f'Merge with medications: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.diseases_df, on='eid', how='inner')
        self.logger.info(f'Merge with disease history: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.genomics_df, on='eid', how='inner')
        self.logger.info(f'Merge with genomics data: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.metabolomics_df, on='eid', how='inner')
        self.logger.info(f'Merge with metabolomics data: {len(self.merge_df["eid"].unique())}.')
        self.merge_df = pd.merge(self.merge_df, self.proteomics_df, on='eid', how='inner')
        self.logger.info(f'Merge with proteomics data: {len(self.merge_df["eid"].unique())}.\n')   
    
    # Exclude participants with pre-existing cardiovascular diseases at baseline and missing >50% proteomics data
    def exclusion(self):
        self.merge_excluded_df = self.merge_df[self.merge_df['cvd_hist'] == 0]
        self.logger.info(f'Exclude participants with pre-existing cardiovascular diseases at baseline: {len(self.merge_excluded_df["eid"].unique())}.')
        missing_percentage = self.merge_excluded_df[self.dict_predictor['proteomics']].isnull().mean(axis=1) * 100
        self.merge_excluded_df = self.merge_excluded_df[missing_percentage <= 50]
        self.logger.info(f'Exclude participants missing <=50% proteomic data: {len(self.merge_excluded_df["eid"].unique())}.\n')
        
    # Combine the outcomes with the predictors
    def combine_outcomes(self, outcomes_list, outcomes_mapping):
        self.logger.info('Combining the outcomes with the predictors.')
        self.merge_excluded_df, self.outcomes_columns = self._combine_outcomes(self.merge_excluded_df, self.outcomes_df, outcomes_list, outcomes_mapping)
    
    # Combine the outcomes with the predictors
    def _combine_outcomes(self, predictors_df, outcomes_df, outcomes_list, mapping):
        selected_columns = ['eid']
        
        for outcome in outcomes_list:
            selected_columns.append(f'bl2{outcome}_yrs')
            selected_columns.append(outcome)
        
        outcomes_df_selected = outcomes_df[selected_columns]
        combined_df = pd.merge(predictors_df, outcomes_df_selected, on='eid', how='left')
        
        for outcome in outcomes_list:
            incident_cases = combined_df[outcome].sum()
            if outcome == 'cved':
                self.logger.info(f"The incident cases of {mapping[outcome]} is {incident_cases}.\n")
            else:
                self.logger.info(f"The incident cases of {mapping[outcome]} is {incident_cases}.")
        
        return combined_df, selected_columns
    
    # Get the merged data
    def get_merged_data(self, excluded=False):
        if excluded:
            return(self.merge_excluded_df)
        else:
            return(self.merge_df)


# Function to load a module from a given path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the util module
util_path = '/your path/multiomics-cardiovascular-disease/utils/util.py'
util_module = load_module_from_path('util', util_path)
setup_logger = util_module.setup_logger

if __name__ == '__main__':
    data_dir = '/your path/multiomics-cardiovascular-disease/data/'
    log_dir = '/your path/multiomics-cardiovascular-disease/saved/log'
    
    split_times = 10
    val_size = 0.1
    shuffle = True
    seed = 241104
    
    log_filename = os.path.join(log_dir, 'DataPreparation/UKBiobankData.log')
    logger = setup_logger(log_filename)
    
    logger.info('------------------Preparing UKBiobank Datasets with different predictor sets--------------------\n')
    data = UKBiobankData(data_dir=data_dir, split_times=split_times, val_size=val_size, shuffle=shuffle, seed=seed, logger=logger)
    logger.info('------------------All UKBiobank Datasets are prepared--------------------')

# if __name__ == '__main__':
#     data_dir = '/your path/multiomics-cardiovascular-disease/data/'
#     log_dir = '/your path/multiomics-cardiovascular-disease/saved/log'
    
#     log_filename = os.path.join(log_dir, 'DataPreparation/UKBiobankDataMerge.log')
#     logger = setup_logger(log_filename)
    
#     logger.info('------------------Preparing UKBiobank Datasets with different predictor sets--------------------\n')
#     data = UKBiobankDataMerge(data_dir=data_dir, logger=logger)
#     logger.info('------------------All UKBiobank Datasets are prepared--------------------')
#     ukb_df = data.get_merged_data(excluded=True)
#     ukb_df.to_csv('/your path/multiomics-cardiovascular-disease/data/processed/ukb_merged.csv', index=False)
#     ukb_dict = data.dict_predictor
#     output_filename = '/your path/multiomics-cardiovascular-disease/data/processed/ukb_dict.json'
#     with open(output_filename, 'w') as json_file:
#         json.dump(ukb_dict, json_file, indent=4)