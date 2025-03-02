import time
import numpy as np
import pandas as pd
from os.path import join
from lifelines.utils import concordance_index

class ASCVD():
    def __init__(self, 
                 data_dir,
                 results_dir,
                 split_times,
                 seed_to_split,
                 logger):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.split_times = split_times
        self.seed_to_split = seed_to_split
        self.logger = logger
        
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        
        self.nested_cross_validation()
        
    def nested_cross_validation(self):
        self.logger.info('Nested cross validation for ASCVD algorithm.')
        start_time = time.time()
        
        self.outcomes_mapping = {
            'mace': 'Major adverse cardiovascular events',
            'cad': 'Coronary artery diseases',
            'stroke': 'Stroke',
            'hf': 'Heart failure',
            'af': 'Atrial fibrillation',
            'va': 'Ventricular arrhythmias',
            'pad': 'Peripheral artery diseases',
            'aaa': 'Abdominal aortic aneurysm',
            'vt': 'Vein thrombosis',
            'cvd_death': 'Cardiovascular death'
        }
        
        outcomes_list = ['mace', 'cad', 'stroke', 'hf', 'af', 'va', 'pad', 'aaa', 'vt', 'cvd_death']
        results = []
        for outcome in outcomes_list:
            self.logger.info('\n')
            self.logger.info(f'Processing outcome: {self.outcomes_mapping[outcome]}')
            for fold in range(self.split_times):
                # Read data
                X_test = pd.read_feather(self.split_seed_dir + f'X_test_ASCVD_K{fold}.feather')
                y_test = pd.read_feather(self.split_seed_dir + f'y_test_K{fold}.feather')
                y_test = y_test[outcome]
                e_test = pd.read_feather(self.split_seed_dir + f'e_test_K{fold}.feather')
                e_test = e_test[outcome]
                num_cases = e_test.sum()
                
                # Calculate ASCVD score
                X_test_ascvd = self._ascvd_calculation(X_test, type='absolute_risk')
                
                # Calculate C-index
                test_cindex = 1 - concordance_index(event_times=y_test.values, predicted_scores=X_test_ascvd['ascvd'].values, event_observed=e_test.values)
                self.logger.info(f'Fold {fold}: Incident cases: {num_cases}, Test Shape: {X_test.shape}, Test C-index: {test_cindex}')
                
                results.append([outcome, fold, test_cindex])
        
        os.makedirs(self.results_dir, exist_ok=True)
        self.test_cindex_df = pd.DataFrame(results, columns=['outcome', 'fold', 'cindex'])
        self.test_cindex_df.to_csv(self.results_dir + 'ASCVD_test_cindex.csv', index=False)
           
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Nested cross validation for ASCVD algorithm completed in {elapsed_time:.2f} seconds.\n')
        
    def _ascvd_calculation(self, df, type):
        '''
        
        df is the dataframe with tranformed variables. 
        type could be "linear_predictor" or "absolute_risk". 
            If type is linear_predictor, the output is the Xβ. 
            If type is absolute_risk, the output is the 10-year CVD risk estimated by 1/(1+exp(linear_predictor)).
        
        Coefficients are from the following paper:
        
        Yadlowsky S, Hayward RA, Sussman JB, McClelland RL, Min YI, Basu S. 
        Clinical Implications of Revised Pooled Cohort Equations for Estimating Atherosclerotic Cardiovascular Disease Risk. 
        Ann Intern Med. 2018;169(1):20-29. doi:10.7326/M17-3011
        
        Appendix Table. Example Calculation for Model Set 2, the Proposed Revision of the PCEs for Estimating ASCVD Risk
        https://www.msdmanuals.com/professional/multimedia/clinical-calculator/cardiovascular-risk-assessment-10-year-revised-pooled-cohort-equations-2018
        
        '''
        df = df.copy()

        coefficients = {
            'male': {
                'age': 0.064200, 'sbp': 0.038950 , 'current_smoking': 0.895589, 'diab_hist': 0.842209, 'antihypt': 2.055533, 'black': 0.482835,
                'totalcl_hdlcl': 0.193307, 'sbp2': -0.000061, 'sbp_antihypt': -0.014207, 'black_age': 0, 'black_sbp': 0.011609,
                'black_antihypt': 0.119460, 'black_diab_hist': -0.077214, 'black_current_smoking': -0.226771,
                'black_totalcl_hdlcl': -0.117749, 'black_sbp_antihypt': 0.004190, 'black_age_sbp': -0.000199, 'age_sbp': 0.000025
            },
            'female': {
                'age': 0.106501, 'sbp': 0.017666, 'current_smoking': 1.009790, 'diab_hist': 0.943970, 'antihypt': 0.731678, 'black': 0.432440,
                'totalcl_hdlcl': 0.151318, 'sbp2': 0.000056, 'sbp_antihypt': -0.003647, 'black_age': -0.008580, 'black_sbp': 0.006208,
                'black_antihypt': 0.152968, 'black_diab_hist': 0.115232, 'black_current_smoking': -0.092231,
                'black_totalcl_hdlcl': 0.070498, 'black_sbp_antihypt': -0.000173, 'black_age_sbp': -0.000094, 'age_sbp': -0.000153
            }
        }
        
        male_coefficients = coefficients['male']
        female_coefficients = coefficients['female']
        
        male_indices = df[df['male']==1].index
        female_indices = df[df['male']==0].index
        
        df['ascvd'] = 0.0
        for feature, male_coef in male_coefficients.items():
            df.loc[male_indices, 'ascvd'] += male_coef * df.loc[male_indices, feature]
        df.loc[male_indices, 'ascvd'] = df.loc[male_indices, 'ascvd'] - 11.679980
        
        for feature, female_coef in female_coefficients.items():
            df.loc[female_indices, 'ascvd'] += female_coef * df.loc[female_indices, feature]
        df.loc[female_indices, 'ascvd'] = df.loc[female_indices, 'ascvd'] - 12.823110
        
        if type == 'linear_predictor':
            return df
        elif type == 'absolute_risk':
            df['ascvd'] = 1/(1 + np.exp(-df['ascvd']))
            return df


class SCORE2():
    def __init__(self, 
                 data_dir,
                 results_dir,
                 split_times,
                 seed_to_split,
                 logger):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.split_times = split_times
        self.seed_to_split = seed_to_split
        self.logger = logger
        
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        
        self.nested_cross_validation()
    
    def nested_cross_validation(self):
        self.logger.info('Nested cross validation for SCORE2 algorithm.')
        start_time = time.time()
        
        self.outcomes_mapping = {
            'mace': 'Major adverse cardiovascular events',
            'cad': 'Coronary artery diseases',
            'stroke': 'Stroke',
            'hf': 'Heart failure',
            'af': 'Atrial fibrillation',
            'va': 'Ventricular arrhythmias',
            'pad': 'Peripheral artery diseases',
            'aaa': 'Abdominal aortic aneurysm',
            'vt': 'Vein thrombosis',
            'cvd_death': 'Cardiovascular death'
        }
        
        outcomes_list = ['mace', 'cad', 'stroke', 'hf', 'af', 'va', 'pad', 'aaa', 'vt', 'cvd_death']
        results = []
        for outcome in outcomes_list:
            self.logger.info('\n')
            self.logger.info(f'Processing outcome: {self.outcomes_mapping[outcome]}')
            for fold in range(self.split_times):
                # Read data
                X_test = pd.read_feather(self.split_seed_dir + f'X_test_SCORE2_K{fold}.feather')
                y_test = pd.read_feather(self.split_seed_dir + f'y_test_K{fold}.feather')
                y_test = y_test[outcome]
                e_test = pd.read_feather(self.split_seed_dir + f'e_test_K{fold}.feather')
                e_test = e_test[outcome]
                num_cases = e_test.sum()
                
                # Calculate SCORE2 score
                X_test_score2 = self._score2_calculation(X_test, type='absolute_risk')
                
                # Calculate C-index
                test_cindex = 1 - concordance_index(event_times=y_test.values, predicted_scores=X_test_score2['score2'].values, event_observed=e_test.values)
                self.logger.info(f'Fold {fold}: Incident cases: {num_cases}, Test Shape: {X_test.shape}, Test C-index: {test_cindex}')
                
                results.append([outcome, fold, test_cindex])
        
        os.makedirs(self.results_dir, exist_ok=True)
        self.test_cindex_df = pd.DataFrame(results, columns=['outcome', 'fold', 'cindex'])
        self.test_cindex_df.to_csv(self.results_dir + 'SCORE2_test_cindex.csv', index=False)
           
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Nested cross validation for SCORE2 algorithm completed in {elapsed_time:.2f} seconds.\n')
        
    def _score2_calculation(self, df, type):
        '''
        
        df is the dataframe with tranformed variables. 
        male_df is the dataframe indicating whether the participant is male.
        type could be "linear_predictor" or "absolute_risk". 
            If type is linear_predictor, the output is the Xβ. 
            If type is absolute_risk, the output is the 10-year CVD risk estimated by 1-(base_survival)^(exp(linear_predictor)).
        
        Coefficients are from the following paper:
        
        SCORE2 working group and ESC Cardiovascular risk collaboration. 
        SCORE2 risk prediction algorithms: new models to estimate 10-year risk of cardiovascular disease in Europe. 
        Eur Heart J. 2021;42(25):2439-2454. doi:10.1093/eurheartj/ehab309
        
        Supplementary methods Table 2: Model coefficients and baseline survival of the SCORE2 algorithm
        Supplementary Table 7: Summary of subdistribution hazard ratios for predictor variables in the SCORE2 risk models
        
        '''
        df = df.copy()

        coefficients = {
            'male': {
                'age': 0.3742, 'current_smoking': 0.6012, 'sbp': 0.2777, 'diab_hist': 0.6457, 
                'total_cl': 0.1458, 'hdl_cl': -0.2698, 'age_current_smoking': -0.0755, 
                'age_sbp': -0.0255, 'age_total_cl': -0.0281, 'age_hdl_cl': 0.0426, 
                'age_diab_hist': -0.0983
            },
            'female': {
                'age': 0.4648, 'current_smoking': 0.7744, 'sbp': 0.3131, 'diab_hist': 0.8096, 
                'total_cl': 0.1002, 'hdl_cl': -0.2606, 'age_current_smoking': -0.1088, 
                'age_sbp': -0.0277, 'age_total_cl': -0.0226, 'age_hdl_cl': 0.0613, 
                'age_diab_hist': -0.1272
            }
        }
        
        male_coefficients = coefficients['male']
        female_coefficients = coefficients['female']
        
        male_indices = df[df['male']==1].index
        female_indices = df[df['male']==0].index
        
        df['score2'] = 0.0
        for feature, male_coef in male_coefficients.items():
            df.loc[male_indices, 'score2'] += male_coef * df.loc[male_indices, feature]

        for feature, female_coef in female_coefficients.items():
            df.loc[female_indices, 'score2'] += female_coef * df.loc[female_indices, feature]
        
        if type == 'linear_predictor':
            return df
        else:
            male_base = 0.9605
            female_base = 0.9776
            df.loc[male_indices, 'score2'] = 1 - (male_base ** np.exp(df.loc[male_indices, 'score2']))
            df.loc[female_indices, 'score2'] = 1 - (female_base ** np.exp(df.loc[female_indices, 'score2']))
            return df
        

import os
import importlib.util

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
    # Define directories and parameters
    data_dir = '/your path/multiomics-cardiovascular-disease/data/'
    results_dir = '/your path/multiomics-cardiovascular-disease/saved/results/Cindex/All/'
    log_dir = '/your path/multiomics-cardiovascular-disease/saved/log'
    
    #---------------------------- ASCVD ----------------------------
    ascvd_log_filename = os.path.join(log_dir, "Model/ClinicalScores/ASCVD.log")
    ascvd_logger = setup_logger(ascvd_log_filename)
    ASCVD(data_dir=data_dir, results_dir=results_dir, split_times=10, seed_to_split=241104, logger=ascvd_logger)
    
    #---------------------------- SCORE2 ----------------------------
    score2_log_filename = os.path.join(log_dir, "Model/ClinicalScores/SCORE2.log")
    score2_logger = setup_logger(score2_log_filename)
    SCORE2(data_dir=data_dir, results_dir=results_dir, split_times=10, seed_to_split=241104, logger=score2_logger)