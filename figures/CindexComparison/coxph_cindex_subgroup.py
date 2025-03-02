import os
import time
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from concurrent.futures import ProcessPoolExecutor, as_completed
from tabulate import tabulate
import importlib.util
from sklearn.preprocessing import StandardScaler

class CoxPHSubgroup():
    def __init__(self, 
                 data_dir,
                 results_dir,
                 split_times,
                 seed_to_split,
                 predictor_set,
                 subgroup,
                 num_workers,
                 logger):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.split_times = split_times
        self.seed_to_split = seed_to_split
        self.predictor_set = predictor_set
        self.subgroup = subgroup
        self.num_workers = num_workers
        self.logger = logger
        
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = os.path.join(self.data_dir, self.split_seed_filename)
        
        self.nested_cross_validation()

    def nested_cross_validation(self):
        self.logger.info('Nested cross validation for CoxPH models.')
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
        
        outcomes_list = ['cad', 'stroke', 'hf', 'af', 'va', 'aaa', 'pad', 'vt', 'cvd_death']
        results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for outcome in outcomes_list:
                for fold in range(self.split_times):
                    futures.append(executor.submit(self._process_fold, outcome, fold, self.subgroup))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing fold: {e}")

        self.test_cindex_df = pd.DataFrame(results, columns=['outcome', 'fold', 'cindex'])
        
        subgroup_mapping = {
            '<60': 'Age_under_60',
            '>=60': 'Age_over_60',
            'male': 'Male',
            'female': 'Female',
            'white': 'White'
        }
        subgroup_folder = subgroup_mapping.get(self.subgroup, 'Unknown')
        output_dir = os.path.join(self.results_dir, 'Cindex', subgroup_folder)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{'_'.join(self.predictor_set)} (CoxPH)_test_cindex.csv"
        self.test_cindex_df.to_csv(os.path.join(output_dir, output_filename), index=False)
        # output_filename = f"Cindex/{subgroup_folder}/{'_'.join(self.predictor_set)} (CoxPH)_test_cindex.csv"
        # os.makedirs(self.results_dir, exist_ok=True)
        # self.test_cindex_df.to_csv(os.path.join(self.results_dir, output_filename), index=False)
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Nested cross validation for CoxPH models completed in {elapsed_time:.2f} seconds.\n')
        
    def _process_fold(self, outcome, fold, subgroup):
        fold_start_time = time.time()
        
        # Read adn merge train data
        prefix = 'train'
        X_train = self.__load_predictors(prefix, outcome, fold, subgroup)
        y_train = pd.read_feather(os.path.join(self.split_seed_dir + f"y_{prefix}_K{fold}.feather"))
        y_train = y_train[['eid'] + [outcome]]
        e_train = pd.read_feather(os.path.join(self.split_seed_dir + f"e_{prefix}_K{fold}.feather"))
        e_train = e_train[['eid'] + [outcome]]

        X_train = pd.merge(X_train, y_train, on='eid', how='left')
        X_train.rename(columns={outcome: 'duration'}, inplace=True)
        X_train = pd.merge(X_train, e_train, on='eid', how='left')
        X_train.rename(columns={outcome: 'event'}, inplace=True)
        X_train = X_train.drop(columns=['eid'])
        
        # Fit the CoxPH model
        cox = CoxPHFitter(penalizer=0.03)
        cox.fit(X_train, duration_col='duration', event_col='event')
        
        # Read adn merge test data
        prefix = 'test'
        X_test = self.__load_predictors(prefix, outcome, fold, subgroup)
        y_test = pd.read_feather(os.path.join(self.split_seed_dir + f"y_{prefix}_K{fold}.feather"))
        y_test = y_test[['eid'] + [outcome]]
        e_test = pd.read_feather(os.path.join(self.split_seed_dir + f"e_{prefix}_K{fold}.feather"))
        e_test = e_test[['eid'] + [outcome]]

        X_test = pd.merge(X_test, y_test, on='eid', how='left')
        X_test.rename(columns={outcome: 'duration'}, inplace=True)
        X_test = pd.merge(X_test, e_test, on='eid', how='left')
        X_test.rename(columns={outcome: 'event'}, inplace=True)
        
        X_copy = X_test.copy()
        X_copy = X_copy.drop(columns=['eid', 'duration', 'event'])
        log_partial_hazard = cox.predict_log_partial_hazard(X_copy)
        cindex = 1 - concordance_index(event_times=X_test['duration'], predicted_scores=log_partial_hazard, event_observed=X_test['event'])

        fold_end_time = time.time()

        table_data = [
            ["Outcome", self.outcomes_mapping[outcome]],
            ["Fold", fold],
            ["Train sample size", X_train.shape[0]],
            ["Number of cases (train)", np.sum(e_train[[outcome]].values)],
            ["Test sample size", X_test.shape[0]],
            ["Number of cases (test)", np.sum(e_test[[outcome]].values)],
            ["Test C-index", cindex],
            ["Time taken (seconds)", f"{fold_end_time - fold_start_time:.2f}"]
        ]
        self.logger.info("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        return [outcome, fold, cindex]
    
    def __load_predictors(self, prefix, outcome, fold, subgroup):
        age_sex_ethnicity = pd.read_csv('/your path/multiomics-cardiovascular-disease/data/processed/ukb_merged.csv', usecols=['eid', 'age', 'male', 'ethnicity'])
        age_under_60_eids = age_sex_ethnicity[age_sex_ethnicity['age'] < 60]['eid']
        age_over_60_eids = age_sex_ethnicity[age_sex_ethnicity['age'] >= 60]['eid']
        male_eids = age_sex_ethnicity[age_sex_ethnicity['male'] == 1]['eid']
        female_eids = age_sex_ethnicity[age_sex_ethnicity['male'] == 0]['eid']
        white_eids = age_sex_ethnicity[age_sex_ethnicity['ethnicity'] == 0]['eid']
        other_eids = age_sex_ethnicity[age_sex_ethnicity['ethnicity'] != 0]['eid']
        
        # Initialize an empty DataFrame for predictors
        X = pd.DataFrame()

        # Load base predictor data
        if 'AgeSex' in self.predictor_set:
            age_sex = pd.read_feather(os.path.join(self.split_seed_dir + f"X_{prefix}_AgeSex_K{fold}.feather"))
            X = age_sex if X.empty else pd.merge(X, age_sex, on='eid', how='left')

        if 'Clinical' in self.predictor_set:
            clinical = pd.read_feather(os.path.join(self.split_seed_dir + f"X_{prefix}_Clinical_K{fold}.feather"))
            X = clinical if X.empty else pd.merge(X, clinical, on='eid', how='left')

        if 'PANEL' in self.predictor_set:
            panel = pd.read_feather(os.path.join(self.split_seed_dir + f"X_{prefix}_PANEL_K{fold}.feather"))
            X = panel if X.empty else pd.merge(X, panel, on='eid', how='left')

        if 'ethnicity_1.0' in X.columns and 'ethnicity_4.0' in X.columns:
            X['ethnicity_1.0'] = X['ethnicity_1.0'].astype(int)
            X['ethnicity_4.0'] = X['ethnicity_4.0'].astype(int)
            X['ethnicity_1.0'] = X['ethnicity_1.0'] | X['ethnicity_4.0']
            X.drop(columns=['ethnicity_4.0'], inplace=True)

        # Load Genomics data if specified
        if 'Genomics' in self.predictor_set:
            genomics = pd.read_feather(os.path.join(self.split_seed_dir + f"X_{prefix}_Genomics_K{fold}.feather"))
            
            # Disease-specific PRSs
            if outcome == 'cad' or outcome == 'cvd_death':
                genomics = genomics[['eid', 'prs_cad']]
            elif outcome == 'stroke':
                genomics = genomics[['eid', 'prs_stroke']]
            elif outcome == 'hf':
                genomics = genomics[['eid', 'prs_hf']]
            elif outcome == 'af' or outcome == 'va':
                genomics = genomics[['eid', 'prs_af']]
            elif outcome == 'pad':
                genomics = genomics[['eid', 'prs_pad']]
            elif outcome == 'aaa':
                genomics = genomics[['eid', 'prs_aaa']]
            elif outcome == 'vt':
                genomics = genomics[['eid', 'prs_vte']]
            X = genomics if X.empty else pd.merge(X, genomics, on='eid', how='left')

        # Load Metabolomics data if specified
        if 'Metabolomics' in self.predictor_set:
            metabolomics = pd.read_csv(os.path.join(self.results_dir + f"MetScore/{prefix}_scores_K{fold}.csv"))
            metabolomics = metabolomics[['eid'] + [outcome]]
            scaler = StandardScaler()
            metabolomics[outcome] = scaler.fit_transform(metabolomics[[outcome]])
            metabolomics.rename(columns={outcome: 'metscore'}, inplace=True)
            X = metabolomics if X.empty else pd.merge(X, metabolomics, on='eid', how='left')

        # Load Proteomics data if specified
        if 'Proteomics' in self.predictor_set:
            proteomics = pd.read_csv(os.path.join(self.results_dir + f"ProScore/{prefix}_scores_K{fold}.csv"))
            proteomics = proteomics[['eid'] + [outcome]]
            scaler = StandardScaler()
            proteomics[outcome] = scaler.fit_transform(proteomics[[outcome]])
            proteomics.rename(columns={outcome: 'proscore'}, inplace=True)
            X = proteomics if X.empty else pd.merge(X, proteomics, on='eid', how='left')

        if 'NTproBNP' in self.predictor_set:
            ntprobnp = pd.read_feather(os.path.join(self.split_seed_dir + f"X_{prefix}_Proteomics_K{fold}.feather"))
            ntprobnp = ntprobnp[['eid'] + ['NTproBNP']]
            X = ntprobnp if X.empty else pd.merge(X, ntprobnp, on='eid', how='left')
            
        # Check if X is still empty, which means no predictors were loaded
        if X.empty:
            raise ValueError('Predictor set not recognized or empty')
        
        if subgroup == "<60": 
            X = X[X['eid'].isin(age_under_60_eids)]
            if 'age' in X.columns:
                X.drop(columns=['age'], inplace=True)
        elif subgroup == ">=60":
            X = X[X['eid'].isin(age_over_60_eids)]
            if 'age' in X.columns:
                X.drop(columns=['age'], inplace=True)
        elif subgroup == "male":
            X = X[X['eid'].isin(male_eids)]
            if 'male_1.0' in X.columns:
                X.drop(columns=['male_1.0'], inplace=True)
        elif subgroup == "female":
            X = X[X['eid'].isin(female_eids)]
            if 'male_1.0' in X.columns:
                X.drop(columns=['male_1.0'], inplace=True)
        elif subgroup == "white":
            X = X[X['eid'].isin(white_eids)]
            if all(col in X.columns for col in ['ethnicity_1.0', 'ethnicity_2.0', 'ethnicity_3.0']):
                X.drop(columns=['ethnicity_1.0', 'ethnicity_2.0', 'ethnicity_3.0'], inplace=True)

        return X


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
    results_dir = '/your path/multiomics-cardiovascular-disease/saved/results/'
    log_dir = '/your path/multiomics-cardiovascular-disease/saved/log'
    split_times = 10
    seed_to_split = 241104
    num_workers = 16
    
    predictor_sets = [
        ['AgeSex'],
        ['AgeSex', 'Genomics'],
        ['AgeSex', 'Metabolomics'],
        ['AgeSex', 'Proteomics'],
        ['AgeSex', 'Genomics', 'Metabolomics'],
        ['AgeSex', 'Genomics', 'Proteomics'],
        ['AgeSex', 'Metabolomics', 'Proteomics'],
        ['AgeSex', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['Clinical'],
        ['Clinical', 'Genomics'],
        ['Clinical', 'Metabolomics'],
        ['Clinical', 'Proteomics'],
        ['Clinical', 'Genomics', 'Metabolomics'],
        ['Clinical', 'Genomics', 'Proteomics'],
        ['Clinical', 'Metabolomics', 'Proteomics'],
        ['Clinical', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['PANEL'],
        ['PANEL', 'Genomics'],
        ['PANEL', 'Metabolomics'],
        ['PANEL', 'Proteomics'],
        ['PANEL', 'Genomics', 'Metabolomics'],
        ['PANEL', 'Genomics', 'Proteomics'],
        ['PANEL', 'Metabolomics', 'Proteomics'],
        ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['PANELBlood'],
        ['PANELBlood', 'Genomics'],
        ['PANELBlood', 'Metabolomics'],
        ['PANELBlood', 'Proteomics'],
        ['PANELBlood', 'Genomics', 'Metabolomics'],
        ['PANELBlood', 'Genomics', 'Proteomics'],
        ['PANELBlood', 'Metabolomics', 'Proteomics'],
        ['PANELBlood', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['Genomics'],
        ['Metabolomics'],
        ['Proteomics'],
        ['Genomics', 'Metabolomics'],
        ['Genomics', 'Proteomics'],
        ['Metabolomics', 'Proteomics'],
        ['Genomics', 'Metabolomics', 'Proteomics'],
        ['NTproBNP'],
        ['AgeSex', 'NTproBNP'],
        ['Clinical', 'NTproBNP'],
        ['PANEL', 'NTproBNP'],
        ['PANELBlood', 'NTproBNP'],
        ['NPPB'],
        ['AgeSex', 'NPPB'],
        ['Clinical', 'NPPB'],
        ['PANEL', 'NPPB'],
        ['PANELBlood', 'NPPB'],
        ['Creatinine'],
        ['AgeSex', 'Creatinine'],
        ['Clinical', 'Creatinine'],
        ['PANEL', 'Creatinine'],
        ['PANELBlood', 'Creatinine'],
        ['Albumin'],
        ['AgeSex', 'Albumin'],
        ['Clinical', 'Albumin'],
        ['PANEL', 'Albumin'],
        ['PANELBlood', 'Albumin'],
        ['Gly'],
        ['AgeSex', 'Gly'],
        ['Clinical', 'Gly'],
        ['PANEL', 'Gly'],
        ['PANELBlood', 'Gly'],
        ['GlycA'],
        ['AgeSex', 'GlycA'],
        ['Clinical', 'GlycA'],
        ['PANEL', 'GlycA'],
        ['PANELBlood', 'GlycA']
    ]

    subgroups = ['<60', '>=60', 'male', 'female', 'white']
    subgroup_mapping = {
        '<60': 'Age_under_60',
        '>=60': 'Age_over_60',
        'male': 'Male',
        'female': 'Female',
        'white': 'White'
    }
    for subgroup in subgroups:
        for predictor_set in predictor_sets:
            subgroup_folder = subgroup_mapping.get(subgroup, 'Unknown')
            log_filename = os.path.join(log_dir, f"Model/CoxPH/{subgroup_folder}/{'_'.join(predictor_set)}.log")
            logger = setup_logger(log_filename)
            
            # Log the current predictor set
            logger.info(f'Predictor set: {predictor_set}')
            coxph = CoxPHSubgroup(data_dir, results_dir, split_times, seed_to_split, predictor_set, subgroup, num_workers, logger)
