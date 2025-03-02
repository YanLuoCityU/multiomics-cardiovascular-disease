import time
import numpy as np
import pandas as pd
from os.path import join
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from lifelines.utils import concordance_index
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import importlib.util
from tabulate import tabulate

class LassoCox():
    def __init__(self, 
                 data_dir,
                 results_dir,
                 split_times,
                 seed_to_split,
                 predictor_set,
                 num_workers,
                 logger):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.split_times = split_times
        self.seed_to_split = seed_to_split
        self.predictor_set = predictor_set
        self.num_workers = num_workers
        self.logger = logger
        
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        
        self.nested_cross_validation()
        
    def nested_cross_validation(self):
        self.logger.info('Nested cross validation for Lasso Cox models.')
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

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for outcome in outcomes_list:
                for fold in range(self.split_times):
                    futures.append(executor.submit(self._process_fold, outcome, fold))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing fold: {e}")

        self.test_cindex_df = pd.DataFrame(results, columns=['outcome', 'fold', 'cindex'])
        output_filename = f"{'_'.join(self.predictor_set)} (LassoCox)_test_cindex.csv"
        os.makedirs(self.results_dir, exist_ok=True)
        self.test_cindex_df.to_csv(os.path.join(self.results_dir, output_filename), index=False)
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Nested cross validation for Lasso Cox models completed in {elapsed_time:.2f} seconds.\n')

    def _process_fold(self, outcome, fold):
        fold_start_time = time.time()
        
        # Read data
        X_train = self.__load_predictors(fold, 'train', outcome)
        X_val = self.__load_predictors(fold, 'val', outcome)
        X_test = self.__load_predictors(fold, 'test', outcome)
        
        y_train = pd.read_feather(self.split_seed_dir + f'y_train_K{fold}.feather')
        y_train = y_train[outcome]
        y_val = pd.read_feather(self.split_seed_dir + f'y_val_K{fold}.feather')
        y_val = y_val[outcome]
        y_test = pd.read_feather(self.split_seed_dir + f'y_test_K{fold}.feather')
        y_test = y_test[outcome]
        
        e_train = pd.read_feather(self.split_seed_dir + f'e_train_K{fold}.feather')
        e_train = e_train[outcome]
        e_val = pd.read_feather(self.split_seed_dir + f'e_val_K{fold}.feather')
        e_val = e_val[outcome]
        e_test = pd.read_feather(self.split_seed_dir + f'e_test_K{fold}.feather')
        e_test = e_test[outcome]
        
        # Create Lasso Cox model
        cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, max_iter=1000)
        
        # Hyperparameter tuning
        best_params, best_score, estimated_alphas = self.__hyperparameter_tuning(cox_lasso, X_train, y_train, e_train, X_val, y_val, e_val)
        
        # Refit with best hyperparameters
        cox_lasso.set_params(alphas=best_params)
        target_train = Surv.from_arrays(event=e_train.values.flatten(), time=y_train.values.flatten())
        cox_lasso.fit(X_train, target_train)
        
        # Predict and evaluate on test data
        test_pred = cox_lasso.predict(X_test)
        test_cindex = 1 - concordance_index(event_times=y_test.values, predicted_scores=test_pred, event_observed=e_test.values)
        
        fold_end_time = time.time()
        
        table_data = [
            ["Outcome", self.outcomes_mapping[outcome]],
            ["Fold", fold],
            ["Number of hyperparameters", len(estimated_alphas)],
            ["Best parameters", best_params],
            ["Best validation score", best_score],
            ["Best model", cox_lasso],
            ["Test Shape", X_test.shape],
            ["Number of cases", np.sum(e_test.values)],
            ["Test C-index", test_cindex],
            ["Time taken (seconds)", f"{fold_end_time - fold_start_time:.2f}"]
        ]
        self.logger.info("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        return [outcome, fold, test_cindex]

    def __load_predictors(self, fold, prefix, outcome):
        dfs = []

        # Load clinical data based on predictor set
        if 'AgeSex' in self.predictor_set:
            clinical = pd.read_feather(self.split_seed_dir + f"X_{prefix}_AgeSex_K{fold}.feather")
            clinical.drop(columns=['eid'], inplace=True)
            dfs.append(clinical)
        elif 'Clinical' in self.predictor_set:
            clinical = pd.read_feather(self.split_seed_dir + f"X_{prefix}_Clinical_K{fold}.feather")
            clinical.drop(columns=['eid'], inplace=True)
            dfs.append(clinical)
        elif 'PANEL' in self.predictor_set:
            clinical = pd.read_feather(self.split_seed_dir + f"X_{prefix}_PANEL_K{fold}.feather")
            clinical.drop(columns=['eid'], inplace=True)
            dfs.append(clinical)
        elif 'PANELBlood' in self.predictor_set:
            clinical = pd.read_feather(self.split_seed_dir + f"X_{prefix}_PANELBlood_K{fold}.feather")
            clinical.drop(columns=['eid'], inplace=True)
            dfs.append(clinical)
        
        # Load metabolomics data if specified
        if 'Genomics' in self.predictor_set:
            genomics = pd.read_feather(self.split_seed_dir + f"X_{prefix}_Genomics_K{fold}.feather")
            
            if outcome == 'cad' or outcome == 'cvd_death':
                genomics = genomics[['eid', 'prs_cad']]
            elif outcome == 'stroke':
                genomics = genomics[['eid', 'prs_stroke']]
            elif outcome == 'hf':
                genomics = genomics[['eid', 'prs_hf']]
            elif outcome == 'af':
                genomics = genomics[['eid', 'prs_af']]
            elif outcome == 'va':
                genomics = genomics[['eid', 'prs_va']]
            elif outcome == 'pad':
                genomics = genomics[['eid', 'prs_pad']]
            elif outcome == 'aaa':
                genomics = genomics[['eid', 'prs_aaa']]
            elif outcome == 'vt':
                genomics = genomics[['eid', 'prs_vte']]
            
            genomics.drop(columns=['eid'], inplace=True)
            dfs.append(genomics)
            
        # Load metabolomics data if specified
        if 'Metabolomics' in self.predictor_set:
            metabolomics = pd.read_feather(self.split_seed_dir + f"X_{prefix}_Metabolomics_K{fold}.feather")
            ukb_nmr_info_df = pd.read_csv('/home/ukb/data/resources/NMR/nmr_info.csv')
            filtered_df = ukb_nmr_info_df[
                (ukb_nmr_info_df['Nightingale'] == True) & 
                (ukb_nmr_info_df['Type'].isin(['Non-derived', 'Composite'])) &
                (~ukb_nmr_info_df['Description'].isin(['Spectrometer-corrected alanine', 'Glucose-lactate']))
            ]
            original_nmr_list = filtered_df['Biomarker'].tolist()
            metabolomics = metabolomics.filter(items=original_nmr_list)
            dfs.append(metabolomics)
        
        # Load proteomics data if specified
        if 'Proteomics' in self.predictor_set:
            proteomics = pd.read_feather(self.split_seed_dir + f"X_{prefix}_Proteomics_K{fold}.feather")
            proteomics.drop(columns=['eid'], inplace=True)
            dfs.append(proteomics)
        
        # Concatenate all predictors into one DataFrame
        X = pd.concat(dfs, axis=1)
        
        return X
    
    
    def __hyperparameter_tuning(self, model, X_train, y_train, e_train, X_val, y_val, e_val):
        best_score = -np.inf
        best_params = None
        
        # Convert survival data for training
        target_train = Surv.from_arrays(event=e_train.values.flatten(), time=y_train.values.flatten())
        
        # Fit model and get estimated alphas
        lasso_cox = CoxnetSurvivalAnalysis(n_alphas=20, l1_ratio=1.0, max_iter=1000)
        lasso_cox.fit(X_train, target_train)
        estimated_alphas = lasso_cox.alphas_
        param_grid = {"alphas": [[v] for v in estimated_alphas]}
        
        # Perform grid search over hyperparameters
        for params in param_grid['alphas']:
            # Set parameters and fit model
            model.set_params(alphas=params)
            try:
                model.fit(X_train, target_train)
                
                # Predict on validation data and evaluate
                pred = model.predict(X_val)
                score = 1 - concordance_index(event_times=y_val.values, predicted_scores=pred, event_observed=e_val.values)
                
                # Update best model if current model is better
                if score > best_score:
                    best_score = score
                    best_params = params

            except ArithmeticError as e:
                self.logger.error(f"Arithmetic error with alpha {params}: {e}")
                continue
        
        return best_params, best_score, estimated_alphas



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
    results_dir = '/your path/multiomics-cardiovascular-disease/saved/results/Cindex/'
    log_dir = '/your path/multiomics-cardiovascular-disease/saved/log'
    split_times = 10
    seed_to_split = 241104
    num_workers = 16
    
    # Without proteomics
    predictor_sets = [
        ['AgeSex'],
        ['AgeSex', 'Genomics'],
        ['AgeSex', 'Metabolomics'],
        ['AgeSex', 'Genomics', 'Metabolomics'],
        ['Clinical'],
        ['Clinical', 'Genomics'],
        ['Clinical', 'Metabolomics'],
        ['Clinical', 'Genomics', 'Metabolomics'],
        ['PANEL'],
        ['PANEL', 'Genomics'],
        ['PANEL', 'Metabolomics'],
        ['PANEL', 'Genomics', 'Metabolomics'],
        ['Genomics'],
        ['Metabolomics'],
        ['Genomics', 'Metabolomics']
    ]

    # With proteomics
    predictor_sets = [
        ['AgeSex', 'Proteomics'],
        ['AgeSex', 'Genomics', 'Proteomics'],
        ['AgeSex', 'Metabolomics', 'Proteomics'],
        ['AgeSex', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['Clinical', 'Proteomics'],
        ['Clinical', 'Genomics', 'Proteomics'],
        ['Clinical', 'Metabolomics', 'Proteomics'],
        ['Clinical', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['PANEL', 'Proteomics'],
        ['PANEL', 'Genomics', 'Proteomics'],
        ['PANEL', 'Metabolomics', 'Proteomics'],
        ['PANEL', 'Genomics', 'Metabolomics', 'Proteomics'],
        ['Proteomics'],
        ['Genomics', 'Proteomics'],
        ['Metabolomics', 'Proteomics'],
        ['Genomics', 'Metabolomics', 'Proteomics']
    ]
    
    for predictor_set in predictor_sets:
        log_filename = os.path.join(log_dir, f"Model/LassoCox/{'_'.join(predictor_set)}.log")
        logger = setup_logger(log_filename)
        
        # Log the current predictor set
        logger.info(f'Predictor set: {predictor_set}')
        lasso_cox = LassoCox(data_dir, results_dir, split_times, seed_to_split, predictor_set, num_workers, logger)
