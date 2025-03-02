import os
import sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from dcurves import dca, plot_graphs
from sklearn.preprocessing import StandardScaler

# Append project paths
project_path = '/your path/multiomics-cardiovascular-disease'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, 'data_loader'))
print(sys.path)

def load_predictors(predictor_set, split_seed_dir, results_dir, prefix, outcome, fold):
    # Initialize an empty DataFrame for predictors
    X = pd.DataFrame()

    # Load base predictor data
    if 'AgeSex' in predictor_set:
        age_sex = pd.read_feather(os.path.join(split_seed_dir + f"X_{prefix}_AgeSex_K{fold}.feather"))
        X = age_sex if X.empty else pd.merge(X, age_sex, on='eid', how='left')

    if 'Clinical' in predictor_set:
        clinical = pd.read_feather(os.path.join(split_seed_dir + f"X_{prefix}_Clinical_K{fold}.feather"))
        X = clinical if X.empty else pd.merge(X, clinical, on='eid', how='left')

    if 'PANEL' in predictor_set:
        panel = pd.read_feather(os.path.join(split_seed_dir + f"X_{prefix}_PANEL_K{fold}.feather"))
        X = panel if X.empty else pd.merge(X, panel, on='eid', how='left')

    if 'PANELBlood' in predictor_set:
        panel = pd.read_feather(os.path.join(split_seed_dir + f"X_{prefix}_PANELBlood_K{fold}.feather"))
        X = panel if X.empty else pd.merge(X, panel, on='eid', how='left')
        
    if 'ethnicity_1.0' in X.columns and 'ethnicity_4.0' in X.columns:
        X['ethnicity_1.0'] = X['ethnicity_1.0'].astype(int)
        X['ethnicity_4.0'] = X['ethnicity_4.0'].astype(int)
        X['ethnicity_1.0'] = X['ethnicity_1.0'] | X['ethnicity_4.0']
        X.drop(columns=['ethnicity_4.0'], inplace=True)

    # Load Genomics data if specified
    if 'Genomics' in predictor_set:
        genomics = pd.read_feather(os.path.join(split_seed_dir + f"X_{prefix}_Genomics_K{fold}.feather"))
        
        # Disease-specific PRSs
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
        
        X = genomics if X.empty else pd.merge(X, genomics, on='eid', how='left')

    # Load Metabolomics data if specified
    if 'Metabolomics' in predictor_set:
        metabolomics = pd.read_csv(os.path.join(results_dir + f"MetScore/{prefix}_scores_K{fold}.csv"))
        metabolomics = metabolomics[['eid'] + [outcome]]
        scaler = StandardScaler()
        metabolomics[outcome] = scaler.fit_transform(metabolomics[[outcome]])
        metabolomics.rename(columns={outcome: 'metscore'}, inplace=True)
        X = metabolomics if X.empty else pd.merge(X, metabolomics, on='eid', how='left')

    # Load Proteomics data if specified
    if 'Proteomics' in predictor_set:
        proteomics = pd.read_csv(os.path.join(results_dir + f"ProScore/{prefix}_scores_K{fold}.csv"))
        proteomics = proteomics[['eid'] + [outcome]]
        scaler = StandardScaler()
        proteomics[outcome] = scaler.fit_transform(proteomics[[outcome]])
        proteomics.rename(columns={outcome: 'proscore'}, inplace=True)
        X = proteomics if X.empty else pd.merge(X, proteomics, on='eid', how='left')

    # Check if X is still empty, which means no predictors were loaded
    if X.empty:
        raise ValueError('Predictor set not recognized or empty')

    return X

def process_fold(predictor_set, split_seed_dir, results_dir, outcome, fold):
    # Read adn merge train data
    prefix = 'train'
    X_train = load_predictors(predictor_set, split_seed_dir, results_dir, prefix, outcome, fold)
    y_train = pd.read_feather(os.path.join(split_seed_dir + f"y_{prefix}_K{fold}.feather"))
    y_train = y_train[['eid'] + [outcome]]
    e_train = pd.read_feather(os.path.join(split_seed_dir + f"e_{prefix}_K{fold}.feather"))
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
    X_test = load_predictors(predictor_set, split_seed_dir, results_dir, prefix, outcome, fold)
    y_test = pd.read_feather(os.path.join(split_seed_dir + f"y_{prefix}_K{fold}.feather"))
    y_test = y_test[['eid'] + [outcome]]
    e_test = pd.read_feather(os.path.join(split_seed_dir + f"e_{prefix}_K{fold}.feather"))
    e_test = e_test[['eid'] + [outcome]]

    X_test = pd.merge(X_test, y_test, on='eid', how='left')
    X_test.rename(columns={outcome: 'duration'}, inplace=True)
    X_test = pd.merge(X_test, e_test, on='eid', how='left')
    X_test.rename(columns={outcome: 'event'}, inplace=True)
    
    X_copy = X_test.copy()
    X_copy = X_copy.drop(columns=['eid', 'duration', 'event'])
    X_test['predicted_prob15'] = 1 - cox.predict_survival_function(X_copy, times=[15]).T # Predict 15-year disease probability
    X_test['predicted_prob10'] = 1 - cox.predict_survival_function(X_copy, times=[10]).T # Predict 15-year disease probability
    X_test['outcome'] = outcome
    X_test['fold'] = fold
    
    return X_test[['eid', 'outcome', 'fold', 'duration', 'event', 'predicted_prob15', 'predicted_prob10']]

data_dir = '/your path/multiomics-cardiovascular-disease/data/'
results_dir = '/your path/multiomics-cardiovascular-disease/saved/results/'
split_times = 10
seed_to_split = 241104
num_workers = 24

split_seed_filename = f"split_seed-{seed_to_split}/"
split_seed_dir = os.path.join(data_dir, split_seed_filename)

outcomes_list = ['cad', 'stroke', 'hf', 'af', 'va', 'aaa', 'pad', 'vt', 'cvd_death']
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
    ['Genomics', 'Metabolomics', 'Proteomics']
]

for predictor_set in predictor_sets:
    all_folds_data = []
    
    for outcome in outcomes_list:
        for fold in range(split_times):
            fold_data = process_fold(predictor_set, split_seed_dir, results_dir, outcome, fold)
            all_folds_data.append(fold_data)

    all_folds_data = pd.concat(all_folds_data, ignore_index=True)
    output_filename = f"Pred/{'_'.join(predictor_set)} (CoxPH)_test_pred.csv"
    os.makedirs(results_dir, exist_ok=True)
    all_folds_data.to_csv(os.path.join(results_dir, output_filename), index=False)