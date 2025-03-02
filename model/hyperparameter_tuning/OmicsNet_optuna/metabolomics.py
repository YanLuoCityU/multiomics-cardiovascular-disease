import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import json
import logging
import time
import random
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from tensorboardX import SummaryWriter
import optuna

# Append project paths
project_path = '/your path/multiomics-cardiovascular-disease'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, 'data_loader'))
print(sys.path)

from data_loader import UKBiobankDataLoader
from model.model import *
from model.loss import *
from model.trainer import *
from utils import count_parameters

def process_batch(device, batch):
    eid, clinical_data, genomics_data, metabolomics_data, proteomics_data, y, e = batch
        
    return eid, clinical_data.to(device).float(), genomics_data.to(device).float(), metabolomics_data.to(device).float(), \
        proteomics_data.to(device).float(), y.to(device).float(), e.to(device).float()
        
def calculate_loss_dict(config, outcome_column_indices, criterion, outputs, e):
    loss_dict = {}
    for outcome in config['outcomes_list']:
        # Get predicted risk, true event indicator for the specific outcome
        predicted_risk = outputs[outcome]
        event = e[:, outcome_column_indices[outcome]].unsqueeze(1)
        
        # Calculate loss for the specific outcome
        loss = criterion(predicted_risk, event)
        loss_dict[outcome] = loss
    return loss_dict

def calculate_loss(loss, mean=True):
    if isinstance(loss, dict):
        total_loss = sum(loss.values())
        num_losses = len(loss)
    elif isinstance(loss, list):
        total_loss = sum(loss)
        num_losses = len(loss)
    else:
        raise TypeError("loss should be either a dict or a list")

    return total_loss / num_losses if mean else total_loss

def test_model(config, device, model, dataloader, return_y_e=False):
    model.eval()
    outputs_dict = defaultdict(list)
    eid_list = []
    y_list = []
    e_list = []

    with torch.no_grad():
        for batch in dataloader:
            eid, _, _, metabolomics_data, proteomics_data, y, e = process_batch(device, batch)

            # Select omics data to be used
            if 'Metabolomics' in config['predictor_set']:
                outputs = model(metabolomics_data)
            elif 'Proteomics' in config['predictor_set']:
                outputs = model(proteomics_data)
                
            for key, value in outputs.items():
                outputs_dict[key].append(value.cpu())
            eid_list.append(eid)
            if return_y_e:
                y_list.append(y.cpu())
                e_list.append(e.cpu())

    for key in outputs_dict:
        outputs_dict[key] = torch.cat(outputs_dict[key])
    eid_list = torch.cat(eid_list)
    
    if return_y_e:
        y_list = torch.cat(y_list)
        e_list = torch.cat(e_list)
        return outputs_dict, eid_list, y_list, e_list
    else:
        return outputs_dict, eid_list
        
def get_test_cindex(config, device, outcome_column_indices, model, dataloader):
    test_outputs, test_eid, test_y, test_e = test_model(config, device, model, dataloader, return_y_e=True)

    cindex_dict = {'outcome': [], 'cindex': []}
    for outcome, index in outcome_column_indices.items():
        df = pd.DataFrame(test_outputs[outcome].numpy(), columns=['prediction'])
        df['duration'] = test_y[:, index].numpy()
        df['event'] = test_e[:, index].numpy()
        cindex = 1 - concordance_index(event_times=df['duration'], predicted_scores=df['prediction'], event_observed=df['event'])
        cindex_dict['outcome'].append(outcome)
        cindex_dict['cindex'].append(cindex)

    test_cindex = pd.DataFrame(cindex_dict).set_index('outcome').transpose()
    mean_cindex = test_cindex.mean(axis=1).values[0]

    return test_cindex, mean_cindex  

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Configuration dictionary
config = {
    'name': 'OmicsNet',
    'data_dir': '/your path/multiomics-cardiovascular-disease/data/',
    'model_dir': '/your path/multiomics-cardiovascular-disease/saved/models/',
    'log_dir': '/your path/multiomics-cardiovascular-disease/saved/log',
    'results_dir': '/your path/multiomics-cardiovascular-disease/saved/results/',
    'predictor_set': ['Metabolomics'],
    'outcomes_list': ['cad', 'stroke', 'hf', 'af', 'va', 'aaa', 'pad', 'vt', 'cvd_death'],
    'model': {
        'shared_mlp_kwargs': {
            'snn_init': False,
            'input_dim': 168, 
            'output_dim': 128, 
            'hidden_dim': 256, 
            'n_hidden_layers': 2,
            'activation': 'nn.ReLU',
            'dropout_fn': 'nn.Dropout', 
            'norm_fn': 'nn.BatchNorm1d',
            'norm_layer': 'all',
            'input_norm': False,
            'final_norm': True,
            'final_activation': 'nn.ReLU',
            'final_dropout': True,
            'dropout': 0.5
            },
        'skip_connection_mlp_kwargs': {
            'snn_init': False,
            'input_dim': None, 
            'output_dim': 128, 
            'hidden_dim': 256, 
            'n_hidden_layers': 2,
            'activation': 'nn.ReLU',
            'dropout_fn': 'nn.Dropout', 
            'norm_fn': 'nn.BatchNorm1d',
            'norm_layer': 'all',
            'input_norm': False,
            'final_norm': True,
            'final_activation': 'nn.ReLU',
            'final_dropout': True,
            'dropout': 0.5
            },
        'predictor_mlp_kwargs': {
            'snn_init': False,
            'input_dim': None, 
            'output_dim': 1, 
            'hidden_dim': 256, 
            'n_hidden_layers': 2,
            'activation': 'nn.ReLU',
            'dropout_fn': 'nn.Dropout', 
            'norm_fn': 'nn.BatchNorm1d',
            'norm_layer': 'all',
            'input_norm': False,
            'final_norm': False,
            'final_activation': None,
            'final_dropout': False,
            'dropout': 0.5
            }
        },
    'data_loader': {
        'batch_size': {
            'train': 1024,
            'validate': 1024,
            'test': None
            },
        'shuffle': True,
        'num_workers': 8,
        'seed_to_split': 241104,
        },
    'trainer': {
        'epochs': 100,
        'lr': 1e-2,
        'weight_decay': 0
        },
    'seed': 24101801
}

# Set seed
SEED = config['seed']
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Logging settings
log_filename = os.path.join(config['log_dir'], f"Model/{config['name']}/Hyperparameter_tuning/{'_'.join(config['predictor_set'])}/Seed_{SEED}/hyperparameter_tuning.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize data loader
fold = 0
data_loader = UKBiobankDataLoader(
    data_dir=config['data_dir'], predictor_set=config['predictor_set'],
    seed_to_split=config['data_loader']['seed_to_split'], batch_size=config['data_loader']['batch_size'], 
    shuffle=config['data_loader']['shuffle'], num_workers=config['data_loader']['num_workers'], logger=logger)
train_dataloader, validate_dataloader, test_dataloader = data_loader.get_dataloader(fold=fold, outcomes_list=config['outcomes_list'])
first_batch = next(iter(test_dataloader))
logger.info(f"Fold {fold}: First test batch EIDs: \n{first_batch[0]}")
outcome_column_indices = data_loader.get_outcome_column_indices(fold=fold, outcomes_list=config['outcomes_list'])

            
def objective(trial):
    shared_mlp_output_dim = trial.suggest_categorical('shared_mlp_output_dim', [32, 64, 128, 256])
    shared_mlp_hidden_dim = trial.suggest_categorical('shared_mlp_hidden_dim', [32, 64, 128, 256, 512])
    shared_mlp_n_hidden_layers = trial.suggest_int('shared_mlp_n_hidden_layers', 2, 5)
    shared_mlp_dropout = trial.suggest_categorical('shared_mlp_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    skip_connection_mlp_output_dim = trial.suggest_categorical('skip_connection_mlp_output_dim', [32, 64, 128])
    skip_connection_mlp_hidden_dim = trial.suggest_categorical('skip_connection_mlp_hidden_dim', [32, 64, 128, 256, 512])
    skip_connection_mlp_n_hidden_layers = trial.suggest_int('skip_connection_mlp_n_hidden_layers', 2, 5)
    skip_connection_mlp_dropout = trial.suggest_categorical('skip_connection_mlp_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    predictor_mlp_hidden_dim = trial.suggest_categorical('predictor_mlp_hidden_dim', [32, 64, 128, 256, 512])
    predictor_mlp_n_hidden_layers = trial.suggest_int('predictor_mlp_n_hidden_layers', 1, 5)
    predictor_mlp_dropout = trial.suggest_categorical('predictor_mlp_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    activation = trial.suggest_categorical('activation', ['nn.ReLU', 'nn.LeakyReLU', 'nn.ELU', 'nn.Tanh'])
    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    epochs = trial.suggest_categorical('epochs', [25, 50, 75])

    config['model']['shared_mlp_kwargs']['output_dim'] = shared_mlp_output_dim
    config['model']['shared_mlp_kwargs']['hidden_dim'] = shared_mlp_hidden_dim
    config['model']['shared_mlp_kwargs']['n_hidden_layers'] = shared_mlp_n_hidden_layers
    config['model']['shared_mlp_kwargs']['dropout'] = shared_mlp_dropout
    config['model']['shared_mlp_kwargs']['activation'] = activation
    config['model']['shared_mlp_kwargs']['final_activation'] = activation
    
    config['model']['skip_connection_mlp_kwargs']['input_dim'] = config['model']['shared_mlp_kwargs']['input_dim']
    config['model']['skip_connection_mlp_kwargs']['output_dim'] = skip_connection_mlp_output_dim
    config['model']['skip_connection_mlp_kwargs']['hidden_dim'] = skip_connection_mlp_hidden_dim
    config['model']['skip_connection_mlp_kwargs']['n_hidden_layers'] = skip_connection_mlp_n_hidden_layers
    config['model']['skip_connection_mlp_kwargs']['dropout'] = skip_connection_mlp_dropout
    config['model']['skip_connection_mlp_kwargs']['activation'] = activation
    config['model']['skip_connection_mlp_kwargs']['final_activation'] = activation
    
    config['model']['predictor_mlp_kwargs']['input_dim'] = shared_mlp_output_dim + skip_connection_mlp_output_dim
    config['model']['predictor_mlp_kwargs']['hidden_dim'] = predictor_mlp_hidden_dim
    config['model']['predictor_mlp_kwargs']['n_hidden_layers'] = predictor_mlp_n_hidden_layers
    config['model']['predictor_mlp_kwargs']['dropout'] = predictor_mlp_dropout
    config['model']['predictor_mlp_kwargs']['activation'] = activation
 
    config['trainer']['lr'] = lr
    config['trainer']['weight_decay'] = weight_decay
    config['trainer']['epochs'] = epochs
    
    logger.info(f"Trial {trial.number}: {trial.params}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OmicsNet(
        outcomes_list=config['outcomes_list'],
        shared_mlp_kwargs=config['model']['shared_mlp_kwargs'],
        skip_connection_mlp_kwargs=config['model']['skip_connection_mlp_kwargs'],
        predictor_mlp_kwargs=config['model']['predictor_mlp_kwargs']
    ).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['trainer']['lr'], weight_decay=config['trainer']['weight_decay'])
    
    for epoch in range(1, config['trainer']['epochs'] + 1):
        model.train()
        
        for batch in train_dataloader:
            _, _, _, metabolomics_data, proteomics_data, y, e = process_batch(device, batch)
            
            if 'Metabolomics' in config['predictor_set']:
                outputs = model(metabolomics_data)
            elif 'Proteomics' in config['predictor_set']:
                outputs = model(proteomics_data)
            
            optimizer.zero_grad()
            train_loss_dict = calculate_loss_dict(config, outcome_column_indices, criterion, outputs, e)
            train_loss = calculate_loss(loss=train_loss_dict, mean=True)
            train_loss.backward()
            optimizer.step()
                    
    model.eval()
    _, mean_cindex = get_test_cindex(config, device, outcome_column_indices, model, validate_dataloader)
            
    trial.report(mean_cindex, epoch)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return mean_cindex

start_time = time.time()
optuna_dir = os.path.join(config['log_dir'], f"Model/{config['name']}/Hyperparameter_tuning/{'_'.join(config['predictor_set'])}/Seed_{SEED}")
os.makedirs(optuna_dir, exist_ok=True)
optuna_filename = os.path.join(optuna_dir, 'optuna_results.db')

study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=SEED), 
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
    storage=f"sqlite:///{optuna_filename}",
    study_name=f"Optuna_hyperparameter_tuning_{SEED}",
    load_if_exists=True
)
study.optimize(objective, n_trials=200)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Total time for hyperparameter tuning: {elapsed_time:.2f} seconds")

trial = study.best_trial
logger.info(f"Best parameters found: {trial.params}")
logger.info(f"Best average c-index across all outcomes for fold {fold}: {trial.value}")