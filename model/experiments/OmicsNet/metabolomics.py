import os
import sys
import json
import logging
import time
import torch
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from tensorboardX import SummaryWriter
import shap

# Append project paths
project_path = '/your path/multiomics-cardiovascular-disease'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, 'data_loader'))
print(sys.path)

from data_loader import UKBiobankDataLoader
from model.model import *
from model.loss import *
from model.trainer import *
from utils.util import *

class OutcomeSpecificNet(nn.Module): 
    def __init__(self, base_model, outcome):
        super(OutcomeSpecificNet, self).__init__()
        self.shared_mlp = base_model.shared_mlp
        self.outcome_mlp = base_model.output_layers[outcome]

    def forward(self, omics_data=None):
        shared_fts = self.shared_mlp(omics_data)
        outcome_output = self.outcome_mlp(shared_fts, omics_data)
        
        return outcome_output
    
# Configuration dictionary
with open('/your path/multiomics-cardiovascular-disease/config/OmicsNet/Metabolomics.json', 'r') as file:
    config = json.load(file)

# Set seed
SEED = config['seed']
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Logging settings
log_filename = os.path.join(config['log_dir'], f"Model/{config['name']}/{'_'.join(config['predictor_set'])}/{'_'.join(config['predictor_set'])}.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


start_time = time.time()

# List to store test cindex
test_cindices = []

# Iterate over all folds for training and testing
for fold in range(10):
    logger.info(f"Processing fold: {fold}/9")
    
    # Initialize data loader
    data_loader = UKBiobankDataLoader(data_dir=config['data_dir'], predictor_set=config['predictor_set'],
                                      seed_to_split=config['data_loader']['seed_to_split'], batch_size=config['data_loader']['batch_size'], 
                                      shuffle=config['data_loader']['shuffle'], num_workers=config['data_loader']['num_workers'], logger=logger)
    
    
    train_dataloader, validate_dataloader, test_dataloader = data_loader.get_dataloader(fold=fold, outcomes_list=config['outcomes_list'])
    first_batch = next(iter(test_dataloader))
    logger.info(f"Fold {fold}: First test batch EIDs: \n{first_batch[0]}")
    
    outcome_column_indices = data_loader.get_outcome_column_indices(fold=fold, outcomes_list=config['outcomes_list'])
    metabolomics_input_dim = data_loader.get_feature(fold=fold, outcomes_list=config['outcomes_list'], predictor='Metabolomics').shape[1]
    metabolomics_features = data_loader.get_feature_names(fold=fold, outcomes_list=config['outcomes_list'], predictor='Metabolomics')
    
    # Initialize a new SummaryWriter for each fold
    writer = SummaryWriter(os.path.join(config['log_dir'], f"Model/{config['name']}/{'_'.join(config['predictor_set'])}/K{fold}")) 
    
    # Initialize model
    model = OmicsNet(
        outcomes_list=config['outcomes_list'],
        shared_mlp_kwargs=config['model']['shared_mlp_kwargs'],
        skip_connection_mlp_kwargs=config['model']['skip_connection_mlp_kwargs'],
        predictor_mlp_kwargs=config['model']['predictor_mlp_kwargs']
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['trainer']['lr'], weight_decay=config['trainer']['weight_decay'])    
    trainer = MLPTrainer(
        config=config, model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=None,
        train_dataloader=train_dataloader, validate_dataloader=validate_dataloader, test_dataloader=test_dataloader, 
        logger=logger, writer=writer, fold=fold, outcome_column_indices=outcome_column_indices, tuning=False)  
        
    logger.info('------------------Start training-----------------------')
    trainer.train()
    logger.info('------------------Finish training---------------------')
    
    # Close the SummaryWriter to release file handles
    writer.close()
    
    # Evaluate the best model on the test set
    logger.info('------------------Start testing-----------------------')
    best_model_path = os.path.join(config['model_dir'], f"{config['name']}/{'_'.join(config['predictor_set'])}/K{fold}.pth")
    
    # Load model weights
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model'])
    
    # Log test cindex
    test_cindex, mean_cindex = trainer.get_test_cindex(model, test_dataloader)
    for i, outcome in enumerate(config['outcomes_list']):
        cindex = test_cindex[outcome].values[0]
        test_cindices.append(f"{outcome},{fold},{cindex}")
    
    logger.info(f"Average testing c-index across all outcomes:\n{mean_cindex}")
    logger.info(f"Testing c-index for each outcome:\n{test_cindex}")
    
    # Log and save train and test Met scores
    train_scores = trainer.get_test_scores(model, train_dataloader)
    logger.info(train_scores.head())
    train_scores_filename = os.path.join(config['results_dir'], f"MetScore/train_scores_K{fold}.csv")
    os.makedirs(os.path.dirname(train_scores_filename), exist_ok=True)
    train_scores.to_csv(train_scores_filename, index=False)
    
    test_scores = trainer.get_test_scores(model, test_dataloader)
    logger.info(test_scores.head())
    test_scores_filename = os.path.join(config['results_dir'], f"MetScore/test_scores_K{fold}.csv")
    os.makedirs(os.path.dirname(test_scores_filename), exist_ok=True)
    test_scores.to_csv(test_scores_filename, index=False)

    # SHAP value computation
    background = generate_background_samples(train_dataloader, 'Metabolomics', num_samples=1000)
    test_batch = next(iter(test_dataloader))
    _, _, _, metabolomics_data, _, _, _ = test_batch
    X_test = metabolomics_data.to('cpu').float()
    X_test_df = pd.DataFrame(X_test.numpy(), columns=metabolomics_features)
    xtest_filename = os.path.join(config['results_dir'], f"SHAP/{'_'.join(config['predictor_set'])}/xtest_K{fold}.csv")
    os.makedirs(os.path.dirname(xtest_filename), exist_ok=True)
    X_test_df.to_csv(xtest_filename, index=False)
    
    for outcome in config['outcomes_list']:
        model_outcome = OutcomeSpecificNet(base_model=model, outcome=outcome).to('cpu')
        explainer = shap.DeepExplainer(model_outcome, background)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        shap_filename = os.path.join(config['results_dir'], f"SHAP/{'_'.join(config['predictor_set'])}/shap_{outcome}_K{fold}.csv")
        os.makedirs(os.path.dirname(shap_filename), exist_ok=True)
        shap_values_df = pd.DataFrame(shap_values.squeeze(axis=-1), columns=metabolomics_features)
        shap_values_df.to_csv(shap_filename, index=False)
        logger.info(f"Saved SHAP values for {outcome} in fold {fold}")
    
    logger.info('------------------Finish testing----------------------\n')

end_time = time.time()
elapsed_time = end_time - start_time
logger.info(f"Total time for training and testing: {elapsed_time:.2f} seconds")

# Save test cindex
test_cindices_filename = os.path.join(config['results_dir'], f"Cindex/{'_'.join(config['predictor_set'])} (Deep)_test_cindex.csv")
os.makedirs(os.path.dirname(test_cindices_filename), exist_ok=True)
with open(test_cindices_filename, 'w') as f:
    f.write("outcome,fold,cindex\n")
    for result in test_cindices:
        f.write(result + "\n")