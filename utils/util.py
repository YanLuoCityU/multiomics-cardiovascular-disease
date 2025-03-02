import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
from lifelines.utils import concordance_index

import os
import logging

def setup_logger(log_filename):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    # Create a new logger
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler with 'w' mode (overwrite mode)
    handler = logging.FileHandler(log_filename, mode='w')
    handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_background_samples(dataloader, predictor_set, num_samples=1000):
    clinical_data = []
    genomics_data = []
    metabolomics_data = []
    proteomics_data = []
    for batch in dataloader:
        eid, clinical, genomics, metabolomics, proteomics, y, e = batch
        clinical_data.append(clinical)
        genomics_data.append(genomics)
        metabolomics_data.append(metabolomics)
        proteomics_data.append(proteomics)
    clinical_data = torch.cat(clinical_data, dim=0)
    genomics_data = torch.cat(genomics_data, dim=0)
    metabolomics_data = torch.cat(metabolomics_data, dim=0)
    proteomics_data = torch.cat(proteomics_data, dim=0)
    
    total_samples = clinical_data.size(0)
    print(f"Total samples available: {total_samples}")
    num_samples = min(num_samples, total_samples)
    
    indices = torch.randint(0, total_samples, (num_samples,))
    if predictor_set == 'Clinical':
        background = clinical_data[indices].to('cpu').float()
    elif predictor_set == 'Genomics':
        background = genomics_data[indices].to('cpu').float()
    elif predictor_set == 'Metabolomics':
        background = metabolomics_data[indices].to('cpu').float()
    elif predictor_set == 'Proteomics':
        background = proteomics_data[indices].to('cpu').float()
    else:
        raise ValueError(f"Invalid predictor_set: {predictor_set}. Expected 'Clinical', 'Genomics', 'Metabolomics', or 'Proteomics'.")
    print(background.shape)
    return background
