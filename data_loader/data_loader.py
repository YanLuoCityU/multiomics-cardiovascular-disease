import os
from os.path import join, exists
import pandas as pd
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from dataset import *

class UKBiobankDataLoader():
    '''
    - predictor_set: str. The predictor set to use. It can be 'ASCVD', 'SCORE2, 'AgeSex', 'Clinical', 'PANEL, 'PANELBlood, 'Genomics, 'Metabolomics', 'Proteomics'.
    - batch_size: dict. The batch size for training, validation, and testing datasets.
    '''
    def __init__(self, 
                 data_dir=None,
                 predictor_set=None,
                 seed_to_split=None,
                 batch_size=None,
                 shuffle=None,
                 num_workers=None,
                 logger=None):
        self.data_dir = data_dir
        self.predictor_set = predictor_set
        self.seed_to_split = seed_to_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.logger = logger
    
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        
        self._check_files_exist()
        
        self.data_loaded = False
        
    def _check_files_exist(self):
        required_files = [
            join(self.split_seed_dir, 'X_test_AgeSex_K0.feather'), 
            join(self.split_seed_dir, 'y_test_K0.feather'), 
            join(self.split_seed_dir, 'e_test_K0.feather')
        ]
        for file in required_files:
            if not exists(file):
                self.logger.error(f'File not found: {file}')
                raise FileNotFoundError(f'Required file {file} not found. Please load UKBiobank dataset first.')
            
    
    def get_data(self, fold, outcomes_list):
        if self.data_loaded:
            return
        
        if fold == 0:
            self.logger.info(f'Loading training, validation, and testing datasets from {self.split_seed_dir}.')

        # Load predictors
        self.eid_train, self.clinical_train, self.genomics_train, self.metabolomics_train, self.proteomics_train = self._load_predictors(fold, 'train')
        self.eid_val, self.clinical_val, self.genomics_val, self.metabolomics_val, self.proteomics_val = self._load_predictors(fold, 'val')
        self.eid_test, self.clinical_test, self.genomics_test, self.metabolomics_test, self.proteomics_test = self._load_predictors(fold, 'test')
        
        # Feature names
        self.clinical_features = self.clinical_train.columns.tolist()
        self.genomics_features = self.genomics_train.columns.tolist()
        self.metabolomics_features = self.metabolomics_train.columns.tolist()
        self.proteomics_features = self.proteomics_train.columns.tolist()
        
        # Load survival time and event indicator
        self.y_train, self.e_train = self._load_outcome_data(fold, outcomes_list, 'train')
        self.y_val, self.e_val = self._load_outcome_data(fold, outcomes_list, 'val')
        self.y_test, self.e_test = self._load_outcome_data(fold, outcomes_list, 'test')
        
        if fold == 0:
            self.logger.info(f'Columns order in y_*: {list(self.y_train.columns)}.')
            self.logger.info(f"The number of samples in the training dataset: {self.y_train[outcomes_list[0]].shape[0]}.")
            self.logger.info(f"The number of samples in the validation dataset: {self.y_val[outcomes_list[0]].shape[0]}.")
            self.logger.info(f"The number of samples in the testing dataset: {self.y_test[outcomes_list[0]].shape[0]}.\n")
        
        # Set flag to indicate data has been loaded
        self.data_loaded = True
        
    def _load_predictors(self, fold, prefix):
        # Load clinical data
        clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Clinical_K{fold}.feather"))
        if 'AgeSex' in self.predictor_set:
            clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_AgeSex_K{fold}.feather"))
        elif 'PANEL' in self.predictor_set:
            clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_PANEL_K{fold}.feather"))
        elif 'PANELBlood' in self.predictor_set:
            clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_PANELBlood_K{fold}.feather"))

        # Load genomics data
        genomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Genomics_K{fold}.feather"))
        
        # Load metabolomics data
        metabolomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Metabolomics_K{fold}.feather"))
        ## Keep 168 non-derived and composite metabolic biomarkers
        ukb_nmr_info_df = pd.read_csv('/home/ukb/data/resources/NMR/nmr_info.csv')
        filtered_df = ukb_nmr_info_df[
            (ukb_nmr_info_df['Nightingale'] == True) & 
            (ukb_nmr_info_df['Type'].isin(['Non-derived', 'Composite'])) &
            (~ukb_nmr_info_df['Description'].isin(['Spectrometer-corrected alanine', 'Glucose-lactate']))
        ]
        original_nmr_list = filtered_df['Biomarker'].tolist()
        metabolomics = metabolomics.filter(items=original_nmr_list + ['eid'])
        
        # Load proteomics data
        proteomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Proteomics_K{fold}.feather")) # 2923 - 3 = 2920. Exclude 3 proteins with more than 50% missingness
        
        if fold == 0:
            self.logger.info(f'The number of clinical features in the {prefix} data: {clinical.shape[1]-1}.')
            self.logger.info(f'The number of genetic features in the {prefix} data: {genomics.shape[1]-1}.')
            self.logger.info(f'The number of metabolites in the {prefix} data: {metabolomics.shape[1]-1}.')
            self.logger.info(f'The number of proteins in the {prefix} data: {proteomics.shape[1]-1}.')
        
        eid = clinical['eid'].tolist() # Have manually checked that the order of eids in all predictors are the same
        clinical.drop(columns=['eid'], inplace=True)
        genomics.drop(columns=['eid'], inplace=True)
        metabolomics.drop(columns=['eid'], inplace=True)
        proteomics.drop(columns=['eid'], inplace=True)
        
        return eid, clinical, genomics, metabolomics, proteomics
    
    def _load_outcome_data(self, fold, outcomes_list, prefix):
        y = pd.read_feather(join(self.split_seed_dir, f'y_{prefix}_K{fold}.feather'))
        e = pd.read_feather(join(self.split_seed_dir, f'e_{prefix}_K{fold}.feather'))
        
        y = y[outcomes_list]
        e = e[outcomes_list]
        return y, e
    
    
    
    def get_dataloader(self, fold, outcomes_list):
        self.get_data(fold, outcomes_list)
        
        train_dataloader = self._create_dataloader(
            batch_size=self.batch_size['train'],
            shuffle=self.shuffle,
            eid_list=self.eid_train,
            clinical_data=self.clinical_train, 
            genomics_data=self.genomics_train,
            metabolomics_data=self.metabolomics_train, 
            proteomics_data=self.proteomics_train, 
            y=self.y_train, 
            e=self.e_train)
        
        validate_dataloader = self._create_dataloader(
            batch_size=self.batch_size['validate'],
            shuffle=self.shuffle,
            eid_list=self.eid_val,
            clinical_data=self.clinical_val, 
            genomics_data=self.genomics_val,
            metabolomics_data=self.metabolomics_val, 
            proteomics_data=self.proteomics_val, 
            y=self.y_val, 
            e=self.e_val)
        
        test_dataloader = self._create_dataloader(
            batch_size=self.batch_size['test'],
            shuffle=False,
            eid_list=self.eid_test,
            clinical_data=self.clinical_test, 
            genomics_data=self.genomics_test,
            metabolomics_data=self.metabolomics_test, 
            proteomics_data=self.proteomics_test, 
            y=self.y_test, 
            e=self.e_test)
        
        return train_dataloader, validate_dataloader, test_dataloader
    
    def _create_dataloader(self, batch_size, shuffle, eid_list, clinical_data, genomics_data, metabolomics_data, proteomics_data, y, e):
        dataset = UKBiobankDataset(eid_list=eid_list, clinical_data=clinical_data, genomics_data=genomics_data, metabolomics_data=metabolomics_data, proteomics_data=proteomics_data, y=y, e=e, ppi=None)
        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=self.num_workers, collate_fn=self.__collate_fn)
        return dataloader
    
    def __collate_fn(self, data_list):
        eid = []
        clinical = []
        genomics =  []
        metabolomics = []
        proteomics = []
        survival_times = []
        event_indicators = []

        for data in data_list:
            eid_tensor, clinical_tensor, genomics_tensor, metabolomics_tensor, proteomics_tensor, y_tensor, e_tensor = data
            eid.append(eid_tensor)
            clinical.append(clinical_tensor)
            genomics.append(genomics_tensor)
            metabolomics.append(metabolomics_tensor)
            proteomics.append(proteomics_tensor)
            survival_times.append(y_tensor)
            event_indicators.append(e_tensor)
        
        eid_batch = torch.stack(eid)
        clinical_batch = torch.stack(clinical)
        genomics_batch = torch.stack(genomics)
        metabolomics_batch = torch.stack(metabolomics)
        proteomics_batch = torch.stack(proteomics)
        survival_times = torch.stack(survival_times)
        event_indicators = torch.stack(event_indicators)

        return eid_batch, clinical_batch, genomics_batch, metabolomics_batch, proteomics_batch, survival_times, event_indicators


    def get_outcome_column_indices(self, fold, outcomes_list):
        self.get_data(fold, outcomes_list)
        
        self.outcome_column_indices = {outcome: self.y_train.columns.get_loc(outcome) for outcome in outcomes_list}
        
        return self.outcome_column_indices
    
    
    def get_feature(self, fold, outcomes_list, predictor):
        self.get_data(fold, outcomes_list)
        
        if predictor == 'Clinical':
            return self.clinical_train
        elif predictor == 'Genomics':
            return self.genomics_train
        elif predictor == 'Metabolomics':
            return self.metabolomics_train
        elif predictor == 'Proteomics':
            return self.proteomics_train
        
    def get_feature_names(self, fold, outcomes_list, predictor):
        self.get_data(fold, outcomes_list)
        
        if predictor == 'Clinical':
            return self.clinical_features
        elif predictor == 'Genomics':
            return self.genomics_features
        elif predictor == 'Metabolomics':
            return self.metabolomics_features
        elif predictor == 'Proteomics':
            return self.proteomics_features



class UKBiobankGraphDataLoader():
    '''
    - predictor_set: str. The predictor set to use. It can be 'ASCVD', 'SCORE2, 'AgeSex', 'Clinical', 'PANEL, 'Genomics, 'Metabolomics', 'Proteomics'.
    - batch_size: dict. The batch size for training, validation, and testing datasets.
    '''
    def __init__(self, 
                 data_dir=None,
                 predictor_set=None,
                 ppi_require=True,
                 seed_to_split=None,
                 batch_size=None,
                 shuffle=None,
                 num_workers=None,
                 num_gpus=1,
                 drop_last=False,
                 logger=None):
        self.data_dir = data_dir
        self.predictor_set = predictor_set
        self.ppi_require = ppi_require
        self.seed_to_split = seed_to_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.drop_last = drop_last
        self.logger = logger
    
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        
        self._check_files_exist()
        
        self.data_loaded = False
        
    def _check_files_exist(self):
        required_files = [
            join(self.split_seed_dir, 'X_test_AgeSex_K0.feather'), 
            join(self.split_seed_dir, 'y_test_K0.feather'), 
            join(self.split_seed_dir, 'e_test_K0.feather')
        ]
        for file in required_files:
            if not exists(file):
                self.logger.error(f'File not found: {file}')
                raise FileNotFoundError(f'Required file {file} not found. Please load UKBiobank dataset first.')
            
    
    def get_data(self, fold, outcomes_list):
        if self.data_loaded:
            return
        
        if fold == 0:
            self.logger.info(f'Loading training, validation, and testing datasets from {self.split_seed_dir}.')
            
        # Load network
        self.logger.info('Loading protein-protein interaction (PPI) network.')
        self.ppi = pd.read_csv(join(self.data_dir, 'processed/networks/ppi.csv'), usecols=['protein1_name', 'protein2_name'])
        
        # Load predictors
        self.eid_train, self.clinical_train, self.genomics_train, self.metabolomics_train, self.proteomics_train = self._load_predictors(fold, 'train')
        self.eid_val, self.clinical_val, self.genomics_val, self.metabolomics_val, self.proteomics_val = self._load_predictors(fold, 'val')
        self.eid_test, self.clinical_test, self.genomics_test, self.metabolomics_test, self.proteomics_test = self._load_predictors(fold, 'test')
        
        # Feature names
        self.clinical_features = self.clinical_train.columns.tolist()
        self.genomics_features = self.genomics_train.columns.tolist()
        self.metabolomics_features = self.metabolomics_train.columns.tolist()
        self.proteomics_features = self.proteomics_train.columns.tolist()
        
        # Load survival time and event indicator
        self.y_train, self.e_train = self._load_outcome_data(fold, outcomes_list, 'train')
        self.y_val, self.e_val = self._load_outcome_data(fold, outcomes_list, 'val')
        self.y_test, self.e_test = self._load_outcome_data(fold, outcomes_list, 'test')
        
        if fold == 0:
            self.logger.info(f'Columns order in y_*: {list(self.y_train.columns)}.')
            self.logger.info(f"The number of samples in the training dataset: {self.y_train[outcomes_list[0]].shape[0]}.")
            self.logger.info(f"The number of samples in the validation dataset: {self.y_val[outcomes_list[0]].shape[0]}.")
            self.logger.info(f"The number of samples in the testing dataset: {self.y_test[outcomes_list[0]].shape[0]}.\n")
        
        # Set flag to indicate data has been loaded
        self.data_loaded = True
        
    def _load_predictors(self, fold, prefix):
        # Load clinical data
        clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Clinical_K{fold}.feather"))
        if 'AgeSex' in self.predictor_set:
            clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_AgeSex_K{fold}.feather"))
        elif 'PANEL' in self.predictor_set:
            clinical = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_PANEL_K{fold}.feather"))

        # Load genomics data
        genomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Genomics_K{fold}.feather"))
        
        # Load metabolomics data
        metabolomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Metabolomics_K{fold}.feather"))
        ## Keep 168 non-derived and composite metabolic biomarkers
        ukb_nmr_info_df = pd.read_csv('/home/ukb/data/resources/NMR/nmr_info.csv')
        filtered_df = ukb_nmr_info_df[
            (ukb_nmr_info_df['Nightingale'] == True) & 
            (ukb_nmr_info_df['Type'].isin(['Non-derived', 'Composite'])) &
            (~ukb_nmr_info_df['Description'].isin(['Spectrometer-corrected alanine', 'Glucose-lactate']))
        ]
        original_nmr_list = filtered_df['Biomarker'].tolist()
        metabolomics = metabolomics.filter(items=original_nmr_list + ['eid'])
        
        # Load proteomics data
        proteomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Proteomics_K{fold}.feather"))
        ukb_olink = pd.read_csv('/your path/multiomics-cardiovascular-disease/data/network_raw/ukb_olink_mapped_entrez.csv')
        ukb_olink = ukb_olink.dropna(subset=['entrezgene_id']) # Exclude 15 proteins with missing Entrez IDs: 
                                                               # DEFA1_DEFA1B, DEFB4A_DEFB4B, EBI3_IL27, FUT3_FUT5, IL12A_IL12B, LGALS7_LGALS7B, MICB_MICA, AMY1A_AMY1B_AMY1C, 
                                                               # BOLA2_BOLA2B, CGB3_CGB5_CGB8, CTAG1A_CTAG1B, DEFB103A_DEFB103B, DEFB104A_DEFB104B, SPACA5_SPACA5B, CKMT1A_CKMT1B
        ukb_olink = ukb_olink[~ukb_olink['Assay'].isin(['GLIPR1', 'NPM1', 'PCOLCE'])] # Exclude 3 proteins with more than 50% missingness
        protein_list = ukb_olink['Assay'].tolist()
        proteomics = proteomics.filter(items=protein_list + ['eid']) # 2923 - 15 - 3 = 2905 proteins
        
        if fold == 0:
            self.logger.info(f'The number of clinical features in the {prefix} data: {clinical.shape[1]-1}.')
            self.logger.info(f'The number of genetic features in the {prefix} data: {genomics.shape[1]-1}.')
            self.logger.info(f'The number of metabolites in the {prefix} data: {metabolomics.shape[1]-1}.')
            self.logger.info(f'The number of proteins in the {prefix} data: {proteomics.shape[1]-1}.')
        
        eid = clinical['eid'].tolist() # Have manually checked that the order of eids in all predictors are the same
        clinical.drop(columns=['eid'], inplace=True)
        genomics.drop(columns=['eid'], inplace=True)
        metabolomics.drop(columns=['eid'], inplace=True)
        proteomics.drop(columns=['eid'], inplace=True)
        
        return eid, clinical, genomics, metabolomics, proteomics
    
    def _load_outcome_data(self, fold, outcomes_list, prefix):
        y = pd.read_feather(join(self.split_seed_dir, f'y_{prefix}_K{fold}.feather'))
        e = pd.read_feather(join(self.split_seed_dir, f'e_{prefix}_K{fold}.feather'))
        
        y = y[outcomes_list]
        e = e[outcomes_list]
        return y, e
    
    
    
    def get_dataloader(self, fold, outcomes_list):
        self.get_data(fold, outcomes_list)
        
        train_dataloader = self._create_dataloader(
            batch_size=self.batch_size['train'],
            shuffle=self.shuffle,
            eid_list=self.eid_train,
            clinical_data=self.clinical_train, 
            genomics_data=self.genomics_train,
            metabolomics_data=self.metabolomics_train, 
            proteomics_data=self.proteomics_train, 
            y=self.y_train, 
            e=self.e_train,
            ppi=self.ppi)
        
        validate_dataloader = self._create_dataloader(
            batch_size=self.batch_size['validate'],
            shuffle=self.shuffle,
            eid_list=self.eid_val,
            clinical_data=self.clinical_val, 
            genomics_data=self.genomics_val,
            metabolomics_data=self.metabolomics_val, 
            proteomics_data=self.proteomics_val, 
            y=self.y_val, 
            e=self.e_val,
            ppi=self.ppi)
        
        test_dataloader = self._create_dataloader(
            batch_size=self.batch_size['test'],
            shuffle=False,
            eid_list=self.eid_test,
            clinical_data=self.clinical_test, 
            genomics_data=self.genomics_test,
            metabolomics_data=self.metabolomics_test, 
            proteomics_data=self.proteomics_test, 
            y=self.y_test, 
            e=self.e_test,
            ppi=self.ppi)
        
        return train_dataloader, validate_dataloader, test_dataloader
    
    def _create_dataloader(self, batch_size, shuffle, eid_list, clinical_data, genomics_data, metabolomics_data, proteomics_data, y, e, ppi=None):
        dataset = UKBiobankDataset(eid_list=eid_list, clinical_data=clinical_data, genomics_data=genomics_data, metabolomics_data=metabolomics_data, proteomics_data=proteomics_data, y=y, e=e, ppi=ppi)
        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size // self.num_gpus,
                                shuffle=shuffle, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.__collate_fn)
        return dataloader
    
    def __collate_fn(self, data_list):
        eid = []
        clinical = []
        genomics =  []
        metabolomics = []
        proteomics = []
        proteomics_graph = []
        survival_times = []
        event_indicators = []

        for data in data_list:
            if self.ppi_require:
                eid_tensor, clinical_tensor, genomics_tensor, metabolomics_tensor, proteomics_tensor, proteomics_graph_tensor, y_tensor, e_tensor = data
                proteomics_graph.append(proteomics_graph_tensor) 
            else:
                eid_tensor, clinical_tensor, genomics_tensor, metabolomics_tensor, proteomics_tensor, y_tensor, e_tensor = data
            eid.append(eid_tensor)
            clinical.append(clinical_tensor)
            genomics.append(genomics_tensor)
            metabolomics.append(metabolomics_tensor)
            proteomics.append(proteomics_tensor)
            survival_times.append(y_tensor)
            event_indicators.append(e_tensor)
        
        eid_batch = torch.stack(eid)
        clinical_batch = torch.stack(clinical)
        genomics_batch = torch.stack(genomics)
        metabolomics_batch = torch.stack(metabolomics)
        proteomics_batch = torch.stack(proteomics)
        survival_times = torch.stack(survival_times)
        event_indicators = torch.stack(event_indicators)
        
        if self.ppi_require:
            proteomics_graph_batch = Batch.from_data_list(proteomics)
            return eid_batch, clinical_batch, genomics_batch, metabolomics_batch, proteomics_batch, proteomics_graph_batch, survival_times, event_indicators
        else:
            return eid_batch, clinical_batch, genomics_batch, metabolomics_batch, proteomics_batch, survival_times, event_indicators


    def get_outcome_column_indices(self, fold, outcomes_list):
        self.get_data(fold, outcomes_list)
        
        self.outcome_column_indices = {outcome: self.y_train.columns.get_loc(outcome) for outcome in outcomes_list}
        
        return self.outcome_column_indices
    
    
    def get_feature(self, fold, outcomes_list, predictor):
        self.get_data(fold, outcomes_list)
        
        if predictor == 'Clinical':
            return self.clinical_train
        elif predictor == 'Genomics':
            return self.genomics_train
        elif predictor == 'Metabolomics':
            return self.metabolomics_train
        elif predictor == 'Proteomics':
            return self.proteomics_train
        
    def get_feature_names(self, fold, outcomes_list, predictor):
        self.get_data(fold, outcomes_list)
        
        if predictor == 'Clinical':
            return self.clinical_features
        elif predictor == 'Genomics':
            return self.genomics_features
        elif predictor == 'Metabolomics':
            return self.metabolomics_features
        elif predictor == 'Proteomics':
            return self.proteomics_features
        

class UKBiobankOmicsDataLoader():
    def __init__(self, 
                 data_dir=None,
                 predictor_set=None,
                 ppi_require=True,
                 seed_to_split=None,
                 batch_size=None,
                 shuffle=None,
                 num_workers=None,
                 logger=None):
        self.data_dir = data_dir
        self.predictor_set = predictor_set
        self.ppi_require = ppi_require
        self.seed_to_split = seed_to_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.logger = logger
    
        self.split_seed_filename = f"split_seed-{self.seed_to_split}/"
        self.split_seed_dir = join(self.data_dir, self.split_seed_filename)
        
        self._check_files_exist()
        
        self.data_loaded = False
    
    def _check_files_exist(self):
        required_files = [
            join(self.split_seed_dir, f'X_test_{self.predictor_set[0]}_K0.feather'), 
            join(self.split_seed_dir, 'y_test_K0.feather'), 
            join(self.split_seed_dir, 'e_test_K0.feather')
        ]
        for file in required_files:
            if not exists(file):
                self.logger.error(f'File not found: {file}')
                raise FileNotFoundError(f'Required file {file} not found. Please load UKBiobank dataset first.')


    def get_data(self, fold, outcomes_list):
        if self.data_loaded:
            return
        
        if fold == 0:
            self.logger.info(f'Loading training and testing datasets from {self.split_seed_dir}.')
        
        # Load network
        self.logger.info('Loading protein-protein interaction (PPI) network.')
        self.ppi = pd.read_csv(join(self.data_dir, 'processed/networks/ppi.csv'), usecols=['protein1_name', 'protein2_name'])
        
        # Load predictors
        self.eid_train, self.metabolomics_train, self.proteomics_train = self._load_predictors(fold, 'train')
        self.eid_test, self.metabolomics_test, self.proteomics_test = self._load_predictors(fold, 'test')
        
        # Feature names
        self.metabolomics_features = self.metabolomics_train.columns.tolist() if self.metabolomics_train is not None else None
        self.proteomics_features = self.proteomics_train.columns.tolist() if self.proteomics_train is not None else None
        
        # Load survival time and event indicator
        self.y_train, self.e_train = self._load_outcome_data(fold, outcomes_list, 'train')
        self.y_test, self.e_test = self._load_outcome_data(fold, outcomes_list, 'test')
        
        if fold == 0:
            self.logger.info(f'Columns order in y_*: {list(self.y_train.columns)}.')
            self.logger.info(f"The number of samples in the training dataset: {self.y_train[outcomes_list[0]].shape[0]}.")
            self.logger.info(f"The number of samples in the testing dataset: {self.y_test[outcomes_list[0]].shape[0]}.\n")
        
        # Set flag to indicate data has been loaded
        self.data_loaded = True
        
    def _load_predictors(self, fold, prefix):          
        metabolomics = None
        if 'Metabolomics' in self.predictor_set:
            # Load metabolomics data
            metabolomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Metabolomics_K{fold}.feather"))
            # Keep 168 non-derived and composite metabolic biomarkers
            ukb_nmr_info_df = pd.read_csv('/home/ukb/data/resources/NMR/nmr_info.csv')
            filtered_df = ukb_nmr_info_df[
                (ukb_nmr_info_df['Nightingale'] == True) & 
                (ukb_nmr_info_df['Type'].isin(['Non-derived', 'Composite'])) &
                (~ukb_nmr_info_df['Description'].isin(['Spectrometer-corrected alanine', 'Glucose-lactate']))
            ]
            original_nmr_list = filtered_df['Biomarker'].tolist()
            metabolomics = metabolomics.filter(items=original_nmr_list + ['eid'])
            
            if fold == 0:
                self.logger.info(f'The number of metabolites in the {prefix} data: {metabolomics.shape[1]-1}.')
                
            eid = metabolomics['eid'].tolist()
            metabolomics.drop(columns=['eid'], inplace=True)
        
        proteomics = None
        if 'Proteomics' in self.predictor_set:
            # Load proteomics data
            proteomics = pd.read_feather(join(self.split_seed_dir, f"X_{prefix}_Proteomics_K{fold}.feather"))
            ukb_olink = pd.read_csv('/your path/multiomics-cardiovascular-disease/data/network_raw/ukb_olink_mapped_entrez.csv')
            ukb_olink = ukb_olink.dropna(subset=['entrezgene_id']) # Exclude 15 proteins with missing Entrez IDs: 
                                                                # DEFA1_DEFA1B, DEFB4A_DEFB4B, EBI3_IL27, FUT3_FUT5, IL12A_IL12B, LGALS7_LGALS7B, MICB_MICA, AMY1A_AMY1B_AMY1C, 
                                                                # BOLA2_BOLA2B, CGB3_CGB5_CGB8, CTAG1A_CTAG1B, DEFB103A_DEFB103B, DEFB104A_DEFB104B, SPACA5_SPACA5B, CKMT1A_CKMT1B
            ukb_olink = ukb_olink[~ukb_olink['Assay'].isin(['GLIPR1', 'NPM1', 'PCOLCE'])] # Exclude 3 proteins with more than 50% missingness
            protein_list = ukb_olink['Assay'].tolist()
            proteomics = proteomics.filter(items=protein_list + ['eid']) # 2923 - 15 - 3 = 2905 proteins
            
            if fold == 0:
                self.logger.info(f'The number of proteins in the {prefix} data: {proteomics.shape[1]-1}.')
                
            eid = proteomics['eid'].tolist()
            proteomics.drop(columns=['eid'], inplace=True)
            
        return eid, metabolomics, proteomics
    
    def _load_outcome_data(self, fold, outcomes_list, prefix):
        y = pd.read_feather(join(self.split_seed_dir, f'y_{prefix}_K{fold}.feather'))
        e = pd.read_feather(join(self.split_seed_dir, f'e_{prefix}_K{fold}.feather'))
        
        y = y[outcomes_list]
        e = e[outcomes_list]
        return y, e


    def get_dataloader(self, fold, outcomes_list):
        self.get_data(fold, outcomes_list)
        
        train_dataloader = self._create_dataloader(
            batch_size=self.batch_size['train'],
            shuffle=self.shuffle,
            eid_list=self.eid_train,
            metabolomics_data=self.metabolomics_train, 
            proteomics_data=self.proteomics_train, 
            y=self.y_train, 
            e=self.e_train,
            ppi=self.ppi if 'Proteomics' in self.predictor_set and self.ppi_require else None
        )
        
        validate_dataloader = self._create_dataloader(
            batch_size=self.batch_size['test'],
            shuffle=self.shuffle,
            eid_list=self.eid_test,
            metabolomics_data=self.metabolomics_test, 
            proteomics_data=self.proteomics_test, 
            y=self.y_test, 
            e=self.e_test,
            ppi=self.ppi if 'Proteomics' in self.predictor_set and self.ppi_require else None
        )
        
        return train_dataloader, validate_dataloader
    
    def _create_dataloader(self, batch_size, shuffle, eid_list, metabolomics_data, proteomics_data, y, e, ppi=None):
        dataset = UKBiobankOmicsDataset(eid_list=eid_list, y=y, e=e, metabolomics_data=metabolomics_data, proteomics_data=proteomics_data, ppi=ppi)
        if batch_size is None:
            batch_size = len(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=self.num_workers, collate_fn=self.__collate_fn)
        return dataloader
    
    def __collate_fn(self, data_list):
        eid = []
        survival_times = []
        event_indicators = []
        
        metabolomics = []
        proteomics = []
        proteomics_graph = []
        
        for data in data_list:
            if 'Metabolomics' in self.predictor_set:
                eid_tensor, metabolomics_tensor, y_tensor, e_tensor = data
                metabolomics.append(metabolomics_tensor)
            elif 'Proteomics' in self.predictor_set:
                if self.ppi_require:
                    eid_tensor, proteomics_tensor, proteomics_graph_tensor, y_tensor, e_tensor = data
                    proteomics_graph.append(proteomics_graph_tensor)
                    proteomics.append(proteomics_tensor)
                else:
                    eid_tensor, proteomics_tensor, y_tensor, e_tensor = data
                    proteomics.append(proteomics_tensor)
            eid.append(eid_tensor)
            survival_times.append(y_tensor)
            event_indicators.append(e_tensor)
        
        eid_batch = torch.stack(eid)
        survival_times = torch.stack(survival_times)
        event_indicators = torch.stack(event_indicators)
        
        metabolomics_batch = torch.stack(metabolomics) if 'Metabolomics' in self.predictor_set else None
        proteomics_batch = torch.stack(proteomics) if 'Proteomics' in self.predictor_set else None
        proteomics_graph_batch = Batch.from_data_list(proteomics_graph) if 'Proteomics' in self.predictor_set and self.ppi_require else None

        return eid_batch, metabolomics_batch, proteomics_batch, proteomics_graph_batch, survival_times, event_indicators

    def get_outcome_column_indices(self, fold, outcomes_list):
        self.get_data(fold, outcomes_list)
        
        self.outcome_column_indices = {outcome: self.y_train.columns.get_loc(outcome) for outcome in outcomes_list}
        
        return self.outcome_column_indices
    
    def get_feature(self, fold, outcomes_list, predictor):
        self.get_data(fold, outcomes_list)
        
        if predictor == 'Clinical':
            return self.clinical_train
        elif predictor == 'Metabolomics':
            return self.metabolomics_train
        elif predictor == 'Proteomics':
            return self.proteomics_train
        
    def get_feature_names(self, fold, outcomes_list, predictor):
        self.get_data(fold, outcomes_list)
        
        if predictor == 'Clinical':
            return self.clinical_features
        elif predictor == 'Metabolomics':
            return self.metabolomics_features
        elif predictor == 'Proteomics':
            return self.proteomics_features