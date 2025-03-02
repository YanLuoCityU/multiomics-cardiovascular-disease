import os
import torch

import pandas as pd
from collections import defaultdict
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

class BaseTrainer:
    def __init__(self, config, model, criterion, optimizer, lr_scheduler,
                 train_dataloader, validate_dataloader, test_dataloader, 
                 logger, writer, fold, outcome_column_indices, tuning=False):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader
        
        self.logger = logger
        self.writer = writer
        
        self.fold = fold
        self.outcome_column_indices = outcome_column_indices
        
        self.tuning = tuning
        self.epochs = self.config['trainer']['epochs']
        self.early_stop = self.config['trainer']['early_stop']

    def train(self):
        best_cindex = -1
        not_improved_count = 0

        for epoch in range(1, self.epochs + 1):
            avg_train_loss, avg_train_loss_dict = self._train_epoch()
            avg_val_loss, avg_val_loss_dict = self._validate_epoch()

            # Log training and validation loss
            self._log_losses(avg_train_loss, avg_val_loss, avg_train_loss_dict, avg_val_loss_dict, epoch)
            
            
            train_cindex, mean_train_cindex = self.get_test_cindex(self.model, self.train_dataloader)
            val_cindex, mean_val_cindex = self.get_test_cindex(self.model, self.validate_dataloader)
            test_cindex, mean_test_cindex = self.get_test_cindex(self.model, self.test_dataloader)
            
            # Early stopping logic
            if mean_val_cindex > best_cindex:
                best_cindex = mean_val_cindex
                not_improved_count = 0
                self._save_checkpoint(epoch)
                self.logger.info(f"Saving model at epoch {epoch} with the best validation C-index {best_cindex}.")
            else:
                not_improved_count += 1
                if not_improved_count >= self.early_stop:
                    self.logger.info(f"Early stopping at epoch {epoch} with the best validation C-index {best_cindex}.")
                    break
                
            # Adjust the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
            # Log testing C-index
            self._log_cindex(train_cindex, val_cindex, test_cindex, mean_train_cindex, mean_val_cindex, mean_test_cindex, epoch)
            
            if epoch % 3 == 0:
                self.logger.info(f"Epoch {epoch} Fold {self.fold}: Average training loss = {avg_train_loss}, Validation loss = {avg_val_loss},\n     Training c-index = {mean_train_cindex}, Validation c-index = {mean_val_cindex}, Testing c-index = {mean_test_cindex}") 
    
    def _train_epoch(self):
        """Implement training logic for one epoch."""
        raise NotImplementedError

    def _validate_epoch(self):
        """Implement validation logic for one epoch."""
        raise NotImplementedError

    def _test_model(self, model, dataloader, return_y_e=False):
        model.eval()
        outputs_dict = defaultdict(list)
        eid_list = []
        y_list = []
        e_list = []

        with torch.no_grad():
            for batch in dataloader:
                eid, outputs, y, e = self._model_batch(batch, model)
                
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

    def _model_batch(self):
        raise NotImplementedError
    
    def get_test_scores(self, model, dataloader):
        test_outputs, test_eid = self._test_model(model, dataloader, return_y_e=False)

        test_scores = pd.DataFrame(test_eid.numpy(), columns=['eid'])
        for outcome in test_outputs:
            test_scores[outcome] = test_outputs[outcome].cpu().numpy()

        return test_scores
    
    def get_test_cindex(self, model, dataloader):
        test_outputs, test_eid, test_y, test_e = self._test_model(model, dataloader, return_y_e=True)

        cindex_dict = {'outcome': [], 'cindex': []}
        for outcome, index in self.outcome_column_indices.items():
            df = pd.DataFrame(test_outputs[outcome].numpy(), columns=['prediction'])
            df['duration'] = test_y[:, index].numpy()
            df['event'] = test_e[:, index].numpy()
            cindex = 1 - concordance_index(event_times=df['duration'], predicted_scores=df['prediction'], event_observed=df['event'])
            cindex_dict['outcome'].append(outcome)
            cindex_dict['cindex'].append(cindex)

        test_cindex = pd.DataFrame(cindex_dict).set_index('outcome').transpose()
        mean_cindex = test_cindex.mean(axis=1).values[0]

        return test_cindex, mean_cindex    
    
    def _log_losses(self, avg_train_loss, avg_val_loss, avg_train_loss_dict, avg_val_loss_dict, epoch):
        self.writer.add_scalars(f'Fold_{self.fold}/Losses', {
            'Train': avg_train_loss,
            'Validation': avg_val_loss
        }, epoch)
        
        for outcome in self.config['outcomes_list']:
            self.writer.add_scalars(f'Fold_{self.fold}/Losses_{outcome}', {
                'Train': avg_train_loss_dict[outcome], 
                'Validation': avg_val_loss_dict[outcome]
            }, epoch)
            
    def _log_cindex(self, train_cindex, val_cindex, test_cindex, mean_train_cindex, mean_val_cindex, mean_test_cindex, epoch):
        self.writer.add_scalars(f'Fold_{self.fold}/Cindex', {
            'Train': mean_train_cindex,
            'Validation': mean_val_cindex,
            'Test': mean_test_cindex
        }, epoch)
        
        for outcome in self.config['outcomes_list']:
            self.writer.add_scalars(f'Fold_{self.fold}/Cindex_{outcome}', {
                'Train': train_cindex[outcome].values[0], 
                'Validation': val_cindex[outcome].values[0],
                'Test': test_cindex[outcome].values[0]
            }, epoch)
        
    def _calculate_loss_dict(self, outputs, e):
        loss_dict = {}
        for outcome in self.config['outcomes_list']:
            # Get predicted risk, true event indicator for the specific outcome
            predicted_risk = outputs[outcome]
            event = e[:, self.outcome_column_indices[outcome]].unsqueeze(1)
            
            # Calculate loss for the specific outcome
            loss = self.criterion(predicted_risk, event)
            loss_dict[outcome] = loss
        return loss_dict
    
    def _calculate_loss(self, loss, mean=True):
        if isinstance(loss, dict):
            total_loss = sum(loss.values())
            num_losses = len(loss)
        elif isinstance(loss, list):
            total_loss = sum(loss)
            num_losses = len(loss)
        else:
            raise TypeError("loss should be either a dict or a list")

        return total_loss / num_losses if mean else total_loss

    def _save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        model_dir = f"{self.config['model_dir']}/{self.config['name']}/{'_'.join(self.config['predictor_set'])}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.tuning:
            filename = f"{model_dir}/K{self.fold}_tuning.pth"
        else:
            filename = f"{model_dir}/K{self.fold}.pth"
        torch.save(state, filename)

'''
Trainer for MLP-based models
'''
class MLPTrainer(BaseTrainer):
    def _train_epoch(self):
        self.model.train()
        train_losses = []
        cumulative_loss_dict = {outcome: 0.0 for outcome in self.config['outcomes_list']}
        
        for batch in self.train_dataloader:
            _, _, _, metabolomics_data, proteomics_data, y, e = self._process_batch(batch)
            if 'Metabolomics' in self.config['predictor_set']:
                outputs = self.model(metabolomics_data)
            elif 'Proteomics' in self.config['predictor_set']:
                outputs = self.model(proteomics_data)
            
            self.optimizer.zero_grad()
            train_loss_dict = self._calculate_loss_dict(outputs=outputs, e=e)
            train_loss = self._calculate_loss(loss=train_loss_dict, mean=True)
            train_losses.append(train_loss.item())
            train_loss.backward()
            self.optimizer.step()
            
            for outcome, loss in train_loss_dict.items():
                cumulative_loss_dict[outcome] += loss.item()
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_loss_dict = {outcome: cumulative_loss / len(self.train_dataloader) for outcome, cumulative_loss in cumulative_loss_dict.items()}
        return avg_train_loss, avg_train_loss_dict

    def _validate_epoch(self):
        self.model.eval()
        val_losses = []
        cumulative_loss_dict = {outcome: 0.0 for outcome in self.config['outcomes_list']}
        
        with torch.no_grad():
            for batch in self.validate_dataloader:
                _, _, _, metabolomics_data, proteomics_data, y, e = self._process_batch(batch)
                if 'Metabolomics' in self.config['predictor_set']:
                    outputs = self.model(metabolomics_data)
                elif 'Proteomics' in self.config['predictor_set']:
                    outputs = self.model(proteomics_data)

                val_loss_dict = self._calculate_loss_dict(outputs=outputs, e=e)
                val_loss = self._calculate_loss(loss=val_loss_dict, mean=True)
                val_losses.append(val_loss.item())
                
                for outcome, loss in val_loss_dict.items():
                    cumulative_loss_dict[outcome] += loss.item()
                    
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_loss_dict = {outcome: cumulative_loss / len(self.validate_dataloader) for outcome, cumulative_loss in cumulative_loss_dict.items()}
        return avg_val_loss, avg_val_loss_dict

    def _process_batch(self, batch):
        eid, clinical_data, genomics_data, metabolomics_data, proteomics_data, y, e = batch
            
        return eid, clinical_data.to(self.device).float(), genomics_data.to(self.device).float(), metabolomics_data.to(self.device).float(), \
            proteomics_data.to(self.device).float(), y.to(self.device).float(), e.to(self.device).float()
            
    def _model_batch(self, batch, model):
        eid, _, _, metabolomics_data, proteomics_data, y, e = self._process_batch(batch)
        if 'Metabolomics' in self.config['predictor_set']:
            outputs = model(metabolomics_data)
        elif 'Proteomics' in self.config['predictor_set']:
            outputs = model(proteomics_data)
        return eid, outputs, y, e
