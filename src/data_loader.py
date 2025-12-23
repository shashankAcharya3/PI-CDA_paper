import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class GasDataset(Dataset):
    def __init__(self, dataframe, batch_id=None):
        if batch_id is not None:
            self.data = dataframe[dataframe['Batch_ID'] == batch_id].reset_index(drop=True)
        else:
            self.data = dataframe
            
        feat_cols = [c for c in self.data.columns if 'feat_' in c]
        self.features = self.data[feat_cols].values.astype(np.float32)
        
        labels = self.data['Gas_Class'].values.astype(np.int64)
        if labels.min() == 1: labels = labels - 1
        self.labels = labels
        
        self.concs = self.data['Concentration'].values.astype(np.float32) if 'Concentration' in self.data.columns else np.zeros(len(self.data), dtype=np.float32)
        
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.concs[idx]