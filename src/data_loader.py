import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def parse_uci_dataset(raw_path, save_path):
    """
    Reads raw UCI .dat files and converts them into a single CSV.
    Fixes the '1, 10, 2' sorting bug by enforcing numerical order.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    output_file = os.path.join(save_path, "gas_data_full.csv")
    
    # Optional: Delete old file to force re-parsing
    if os.path.exists(output_file):
        os.remove(output_file) 
        print("Removed old out-of-order CSV. Re-parsing...")

    print("Parsing raw .dat files...")
    all_data = []
    
    # 1. Get all files
    file_list = glob.glob(os.path.join(raw_path, "batch*.dat"))
    
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .dat files found in {raw_path}")

    # 2. THE FIX: Sort Numerically (1, 2, ..., 9, 10)
    # We extract the number from "batch10.dat" -> 10 and sort by that.
    file_list.sort(key=lambda f: int(os.path.basename(f).lower().replace("batch", "").replace(".dat", "")))
    
    print(f"Processing order: {[os.path.basename(f) for f in file_list]}")

    # 3. Loop through strictly ordered files
    for file_path in file_list:
        filename = os.path.basename(file_path)
        # Extract Batch ID safely
        batch_id = int(filename.lower().replace("batch", "").replace(".dat", ""))
        
        print(f"  -> Parsing Batch {batch_id}...")

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                # Header Split
                header = parts[0]
                if ';' in header:
                    class_label, conc = header.split(';')
                else:
                    class_label, conc = header, -1.0
                
                # Sparse to Dense Features
                features = np.zeros(128)
                for item in parts[1:]:
                    if ':' in item:
                        idx_str, val_str = item.split(':')
                        feat_idx = int(idx_str) - 1 
                        if 0 <= feat_idx < 128:
                            features[feat_idx] = float(val_str)
                
                # Build Row
                row = {
                    'Batch_ID': batch_id,
                    'Gas_Class': int(class_label),
                    'Concentration': float(conc)
                }
                for i in range(128):
                    row[f'feat_{i}'] = features[i]
                
                all_data.append(row)
    
    # 4. Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(df)} rows to {output_file} in chronological order.")
    return df

class GasDataset(Dataset):
    def __init__(self, dataframe, batch_id=None):
        if batch_id is not None:
            self.data = dataframe[dataframe['Batch_ID'] == batch_id].reset_index(drop=True)
        else:
            self.data = dataframe
            
        self.features = self.data.iloc[:, 3:].values.astype(np.float32)
        self.labels = self.data['Gas_Class'].values.astype(np.int64) - 1
        self.concs = self.data['Concentration'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.concs[idx]

if __name__ == "__main__":
    raw_dir = "./raw_data"
    proc_dir = "./processed_data"
    parse_uci_dataset(raw_dir, proc_dir)