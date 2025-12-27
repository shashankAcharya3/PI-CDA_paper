import numpy as np
import os
from src.config import NUM_FEATURES

def parse_custom_line(line):
    """
    Parses your specific line format:
    '1;10.000000 1:15596.16... 2:1.86...'
    """
    parts = line.strip().split()
    if not parts:
        return None, None

    # --- 1. Parse Label ---
    # The first token is '1;10.000000' (Label;Concentration)
    first_token = parts[0]
    
    if ';' in first_token:
        # Split '1;10.000' -> ['1', '10.000']
        label_str = first_token.split(';')[0]
        label = int(float(label_str))
    else:
        # Fallback if no semicolon
        label = int(float(first_token))

    # --- 2. Parse Features ---
    # The rest of the tokens are 'index:value'
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    
    for p in parts[1:]:
        if ':' in p:
            idx_str, val_str = p.split(':')
            # Convert 1-based index (UCI style) to 0-based (Python style)
            idx = int(idx_str) - 1
            if 0 <= idx < NUM_FEATURES:
                features[idx] = float(val_str)
                
    return features, label

def apply_polarity_correction(X):
    """
    Implements Eq (1) from the paper.
    Ensures sensor readings follow physical direction constraints.
    """
    X_corr = X.copy()
    for i in range(NUM_FEATURES):
        # Paper uses 1-based indexing for this rule:
        # if (i%8) is 1..5 -> Positive
        # else -> Negative
        
        feat_idx_1based = i + 1
        mod_val = feat_idx_1based % 8
        
        if 1 <= mod_val <= 5:
            X_corr[:, i] = np.abs(X_corr[:, i])
        else:
            X_corr[:, i] = -np.abs(X_corr[:, i])
            
    return X_corr

def load_batch(file_path):
    data = []
    labels = []
    
    print(f"Loading {os.path.basename(file_path)}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None

    with open(file_path, 'r') as f:
        for line in f:
            feat, lbl = parse_custom_line(line)
            if feat is not None:
                data.append(feat)
                labels.append(lbl)
                
    if not data:
        raise ValueError("File loaded but no valid data found.")

    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    # Apply the physics correction immediately
    X = apply_polarity_correction(X)
    
    return X, y