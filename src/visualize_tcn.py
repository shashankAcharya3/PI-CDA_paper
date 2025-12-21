import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from models import SiameseEncoder

def visualize_tcn_flow():
    print("--- VISUALIZING TCN DATA FLOW ---")
    
    # 1. Load the Normalized Data
    # Try both common paths
    paths = ["processed_data/gas_data_normalized.csv", "gas_data_normalized.csv"]
    csv_path = None
    for p in paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if csv_path is None:
        print("❌ Error: gas_data_normalized.csv not found. Run strict_preprocessing first!")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Extract Samples
    # Sample A: Batch 1 (Source)
    # Sample B: Batch 10 (Target - Drifted)
    # We pick Class 1 (Ethanol) for both to compare apples to apples
    src_sample = df[(df['Batch_ID'] == 1) & (df['Gas_Class'] == 1)].iloc[0]
    tgt_sample = df[(df['Batch_ID'] == 10) & (df['Gas_Class'] == 1)].iloc[0]
    
    feat_cols = [c for c in df.columns if 'feat_' in c]
    
    src_data = torch.tensor(src_sample[feat_cols].values.astype(np.float32)).unsqueeze(0)
    tgt_data = torch.tensor(tgt_sample[feat_cols].values.astype(np.float32)).unsqueeze(0)
    
    # 3. Run Model
    model = SiameseEncoder()
    model.eval()
    
    with torch.no_grad():
        z_src, f1_src, _, _ = model(src_data)
        z_tgt, f1_tgt, _, _ = model(tgt_data)

    # 4. Plot Comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # PLOT 1: Input Comparison (Drift Visible?)
    axes[0].plot(src_data[0].numpy(), color='blue', alpha=0.7, label='Batch 1 (Source)')
    axes[0].plot(tgt_data[0].numpy(), color='red', alpha=0.7, label='Batch 10 (Target)')
    axes[0].set_title("1. INPUT: Normalized Features (Ethanol)\nNotice the Red line (Batch 10) is shifted/lower due to Drift", fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PLOT 2: Layer 1 Activations (Sensor Alignment)
    # We check if the TCN still finds the same active sensors despite drift
    l1_src = f1_src[0].mean(dim=0).numpy()
    l1_tgt = f1_tgt[0].mean(dim=0).numpy()
    
    axes[1].plot(l1_src, marker='o', color='blue', label='Batch 1 Activation')
    axes[1].plot(l1_tgt, marker='x', color='red', linestyle='--', label='Batch 10 Activation')
    axes[1].set_title("2. TCN LAYER 1: Sensor Alignment (16 Points)\nDo the peaks align? (e.g. if Sensor 4 is high in Blue, is it high in Red?)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Sensor Index (0-15)")
    axes[1].set_xticks(range(16))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # PLOT 3: Latent Output z
    axes[2].bar(np.arange(64) - 0.2, z_src[0].numpy(), width=0.4, color='blue', label='Batch 1 z')
    axes[2].bar(np.arange(64) + 0.2, z_tgt[0].numpy(), width=0.4, color='red', label='Batch 10 z')
    axes[2].set_title("3. OUTPUT: Latent Code z (Untrained)\nBefore training, these look different. Our goal is to make them match.", fontsize=11, fontweight='bold')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    print("✅ Visualization Complete.")

if __name__ == "__main__":
    visualize_tcn_flow()