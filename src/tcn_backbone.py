import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. THE SENSOR-ALIGNED TCN (Optimized Configuration) ---
class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        
        # LAYER 1: Sensor Alignment
        # Input: (Batch, 1, 128) -> Output: (Batch, 32, 16)
        # Kernel=9 covers one full sensor (8 features) + 1 neighbor overlap.
        # Stride=8 jumps exactly one sensor at a time.
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 2: Array-Wide Context (Dilation)
        # Input: (Batch, 32, 16) -> Output: (Batch, 64, 16)
        # Dilation=4 compares Sensor 1 with Sensor 5
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 3: Refinement
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Flatten and Project
        self.flatten_dim = 64 * 16 
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

    def forward(self, x):
        # Reshape for TCN: (Batch, 128) -> (Batch, 1, 128)
        x = x.unsqueeze(1) 
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        flat = f3.view(f3.size(0), -1) 
        z = self.fc(flat)
        return z, f1, f2, f3

# --- 2. VISUALIZATION FUNCTION (USING REAL DATA) ---
def visualize_real_data():
    print("Loading real data from processed_data/gas_data_full.csv...")
    
    # Path handling: works if running from 'src' or project root
    file_path = "processed_data/gas_data_full.csv"
    if not os.path.exists(file_path):
        file_path = "../processed_data/gas_data_full.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find gas_data_full.csv. Expected at {file_path}")

    # Load Data
    df = pd.read_csv(file_path)
    
    # FILTER: Get a sample of Ethanol (Class 1) from Batch 1
    # Note: In our parser, Gas_Class is 1-6.
    sample_row = df[(df['Batch_ID'] == 1) & (df['Gas_Class'] == 1)].iloc[0]
    
    print(f"Visualizing Sample: Batch {int(sample_row['Batch_ID'])}, Gas Class {int(sample_row['Gas_Class'])}, Conc {sample_row['Concentration']}")

    # Extract Features (feat_0 to feat_127)
    # We iterate 0 to 127 to ensure order
    feature_cols = [f'feat_{i}' for i in range(128)]
    input_numpy = sample_row[feature_cols].values.astype(np.float32)
    
    # Convert to Tensor (Batch Size 1)
    input_tensor = torch.tensor(input_numpy).unsqueeze(0) # Shape: (1, 128)

    # Init Model
    model = SiameseEncoder()
    model.eval()

    # Run Inference
    with torch.no_grad():
        z, f1, f2, f3 = model(input_tensor)

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # PLOT A: Raw Real Data
    axes[0].plot(input_numpy, color='black', alpha=0.8, label='Raw Sensor Features')
    axes[0].set_title(f"1. REAL INPUT: 128 Features (Ethanol, Batch 1)\nNotice the repeating patterns every 8 features (Sensor blocks)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Feature Index (0-127)")
    # Draw vertical lines every 8 features to show sensor boundaries
    for i in range(0, 128, 8):
        axes[0].axvline(i, color='grey', linestyle='--', alpha=0.2)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # PLOT B: TCN Layer 1 Output (Sensor Alignment)
    # We plot the mean activation across the 32 filters
    layer1_act = torch.mean(f1[0], dim=0).numpy()
    
    axes[1].plot(layer1_act, color='blue', marker='o', linewidth=2, label='Filter Activation')
    axes[1].set_title("2. INSIDE TCN (Layer 1): Sensor-Aligned Feature Map\nThe 128 features are compressed to 16 points (One per Sensor)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Sensor Index (0-15)")
    axes[1].set_xticks(range(16))
    axes[1].set_ylabel("Activation Intensity")
    axes[1].grid(True, alpha=0.3)
    
    # Highlight highest activation (Strongest reacting sensor)
    max_idx = np.argmax(layer1_act)
    axes[1].annotate(f'Strongest Sensor #{max_idx+1}', xy=(max_idx, layer1_act[max_idx]), 
                     xytext=(max_idx, layer1_act[max_idx]+0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    # PLOT C: Latent Output
    axes[2].bar(range(64), z[0].numpy(), color='purple', alpha=0.7)
    axes[2].set_title("3. OUTPUT: Latent Vector 'z' (64 Dim)\nReady for Physics Constraints", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Latent Dimension (0-63)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_real_data()