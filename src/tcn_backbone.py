import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. THE TCN ARCHITECTURE (The "Brain") ---
class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        
        # LAYER 1: Sensor Alignment (The Physical Step)
        # Input: (Batch, 1, 128) -> Output: (Batch, 32, 16)
        # Why Kernel=9? It covers 8 features (1 sensor) + 1 neighbor.
        # Why Stride=8? It jumps exactly 1 sensor at a time.
        # Result: We convert "128 numbers" into "16 Sensor Embeddings".
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 2: Global Context (The Temporal Step)
        # Input: (Batch, 32, 16) -> Output: (Batch, 64, 16)
        # Why Dilation=4? It compares Sensor i with Sensor i+4.
        # Result: It detects "Array-Wide" patterns (like global drift).
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 3: Feature Refinement
        # Input: (Batch, 64, 16) -> Output: (Batch, 64, 16)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # LAYER 4: Latent Projection
        # Flatten: 64 channels * 16 sensors = 1024 features
        self.flatten_dim = 64 * 16 
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim) # Normalize for Contrastive Loss
        )

    def forward(self, x):
        # 1. Reshape: (Batch, 128) -> (Batch, 1, 128)
        # The TCN needs a "Channel" dimension.
        x = x.unsqueeze(1)
        
        # 2. Pass through layers
        f1 = self.conv1(x) # Shape: (B, 32, 16)
        f2 = self.conv2(f1) # Shape: (B, 64, 16)
        f3 = self.conv3(f2) # Shape: (B, 64, 16)
        
        # 3. Flatten and Project
        flat = f3.view(f3.size(0), -1) 
        z = self.fc(flat)
        
        return z, f1, f2, f3 # Return intermediates for visualization

# --- 2. VISUALIZATION LOGIC ---
def visualize_tcn_on_real_data():
    print("Running TCN Visualization on gas_data_scaled.csv...")
    
    # Path Handling
    csv_path = 'processed_data/gas_data_scaled.csv'
    if not os.path.exists(csv_path):
        # Fallback if running from root
        csv_path = 'gas_data_scaled.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}. Run preprocess.py first!")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    
    # Pick a sample: Batch 1, Ethanol (Class 1)
    # Note: Adjust logic if your classes are 0-5
    sample = df[(df['Batch_ID'] == 1) & (df['Gas_Class'] == 1)].iloc[0]
    
    # Extract 128 features
    feat_cols = [c for c in df.columns if 'feat_' in c]
    input_data = sample[feat_cols].values.astype(np.float32)
    
    print(f"Visualizing Sample: Batch {int(sample['Batch_ID'])}, Gas {int(sample['Gas_Class'])}")
    
    # Init Model
    model = SiameseEncoder()
    model.eval()
    
    # Run Inference
    with torch.no_grad():
        input_tensor = torch.tensor(input_data).unsqueeze(0) # (1, 128)
        z, f1, f2, f3 = model(input_tensor)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot A: The Scaled Input
    axes[0].plot(input_data, color='black', linewidth=1.5)
    axes[0].set_title("1. TCN INPUT: Scaled Features (Ethanol, Batch 1)\n(Values are normalized to approx -2 to 2)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Feature Index (0-127)")
    axes[0].grid(True, alpha=0.3)
    # Draw red lines to show sensor boundaries (every 8 features)
    for i in range(0, 128, 8):
        axes[0].axvline(i, color='red', linestyle=':', alpha=0.3)

    # Plot B: Layer 1 Output (Sensor Alignment)
    # We plot the mean activation of the 32 filters
    l1_act = f1[0].mean(dim=0).numpy()
    axes[1].plot(l1_act, marker='o', color='blue', linewidth=2)
    axes[1].set_title("2. TCN LAYER 1: Sensor-Aligned Features (16 Points)\n(The 128 raw features are compressed into 16 'Sensor Embeddings')", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Sensor Index (0-15)")
    axes[1].set_xticks(range(16))
    axes[1].grid(True, alpha=0.3)
    
    # Plot C: Latent Output
    axes[2].bar(range(64), z[0].numpy(), color='purple', alpha=0.7)
    axes[2].set_title("3. TCN OUTPUT: Latent Vector 'z' (64 Dimensions)\n(Abstract representation ready for classification)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Latent Dimension (0-63)")
    
    plt.tight_layout()
    plt.show()
    print("✅ Visualization Complete.")

if __name__ == "__main__":
    visualize_tcn_on_real_data()