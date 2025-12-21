import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# ==========================================
# 1. MODEL DEFINITIONS (TCN + Classifier)
# ==========================================

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        
        # LAYER 1: Sensor Alignment (Kernel=9, Stride=8)
        # Input: (Batch, 1, 128) -> Output: (Batch, 32, 16)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 2: Global Context (Dilation=4)
        # Input: (Batch, 32, 16) -> Output: (Batch, 64, 16)
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
        
        # LATENT PROJECTION
        self.flatten_dim = 64 * 16 
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

    def forward(self, x):
        # Reshape: (Batch, 128) -> (Batch, 1, 128)
        x = x.unsqueeze(1)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        flat = f3.view(f3.size(0), -1) 
        z = self.fc(flat)
        return z

class TaskClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=6):
        super(TaskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
    def forward(self, z):
        return self.net(z)

# ==========================================
# 2. DATA HANDLING
# ==========================================

class GasDataset(Dataset):
    def __init__(self, dataframe):
        # Extract features (feat_0 to feat_127)
        feat_cols = [c for c in dataframe.columns if 'feat_' in c]
        self.features = dataframe[feat_cols].values.astype(np.float32)
        
        # Extract Labels (Ensure 0-5 range)
        # If labels are 1-6, subtract 1. If 0-5, keep as is.
        labels = dataframe['Gas_Class'].values.astype(np.int64)
        if labels.min() == 1:
            labels = labels - 1
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==========================================
# 3. TEST LOGIC
# ==========================================

def test_classifier_head():
    print("--- TESTING HEAD 1: TASK CLASSIFIER ---")
    
    # 1. Load Data
    # Check multiple locations for the file
    paths = ["processed_data/gas_data_normalized.csv", "gas_data_normalized.csv"]
    csv_path = None
    for p in paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if csv_path is None:
        print("❌ Error: gas_data_normalized.csv not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for Batch 1 (Source Domain)
    batch1_df = df[df['Batch_ID'] == 1].copy()
    print(f"Loaded Batch 1: {len(batch1_df)} samples.")
    
    # Setup Loader
    dataset = GasDataset(batch1_df)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Models
    encoder = SiameseEncoder()
    classifier = TaskClassifier()
    
    # 3. UNTRAINED EVALUATION (Baseline)
    print("\n[Step A] Measuring Untrained Performance...")
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for x, y in loader:
            z = encoder(x)
            logits = classifier(z)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_true.extend(y.numpy())
            
    acc_untrained = accuracy_score(all_true, all_preds)
    print(f"  > Untrained Accuracy: {acc_untrained:.2%} (Should be ~16%)")
    
    # 4. TRAINING LOOP (Proof of Concept)
    print("\n[Step B] Training for 15 Epochs to verify learning...")
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    encoder.train()
    classifier.train()
    
    for epoch in range(15):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for x, y in loader:
            optimizer.zero_grad()
            
            # Forward
            z = encoder(x)
            logits = classifier(z)
            
            # Loss
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        avg_loss = epoch_loss / len(loader)
        acc = correct / total
        loss_history.append(avg_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss {avg_loss:.4f} | Acc {acc:.2%}")
            
    # 5. TRAINED EVALUATION
    print("\n[Step C] Measuring Trained Performance...")
    encoder.eval()
    classifier.eval()
    
    all_preds_final = []
    all_true_final = []
    
    with torch.no_grad():
        for x, y in loader:
            z = encoder(x)
            logits = classifier(z)
            preds = logits.argmax(dim=1)
            all_preds_final.extend(preds.numpy())
            all_true_final.extend(y.numpy())
            
    acc_final = accuracy_score(all_true_final, all_preds_final)
    print(f"  > Trained Accuracy: {acc_final:.2%} (Should be > 95%)")
    
    # 6. PLOTTING
    print("\nGenerating Report Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Loss
    axes[0].plot(range(1, 16), loss_history, marker='o', color='red')
    axes[0].set_title("Training Loss (Batch 1)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    
    # Plot Untrained Matrix
    cm_un = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm_un, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1])
    axes[1].set_title(f"Untrained Confusion Matrix\nAcc: {acc_untrained:.1%}")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True Label")
    
    # Plot Trained Matrix
    cm_tr = confusion_matrix(all_true_final, all_preds_final)
    sns.heatmap(cm_tr, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[2])
    axes[2].set_title(f"Trained Confusion Matrix\nAcc: {acc_final:.1%}")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True Label")
    
    plt.tight_layout()
    plt.savefig("head1_test_results.png")
    plt.show()
    print("✅ Test Complete. Results saved to head1_test_results.png")

if __name__ == "__main__":
    test_classifier_head()