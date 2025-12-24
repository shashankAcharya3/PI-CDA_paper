import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

# Import your modules
sys.path.append(os.path.abspath("."))
from src.data_loader import GasDataset
from src.models import SiameseEncoder, TaskClassifier, PhysicsHead

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "processed_data/gas_data_normalized.csv"
if not os.path.exists(CSV_PATH): CSV_PATH = "gas_data_normalized.csv"

def load_source_model():
    print("Loading Source Model (Batch 1)...")
    path = "checkpoints/source_model.pth"
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found. Run Phase 1 training first.")
        sys.exit()
        
    # Init Models
    enc = SiameseEncoder(128, 64).to(DEVICE)
    cls = TaskClassifier(64, 6).to(DEVICE)
    phy = PhysicsHead(64).to(DEVICE)
    
    # Load Weights
    ckpt = torch.load(path, map_location=DEVICE)
    enc.load_state_dict(ckpt['enc'])
    cls.load_state_dict(ckpt['cls'])
    phy.load_state_dict(ckpt['phy'])
    
    enc.eval(); cls.eval(); phy.eval()
    return enc, cls, phy

def analyze_batch(enc, cls, batch_id, df):
    print(f"\n--- ANALYZING BATCH {batch_id} ---")
    ds = GasDataset(df, batch_id=batch_id)
    if len(ds) == 0: return
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            z = enc(x)
            logits = cls(z)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())
            all_confs.extend(conf.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)
    
    # 1. Overall Accuracy
    acc = (all_preds == all_labels).mean() * 100
    print(f"  > Raw Accuracy: {acc:.2f}%")
    
    # 2. The "Lie Detector" (Confidence Analysis)
    # We check accuracy at different confidence levels
    high_conf_mask = all_confs > 0.8
    if high_conf_mask.sum() > 0:
        high_conf_acc = (all_preds[high_conf_mask] == all_labels[high_conf_mask]).mean() * 100
        print(f"  > High Conf (>0.8) Samples: {high_conf_mask.sum()} / {len(all_labels)}")
        print(f"  > Accuracy on High Conf: {high_conf_acc:.2f}%")
        
        if high_conf_acc < 50.0:
            print("    ⚠️ CRITICAL: Model is Hallucinating! (Confident but Wrong)")
    else:
        print("  > No samples with confidence > 0.8")

    # 3. Class Confusion (Which class disappears?)
    unique, counts = np.unique(all_preds, return_counts=True)
    print(f"  > Predicted Class Distribution: {dict(zip(unique, counts))}")
    if len(unique) < 6:
        print(f"    ⚠️ MODE COLLAPSE WARNING: Only {len(unique)}/6 classes predicted.")

    return acc

def main():
    if not os.path.exists("plots"): os.makedirs("plots")
    
    df = pd.read_csv(CSV_PATH)
    enc, cls, phy = load_source_model()
    
    accuracies = []
    
    for b_id in range(2, 11):
        acc = analyze_batch(enc, cls, b_id, df)
        accuracies.append(acc)
        
    print("\n--- DIAGNOSIS SUMMARY ---")
    print("If 'Raw Accuracy' > 'Your Training Accuracy', your training is hurting.")
    print("If 'High Conf Accuracy' is low, your pseudo-labels are poisoning the model.")

if __name__ == "__main__":
    main()