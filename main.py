import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import GasDataset
from src.models import SiameseEncoder, TaskClassifier, DomainDiscriminator, PhysicsHead
from src.loss import PICDA_Loss
from src.trainer import PICDATrainer

# --- CONFIGURATION ---
CSV_PATH = "processed_data/gas_data_normalized.csv" 
CSV_PATH_ALT = "gas_data_normalized.csv"
SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HYPERPARAMETERS (TUNED FOR STABILITY) ---
BATCH_SIZE = 64
LR_SOURCE = 0.001
# Keep Adaptation LR low to prevent destroying previous knowledge
LR_ADAPT = 0.00001 

EPOCHS_SOURCE = 20
EPOCHS_ADAPT = 15

def calculate_drift_direction(df):
    b1_mean = df[df['Batch_ID'] == 1]['feat_0'].mean()
    b10_mean = df[df['Batch_ID'] == 10]['feat_0'].mean()
    diff = b10_mean - b1_mean
    direction = 1.0 if diff > 0 else -1.0
    print(f"\n[PHYSICS CHECK] Drift Direction: {direction} (Diff: {diff:.4f})")
    return direction

def main():
    print("--- STARTING PI-CDA EXPERIMENT (GRADIENT CLIPPING ENABLED) ---")
    
    # 1. LOAD DATA
    if os.path.exists(CSV_PATH): df = pd.read_csv(CSV_PATH)
    elif os.path.exists(CSV_PATH_ALT): df = pd.read_csv(CSV_PATH_ALT)
    else: return
        
    DRIFT_DIR = calculate_drift_direction(df)
    
    # 2. MODELS
    encoder = SiameseEncoder(input_dim=128, latent_dim=64)
    classifier = TaskClassifier(latent_dim=64, num_classes=6)
    discriminator = DomainDiscriminator(latent_dim=64)
    phy_head = PhysicsHead(latent_dim=64)
    
    # 3. LOSS CONFIGURATION
    # We significantly lower lambda_adv to prevent the "86.5" loss explosion
    criterion = PICDA_Loss(
        lambda_cont=1.0,   # Strong Anchor (Keep this high)
        lambda_power=0.1,  # Physics
        lambda_adv=0.05,   # LOWERED from 0.2 -> 0.05 (Gentle Nudge)
        lambda_mono=1.0    # Physics
    )
    
    trainer = PICDATrainer(
        encoder, classifier, discriminator, phy_head, criterion, 
        DEVICE, SAVE_DIR
    )
    
    # --- PHASE 1 ---
    print("\n[PHASE 1] Source Training")
    ds_src = GasDataset(df, batch_id=1)
    loader_src = DataLoader(ds_src, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    prev_baseline = trainer.train_source_phase(loader_src, epochs=EPOCHS_SOURCE, lr=LR_SOURCE)
    trainer.evaluate(loader_src)

    # --- PHASE 2 ---
    print("\n[PHASE 2] Sequential Adaptation")
    results = []
    
    for b_id in range(2, 11):
        ds_tgt = GasDataset(df, batch_id=b_id)
        if len(ds_tgt) < BATCH_SIZE: continue
        loader_tgt = DataLoader(ds_tgt, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        acc_before = trainer.evaluate(loader_tgt)
        
        new_baseline = trainer.adapt_target_phase(
            loader_src, loader_tgt, b_id, prev_baseline, DRIFT_DIR, 
            epochs=EPOCHS_ADAPT, lr=LR_ADAPT
        )
        
        acc_after = trainer.evaluate(loader_tgt)
        print(f"Batch {b_id} Result: {acc_before:.1f}% -> {acc_after:.1f}%")
        
        results.append({
            'Batch': b_id,
            'Acc_Before': acc_before,
            'Acc_After': acc_after,
            'Baseline': new_baseline
        })
        prev_baseline = new_baseline

    # --- FINAL REPORTING ---
    print("\n" + "="*40)
    print(" FINAL RESULTS SUMMARY")
    print("="*40)
    results_df = pd.DataFrame(results)
    
    # Display Table
    print(results_df[['Batch', 'Acc_Before', 'Acc_After', 'Baseline']].to_string(index=False))
    
    # Calculate Averages
    avg_before = results_df['Acc_Before'].mean()
    avg_after = results_df['Acc_After'].mean()
    
    print("-" * 40)
    print(f"AVERAGE ACCURACY (Before Adapt): {avg_before:.2f}%")
    print(f"AVERAGE ACCURACY (After Adapt):  {avg_after:.2f}%")
    print("-" * 40)
    
    if not os.path.exists("results"): os.makedirs("results")
    results_df.to_csv("results/final_accuracy.csv", index=False)

if __name__ == "__main__":
    main()