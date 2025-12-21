import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

# Import Modules
from src.data_loader import GasDataset
from src.models import SiameseEncoder, TaskClassifier, DomainDiscriminator, PhysicsHead
from src.loss import PICDA_Loss
from src.trainer import PICDATrainer

# --- CONFIG ---
CSV_PATH = "processed_data/gas_data_scaled.csv"
SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

def calculate_drift_direction(df):
    """
    Determines if drift is Positive (+1) or Negative (-1)
    based on the difference between Batch 10 and Batch 1.
    """
    b1_mean = df[df['Batch_ID'] == 1]['feat_0'].mean()
    b10_mean = df[df['Batch_ID'] == 10]['feat_0'].mean()
    
    diff = b10_mean - b1_mean
    direction = 1.0 if diff > 0 else -1.0
    print(f"\n[PHYSICS CHECK] Drift ({b10_mean:.2f} - {b1_mean:.2f}) = {diff:.2f}")
    print(f"VERDICT: Using Drift Direction = {direction}")
    return direction

def main():
    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        # Fallback
        CSV_PATH_ALT = "gas_data_scaled.csv"
        if os.path.exists(CSV_PATH_ALT):
            df = pd.read_csv(CSV_PATH_ALT)
        else:
            print("❌ Data not found. Run preprocess.py!")
            return
    else:
        df = pd.read_csv(CSV_PATH)
        
    print(f"Data Loaded: {len(df)} samples.")
    
    # 2. Physics Setup
    DRIFT_DIR = calculate_drift_direction(df)
    
    # 3. Model Setup
    encoder = SiameseEncoder()
    classifier = TaskClassifier()
    discriminator = DomainDiscriminator()
    phy_head = PhysicsHead()
    criterion = PICDA_Loss()
    
# Reduce lambda_power to 0.1 so it doesn't overpower the classification task
    criterion = PICDA_Loss(lambda_cont=0.1, lambda_power=0.1, lambda_adv=0.5, lambda_mono=1.0)

    trainer = PICDATrainer(encoder, classifier, discriminator, phy_head, criterion, DEVICE, SAVE_DIR)
    
    # --- PHASE 1: SOURCE ANCHORING ---
    ds_src = GasDataset(df, batch_id=1)
    loader_src = DataLoader(ds_src, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Train Batch 1
    prev_baseline = trainer.train_source_phase(loader_src, epochs=20)
    
    # Baseline Accuracy
    acc_b1 = trainer.evaluate(loader_src)
    print(f"Batch 1 Baseline Accuracy: {acc_b1:.2f}%")
    
    # --- PHASE 2: SEQUENTIAL ADAPTATION ---
    print("\n--- STARTING ADAPTATION (Batch 2 -> 10) ---")
    results = []
    
    # Loop through batches 2, 3, ... 10
    for b_id in range(2, 11):
        ds_tgt = GasDataset(df, batch_id=b_id)
        # Skip empty batches if any
        if len(ds_tgt) < BATCH_SIZE: continue
        
        loader_tgt = DataLoader(ds_tgt, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        # Eval Before
        acc_before = trainer.evaluate(loader_tgt)
        
        # Adapt
        new_baseline = trainer.adapt_target_phase(
            loader_src, loader_tgt, 
            batch_id=b_id, 
            prev_baseline=prev_baseline, 
            drift_dir=DRIFT_DIR,
            epochs=15 # Gives enough time to align
        )
        
        # Eval After
        acc_after = trainer.evaluate(loader_tgt)
        print(f"Batch {b_id} Result: {acc_before:.1f}% -> {acc_after:.1f}%")
        
        results.append({'Batch': b_id, 'Acc_Before': acc_before, 'Acc_After': acc_after})
        
        # Update Chain
        prev_baseline = new_baseline
        
    print("\n--- FINAL RESULTS TABLE ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()