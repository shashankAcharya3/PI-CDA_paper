import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your modules
from src.data_loader import GasDataset
from src.models import SiameseEncoder, TaskClassifier, DomainDiscriminator, PhysicsHead
from src.loss import PICDA_Loss
from src.trainer import PICDATrainer

# --- CONFIGURATION ---
CSV_PATH = "processed_data/gas_data_normalized.csv" 
# Fallback path if you didn't create the folder structure exactly
CSV_PATH_ALT = "gas_data_normalized.csv"

SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (tuned for stability)
BATCH_SIZE = 64
LR_SOURCE = 0.001
LR_ADAPT = 0.0001
EPOCHS_SOURCE = 20
EPOCHS_ADAPT = 15

def calculate_drift_direction(df):
    """
    Automatically determines if drift is Positive or Negative.
    Physics: Compares Batch 10 Mean vs Batch 1 Mean.
    """
    # Use Feature 0 (Steady State Resistance) as proxy
    b1_mean = df[df['Batch_ID'] == 1]['feat_0'].mean()
    b10_mean = df[df['Batch_ID'] == 10]['feat_0'].mean()
    
    diff = b10_mean - b1_mean
    
    # If diff is negative, drift is -1.0. If positive, +1.0.
    direction = 1.0 if diff > 0 else -1.0
    
    print(f"\n[PHYSICS CHECK] Global Drift Calc:")
    print(f"  Batch 1 Mean:  {b1_mean:.4f}")
    print(f"  Batch 10 Mean: {b10_mean:.4f}")
    print(f"  Difference:    {diff:.4f}")
    print(f"  VERDICT: Drift Direction = {direction}")
    
    return direction

def main():
    print("--- STARTING PI-CDA EXPERIMENT ---")
    
    # 1. LOAD DATA
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    elif os.path.exists(CSV_PATH_ALT):
        df = pd.read_csv(CSV_PATH_ALT)
    else:
        print("❌ Error: gas_data_normalized.csv not found!")
        print("   Did you run strict_preprocessing() in src/preprocess.py?")
        return
        
    print(f"Data Loaded: {len(df)} samples.")
    
    # 2. PHYSICS CHECK
    DRIFT_DIR = calculate_drift_direction(df)
    
    # 3. SETUP MODELS
    print("\nInitializing Models...")
    encoder = SiameseEncoder(input_dim=128, latent_dim=64)
    classifier = TaskClassifier(latent_dim=64, num_classes=6)
    discriminator = DomainDiscriminator(latent_dim=64)
    phy_head = PhysicsHead(latent_dim=64)
    
    # 4. SETUP LOSS & TRAINER
    # We use lower weights for physics initially to let classification converge
    criterion = PICDA_Loss(
        lambda_cont=0.1, 
        lambda_power=0.1, 
        lambda_adv=0.5, 
        lambda_mono=1.0
    )
    
    trainer = PICDATrainer(
        encoder, classifier, discriminator, phy_head, criterion, 
        DEVICE, SAVE_DIR
    )
    
    # ==========================================
    # PHASE 1: SOURCE ANCHORING (Batch 1)
    # ==========================================
    print("\n" + "="*40)
    print(" PHASE 1: TRAINING ON SOURCE (BATCH 1)")
    print("="*40)
    
    ds_src = GasDataset(df, batch_id=1)
    loader_src = DataLoader(ds_src, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Train
    prev_baseline = trainer.train_source_phase(loader_src, epochs=EPOCHS_SOURCE, lr=LR_SOURCE)
    
    # Baseline Accuracy Check
    acc_b1 = trainer.evaluate(loader_src)
    print(f"✅ Batch 1 Baseline Accuracy: {acc_b1:.2f}%")
    
    if acc_b1 < 90.0:
        print("⚠️ Warning: Source accuracy is low. Check hyperparameters.")

    # ==========================================
    # PHASE 2: SEQUENTIAL ADAPTATION (2 -> 10)
    # ==========================================
    print("\n" + "="*40)
    print(" PHASE 2: SEQUENTIAL ADAPTATION")
    print("="*40)
    
    results = []
    
    for b_id in range(2, 11):
        # Target Data
        ds_tgt = GasDataset(df, batch_id=b_id)
        
        # Skip tiny batches if any
        if len(ds_tgt) < BATCH_SIZE:
            print(f"Skipping Batch {b_id} (Too small)")
            continue
            
        loader_tgt = DataLoader(ds_tgt, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        # 1. Eval Before Adaptation
        acc_before = trainer.evaluate(loader_tgt)
        
        # 2. Adapt
        # We pass loader_src because we need "True Labels" for the Adversarial/Contrastive anchoring
        new_baseline = trainer.adapt_target_phase(
            source_loader=loader_src,
            target_loader=loader_tgt,
            batch_id=b_id,
            prev_baseline=prev_baseline,
            drift_dir=DRIFT_DIR,
            epochs=EPOCHS_ADAPT,
            lr=LR_ADAPT
        )
        
        # 3. Eval After Adaptation
        acc_after = trainer.evaluate(loader_tgt)
        
        print(f"Batch {b_id} Result: {acc_before:.1f}% -> {acc_after:.1f}%")
        
        results.append({
            'Batch': b_id,
            'Accuracy_Before': acc_before,
            'Accuracy_After': acc_after,
            'Baseline_Est': new_baseline
        })
        
        # Update Chain: The new baseline becomes the "previous" for the next batch
        prev_baseline = new_baseline
        
    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "="*40)
    print(" FINAL RESULTS TABLE")
    print("="*40)
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Save results
    if not os.path.exists("results"):
        os.makedirs("results")
    results_df.to_csv("results/final_accuracy.csv", index=False)
    print("\nExperiment Complete. Results saved to results/final_accuracy.csv")

if __name__ == "__main__":
    main()