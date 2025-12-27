import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import *
from src.data_loader import load_batch
from src.correction import IterativeRFCorrector
from src.models import IDAN

# --- CONFIGURATION UPDATES ---
ADAPTATION_STEPS = 5   # NEW: Train on each chunk 5 times
CLIP_VALUE = 5.0       # NEW: Clamp outliers to prevent destabilization

def main():
    # --- STEP 1: Process Batch 1 (Training & Calibration) ---
    print("\n>>> [Phase 1] Initializing on Batch 1...")
    
    X1, y1 = load_batch("raw_data/batch1.dat")
    if X1 is None: return

    # Train Random Forest Corrector
    corrector = IterativeRFCorrector()
    corrector.fit(X1)
    
    # Correct Batch 1
    X1_clean = corrector.correct(X1)
    
    # Fit Scaler
    scaler = MinMaxScaler()
    X1_norm = scaler.fit_transform(X1_clean)
    
    # Prepare Tensors
    y1_t = torch.LongTensor(y1 - 1).to(DEVICE)
    X1_t = torch.FloatTensor(X1_norm).unsqueeze(1).to(DEVICE)

    # --- STEP 2: Pre-train IDAN ---
    print("\n>>> [Phase 2] Pre-training IDAN Network...")
    model = IDAN(num_classes=NUM_CLASSES, initial_domains=1).to(DEVICE)
    optimizer_cls = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_CLS)
    criterion_cls = torch.nn.CrossEntropyLoss()
    
    # Train heavily on Batch 1 to establish a strong baseline
    train_ds = TensorDataset(X1_t, y1_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(30): # Increased from 20 to 30 for solidity
        for bx, by in train_loader:
            optimizer_cls.zero_grad()
            pred_cls, _ = model(bx)
            loss = criterion_cls(pred_cls, by)
            loss.backward()
            optimizer_cls.step()
            
    print("Pre-training Complete.")

    # --- STEP 3: Incremental Adaptation (Batches 2-10) ---
    history_X = X1_t
    history_y = y1_t
    history_d = torch.zeros(len(X1_t), dtype=torch.long).to(DEVICE)
    
    current_max_domain = 0
    results = {}

    for b_idx in range(2, 11):
        print(f"\n>>> [Phase 3] Adapting to Batch {b_idx}...")
        
        X_new, y_new = load_batch(f"raw_data/batch{b_idx}.dat")
        if X_new is None: continue
        
        # 1. Correct & Normalize
        X_new_corr = corrector.correct(X_new)
        X_new_norm = scaler.transform(X_new_corr)
        
        # 2. CLAMPING (Safety Mechanism) [New Measure]
        # Drift can cause values to go to 10.0+ or -10.0, which breaks Neural Nets.
        # We clamp to [-5, 5] (relative to Batch 1's scale)
        X_new_norm = np.clip(X_new_norm, -CLIP_VALUE, 1.0 + CLIP_VALUE)
        
        # 3. Prepare Tensors
        current_max_domain += 1
        X_new_t = torch.FloatTensor(X_new_norm).unsqueeze(1).to(DEVICE)
        y_new_t = torch.LongTensor(y_new - 1).to(DEVICE)
        d_new_t = torch.full((len(X_new_t),), current_max_domain, dtype=torch.long).to(DEVICE)
        
        # 4. Expand Model
        model.expand_domains(current_max_domain + 1)
        optimizer_adapt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_ADAPT)
        criterion_dom = torch.nn.CrossEntropyLoss()
        
        # 5. Aggressive Adaptation Loop
        model.train()
        indices = torch.randperm(len(X_new_t))
        
        for start in range(0, len(X_new_t), CHUNK_SIZE):
            end = start + CHUNK_SIZE
            chunk_indices = indices[start:end]
            
            x_chunk = X_new_t[chunk_indices]
            d_chunk = d_new_t[chunk_indices]
            
            # INNER LOOP (The Fix for Low Accuracy)
            # We optimize multiple times on the same chunk to force alignment
            for step in range(ADAPTATION_STEPS):
                # Replay history
                hist_idx = torch.randint(0, len(history_X), (len(x_chunk),))
                x_hist = history_X[hist_idx]
                y_hist = history_y[hist_idx]
                d_hist = history_d[hist_idx]
                
                # Combine
                x_comb = torch.cat([x_hist, x_chunk])
                d_comb = torch.cat([d_hist, d_chunk])
                
                optimizer_adapt.zero_grad()
                pred_cls, pred_dom = model(x_comb)
                
                # Losses
                loss_class = criterion_cls(pred_cls[:len(x_hist)], y_hist)
                loss_domain = criterion_dom(pred_dom, d_comb)
                
                # Backprop
                loss = loss_class + LAMBDA_DOMAIN * loss_domain
                loss.backward()
                optimizer_adapt.step()
            
        # 6. Evaluation
        model.eval()
        with torch.no_grad():
            preds, _ = model(X_new_t)
            pred_lbls = preds.argmax(dim=1).cpu().numpy()
            
        acc = accuracy_score(y_new - 1, pred_lbls)
        results[b_idx] = acc
        print(f"Batch {b_idx} Accuracy: {acc*100:.2f}%")
        
        # 7. Update History
        history_X = torch.cat([history_X, X_new_t])
        history_y = torch.cat([history_y, y_new_t])
        history_d = torch.cat([history_d, d_new_t])

    print("\n=== Final Accuracy Report ===")
    for b, acc in results.items():
        print(f"Batch {b}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()