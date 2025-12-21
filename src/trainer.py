import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

class PICDATrainer:
    def __init__(self, encoder, classifier, discriminator, physics_head, criterion, device, save_dir):
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.physics_head = physics_head.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # --- PHASE 1: SOURCE ANCHORING ---
    def train_source_phase(self, source_loader, epochs=20, lr=0.001):
        print(f"\n[PHASE 1] Training Source Model (Batch 1)...")
        
        # Optimizer: Train EVERYTHING
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.physics_head.parameters()), 
            lr=lr
        )
        
        self.encoder.train(); self.classifier.train(); self.physics_head.train()
        
        final_baseline_est = 0.0

        for epoch in range(epochs):
            total_loss = 0
            total_task = 0  # <--- Task Loss Tracker
            baseline_accum = []

            for features, labels, concs in source_loader:
                features, labels, concs = features.to(self.device), labels.to(self.device), concs.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                z = self.encoder(features)
                pred_y = self.classifier(z)
                sens_mag, baseline_est = self.physics_head(z)
                
                # Losses
                l_task = nn.CrossEntropyLoss()(pred_y, labels)
                l_cont = self.criterion.contrastive_loss(z, labels)
                l_power = self.criterion.power_law_loss(sens_mag, concs, self.physics_head.w, self.physics_head.b)
                
                loss = l_task + (self.criterion.lambda_cont * l_cont) + (self.criterion.lambda_power * l_power)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_task += l_task.item()
                baseline_accum.append(baseline_est.mean().item())
            
            # Average baseline for this epoch
            final_baseline_est = np.mean(baseline_accum)
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Total {total_loss/len(source_loader):.4f} | Task {total_task/len(source_loader):.4f} | Base {final_baseline_est:.4f}")
        
        # Save Source Model
        self.save_checkpoint("source_model.pth")
        print(f"  > Phase 1 Complete. Learned Baseline: {final_baseline_est:.4f}")
        return final_baseline_est

    # --- PHASE 2: SEQUENTIAL ADAPTATION ---
    def adapt_target_phase(self, source_loader, target_loader, batch_id, prev_baseline, drift_dir, epochs=10, lr=0.0001):
        print(f"\n[PHASE 2] Adapting to Batch {batch_id} (Drift Dir: {drift_dir})...")
        
        # 1. FREEZE Classifier & Physics Params (w, b)
        for param in self.classifier.parameters(): param.requires_grad = False
        self.physics_head.w.requires_grad = False
        self.physics_head.b.requires_grad = False
        
        # 2. Optimizer: Encoder + Discriminator + PhysicsNet (Only Weights)
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.discriminator.parameters()) +
            list(self.physics_head.net.parameters()), 
            lr=lr
        )
        
        self.encoder.train(); self.discriminator.train(); self.physics_head.train()
        self.classifier.eval() # STRICT EVAL MODE
        
        curr_baseline_avg = prev_baseline

        for epoch in range(epochs):
            # We iterate through Target data, cycle through Source data
            min_len = min(len(source_loader), len(target_loader))
            iter_src = iter(source_loader)
            iter_tgt = iter(target_loader)
            
            baseline_accum = []
            
            for _ in range(min_len):
                # Get Data
                x_s, _, _ = next(iter_src)
                x_t, _, _ = next(iter_tgt)
                x_s, x_t = x_s.to(self.device), x_t.to(self.device)
                
                optimizer.zero_grad()
                
                # Source Forward (for Adversarial)
                z_s = self.encoder(x_s)
                d_s = self.discriminator(z_s) # Alpha handled in GRL layer
                
                # Target Forward
                z_t = self.encoder(x_t)
                d_t = self.discriminator(z_t)
                _, base_t = self.physics_head(z_t)
                
                # Losses
                # Adversarial: Source=0, Target=1
                l_adv = nn.BCELoss()(d_s, torch.zeros_like(d_s)) + nn.BCELoss()(d_t, torch.ones_like(d_t))
                
                # Monotonicity
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                
                loss = (self.criterion.lambda_adv * l_adv) + (self.criterion.lambda_mono * l_mono)
                
                loss.backward()
                optimizer.step()
                
                baseline_accum.append(base_t.mean().item())
            
            curr_baseline_avg = np.mean(baseline_accum)
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Adv {l_adv.item():.3f} | Mono {l_mono.item():.4f} | New Base {curr_baseline_avg:.3f}")

        self.save_checkpoint(f"adapted_model_b{batch_id}.pth")
        return curr_baseline_avg

    def evaluate(self, loader):
        self.encoder.eval(); self.classifier.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.classifier(self.encoder(x))
                correct += (pred.argmax(1) == y).sum().item()
                total += x.size(0)
        return 100 * correct / total

    def save_checkpoint(self, name):
        torch.save({
            'enc': self.encoder.state_dict(),
            'cls': self.classifier.state_dict(),
            'phy': self.physics_head.state_dict()
        }, os.path.join(self.save_dir, name))