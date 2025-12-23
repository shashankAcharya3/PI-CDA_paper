import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

class PICDATrainer:
    def __init__(self, encoder, classifier, discriminator, physics_head, criterion, device, save_dir):
        """
        The Engine for Physics-Informed Contrastive Domain Adaptation.
        """
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.physics_head = physics_head.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _get_z(self, x):
        """
        Helper: Handles the case where Encoder returns (z, f1, f2, f3) for visualization.
        We only need 'z' for training.
        """
        out = self.encoder(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    # ====================================================
    # PHASE 1: SOURCE ANCHORING (Supervised + Physics)
    # ====================================================
    def train_source_phase(self, source_loader, epochs=20, lr=0.001):
        print(f"\n[PHASE 1] Training Source Model (Batch 1)...")
        
        # 1. Optimizer: Train Everything
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.physics_head.parameters()), 
            lr=lr
        )
        
        # 2. Set Mode
        self.encoder.train(); self.classifier.train(); self.physics_head.train()
        
        final_baseline_est = 0.0

        for epoch in range(epochs):
            total_loss = 0
            total_task_loss = 0
            baseline_accum = []

            for features, labels, concs in source_loader:
                features, labels, concs = features.to(self.device), labels.to(self.device), concs.to(self.device)
                
                optimizer.zero_grad()
                
                # --- FORWARD PASS ---
                z = self._get_z(features)
                pred_y = self.classifier(z)
                sens_mag, baseline_est = self.physics_head(z)
                
                # --- LOSSES ---
                # A. Task (Cross Entropy)
                l_task = self.criterion.task_loss(pred_y, labels)
                
                # B. Supervised Contrastive (Cluster by Class)
                l_cont = self.criterion.contrastive_loss(z, labels)
                
                # C. Physics (Power Law: Magnitude vs Concentration)
                l_power = self.criterion.power_law_loss(sens_mag, concs, self.physics_head.w, self.physics_head.b)
                
                # Weighted Sum
                loss = l_task + \
                       (self.criterion.lambda_cont * l_cont) + \
                       (self.criterion.lambda_power * l_power)
                
                loss.backward()
                optimizer.step()
                
                # Tracking
                total_loss += loss.item()
                total_task_loss += l_task.item()
                baseline_accum.append(baseline_est.mean().item())
            
            # Epoch Reporting
            avg_base = np.mean(baseline_accum)
            final_baseline_est = avg_base
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Total {total_loss/len(source_loader):.4f} | Task {total_task_loss/len(source_loader):.4f} | Base {avg_base:.4f}")
        
        # Save State
        self.save_checkpoint("source_model.pth")
        print(f"  > Phase 1 Complete. Learned Baseline: {final_baseline_est:.4f}")
        return final_baseline_est

    # ====================================================
    # PHASE 2: SEQUENTIAL ADAPTATION (Pseudo-Labels + Drift)
    # ====================================================
    def adapt_target_phase(self, source_loader, target_loader, batch_id, prev_baseline, drift_dir, epochs=10, lr=0.0001):
        print(f"\n[PHASE 2] Adapting to Batch {batch_id} (Drift Dir: {drift_dir})...")
        
        # 1. FREEZE Classifier & Physics Params
        # We trust the source decision boundaries and material constants
        for param in self.classifier.parameters(): param.requires_grad = False
        self.physics_head.w.requires_grad = False
        self.physics_head.b.requires_grad = False
        
        # 2. Optimizer: Train Encoder, Discriminator, PhysicsNet (weights only)
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.discriminator.parameters()) +
            list(self.physics_head.net.parameters()), 
            lr=lr
        )
        
        # 3. Set Mode
        self.encoder.train(); self.discriminator.train(); self.physics_head.train()
        self.classifier.eval() # STRICT EVAL MODE for Pseudo-labeling
        
        curr_baseline_avg = prev_baseline

        for epoch in range(epochs):
            # Iterate through Target, cycle through Source
            min_len = min(len(source_loader), len(target_loader))
            iter_src = iter(source_loader)
            iter_tgt = iter(target_loader)
            
            baseline_accum = []
            
            for _ in range(min_len):
                # Load Batches
                x_s, y_s, _ = next(iter_src) # Source (True Labels)
                x_t, _, _ = next(iter_tgt)   # Target (No Labels)
                
                x_s, y_s = x_s.to(self.device), y_s.to(self.device)
                x_t = x_t.to(self.device)
                
                optimizer.zero_grad()
                
                # --- FORWARD PASS ---
                z_s = self._get_z(x_s)
                z_t = self._get_z(x_t)
                
                # --- PSEUDO-LABELING ---
                # Use frozen classifier to guess Target labels
                with torch.no_grad():
                    logits_t = self.classifier(z_t)
                    probs_t = torch.softmax(logits_t, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_t, dim=1)
                    
                    # Filter: Only keep confident samples (> 0.8)
                    mask_conf = max_probs > 0.8
                
                # --- LOSSES ---
                
                # A. Adversarial (Align Distributions)
                d_s = self.discriminator(z_s)
                d_t = self.discriminator(z_t)
                # Source=0, Target=1
                l_adv = self.criterion.adversarial_loss(d_s, target_is_real=0) + \
                        self.criterion.adversarial_loss(d_t, target_is_real=1)
                
                # B. Pseudo-Contrastive (Anchor Target to Source)
                if mask_conf.sum() > 0:
                    # Combine Source (True) + Target (Confident Pseudo)
                    z_combined = torch.cat([z_s, z_t[mask_conf]], dim=0)
                    y_combined = torch.cat([y_s, pseudo_labels[mask_conf]], dim=0)
                    l_cont = self.criterion.contrastive_loss(z_combined, y_combined)
                else:
                    l_cont = torch.tensor(0.0, device=self.device)
                
                # C. Monotonicity (Physics: Arrow of Time)
                _, base_t = self.physics_head(z_t)
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                
                # Weighted Sum
                loss = (self.criterion.lambda_adv * l_adv) + \
                       (self.criterion.lambda_cont * l_cont) + \
                       (self.criterion.lambda_mono * l_mono)
                
                loss.backward()
                optimizer.step()
                
                baseline_accum.append(base_t.mean().item())
            
            # Update baseline tracking
            curr_baseline_avg = np.mean(baseline_accum)
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Adv {l_adv.item():.3f} | Cont {l_cont.item():.3f} | Mono {l_mono.item():.4f}")

        # Save State
        self.save_checkpoint(f"adapted_model_b{batch_id}.pth")
        return curr_baseline_avg

    # ====================================================
    # UTILITIES
    # ====================================================
    def evaluate(self, loader):
        """Standard Accuracy Check"""
        self.encoder.eval(); self.classifier.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                z = self._get_z(x)
                pred = self.classifier(z)
                correct += (pred.argmax(1) == y).sum().item()
                total += x.size(0)
        return 100 * correct / total

    def save_checkpoint(self, name):
        torch.save({
            'enc': self.encoder.state_dict(),
            'cls': self.classifier.state_dict(),
            'phy': self.physics_head.state_dict()
        }, os.path.join(self.save_dir, name))