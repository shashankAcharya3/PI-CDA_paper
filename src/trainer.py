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

    def _get_z(self, x):
        out = self.encoder(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    # --- HELPER: FREEZE BATCH NORM ---
    def _freeze_bn_stats(self):
        """
        CRITICAL FIX: Force Batch Norm layers to use saved running stats 
        instead of updating them with target data.
        """
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval() # Sets tracking_running_stats to False behavior

    # --- PHASE 1: SOURCE ANCHORING ---
    def train_source_phase(self, source_loader, epochs=20, lr=0.001):
        print(f"\n[PHASE 1] Training Source Model (Batch 1)...")
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.physics_head.parameters()), 
            lr=lr
        )
        
        # Standard Train Mode
        self.encoder.train(); self.classifier.train(); self.physics_head.train()
        
        final_baseline_est = 0.0

        for epoch in range(epochs):
            total_loss = 0; total_task = 0; baseline_accum = []
            
            for x, y, c in source_loader:
                x, y, c = x.to(self.device), y.to(self.device), c.to(self.device)
                optimizer.zero_grad()
                
                z = self._get_z(x)
                pred = self.classifier(z)
                sens, base = self.physics_head(z)
                
                l_task = self.criterion.task_loss(pred, y)
                l_cont = self.criterion.contrastive_loss(z, y)
                l_power = self.criterion.power_law_loss(sens, c, self.physics_head.w, self.physics_head.b)
                
                loss = l_task + (self.criterion.lambda_cont * l_cont) + (self.criterion.lambda_power * l_power)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_task += l_task.item()
                baseline_accum.append(base.mean().item())
            
            avg_base = np.mean(baseline_accum)
            final_baseline_est = avg_base
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Total {total_loss/len(source_loader):.4f} | Task {total_task/len(source_loader):.4f} | Base {avg_base:.4f}")

        self.save_checkpoint("source_model.pth")
        print(f"  > Phase 1 Complete. Learned Baseline: {final_baseline_est:.4f}")
        return final_baseline_est

    # --- PHASE 2: ADAPTATION (WITH SAFEGUARDS) ---
    def adapt_target_phase(self, source_loader, target_loader, batch_id, prev_baseline, drift_dir, epochs=10, lr=0.0001):
        
        # --- SAFEGUARD 1: Check Accuracy Before Adapting ---
        acc_before = self.evaluate(target_loader)
        print(f"\n[PHASE 2] Adapting to Batch {batch_id} (Initial Acc: {acc_before:.1f}%)")
        
        if acc_before > 75.0:
            print("  >>> SAFETY TRIGGER: Accuracy is already high. Skipping aggressive adaptation to prevent collapse.")
            # We skip training but still return the baseline estimate for the chain
            return self._estimate_baseline_only(target_loader)

        # 1. Setup Optimizer (Weights Only)
        for param in self.classifier.parameters(): param.requires_grad = False
        self.physics_head.w.requires_grad = False; self.physics_head.b.requires_grad = False
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.discriminator.parameters()) +
            list(self.physics_head.net.parameters()), 
            lr=lr
        )
        
        # 2. Modes
        self.encoder.train()
        self._freeze_bn_stats() # <--- CRITICAL FIX: FREEZE BN STATS
        self.discriminator.train(); self.physics_head.train()
        self.classifier.eval() 
        
        curr_baseline_avg = prev_baseline

        for epoch in range(epochs):
            min_len = min(len(source_loader), len(target_loader))
            iter_src = iter(source_loader); iter_tgt = iter(target_loader)
            baseline_accum = []
            
            pseudo_label_counts = torch.zeros(6).to(self.device) # Track diversity

            for _ in range(min_len):
                x_s, y_s, _ = next(iter_src)
                x_t, _, _ = next(iter_tgt)
                x_s, y_s, x_t = x_s.to(self.device), y_s.to(self.device), x_t.to(self.device)
                
                optimizer.zero_grad()
                
                z_s = self._get_z(x_s)
                z_t = self._get_z(x_t)
                
                # Pseudo-Labeling
                with torch.no_grad():
                    logits_t = self.classifier(z_t)
                    probs_t = torch.softmax(logits_t, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_t, dim=1)
                    mask_conf = max_probs > 0.8
                    
                    # Track diversity
                    if mask_conf.sum() > 0:
                        classes, counts = torch.unique(pseudo_labels[mask_conf], return_counts=True)
                        pseudo_label_counts[classes] += counts
                
                # Losses
                d_s = self.discriminator(z_s); d_t = self.discriminator(z_t)
                l_adv = self.criterion.adversarial_loss(d_s, 0) + self.criterion.adversarial_loss(d_t, 1)
                
                if mask_conf.sum() > 0:
                    z_comb = torch.cat([z_s, z_t[mask_conf]], dim=0)
                    y_comb = torch.cat([y_s, pseudo_labels[mask_conf]], dim=0)
                    l_cont = self.criterion.contrastive_loss(z_comb, y_comb)
                else:
                    l_cont = torch.tensor(0.0, device=self.device)
                
                _, base_t = self.physics_head(z_t)
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                
                loss = (self.criterion.lambda_adv * l_adv) + \
                       (self.criterion.lambda_cont * l_cont) + \
                       (self.criterion.lambda_mono * l_mono)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                optimizer.step()
                baseline_accum.append(base_t.mean().item())
            
            curr_baseline_avg = np.mean(baseline_accum)
            
            # Diversity Check
            unique_classes = (pseudo_label_counts > 0).sum().item()
            if unique_classes < 2 and epoch > 1:
                print(f"  >>> WARNING: Mode Collapse Detected (Only {unique_classes} classes found). Stopping early.")
                break

            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Adv {l_adv.item():.3f} | Cont {l_cont.item():.3f} | Mono {l_mono.item():.4f}")

        self.save_checkpoint(f"adapted_model_b{batch_id}.pth")
        return curr_baseline_avg

    def _estimate_baseline_only(self, loader):
        """Helper to get baseline without training"""
        self.encoder.eval(); self.physics_head.eval()
        accum = []
        with torch.no_grad():
            for x, _, _ in loader:
                x = x.to(self.device)
                z = self._get_z(x)
                _, base = self.physics_head(z)
                accum.append(base.mean().item())
        return np.mean(accum)

    def evaluate(self, loader):
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