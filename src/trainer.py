import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os

class PICDATrainer:
    def __init__(self, encoder, classifier, discriminator, physics_head, criterion, device, save_dir):
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.discriminator = discriminator.to(device)
        self.physics_head = physics_head.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.save_dir = save_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)

    def _freeze_bn_stats(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm1d): m.eval()

    def _get_z(self, x):
        out = self.encoder(x)
        if isinstance(out, tuple): return out[0]
        return out

    def train_source_phase(self, source_loader, epochs=20, lr=0.001):
        print(f"\n[PHASE 1] Training Source Model (Batch 1)...")
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()) + list(self.physics_head.parameters()), 
            lr=lr
        )
        self.encoder.train(); self.classifier.train(); self.physics_head.train()
        
        final_base = 0.0
        for epoch in range(epochs):
            accum_base = []
            for x, y, c in source_loader:
                x, y, c = x.to(self.device), y.to(self.device), c.to(self.device)
                optimizer.zero_grad()
                z = self.encoder(x); pred = self.classifier(z); sens, base = self.physics_head(z)
                loss = self.criterion.task_loss(pred, y) + \
                       self.criterion.lambda_cont * self.criterion.contrastive_loss(z, y) + \
                       self.criterion.lambda_power * self.criterion.power_law_loss(sens, c, self.physics_head.w, self.physics_head.b)
                loss.backward(); optimizer.step(); accum_base.append(base.mean().item())
            final_base = np.mean(accum_base)
            if (epoch+1)%5==0: print(f"  Epoch {epoch+1}: Base {final_base:.4f}")
        self.save_checkpoint("source_model.pth")
        return final_base

    def adapt_target_phase(self, source_loader, target_loader, batch_id, prev_baseline, drift_dir, epochs=10, lr=0.0001, use_pseudo_labels=True):
        print(f"\n[PHASE 2] Adapting to Batch {batch_id}...")
        for p in self.classifier.parameters(): p.requires_grad = False
        self.physics_head.w.requires_grad = False; self.physics_head.b.requires_grad = False
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.discriminator.parameters()) + list(self.physics_head.net.parameters()), 
            lr=lr
        )
        
        self.encoder.train(); self._freeze_bn_stats()
        self.discriminator.train(); self.physics_head.train(); self.classifier.eval()
        
        curr_base = prev_baseline

        for epoch in range(epochs):
            min_len = min(len(source_loader), len(target_loader))
            iter_src = iter(source_loader); iter_tgt = iter(target_loader)
            accum_base = []
            
            for _ in range(min_len):
                x_s, y_s, _ = next(iter_src); x_t, _, _ = next(iter_tgt)
                x_s, y_s, x_t = x_s.to(self.device), y_s.to(self.device), x_t.to(self.device)
                
                optimizer.zero_grad()
                z_s = self.encoder(x_s); z_t = self.encoder(x_t)
                
                # Source Replay & Entropy
                pred_s = self.classifier(z_s); l_src = self.criterion.task_loss(pred_s, y_s)
                logits_t = self.classifier(z_t); l_ent = self.criterion.entropy_loss(logits_t)
                
                # Pseudo-Labels (Optional Flag for Rescue Mode)
                l_cont = torch.tensor(0.0, device=self.device)
                mask_count = 0
                
                if use_pseudo_labels:
                    with torch.no_grad():
                        probs = torch.softmax(logits_t, dim=1)
                        conf, plabels = torch.max(probs, dim=1)
                        mask = torch.zeros_like(conf, dtype=torch.bool)
                        for c in range(6):
                            idx = (plabels == c).nonzero(as_tuple=True)[0]
                            if len(idx) > 0:
                                thresh = torch.quantile(conf[idx], 0.5)
                                mask[idx[conf[idx] >= torch.max(thresh, torch.tensor(0.5).to(self.device))]] = True
                    
                    mask_count = mask.sum().item()
                    if mask.sum() > 0:
                        z_c = torch.cat([z_s, z_t[mask]], dim=0); y_c = torch.cat([y_s, plabels[mask]], dim=0)
                        l_cont = self.criterion.contrastive_loss(z_c, y_c)

                d_s = self.discriminator(z_s); d_t = self.discriminator(z_t)
                l_adv = self.criterion.adversarial_loss(d_s, 0) + self.criterion.adversarial_loss(d_t, 1)
                
                _, base_t = self.physics_head(z_t)
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                
                loss = (self.criterion.lambda_adv * l_adv) + \
                       (self.criterion.lambda_cont * l_cont) + \
                       (self.criterion.lambda_mono * l_mono) + \
                       (self.criterion.lambda_ent * l_ent) + \
                       (0.1 * l_src)
                
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step(); accum_base.append(base_t.mean().item())
            
            curr_base = np.mean(accum_base)
            if (epoch+1)%5==0: print(f"  Epoch {epoch+1}: Adv {l_adv:.3f} | Cont {l_cont:.3f} | Mask {mask_count}")
            
        self.save_checkpoint(f"adapted_model_b{batch_id}.pth")
        return curr_base

    def update_physics_only(self, loader, prev_baseline, drift_dir):
        print("  >>> Bridging Gap: Updating Physics Baseline only...")
        self.encoder.eval(); self.classifier.eval(); self.physics_head.train()
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.physics_head.net.parameters(): p.requires_grad = True
        
        optimizer = optim.Adam(self.physics_head.parameters(), lr=0.001)
        accum_base = []
        for _ in range(5):
            for x, _, _ in loader:
                x = x.to(self.device)
                z = self._get_z(x)
                _, base_t = self.physics_head(z)
                l_mono = self.criterion.monotonicity_loss(base_t, prev_baseline, drift_dir)
                optimizer.zero_grad(); l_mono.backward(); optimizer.step()
                accum_base.append(base_t.mean().item())
        
        for p in self.encoder.parameters(): p.requires_grad = True
        return np.mean(accum_base)

    def evaluate(self, loader):
        self.encoder.eval(); self.classifier.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y, _ in loader:
                pred = self.classifier(self.encoder(x.to(self.device)))
                correct += (pred.argmax(1) == y.to(self.device)).sum().item()
                total += x.size(0)
        return 100 * correct / total

    def save_checkpoint(self, name):
        torch.save({'enc': self.encoder.state_dict(), 'cls': self.classifier.state_dict(), 'phy': self.physics_head.state_dict()}, os.path.join(self.save_dir, name))