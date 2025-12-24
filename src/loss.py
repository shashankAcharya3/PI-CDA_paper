import torch
import torch.nn as nn
import torch.nn.functional as F

class PICDA_Loss(nn.Module):
    def __init__(self, lambda_cont=1.0, lambda_power=0.1, lambda_adv=0.1, lambda_mono=1.0, lambda_ent=0.1):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_power = lambda_power
        self.lambda_adv = lambda_adv
        self.lambda_mono = lambda_mono
        self.lambda_ent = lambda_ent 
        
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def task_loss(self, preds, targets):
        return self.ce(preds, targets)

    def entropy_loss(self, logits):
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(p * log_p, dim=1)
        return entropy.mean()

    def contrastive_loss(self, features, labels, temperature=0.07):
        device = features.device
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.size(0)).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        
        # --- CRITICAL FIX: Normalize by positive pairs count ---
        pos_pairs_count = mask.sum(1) + 1e-6
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_pairs_count
        
        return -mean_log_prob_pos.mean()

    def power_law_loss(self, sensitivity_mag, concentrations, w, b):
        log_conc = torch.log(concentrations.view(-1, 1) + 1e-6)
        target_mag = w * log_conc + b
        return self.mse(sensitivity_mag, target_mag)

    def monotonicity_loss(self, current_base, prev_base, drift_dir):
        delta = current_base.mean() - prev_base
        return torch.relu(-delta * drift_dir)

    def adversarial_loss(self, preds, target_is_real):
        targets = torch.ones_like(preds) if target_is_real else torch.zeros_like(preds)
        return self.bce(preds, targets)