import torch
import torch.nn as nn
import torch.nn.functional as F

class PICDA_Loss(nn.Module):
    def __init__(self, lambda_cont=0.1, lambda_power=1.0, lambda_adv=0.5, lambda_mono=1.0):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_power = lambda_power
        self.lambda_adv = lambda_adv
        self.lambda_mono = lambda_mono

    # --- 1. SUPERVISED CONTRASTIVE LOSS (NT-Xent) ---
    def contrastive_loss(self, z, labels, temp=0.07):
        """
        Forces z vectors of the same gas class to cluster together.
        """
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / temp
        
        # Mask for same-class pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity from mask
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(mask.size(0)).view(-1, 1).to(z.device), 
            0
        )
        mask = mask * logits_mask
        
        # Compute Log Prob
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        
        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        return -mean_log_prob_pos.mean()

    # --- 2. YAMAZOE POWER LAW LOSS ---
    def power_law_loss(self, sensitivity_mag, concentrations, w, b):
        """
        Enforces: Magnitude = w * log(Concentration) + b
        """
        # Log of concentration (add epsilon to avoid log(0))
        log_conc = torch.log(concentrations.view(-1, 1) + 1e-6)
        
        # Target Magnitude based on Physics
        target_mag = w * log_conc + b
        
        # MSE Loss between Model Output and Physics Target
        return F.mse_loss(sensitivity_mag, target_mag)

    # --- 3. MONOTONICITY LOSS (The Novelty) ---
    def monotonicity_loss(self, current_baseline_est, prev_baseline_scalar, drift_dir):
        """
        Penalizes drift if it goes in the 'wrong' direction.
        drift_dir: +1.0 (Rising Resistance) or -1.0 (Falling Resistance)
        """
        # Average baseline for the current batch
        curr_avg = current_baseline_est.mean()
        
        # Change from previous batch
        delta = curr_avg - prev_baseline_scalar
        
        # Logic: 
        # If drift_dir is -1.0 (Falling), we expect delta to be Negative.
        # If delta is Positive (Rising), that is a violation.
        # Check: -1.0 * (+5) * (-1.0) = +5 (Penalty)
        # Check: -1.0 * (-5) * (-1.0) = -5 -> ReLU -> 0 (No Penalty)
        violation = -1.0 * delta * drift_dir
        
        return torch.relu(violation)