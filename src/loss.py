import torch
import torch.nn as nn
import torch.nn.functional as F

class PICDA_Loss(nn.Module):
    """
    Centralized Loss Module for PI-CDA.
    """
    def __init__(self, lambda_cont=0.1, lambda_power=0.1, lambda_adv=0.5, lambda_mono=1.0):
        super().__init__()
        # Trade-off hyperparameters
        self.lambda_cont = lambda_cont
        self.lambda_power = lambda_power
        self.lambda_adv = lambda_adv
        self.lambda_mono = lambda_mono
        
        # Standard losses
        self.ce = nn.CrossEntropyLoss() # For Task
        self.bce = nn.BCELoss()         # For Adversarial
        self.mse = nn.MSELoss()         # For Physics

    # --- 1. TASK LOSS (The Missing Wrapper) ---
    def task_loss(self, predictions, targets):
        """
        Standard Cross Entropy for Gas Classification.
        Args:
            predictions: Logits from TaskClassifier [Batch, 6]
            targets: True Class Labels [Batch]
        """
        return self.ce(predictions, targets)

    # --- 2. SUPERVISED CONTRASTIVE LOSS ---
    def contrastive_loss(self, features, labels, temperature=0.07):
        """
        Forces z vectors of the same gas class to cluster together.
        """
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / temperature
        
        # Mask for same-class pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        # Log Probabilities
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        return -mean_log_prob_pos.mean()

    # --- 3. YAMAZOE POWER LAW LOSS ---
    def power_law_loss(self, sensitivity_mag, concentrations, w, b):
        """
        Enforces Physics: Magnitude ~ w * log(Concentration) + b
        """
        log_conc = torch.log(concentrations.view(-1, 1) + 1e-6)
        target_mag = w * log_conc + b
        return self.mse(sensitivity_mag, target_mag)

    # --- 4. MONOTONICITY LOSS ---
    def monotonicity_loss(self, current_baseline_est, prev_baseline_scalar, drift_dir):
        """
        Enforces the "Arrow of Time" (Drift Direction).
        """
        curr_avg = current_baseline_est.mean()
        delta = curr_avg - prev_baseline_scalar
        
        # Penalty if movement is opposite to drift_dir
        penalty = -1.0 * delta * drift_dir
        return torch.relu(penalty)

    # --- 5. ADVERSARIAL LOSS ---
    def adversarial_loss(self, preds, target_is_real):
        """
        Discriminator Loss.
        target_is_real: 1.0 (Target Domain), 0.0 (Source Domain)
        """
        if target_is_real:
            targets = torch.ones_like(preds)
        else:
            targets = torch.zeros_like(preds)
        return self.bce(preds, targets)