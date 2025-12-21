import torch
import torch.nn as nn
from torch.autograd import Function

# --- 1. GRADIENT REVERSAL LAYER (GRL) ---
# Essential for the Adversarial Loss (Head 2)
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient (multiply by -1)
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# --- 2. SIAMESE ENCODER (Validated TCN Backbone) ---
class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        
        # LAYER 1: Sensor Alignment (Kernel=9, Stride=8)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 2: Global Context (Dilation=4)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LAYER 3: Refinement
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Latent Projection
        self.flatten_dim = 64 * 16 
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim) # Normalize for Contrastive Loss
        )

    def forward(self, x):
        # Reshape (Batch, 128) -> (Batch, 1, 128)
        x = x.unsqueeze(1)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        flat = f3.view(f3.size(0), -1) 
        z = self.fc(flat)
        return z

# --- 3. HEAD 1: TASK CLASSIFIER ---
class TaskClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=6):
        super(TaskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes) # Output: 6 Gas Scores
        )
    def forward(self, z):
        return self.net(z)

# --- 4. HEAD 2: DOMAIN DISCRIMINATOR (Adversarial) ---
class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim=64):
        super(DomainDiscriminator, self).__init__()
        self.grl = GRL(alpha=1.0)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1), # Output: 1 score (Source vs Target)
            nn.Sigmoid()      # Probability 0-1
        )
    def forward(self, z):
        z_rev = self.grl(z) # Reverse gradient here!
        return self.net(z_rev)

# --- 5. HEAD 3: PHYSICS PROJECTION HEAD (The Novelty) ---
class PhysicsHead(nn.Module):
    def __init__(self, latent_dim=64):
        super(PhysicsHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
        
        # --- THE FIX IS HERE ---
        # Change -0.5 to 0.5. 
        # Reason: Higher concentration = Stronger Signal = Larger Feature Magnitude
        self.w = nn.Parameter(torch.tensor([0.5])) 
        self.b = nn.Parameter(torch.tensor([0.0])) # Start intercept at 0

    def forward(self, z):
        out = self.net(z)
        
        # Physical Variable 1: Sensitivity Magnitude (|phi|)
        # We use the vector magnitude of z to represent "Signal Strength"
        sensitivity_mag = torch.norm(z, dim=1, keepdim=True)
        
        # Physical Variable 2: Baseline Estimate (Beta)
        # The second output of the linear layer represents the sensor baseline
        baseline_est = out[:, 1].unsqueeze(1)
        
        return sensitivity_mag, baseline_est