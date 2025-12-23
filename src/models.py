import torch
import torch.nn as nn

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        
        # --- LAYER 1: SENSOR ALIGNMENT (The Physical Step) ---
        # Input: (Batch, 1, 128) -> Output: (Batch, 32, 16)
        # Physics: 128 features = 16 sensors * 8 features.
        # Kernel=9: Covers 8 features (1 sensor) + 1 neighbor.
        # Stride=8: Jumps exactly 1 sensor at a time.
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # --- LAYER 2: GLOBAL CONTEXT (The Temporal Step) ---
        # Input: (Batch, 32, 16) -> Output: (Batch, 64, 16)
        # Physics: Compare Sensor i with Sensor i+4 (Global Drift)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # --- LAYER 3: REFINEMENT ---
        # Input: (Batch, 64, 16) -> Output: (Batch, 64, 16)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # --- LAYER 4: LATENT PROJECTION ---
        # Flatten: 64 channels * 16 sensors = 1024 features
        self.flatten_dim = 64 * 16 
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim) # Normalize z for Contrastive Loss
        )

    def forward(self, x):
        # 1. Reshape: (Batch, 128) -> (Batch, 1, 128)
        # Treat the feature vector as a 1D signal
        x = x.unsqueeze(1)
        
        # 2. Pass through TCN
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        
        # 3. Flatten and Project
        flat = f3.view(f3.size(0), -1) 
        z = self.fc(flat)
        
        # return z, f1, f2, f3 # Return intermediates for visualization
        return z # Return only the latent vector for actual use 

# --- NEW: HEAD 1 (TASK CLASSIFIER) ---
class TaskClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=6):
        super(TaskClassifier, self).__init__()
        
        self.net = nn.Sequential(
            # Input: 64-dim Latent Code
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1), # Safety against overfitting
            
            # Output: 6 Gas Class Scores
            nn.Linear(32, num_classes) 
        )

    def forward(self, z):
        # z shape: (Batch, 64)
        # output shape: (Batch, 6)
        return self.net(z)

# --- NEW: HEAD 3 (PHYSICS PROJECTION) ---
class PhysicsHead(nn.Module):
    def __init__(self, latent_dim=64):
        super(PhysicsHead, self).__init__()
        
        # 1. The Baseline Estimator
        # We use a small MLP to guess the "Sensor Baseline" from the latent code.
        # Architecture: 64 -> 32 -> ReLU -> 2 (Output 1 ignored, Output 2 is Baseline)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
        
        # 2. The Learnable Physics Constants (Yamazoe Law)
        # We start w at 0.5 because Concentration and Signal Strength are positively correlated.
        # We learn 'w' and 'b' to fit: Magnitude = w * log(Conc) + b
        self.w = nn.Parameter(torch.tensor([0.5])) 
        self.b = nn.Parameter(torch.tensor([0.0]))

    def forward(self, z):
        # A. Physical Variable 1: Sensitivity Magnitude
        # DESIGN CHOICE: We use the Length (Norm) of the latent vector z itself.
        # Meaning: "Direction" = Gas Type, "Length" = Concentration.
        sensitivity_mag = torch.norm(z, dim=1, keepdim=True)
        
        # B. Physical Variable 2: Baseline Estimate
        out = self.net(z)
        # We pick the second output neuron to represent Baseline
        baseline_est = out[:, 1].unsqueeze(1)
        
        return sensitivity_mag, baseline_est

# --- 0. GRADIENT REVERSAL LAYER (Required for Head 2) ---
from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# --- NEW: HEAD 2 (DOMAIN DISCRIMINATOR) ---
class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim=64):
        super(DomainDiscriminator, self).__init__()
        
        # 1. The Reversal Layer
        # We put it inside the module so it's automatic.
        self.grl = GRL(alpha=1.0)
        
        # 2. The Binary Classifier
        # Input: 64-dim Latent Code
        # Output: 1 Probability Score (0 = Source, 1 = Target)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        # 1. Reverse Gradients
        z_rev = self.grl(z)
        
        # 2. Predict Domain
        return self.net(z_rev)