import torch
import torch.nn as nn
from torch.autograd import Function

# --- KEEP YOUR GRL CODE SAME AS BEFORE ---
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# --- THE NEW TCN ENCODER (Replaces the old MLP) ---
class SiameseEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super(SiameseEncoder, self).__init__()
        
        # TCN Block 1: Feature Expansion
        # Input: (Batch, 1, 128) -> Output: (Batch, 16, 64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # TCN Block 2: Dilated Feature Extraction
        # Input: (Batch, 16, 64) -> Output: (Batch, 32, 32)
        # Dilation=2 lets it look at broader sensor patterns
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # TCN Block 3: Compression
        # Input: (Batch, 32, 32) -> Output: (Batch, 64, 16)
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Final Flattening to Latent Space z
        # We flatten (64 channels * 16 length) -> 1024 features
        # Then map to 64-dim latent space
        self.flatten_dim = 64 * 16 
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim) # Normalize z for contrastive loss
        )

    def forward(self, x):
        # x shape comes in as (Batch_Size, 128)
        # TCN needs (Batch_Size, Channels, Length) -> (Batch, 1, 128)
        x = x.unsqueeze(1) 
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten for the dense layer
        x = x.view(x.size(0), -1) 
        z = self.fc(x)
        
        return z

# --- KEEP THE OTHER HEADS (Classifier, Discriminator, PhysicsHead) SAME ---
class TaskClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=6):
        super(TaskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, z):
        return self.net(z)

class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim=64):
        super(DomainDiscriminator, self).__init__()
        self.grl = GRL(alpha=1.0)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        z_rev = self.grl(z)
        return self.net(z_rev)

class PhysicsHead(nn.Module):
    def __init__(self, latent_dim=64):
        super(PhysicsHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
        self.w = nn.Parameter(torch.tensor([-0.5])) 
        self.b = nn.Parameter(torch.tensor([1.0]))

    def forward(self, z):
        physics_state = self.net(z)
        sensitivity_mag = torch.norm(z, dim=1, keepdim=True) 
        baseline_est = physics_state[:, 1].unsqueeze(1)
        return sensitivity_mag, baseline_est