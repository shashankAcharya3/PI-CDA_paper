import torch
import torch.nn as nn
from torch.autograd import Function

# --- Gradient Reversal Layer ---
# This flips the gradient during backprop, making the feature extractor
# "unlearn" features that are specific to a single batch/domain.
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)

# --- IDAN Architecture ---
class IDAN(nn.Module):
    def __init__(self, num_classes=6, initial_domains=1):
        super().__init__()
        
        # Feature Extractor (CNN)
        # Input: [Batch, 1, 128]
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 128 -> 64
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 64 -> 32
            
            nn.Flatten(),
            nn.Linear(64 * 32, 64),
            nn.ReLU()
        )
        
        # 1. Class Classifier (Predicts Gas Type)
        self.class_classifier = nn.Linear(64, num_classes)
        
        # 2. Domain Classifier (Predicts Batch ID)
        self.grl = GradientReversal()
        self.domain_classifier = nn.Linear(64, initial_domains)

    def forward(self, x, alpha=1.0):
        # x shape: [Batch, 1, 128]
        features = self.feature_extractor(x)
        
        class_out = self.class_classifier(features)
        
        domain_features = self.grl(features, alpha)
        domain_out = self.domain_classifier(domain_features)
        
        return class_out, domain_out

    def expand_domains(self, new_total_domains):
        """
        Incremental Adaptation.
        Expands the output layer to accommodate new batches as they arrive.
        """
        old_layer = self.domain_classifier
        old_out = old_layer.out_features
        
        if new_total_domains <= old_out:
            return # Already big enough

        # Create new bigger layer
        new_layer = nn.Linear(old_layer.in_features, new_total_domains)
        
        # Copy old weights (Preserve knowledge)
        with torch.no_grad():
            new_layer.weight[:old_out] = old_layer.weight
            new_layer.bias[:old_out] = old_layer.bias
            # Init new weights randomly
            nn.init.normal_(new_layer.weight[old_out:], std=0.01)
        
        # Move to same device (GPU/CPU)
        self.domain_classifier = new_layer.to(old_layer.weight.device)