import torch

# --- System Settings ---
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SEED = 42

# --- Data Dimensions ---
NUM_FEATURES = 128
NUM_CLASSES = 6
NUM_BATCHES = 10

# --- Error Correction (Section 3) ---
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 16
CORRECTION_THRESHOLD = 0.05
CORRECTION_ITERATIONS = 2

# --- IDAN Model (Section 4) ---
BATCH_SIZE = 32

# === TUNED FOR STABILITY (Paper Settings) ===
# Paper uses chunk size 50 
CHUNK_SIZE = 50           

# Paper uses Lambda=1.0 [cite: 486]
LAMBDA_DOMAIN = 1.0       

# === OPTIMIZATION (Low & Slow) ===
LEARNING_RATE_CLS = 0.001

# We lower this to 1e-4 to prevent "catastrophic forgetting" of the class labels
# while the domain shifts.
LEARNING_RATE_ADAPT = 0.0001 

# We increase steps to 20 to compensate for the lower learning rate
ADAPTATION_STEPS = 20    
CLIP_VALUE = 5.0