import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.config import RF_ESTIMATORS, RF_MAX_DEPTH, CORRECTION_THRESHOLD

class IterativeRFCorrector:
    def __init__(self):
        # A list to hold 128 distinct Random Forest models
        self.models = [None] * 128
        self.is_fitted = False

    def fit(self, X_train):
        """
        Train the error correction models on Batch 1 (the 'clean' batch).
        """
        print(f"Training 128 Correction Models (this may take a moment)...")
        n_features = X_train.shape[1]
        
        for i in range(n_features):
            # Target: Feature i
            y = X_train[:, i]
            # Predictors: All features EXCEPT i
            X = np.delete(X_train, i, axis=1)
            
            rf = RandomForestRegressor(
                n_estimators=RF_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                n_jobs=-1,  # Use all cores on your M-chip
                random_state=42
            )
            rf.fit(X, y)
            self.models[i] = rf
            
        self.is_fitted = True
        print("Corrector Training Complete.")

    def _correct_single_pass(self, X_input):
        """
        Performs one pass of correction on the dataset.
        """
        X_corrected = X_input.copy()
        n_features = X_input.shape[1]
        
        for i in range(n_features):
            model = self.models[i]
            if model is None: continue
            
            # Predict what Feature i SHOULD be
            X_feats = np.delete(X_input, i, axis=1)
            y_pred = model.predict(X_feats)
            
            # Calculate deviation (Eq 7 in paper)
            observed = X_input[:, i]
            # Avoid divide by zero
            denom = np.abs(observed)
            denom[denom < 1e-9] = 1e-9
            
            relative_residual = np.abs(y_pred - observed) / denom
            
            # If deviation > 5%, replace with prediction (Eq 8)
            mask = relative_residual > CORRECTION_THRESHOLD
            X_corrected[mask, i] = y_pred[mask]
            
        return X_corrected

    def correct(self, X_batch, iterations=2):
        """
        Run correction twice (Iterative) to clean mutual errors.
        """
        if not self.is_fitted:
            print("Warning: Corrector not fitted. Returning original data.")
            return X_batch
            
        X_curr = X_batch.copy()
        for k in range(iterations):
            X_curr = self._correct_single_pass(X_curr)
            
        return X_curr