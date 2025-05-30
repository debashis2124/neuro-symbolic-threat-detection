import numpy as np
import pandas as pd
from autoencoder.nn_score import nn_score
from symbolic_rules.symbolic_engine import sym_score_df
from hybrid_fusion.risk_fusion import compute_hybrid_score

def generate_synthetic(X, noise_level=0.1):
    """Add small Gaussian noise to each feature vector."""
    return X + np.random.normal(scale=noise_level, size=X.shape)

def create_augmented_features(autoencoder, X_synth, feature_cols, alpha=0.6, beta=0.4):
    y_synth = np.ones(X_synth.shape[0])
    f_synth = nn_score(autoencoder, X_synth)
    g_synth = sym_score_df(pd.DataFrame(X_synth, columns=feature_cols))
    R_synth = compute_hybrid_score(f_synth, g_synth, alpha, beta)
    synth_feats = np.column_stack((R_synth, X_synth[:, [7,9,32]]))
    return synth_feats, y_synth