import numpy as np
import pandas as pd
from tensorflow.keras.models import clone_model
from autoencoder.nn_score import nn_score
from hybrid_fusion.risk_fusion import compute_hybrid_score

def fine_tune_autoencoder(autoencoder, X_val, X_train, X_val_raw, X_test_raw, g_sr_train, g_sr_val, g_sr_test, feature_cols, alpha=0.6, beta=0.4):
    ae_ft = clone_model(autoencoder)
    ae_ft.set_weights(autoencoder.get_weights())
    ae_ft.compile(optimizer='adam', loss='mse')

    X_val_lstm = X_val.reshape((-1, 1, X_val.shape[1]))
    ae_ft.fit(X_val_lstm, X_val_lstm, epochs=100, batch_size=256, verbose=1)

    f_ft_train = nn_score(ae_ft, X_train)
    f_ft_val   = nn_score(ae_ft, X_val_raw)
    f_ft_test  = nn_score(ae_ft, X_test_raw)

    R_ft_train = compute_hybrid_score(f_ft_train, g_sr_train, alpha, beta)
    R_ft_val   = compute_hybrid_score(f_ft_val,   g_sr_val,   alpha, beta)
    R_ft_test  = compute_hybrid_score(f_ft_test,  g_sr_test,  alpha, beta)

    train_feats_ft = np.column_stack((R_ft_train, X_train[:, [7,9,32]]))
    val_feats_ft   = np.column_stack((R_ft_val,   X_val_raw[:, [7,9,32]]))
    test_feats_ft  = np.column_stack((R_ft_test,  X_test_raw[:, [7,9,32]]))

    return ae_ft, train_feats_ft, val_feats_ft, test_feats_ft