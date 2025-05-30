import pandas as pd
import numpy as np

from data.load_data import load_and_encode_data
from preprocessing.preprocess import scale_and_split
from autoencoder.lstm_autoencoder import build_autoencoder, AEAccHistory
from autoencoder.nn_score import nn_score
from symbolic_rules.symbolic_engine import sym_score_df
from hybrid_fusion.risk_fusion import compute_hybrid_score
from ml_classifier.train_mlp import train_eval
from augmentation.synthetic_data import generate_synthetic, create_augmented_features
from gan_training.train_gan import setup_and_train_gan
from transfer_learning.fine_tune_ae import fine_tune_autoencoder
from evaluation.results_summary import build_results_table
from visualization.plot_figures import plot_ae_loss, plot_nn_score_accuracy

# --- Load and preprocess data ---
df = load_and_encode_data("data/UNSW_NB15_training-set.csv")
feature_cols = [c for c in df.columns if c not in ('attack_cat', 'label')]

X_train, X_val, X_test, y_train, y_val, y_test, _ = scale_and_split(df, feature_cols)

# --- Train LSTM Autoencoder ---
X_benign = X_train[y_train == 0]
X_ben_lstm = X_benign.reshape((-1, 1, X_benign.shape[1]))

autoencoder = build_autoencoder(X_benign.shape[1])
f_nn_val = nn_score(autoencoder, X_val)
fpr, tpr, ths = roc_curve(y_val, f_nn_val)
best_thr = ths[np.argmax(tpr - fpr)]

ae_acc_cb = AEAccHistory(X_train, y_train, X_val, y_val, best_thr)
autoencoder.fit(X_ben_lstm, X_ben_lstm, epochs=100, batch_size=256,
                validation_split=0.2, callbacks=[ae_acc_cb], verbose=1)

f_nn_train = nn_score(autoencoder, X_train)
f_nn_val = nn_score(autoencoder, X_val)
f_nn_test = nn_score(autoencoder, X_test)

# --- Symbolic Scoring ---
df_train = pd.DataFrame(X_train, columns=feature_cols)
df_val = pd.DataFrame(X_val, columns=feature_cols)
df_test = pd.DataFrame(X_test, columns=feature_cols)
g_sr_train = sym_score_df(df_train)
g_sr_val = sym_score_df(df_val)
g_sr_test = sym_score_df(df_test)

# --- Hybrid Scoring ---
R_train = compute_hybrid_score(f_nn_train, g_sr_train)
R_val = compute_hybrid_score(f_nn_val, g_sr_val)
R_test = compute_hybrid_score(f_nn_test, g_sr_test)

# --- MLP Classifiers ---
train_feats = np.column_stack((R_train, X_train[:, [7,9,32]]))
val_feats = np.column_stack((R_val, X_val[:, [7,9,32]]))
test_feats = np.column_stack((R_test, X_test[:, [7,9,32]]))

clf_nn, res_nn = train_eval("Neural-Only", f_nn_train.reshape(-1,1), y_train, f_nn_val.reshape(-1,1), y_val, f_nn_test.reshape(-1,1), y_test)
clf_sy, res_sy = train_eval("Symbolic-Only", g_sr_train.reshape(-1,1), y_train, g_sr_val.reshape(-1,1), y_val, g_sr_test.reshape(-1,1), y_test)
clf_hy, res_hy = train_eval("Hybrid-Only", R_train.reshape(-1,1), y_train, R_val.reshape(-1,1), y_val, R_test.reshape(-1,1), y_test)
clf_hr, res_hr = train_eval("Hybrid+Raw", train_feats, y_train, val_feats, y_val, test_feats, y_test)

# --- Synthetic Augmentation ---
X_synth = generate_synthetic(X_train)
synth_feats, y_synth = create_augmented_features(autoencoder, X_synth, feature_cols)
clf_aug, res_aug = train_eval("Augmented", np.vstack([train_feats, synth_feats]), np.concatenate([y_train, y_synth]), val_feats, y_val, test_feats, y_test)

# --- GAN Training & Adversarial Data ---
generator, _, _ = setup_and_train_gan(X_ben_lstm, epochs=100)
X_adv_lstm = generator.predict(np.random.normal(size=(len(X_train), 32)), verbose=0)
X_adv = X_adv_lstm.reshape((-1, X_adv_lstm.shape[2]))
y_adv = np.ones_like(y_train)
g_adv = sym_score_df(pd.DataFrame(X_adv, columns=feature_cols))
f_adv = nn_score(autoencoder, X_adv)
R_adv = compute_hybrid_score(f_adv, g_adv)
adv_feats = np.column_stack((R_adv, X_adv[:, [7,9,32]]))
clf_adv, res_adv = train_eval("Adversarial", np.vstack([train_feats, adv_feats]), np.concatenate([y_train, y_adv]), val_feats, y_val, test_feats, y_test)

# --- Transfer Learning ---
ae_ft, train_feats_ft, val_feats_ft, test_feats_ft = fine_tune_autoencoder(autoencoder, X_val, X_train, X_val, X_test, g_sr_train, g_sr_val, g_sr_test, feature_cols)
clf_ft, res_ft = train_eval("Transfer", train_feats_ft, y_train, val_feats_ft, y_val, test_feats_ft, y_test)

# --- Results Summary ---
models = ['Neural-Only','Symbolic-Only','NeuroSymbolic','NSMLP','NSAug','NSAdv','NSTrans']
results = [res_nn, res_sy, res_hy, res_hr, res_aug, res_adv, res_ft]
splits = ['Train', 'Val', 'Test']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
df_summary = build_results_table(models, results, splits, metrics)
print("\n=== Summary Comparison ===")
print(df_summary)