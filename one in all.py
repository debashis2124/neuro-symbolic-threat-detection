import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)

import tensorflow as tf
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import Callback # Import Callback

# --- STEP 1: Load & Preprocess ----------------------------------------------
df = pd.read_csv("/content/UNSW_NB15_training-set.csv")
df.drop(columns=['id'], inplace=True)
for col in ['proto','service','state']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

feature_cols = [c for c in df.columns if c not in ('attack_cat','label')]
X = df[feature_cols].values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tv, X_test, y_tv, y_test = train_test_split(
    X_scaled, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=2/9, stratify=y_tv, random_state=42)

# --- STEP 2: Neural Autoencoder --------------------------------------------
X_benign = X_train[y_train == 0]
X_ben_lstm = X_benign.reshape((-1,1,X_benign.shape[1]))

inp = Input(shape=(1,X_benign.shape[1]))
enc = LSTM(64, activation='relu')(inp)
dec = RepeatVector(1)(enc)
dec = LSTM(64, activation='relu', return_sequences=True)(dec)
out = TimeDistributed(Dense(X_benign.shape[1]))(dec)

autoencoder = Model(inp,out)
autoencoder.compile(optimizer='adam', loss='mse')

# Define and use callback for AE accuracy plotting
class AEAccHistory(Callback):
    def __init__(self, X_train, y_train, X_val, y_val, threshold):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.thr      = threshold
        self.acc_train = []
        self.acc_val   = []
    def on_epoch_end(self, epoch, logs=None):
        # compute reconstruction errors
        # Need the nn_score function defined BEFORE this callback is used
        ftr = nn_score(self.model, self.X_train)
        fvl = nn_score(self.model, self.X_val)
        # classify by threshold
        ytr_pred = (ftr >= self.thr).astype(int)
        yvl_pred = (fvl >= self.thr).astype(int)
        # record accuracy
        self.acc_train.append( accuracy_score(self.y_train, ytr_pred) )
        self.acc_val.  append( accuracy_score(self.y_val,   yvl_pred) )

# Define nn_score function
def nn_score(model, X_arr):
    X3 = X_arr.reshape((-1,1,X_arr.shape[1]))
    recon = model.predict(X3, verbose=0)
    return np.mean((X3 - recon)**2, axis=(1,2))

# Calculate initial scores to determine threshold
f_nn_train = nn_score(autoencoder, X_train)
f_nn_val   = nn_score(autoencoder, X_val)
f_nn_test  = nn_score(autoencoder, X_test)

# Determine a fixed threshold from the final validation scores
fpr, tpr, ths = roc_curve(y_val, f_nn_val)
best_idx   = np.argmax(tpr - fpr)
best_thr   = ths[best_idx]
print(f"Using fixed threshold = {best_thr:.4f}")

# Re-train autoencoder with the callback
ae_acc_cb = AEAccHistory(X_train, y_train, X_val, y_val, best_thr)
history_ae = autoencoder.fit(
    X_ben_lstm, X_ben_lstm,
    epochs=100, batch_size=256,
    validation_split=0.2,
    callbacks=[ae_acc_cb], # Use the callback here
    verbose=1
)


# --- STEP 3: Symbolic Rule Engine -------------------------------------------
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
X_val_df   = pd.DataFrame(X_val,   columns=feature_cols)
X_test_df  = pd.DataFrame(X_test,  columns=feature_cols)

def sym_score_df(df_in):
    s = pd.Series(0.0, index=df_in.index)
    s += (df_in['sbytes'] > 1e4)       * 0.4
    s += (df_in['dbytes'] > 5e4)       * 0.2
    s += (df_in['rate']   > 100)       * 0.3
    s += (df_in['trans_depth'] > 10)   * 0.2
    s += (df_in['response_body_len']>1e5)*0.1
    s += (df_in['ct_state_ttl'] < 5)   * 0.3
    s += (df_in['spkts'] > df_in['dpkts']*5) * 0.2
    s += (df_in['is_sm_ips_ports'] == 1) * 0.2
    return s.values

g_sr_train = sym_score_df(X_train_df)
g_sr_val   = sym_score_df(X_val_df)
g_sr_test  = sym_score_df(X_test_df)

# --- STEP 4: Hybrid Score --------------------------------------------------
alpha, beta = 0.6, 0.4
R_train = alpha * f_nn_train + beta * g_sr_train
R_val   = alpha * f_nn_val   + beta * g_sr_val
R_test  = alpha * f_nn_test  + beta * g_sr_test

# --- STEP 5: Prepare Features for MLP --------------------------------------
train_feats = np.column_stack((R_train, X_train[:, [7,9,32]]))
val_feats   = np.column_stack((R_val,   X_val[:,   [7,9,32]]))
test_feats  = np.column_stack((R_test,  X_test[:,  [7,9,32]]))

# --- STEP 6: Train & Evaluate Classifiers ---------------------------------
def train_eval(name, X_tr, y_tr, X_vl, y_vl, X_te, y_te):
    clf = MLPClassifier((64,32), max_iter=300, early_stopping=True, random_state=42)
    clf.fit(X_tr, y_tr)
    def m(X, y):
        y_p = clf.predict(X)
        y_pr = clf.predict_proba(X)[:,1]
        return [
            accuracy_score(y, y_p),
            precision_score(y, y_p, zero_division=0),
            recall_score(y, y_p, zero_division=0),
            f1_score(y, y_p, zero_division=0),
            roc_auc_score(y, y_pr)
        ]
    return clf, {
        'Train': m(X_tr, y_tr),
        'Val':   m(X_vl, y_vl),
        'Test':  m(X_te, y_te)
    }

# Ensure correct reshaping for 1D inputs for MLPClassifier
clf_nn,  res_nn  = train_eval("Neural-Only",  f_nn_train.reshape(-1,1), y_train, f_nn_val.reshape(-1,1), y_val, f_nn_test.reshape(-1,1), y_test)
clf_sy,  res_sy  = train_eval("Symbolic-Only", g_sr_train.reshape(-1,1), y_train, g_sr_val.reshape(-1,1), y_val, g_sr_test.reshape(-1,1), y_test)
clf_hy,  res_hy  = train_eval("Hybrid-Only",  R_train.reshape(-1,1),    y_train, R_val.reshape(-1,1),    y_val, R_test.reshape(-1,1),    y_test)
clf_hr,  res_hr  = train_eval("Hybrid+Raw",   train_feats,             y_train, val_feats,             y_val, test_feats,             y_test)


# --- Synthetic Data Augmentation (stub) -----------------------------------
def generate_synthetic(X, noise_level=0.1):
    """Add small Gaussian noise to each feature vector."""
    return X + np.random.normal(scale=noise_level, size=X.shape)

X_synth = generate_synthetic(X_train)
# For augmentation, typically generate attacks (label = 1)
y_synth = np.ones_like(y_train)

f_synth = nn_score(autoencoder, X_synth)
g_synth = sym_score_df(pd.DataFrame(X_synth, columns=feature_cols))
R_synth = alpha * f_synth + beta * g_synth
synth_feats = np.column_stack((R_synth, X_synth[:, [7,9,32]]))

# Train classifier on original + synthetic data
clf_aug, res_aug = train_eval(
    "Augmented",
    np.vstack([train_feats, synth_feats]), # Combine original and synthetic features
    np.concatenate([y_train, y_synth]),    # Combine original and synthetic labels
    val_feats, y_val,
    test_feats, y_test
)


# --- GAN Adversarial Examples -----------------------------------------------
def build_generator(latent_dim, feature_dim, timesteps=1):
    """Builds a generator that outputs 3D data (batch, timesteps, features)."""
    z = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(z)
    x = RepeatVector(timesteps)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    # Output tanh activation to generate data within the scaled range (-1, 1)
    return Model(z, TimeDistributed(Dense(feature_dim, activation='tanh'))(x), name="Generator")

def build_discriminator(timesteps, feature_dim):
    """Builds a discriminator that accepts 3D data (batch, timesteps, features)."""
    inp = Input(shape=(timesteps, feature_dim))
    x = LSTM(64, activation='relu')(inp)
    return Model(inp, Dense(1, activation='sigmoid')(x), name="Discriminator")

def setup_and_train_gan(X_real, latent_dim=32, epochs=500, batch_size=128,
                        lr_gen=0.001, lr_disc=0.0005):
    timesteps = X_real.shape[1]
    feature_dim = X_real.shape[2]

    # Build models
    generator = build_generator(latent_dim, feature_dim, timesteps)
    discriminator = build_discriminator(timesteps, feature_dim)

    # Define optimizers with separate learning rates
    opt_gen = tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=0.5)
    opt_disc = tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=0.5)

    # Compile discriminator
    discriminator.compile(optimizer=opt_disc, loss='binary_crossentropy', metrics=['accuracy'])

    # Compile GAN (discriminator is frozen during generator training)
    discriminator.trainable = False
    z = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(z))
    gan_model = Model(z, gan_output, name="GAN")
    gan_model.compile(optimizer=opt_gen, loss='binary_crossentropy')

    print("--- Starting GAN Training ---")
    for epoch in range(epochs):
        # ---------------------
        # Train Discriminator
        # ---------------------
        idx = np.random.randint(0, X_real.shape[0], batch_size)
        real = X_real[idx]

        z = np.random.normal(size=(batch_size, latent_dim))
        fake = generator.predict(z, verbose=0)

        y_real = np.ones((batch_size, 1)) * 0.9
        y_fake = np.zeros((batch_size, 1)) * 0.1

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real, y_real)
        d_loss_fake = discriminator.train_on_batch(fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        # Train Generator
        # ---------------------
        z = np.random.normal(size=(batch_size, latent_dim))
        y_gan = np.ones((batch_size, 1))

        discriminator.trainable = False
        g_loss = gan_model.train_on_batch(z, y_gan)

        if epoch % 100 == 0:
            print(f"[GAN] Epoch {epoch}/{epochs} | "
                  f"D_loss: {d_loss[0]:.4f} (real: {d_loss_real[0]:.4f}, fake: {d_loss_fake[0]:.4f}) | "
                  f"D_acc: {d_loss[1]:.4f} | G_loss: {g_loss:.4f}")

    print("--- GAN Training Finished ---")
    return generator, discriminator, gan_model
 # Optionally return gan_model

# Train GAN on benign data reshaped for LSTM (batch, timesteps, features)
generator, discriminator, gan_model = setup_and_train_gan(X_ben_lstm, epochs=500)

# Generate adversarial examples using the trained generator
# We need to generate as many as the training set to augment it
z_adversarial = np.random.normal(size=(len(X_train), 32))
X_adv_lstm = generator.predict(z_adversarial, verbose=0) # Generator outputs 3D data

# The rest of the pipeline expects 2D data, so reshape back
X_adv = X_adv_lstm.reshape((-1, X_adv_lstm.shape[2]))
y_adv = np.ones_like(y_train) # Adversarial examples are intended to fool, often labeled as attacks

# Calculate scores for adversarial examples and prepare features
f_adv = nn_score(autoencoder, X_adv)
# Need to convert X_adv back to a DataFrame to use sym_score_df
g_adv = sym_score_df(pd.DataFrame(X_adv, columns=feature_cols))
R_adv = alpha * f_adv + beta * g_adv
adv_feats = np.column_stack((R_adv, X_adv[:, [7,9,32]]))

# Train classifier on original + adversarial data
clf_adv, res_adv = train_eval(
    "Adversarial",
    np.vstack([train_feats, adv_feats]), # Combine original and adversarial features
    np.concatenate([y_train, y_adv]),    # Combine original and adversarial labels
    val_feats, y_val,
    test_feats, y_test
)

# --- Transfer Learning (Autoencoder Fine-Tuning) ----------------------------
# Fine-tune the autoencoder on validation data (or a subset of new data)
ae_ft = clone_model(autoencoder)
ae_ft.set_weights(autoencoder.get_weights()) # Start with initial weights
ae_ft.compile(optimizer='adam', loss='mse')

# Reshape validation data for LSTM
X_val_lstm = X_val.reshape((-1, 1, X_val.shape[1]))

print("\n--- Starting Autoencoder Fine-tuning on Validation Data ---")
ae_ft.fit(
    X_val_lstm, X_val_lstm, # Fine-tune on validation data
    epochs=100, batch_size=256, verbose=1
)
print("--- Autoencoder Fine-tuning Finished ---")

# Calculate neural scores using the fine-tuned autoencoder
f_ft_train = nn_score(ae_ft, X_train)
f_ft_val = nn_score(ae_ft, X_val)
f_ft_test = nn_score(ae_ft, X_test)

# Combine with original symbolic scores
R_ft_train = alpha * f_ft_train + beta * g_sr_train
R_ft_val = alpha * f_ft_val + beta * g_sr_val
R_ft_test = alpha * f_ft_test + beta * g_sr_test

# Prepare features for MLP using scores from the fine-tuned AE
train_feats_ft = np.column_stack((R_ft_train, X_train[:, [7,9,32]]))
val_feats_ft = np.column_stack((R_ft_val, X_val[:, [7,9,32]]))
test_feats_ft = np.column_stack((R_ft_test, X_test[:, [7,9,32]]))

# Train a new MLP Classifier on these transfer-learned features
clf_ft = MLPClassifier((64,32), max_iter=300, early_stopping=True, random_state=42)
clf_ft.fit(train_feats_ft, y_train) # Train MLP on fine-tuned features from original train data

# Evaluate this new MLP on the validation and test sets using features from the fine-tuned AE
# Get predictions and probabilities first
y_pred_ft_train = clf_ft.predict(train_feats_ft)
y_prob_ft_train = clf_ft.predict_proba(train_feats_ft)[:, 1] if len(np.unique(y_train)) == 2 else np.nan

y_pred_ft_val = clf_ft.predict(val_feats_ft)
y_prob_ft_val = clf_ft.predict_proba(val_feats_ft)[:, 1] if len(np.unique(y_val)) == 2 else np.nan

y_pred_ft_test = clf_ft.predict(test_feats_ft)
y_prob_ft_test = clf_ft.predict_proba(test_feats_ft)[:, 1] if len(np.unique(y_test)) == 2 else np.nan


res_ft = {
    'Train': [
        accuracy_score(y_train, y_pred_ft_train),
        precision_score(y_train, y_pred_ft_train, zero_division=0),
        recall_score(y_train, y_pred_ft_train, zero_division=0),
        f1_score(y_train, y_pred_ft_train, zero_division=0),
        roc_auc_score(y_train, y_prob_ft_train) if not np.isnan(y_prob_ft_train).all() else np.nan
    ],
    'Val': [
        accuracy_score(y_val, y_pred_ft_val),
        precision_score(y_val, y_pred_ft_val, zero_division=0),
        recall_score(y_val, y_pred_ft_val, zero_division=0),
        f1_score(y_val, y_pred_ft_val, zero_division=0),
        roc_auc_score(y_val, y_prob_ft_val) if not np.isnan(y_prob_ft_val).all() else np.nan
    ],
     'Test': [
        accuracy_score(y_test, y_pred_ft_test),
        precision_score(y_test, y_pred_ft_test, zero_division=0),
        recall_score(y_test, y_pred_ft_test, zero_division=0),
        f1_score(y_test, y_pred_ft_test, zero_division=0),
        roc_auc_score(y_test, y_prob_ft_test) if not np.isnan(y_prob_ft_test).all() else np.nan
    ]
}
# Correct the first metric in 'Val' and 'Test' which was missing y_true
res_ft['Val'][0] = accuracy_score(y_val, clf_ft.predict(val_feats_ft))
res_ft['Test'][0] = accuracy_score(y_test, clf_ft.predict(test_feats_ft))


# --- Rebuild results DataFrame to include all variants ----------------------
models = [
    'Neural-Only',
    'Symbolic-Only',
    'NeuroSymbolic',
    'NSMLP',
    'NSAug',
    'NSAdv',
    'NSTrans'
]
all_res = [res_nn, res_sy, res_hy, res_hr, res_aug, res_adv, res_ft]
splits = ['Train', 'Val', 'Test']
metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']

data_list = []
for model_name, results_dict in zip(models, all_res):
    row_data = {'Model': model_name}
    for split in splits:
        if split in results_dict and len(results_dict[split]) == len(metrics):
            for i, metric in enumerate(metrics):
                row_data[(split, metric)] = results_dict[split][i]
        else:
             # Handle cases where a split might be missing or metrics don't match
             # For 'NeuroTrans', train_eval was structured differently, so handle it explicitly if needed
             # Based on res_ft creation above, all splits should exist and have 5 metrics.
             if split not in results_dict:
                  print(f"Warning: Split '{split}' missing for model '{model_name}'.")
             elif len(results_dict[split]) != len(metrics):
                 print(f"Warning: Mismatch in metrics for model '{model_name}', split '{split}'. Expected {len(metrics)}, got {len(results_dict[split])}.")


    data_list.append(row_data)

df_res = pd.DataFrame(data_list).set_index('Model')
expected_columns = pd.MultiIndex.from_product([splits, metrics])
df_res = df_res.reindex(columns=expected_columns)

# --- STEP 7: Summary Table (using the correctly structured df_res) ---------
print("\n=== Summary Comparison ===")
print(df_res)

##############Figure 2: Neural Risk Scoring (LSTM Autoencoder)#############
# === Figure 2a: Autoencoder Loss ===
def plot_figure_2a(epochs, train_loss, val_loss):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label='Train Loss', color='C0')
    plt.plot(epochs, val_loss, label='Val Loss', color='C0', linestyle='--')
    plt.title('LSTM Autoencoder Reconstruction Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# === Figure 2b: Neural Score Accuracy ===
def plot_figure_2b(epochs, acc_train, acc_val):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, acc_train, label='Train Accuracy', color='C1')
    plt.plot(epochs, acc_val, label='Val Accuracy', color='C1', linestyle='--')
    plt.title('Neural Score Classification Accuracy ($f_{NN}(x)$)', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# === Figure 3: Symbolic Rule Scoring ===
def plot_figure_3(g_sr_test, y_test):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(8,5))

    # Check for variation in symbolic scores
    benign_scores = g_sr_test[y_test == 0]
    attack_scores = g_sr_test[y_test == 1]

    if np.std(benign_scores) > 0:
        sns.kdeplot(benign_scores, label='Symbolic - Benign', linestyle='--', fill=True)
    else:
        plt.axvline(x=benign_scores[0], color='blue', linestyle='--', label='Symbolic - Benign (Const)')

    if np.std(attack_scores) > 0:
        sns.kdeplot(attack_scores, label='Symbolic - Attack', fill=True)
    else:
        plt.axvline(x=attack_scores[0], color='orange', label='Symbolic - Attack (Const)')

    plt.title('Symbolic Score Distribution $g_{SR}(x)$', fontsize=18)
    plt.xlabel('Symbolic Score Value', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Figure 4: Hybrid Risk Fusion ===
def plot_figure_4(f_nn_test, g_sr_test, R_test, y_test):
    plt.figure(figsize=(10,5))
    sns.kdeplot(f_nn_test[y_test==1], label='Neural - Attack')
    sns.kdeplot(g_sr_test[y_test==1], label='Symbolic - Attack')
    sns.kdeplot(R_test[y_test==1], label='Hybrid - Attack')
    plt.title('Score Distribution Comparison – $f_{NN}(x)$, $g_{SR}(x)$, $R(x)$', fontsize=18)
    plt.xlabel('Risk Score', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

################ Figure 5: Threat Classification Performance ################
# 5a: Bar chart
test_df.plot(kind='bar', rot=45, figsize=(10,6), title="Final Threat Prediction Accuracy")
plt.grid(axis='y'); plt.tight_layout(); plt.show()

# 5b: Heatmap
sns.heatmap(test_df.T, annot=True, fmt=".2f", cmap="Blues")
plt.title('Test Set Evaluation Heatmap')
plt.xlabel('Model'); plt.ylabel('Metric')
plt.show()

# 5c: ROC Curve
plt.figure(figsize=(8,6))
for name, clf in classifiers.items():
    feats = val_features_map.get(name)
    if feats is not None:
        y_pr = clf.predict_proba(feats)[:,1]
        fpr, tpr, _ = roc_curve(y_val, y_pr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curves'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(); plt.grid(True); plt.show()

# 5d: PR Curve
plt.figure(figsize=(8,6))
for name, clf in classifiers.items():
    feats = val_features_map.get(name)
    if feats is not None:
        y_pr = clf.predict_proba(feats)[:,1]
        prec, rec, _ = precision_recall_curve(y_val, y_pr)
        plt.plot(rec, prec, label=f'{name} (AP={auc(rec,prec):.2f})')
plt.title('Precision-Recall Curves')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.legend(); plt.grid(True); plt.show()


exclude_symbolic = ['Symbolic-Only']
train_val_models = [m for m in flat.index if m not in exclude_symbolic]

# Prepare subsets
flat_train = flat.loc[train_val_models, [c for c in flat.columns if c.startswith('Train')]]
flat_val = flat.loc[train_val_models, [c for c in flat.columns if c.startswith('Val')]]
flat_test = flat[[c for c in flat.columns if c.startswith('Test')]]  # keep Symbolic-Only for test

# === Figure 6a, 7a, 8a: Training Performance ===
flat_train.plot(kind='bar', rot=45, figsize=(14,6), title='Training Performance Across Models')
plt.grid(axis='y'); plt.tight_layout(); plt.show()

# === Figure 6b, 7b, 8b: Validation Performance ===
flat_val.plot(kind='bar', rot=45, figsize=(14,6), title='Validation Performance Across Models')
plt.grid(axis='y'); plt.tight_layout(); plt.show()

# === Figure 6c, 7c, 8c: Test Performance ===
flat_test.plot(kind='bar', rot=45, figsize=(14,6), title='Test Performance Across Models')
plt.grid(axis='y'); plt.tight_layout(); plt.show()


df_comparison = df_res['Test'].drop('Symbolic-Only', errors='ignore')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

# Create wider and taller figure with larger subplots
fig, axes = plt.subplots(1, len(metrics), figsize=(30, 8))  # Increased from (24, 5) to (30, 8)

for ax, metric in zip(axes, metrics):
    df_comparison[metric].plot(kind='bar', ax=ax)
    ax.set_title(metric, fontsize=20)
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel(metric, fontsize=18)
    ax.tick_params(axis='x', labelrotation=45, labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y')

plt.suptitle('Evaluation Metrics (Test Set) Across All Models', fontsize=24, y=1.10)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Add space for the title
plt.show()