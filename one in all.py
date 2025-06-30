import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.model_selection   import train_test_split
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)

import tensorflow as tf
from tensorflow.keras.models  import Model, clone_model
from tensorflow.keras.layers  import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import Callback


def save_and_show(fname):
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

# --- STEP 1: Load & Preprocess ----------------------------------------------
df = pd.read_csv("/content/UNSW_NB15_training-set.csv")
df.drop(columns=['id'], inplace=True)
for col in ['proto','service','state']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

feature_cols = [c for c in df.columns if c not in ('attack_cat','label')]
X = df[feature_cols].values
y = df['label'].values

# Remove rows with NaN values in the target variable 'y'
nan_indices = np.isnan(y)
X = X[~nan_indices]
y = y[~nan_indices]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tv, X_test, y_tv, y_test = train_test_split(
    X_scaled, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=2/9, stratify=y_tv, random_state=42)


# --- STEP 2: Neural Autoencoder --------------------------------------------
# STEP 2: LSTM AUTOENCODER FOR NEURAL SCORING
# -----------------------------------------------------------------------------
X_ben = X_train[y_train==0]
X_ben_lstm = X_ben.reshape(-1,1,X_ben.shape[1])

inp = Input(shape=(1,X_ben.shape[1]))
enc = LSTM(64,activation="relu")(inp)
dec = RepeatVector(1)(enc)
dec = LSTM(64,activation="relu",return_sequences=True)(dec)
out = TimeDistributed(Dense(X_ben.shape[1]))(dec)
autoencoder = Model(inp,out)
autoencoder.compile(optimizer="adam",loss="mse")

def nn_score(model,X_arr):
    X3 = X_arr.reshape(-1,1,X_arr.shape[1])
    recon = model.predict(X3,verbose=0)
    return np.mean((X3-recon)**2,axis=(1,2))

# find threshold on val
f_val = nn_score(autoencoder,X_val)
fpr,tpr,ths = roc_curve(y_val,f_val)
thr = ths[np.argmax(tpr-fpr)]

class AEAccHistory(Callback):
    def __init__(self,X_tr,y_tr,X_vl,y_vl,thr):
        super().__init__()
        self.X_tr,self.y_tr=X_tr,y_tr
        self.X_vl,self.y_vl=X_vl,y_vl
        self.thr=thr
        self.acc_tr=[]; self.acc_vl=[]
    def on_epoch_end(self,epoch,logs=None):
        tr=nn_score(self.model,self.X_tr)
        vl=nn_score(self.model,self.X_vl)
        self.acc_tr.append(accuracy_score(self.y_tr,(tr>=self.thr).astype(int)))
        self.acc_vl.append(accuracy_score(self.y_vl,(vl>=self.thr).astype(int)))

ae_cb = AEAccHistory(X_train,y_train,X_val,y_val,thr)
history_ae = autoencoder.fit(
    X_ben_lstm, X_ben_lstm,
    epochs=50, batch_size=256,
    validation_split=0.2,
    callbacks=[ae_cb],
    verbose=1
)

# final scores
f_nn_train = nn_score(autoencoder, X_train)
f_nn_val   = nn_score(autoencoder, X_val)
f_nn_test  = nn_score(autoencoder, X_test)

# Determine a fixed threshold from the final validation scores
fpr, tpr, ths = roc_curve(y_val, f_nn_val)
best_thr     = ths[np.argmax(tpr - fpr)]
print(f"Using fixed threshold = {best_thr:.4f}")

# --- STEP 3: Symbolic Rule Engine ------------------------------------------
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
X_val_df   = pd.DataFrame(X_val,   columns=feature_cols)
X_test_df  = pd.DataFrame(X_test,  columns=feature_cols)

def sym_score_df(df):
    s = pd.Series(0.0, index=df.index)
    s += (df['sbytes']            > 1e4) * 0.4
    s += (df['dbytes']            > 5e4) * 0.2
    s += (df['rate']              > 100) * 0.3
    s += (df['trans_depth']       > 10) * 0.2
    s += (df['response_body_len'] >1e5) * 0.1
    s += (df['ct_state_ttl']      < 5) * 0.3
    s += (df['spkts'] > df['dpkts']*5) * 0.2
    s += (df['is_sm_ips_ports']   == 1) * 0.2
    return s.values

g_sr_train = sym_score_df(X_train_df)
g_sr_val   = sym_score_df(X_val_df)
g_sr_test  = sym_score_df(X_test_df)

# --- STEP 4: Hybrid Score -------------------------------------------------
alpha, beta = 0.6, 0.4
R_train = alpha * f_nn_train + beta * g_sr_train
R_val   = alpha * f_nn_val   + beta * g_sr_val
R_test  = alpha * f_nn_test  + beta * g_sr_test

# --- STEP 5: MLP Features ------------------------------------------------
train_feats = np.column_stack((R_train, X_train[:, [7,9,32]]))
val_feats   = np.column_stack((R_val,   X_val[:,   [7,9,32]]))
test_feats  = np.column_stack((R_test,  X_test[:,  [7,9,32]]))


# -----------------------------------------------------------------------------
# STEP 6: TRAIN & EVAL BASELINE MODELS
# -----------------------------------------------------------------------------
def train_eval(X_tr,y_tr,X_vl,y_vl,X_te,y_te):
    clf=MLPClassifier((64,32),max_iter=300,early_stopping=False,random_state=42)
    clf.fit(X_tr,y_tr)
    def scores(X,y):
        p = clf.predict(X); pr = clf.predict_proba(X)[:,1]
        return [
            accuracy_score(y,p),
            precision_score(y,p,zero_division=0),
            recall_score(y,p,zero_division=0),
            f1_score(y,p,zero_division=0),
            roc_auc_score(y,pr)
        ]
    return clf, {'Train':scores(X_tr,y_tr),'Val':scores(X_vl,y_vl),'Test':scores(X_te,y_te)}

clf_nn,res_nn   = train_eval(f_nn_train.reshape(-1,1),y_train,
                            f_nn_val.reshape(-1,1),  y_val,
                            f_nn_test.reshape(-1,1), y_test)

clf_sy,res_sy   = train_eval(g_sr_train.reshape(-1,1),y_train,
                            g_sr_val.reshape(-1,1),  y_val,
                            g_sr_test.reshape(-1,1), y_test)

clf_hy,res_hy   = train_eval(R_train.reshape(-1,1),   y_train,
                            R_val.reshape(-1,1),     y_val,
                            R_test.reshape(-1,1),    y_test)

clf_hr,res_hr   = train_eval(train_feats,y_train,
                            val_feats,  y_val,
                            test_feats, y_test)


# -----------------------------------------------------------------------------
# STEP 7: SYNTHETIC AUGMENTATION
# -----------------------------------------------------------------------------
def gen_synth(X,noise=0.1):
    return X + np.random.normal(scale=noise,size=X.shape)

X_synth = gen_synth(X_train[y_train==0])
y_synth = np.ones_like(y_train[y_train==0])

f_synth = nn_score(autoencoder,X_synth)
g_synth = sym_score_df(pd.DataFrame(X_synth,columns=feature_cols))
R_synth = alpha*f_synth + beta*g_synth
synth_feats = np.column_stack((R_synth,X_synth[:,[7,9,32]]))

clf_aug,res_aug = train_eval(
    np.vstack([train_feats,synth_feats]),
    np.concatenate([y_train,y_synth]),
    val_feats,y_val,
    test_feats,y_test
)


# -----------------------------------------------------------------------------
# STEP 8: ADVERSARIAL VIA SIMPLE GAN
# -----------------------------------------------------------------------------
def build_gen(latent,feat,t=1):
    z=Input(shape=(latent,))
    x=Dense(64,activation='relu')(z)
    x=RepeatVector(t)(x)
    x=LSTM(64,activation='relu',return_sequences=True)(x)
    return Model(z,TimeDistributed(Dense(feat,activation='tanh'))(x))

def build_disc(t,feat):
    inp=Input(shape=(t,feat))
    x=LSTM(64,activation='relu')(inp)
    return Model(inp,Dense(1,activation='sigmoid')(x))

def train_gan(X_real,latent=32,epochs=50,batch=128):
    t,f=X_real.shape[1],X_real.shape[2]
    gen=build_gen(latent,f,t)
    disc=build_disc(t,f)
    optg=tf.keras.optimizers.Adam(1e-3,0.5)
    optd=tf.keras.optimizers.Adam(5e-4,0.5)
    disc.compile(optd,'binary_crossentropy')
    z=Input(shape=(latent,))
    gan=Model(z,disc(gen(z)))
    gan.compile(optg,'binary_crossentropy')
    for ep in range(epochs):
        idx=np.random.randint(0,X_real.shape[0],batch)
        real=X_real[idx]
        noise=np.random.normal(size=(batch,latent))
        fake=gen.predict(noise,verbose=0)
        disc.train_on_batch(real,np.ones((batch,1))*0.9)
        disc.train_on_batch(fake,np.zeros((batch,1))*0.1)
        gan.train_on_batch(noise,np.ones((batch,1)))
    return gen

gen = train_gan(X_ben_lstm,epochs=50)
z_adv=np.random.normal(size=(len(X_train),32))
X_adv_lstm=gen.predict(z_adv,verbose=0)
X_adv=X_adv_lstm.reshape(-1,X_adv_lstm.shape[2])
y_adv=np.ones_like(y_train)

f_adv=nn_score(autoencoder,X_adv)
g_adv=sym_score_df(pd.DataFrame(X_adv,columns=feature_cols))
R_adv=alpha*f_adv+beta*g_adv
adv_feats=np.column_stack((R_adv,X_adv[:,[7,9,32]]))

clf_adv,res_adv = train_eval(
    np.vstack([train_feats,adv_feats]),
    np.concatenate([y_train,y_adv]),
    val_feats,y_val,
    test_feats,y_test
)


# -----------------------------------------------------------------------------
# STEP 9: TRANSFER LEARNING (AE FINE-TUNE)
# -----------------------------------------------------------------------------
ae_ft=clone_model(autoencoder); ae_ft.set_weights(autoencoder.get_weights())
ae_ft.compile('adam','mse')
X_val_lstm=X_val.reshape(-1,1,X_val.shape[1])
ae_ft.fit(X_val_lstm,X_val_lstm,epochs=50,batch_size=256,verbose=1)

f_ft_train=nn_score(ae_ft,X_train)
f_ft_val  =nn_score(ae_ft,X_val)
f_ft_test =nn_score(ae_ft,X_test)
R_ft_train=alpha*f_ft_train+beta*g_sr_train
R_ft_val  =alpha*f_ft_val  +beta*g_sr_val
R_ft_test =alpha*f_ft_test +beta*g_sr_test

train_feats_ft=np.column_stack((R_ft_train,X_train[:,[7,9,32]]))
val_feats_ft  =np.column_stack((R_ft_val,  X_val[:,[7,9,32]]))
test_feats_ft =np.column_stack((R_ft_test, X_test[:,[7,9,32]]))

clf_ft,res_ft = train_eval(
    train_feats_ft,y_train,
    val_feats_ft,  y_val,
    test_feats_ft, y_test
)


# -----------------------------------------------------------------------------
# STEP 10: AGGREGATE RESULTS
# -----------------------------------------------------------------------------
models = ['Neural-Only','Symbolic-Only','NS Only','NS-MLP',
          'NS-Aug','NS-Adv','NS-Trans']
all_res = [res_nn,res_sy,res_hy,res_hr,res_aug,res_adv,res_ft]

splits  = ['Train','Val','Test']
metrics = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']

rows = []
for name, res_dict in zip(models, all_res):
    row = {'Model': name}
    for split in splits:
        # Check if split exists and has the correct number of metrics
        if split in res_dict and len(res_dict[split]) == len(metrics):
            for i, metric in enumerate(metrics):
                row[(split, metric)] = res_dict[split][i]
        else:
            # Handle missing splits or incorrect metric counts gracefully
            print(f"Warning: Data for model '{name}', split '{split}' is missing or incomplete.")
            for metric in metrics:
                row[(split, metric)] = np.nan # Or some other indicator of missing data
    rows.append(row)

df_res = pd.DataFrame(rows).set_index('Model')
df_res = df_res.reindex(columns=pd.MultiIndex.from_product([splits,metrics]))


print("\n=== Summary Comparison ===\n", df_res)
print("\n=== Detailed Results ===\n", df_res.to_string())


# -----------------------------------------------------------------------------
# STEP 11: PLOT EVERYTHING IN YOUR STYLE
# -----------------------------------------------------------------------------
# Prepare classifier & val-features dicts
classifiers = dict(zip(models,[clf_nn,clf_sy,clf_hy,clf_hr,clf_aug,clf_adv,clf_ft]))
val_feats_map= {
    'Neural-Only':   f_nn_val.reshape(-1,1),
    'Symbolic-Only': g_sr_val.reshape(-1,1),
    'NS Only':   R_val.reshape(-1,1),
    'NS-MLP':    val_feats,
    'NS-Aug':     val_feats,
    'NS-Adv':   val_feats,
    'NS-Trans':      val_feats_ft
}
test_df = df_res['Test']
# Figure 2a
def plot_figure_2a():
    ep = range(1,len(history_ae.history['loss'])+1)
    plt.figure(figsize=(8,5))
    plt.plot(ep,history_ae.history['loss'],label='Train Loss')
    plt.plot(ep,history_ae.history['val_loss'],'--',label='Val Loss')
    plt.title('Nural Only Reconstruction Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.grid(); save_and_show("figure_2a_autoencoder_loss.png")

# Figure 2b
def plot_figure_2b():
    ep=range(1,len(ae_cb.acc_tr)+1)
    plt.figure(figsize=(8,5))
    plt.plot(ep,ae_cb.acc_tr,label='Train Acc')
    plt.plot(ep,ae_cb.acc_vl,'--',label='Val Acc')
    plt.title('Neural Score Classification Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(); save_and_show("figure_2b_autoencoder_accuracy.png")

# Figure 3
def plot_figure_3():
    bs=g_sr_test[y_test==0]; at=g_sr_test[y_test==1]
    plt.figure(figsize=(8,5))
    if np.std(bs)>0: sns.kdeplot(bs,linestyle='--',fill=True,label='Benign')
    else: plt.axvline(bs[0],linestyle='--',label='Benign(const)')
    if np.std(at)>0: sns.kdeplot(at,fill=True,label='Attack')
    else: plt.axvline(at[0],label='Attack(const)')
    plt.title('Symbolic Score Distribution'); plt.xlabel('g_SR(x)'); plt.ylabel('Density')
    plt.legend(); plt.grid(); save_and_show("figure_3_score_distribution.png")

# Figure 4
def plot_figure_4():
    plt.figure(figsize=(8,5))
    sns.kdeplot(f_nn_test[y_test==1],label='Neural-Attack')
    sns.kdeplot(g_sr_test[y_test==1],label='Symbolic-Attack')
    sns.kdeplot(R_test[y_test==1],label='NS-Attack')
    plt.title('Risk Scores of Neural vs Symbolic vs NeuroSymbolic'); plt.xlabel('Score'); plt.ylabel('Density')
    plt.legend(); plt.grid(); save_and_show("figure_4_risk_score.png")

# --- After your function definitions ---
plot_figure_2a()
plot_figure_2b()
plot_figure_3()
plot_figure_4()

# Figure 5a
test_df['Accuracy'].plot(kind='bar',rot=45,figsize=(10,6),
                         title="Threat Prediction Accuracy")
plt.ylim(0,1); plt.grid(axis='y'); plt.tight_layout(); save_and_show("figure_5a_threat_prediction_accuracy.png")
# Figure 5b
sns.heatmap(test_df.T,annot=True,fmt=".2f",cmap="Blues")
plt.title('Test Set Evaluation Heatmap'); plt.tight_layout(); save_and_show("figure_5b_evaluation_heatmap.png")
# Figure 5c
plt.figure(figsize=(8,6))
for nm,clf in classifiers.items():
    Xv=val_feats_map[nm]; ypr=clf.predict_proba(Xv)[:,1]
    fpr,tpr,_=roc_curve(y_val,ypr)
    plt.plot(fpr,tpr,label=f"{nm}(AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--'); plt.title('ROC Curves'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(); plt.grid(); plt.tight_layout(); save_and_show("figure_5c_roc_curve.png")
# Figure 5d
plt.figure(figsize=(8,6))
for nm,clf in classifiers.items():
    Xv=val_feats_map[nm]; ypr=clf.predict_proba(Xv)[:,1]
    prec,rec,_=precision_recall_curve(y_val,ypr)
    plt.plot(rec,prec,label=f"{nm}(AP={auc(rec,prec):.2f})")
plt.title('Precision-Recall Curves'); plt.xlabel('Recall'); plt.ylabel('Precision')
plt.legend(); plt.grid(); plt.tight_layout(); save_and_show("figure_5d_precision_recall.png")
# Figures 6/7/8: Training/Validation/Test bar‐grids
flat = pd.DataFrame({
    f"{sp}_{met}": df_res[(sp,met)]
    for sp in ["Train","Val","Test"]
    for met in metrics
})

# Training
flat[[c for c in flat if c.startswith("Train")]].plot(
    kind='bar',rot=45,figsize=(14,6),title='Training Performance')
plt.grid(axis='y'); plt.tight_layout(); save_and_show("figure_training_preformance.png")
# Validation
flat[[c for c in flat if c.startswith("Val")]].plot(
    kind='bar',rot=45,figsize=(14,6),title='Validation Performance')
plt.grid(axis='y'); plt.tight_layout(); save_and_show("figure_validation_preformance.png")
# Test
flat[[c for c in flat if c.startswith("Test")]].plot(
    kind='bar',rot=45,figsize=(14,6),title='Test Performance')
plt.grid(axis='y'); plt.tight_layout(); save_and_show("figure_test_preformance.png")
# =============================================================================
# 1) Helper to build & compile a Keras MLP
# =============================================================================
def build_keras_mlp(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def fit_keras_model(X_tr, y_tr, X_vl, y_vl, epochs=50):
    model = build_keras_mlp(X_tr.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=256,
        callbacks=[es],
        verbose=0
    )
    return model, history

# =============================================================================
# 2) Prepare feature matrices for each variant
#    (uses variables from your pipeline: f_nn_train, f_nn_val, etc.)
# =============================================================================

# Neural-Only
Xn_tr = f_nn_train.reshape(-1,1)
Xn_vl = f_nn_val.reshape(-1,1)

# Symbolic-Only
Xs_tr = g_sr_train.reshape(-1,1)
Xs_vl = g_sr_val.reshape(-1,1)

# Hybrid-Only
Xh_tr = R_train.reshape(-1,1)
Xh_vl = R_val.reshape(-1,1)

# Hybrid+Raw
Xhr_tr = train_feats
Xhr_vl = val_feats

# Augmented
Xaug_tr = np.vstack([train_feats, synth_feats])
yaug_tr = np.concatenate([y_train, y_synth])
Xaug_vl = val_feats
yaug_vl = y_val

# Adversarial
Xadv_tr = np.vstack([train_feats, adv_feats])
yadv_tr = np.concatenate([y_train, y_adv])
Xadv_vl = val_feats
yadv_vl = y_val

# Transfer
Xtrans_tr = train_feats_ft
ytrans_tr = y_train
Xtrans_vl = val_feats_ft
ytrans_vl = y_val

# =============================================================================
# 3) Fit Keras MLPs and collect histories
# =============================================================================
models = {}
histories = {}

models['Neural-Only'],    histories['Neural-Only']    = fit_keras_model(Xn_tr,    y_train, Xn_vl,    y_val)
models['Symbolic-Only'],  histories['Symbolic-Only']  = fit_keras_model(Xs_tr,    y_train, Xs_vl,    y_val)
models['NS Only'],    histories['NS Only']    = fit_keras_model(Xh_tr,    y_train, Xh_vl,    y_val)
models['NS-MLP'],     histories['NS-MLP']     = fit_keras_model(Xhr_tr,   y_train, Xhr_vl,   y_val)
models['NS-Aug'],      histories['NS-Aug']      = fit_keras_model(Xaug_tr,  yaug_tr, Xaug_vl,  yaug_vl)
models['NS-Adv'],    histories['NS-Adv']    = fit_keras_model(Xadv_tr,  yadv_tr, Xadv_vl,  yadv_vl)
models['NS-Trans'],       histories['NS-Trans']       = fit_keras_model(Xtrans_tr,ytrans_tr,Xtrans_vl,ytrans_vl)

# =============================================================================
# 4) Plot epoch‐wise metrics for all variants
# =============================================================================
markers = ['o','s','^','v','d','*','h']
epochs = range(1, len(next(iter(histories.values())).history['loss']) + 1)

def plot_multi_histories(histories, metric, title):
    plt.figure(figsize=(10,6))
    markers = ['o','s','^','v','d','*','h']

    for (name, hist), m in zip(histories.items(), markers):
        # training curve
        y_tr = hist.history[metric]
        x_tr = range(1, len(y_tr) + 1)
        plt.plot(x_tr, y_tr, marker=m, label=f"{name} (train)")

        # validation curve
        y_vl = hist.history.get(f"val_{metric}", [])
        x_vl = range(1, len(y_vl) + 1)
        if y_vl:
            plt.plot(x_vl, y_vl, linestyle='--', marker=m, label=f"{name} (val)")

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_',' ').title())
    plt.grid(True)
    # Option A: automatic best placement
    plt.legend(loc='best', ncol=2, fontsize='small')

    # Option B: place legend outside to the right
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize='small')

    plt.tight_layout()
    save_and_show(f"figure_{metric}_per_epoch.png")


def plot_train_epoch_metrics(histories):
    plt.figure(figsize=(16,12))
    markers = ['o','s','^','v','d','*','h']
    metrics = ['loss','accuracy','precision','recall']

    for i, met in enumerate(metrics, 1):
        ax = plt.subplot(3, 2, i)
        for (name, hist), m in zip(histories.items(), markers):
            y = hist.history[met]
            x = range(1, len(y) + 1)
            ax.plot(x, y, marker=m, label=name)
        ax.set_title(f"Training {met.title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(met.title())
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

    # F1 subplot
    ax = plt.subplot(3, 2, 5)
    for (name, hist), m in zip(histories.items(), markers):
        p = np.array(hist.history['precision'])
        r = np.array(hist.history['recall'])
        f1 = 2*(p*r)/(p+r+1e-8)
        x = range(1, len(f1) + 1)
        ax.plot(x, f1, marker=m, label=name)
    ax.set_title("Training F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    save_and_show("figure_training_epoch_metrics.png")

def plot_val_epoch_metrics(histories):
    plt.figure(figsize=(16,12))
    markers = ['o','s','^','v','d','*','h']
    metrics = ['loss','accuracy','precision','recall']

    for i, met in enumerate(metrics, 1):
        ax = plt.subplot(3, 2, i)
        for (name, hist), m in zip(histories.items(), markers):
            y = hist.history.get(f"val_{met}", [])
            x = range(1, len(y) + 1)
            if y:
                ax.plot(x, y, linestyle='--', marker=m, label=name)
        ax.set_title(f"Validation {met.title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(met.title())
        ax.grid(True)
        ax.legend(loc='best', fontsize='small')

    # F1 subplot
    ax = plt.subplot(3, 2, 5)
    for (name, hist), m in zip(histories.items(), markers):
        p = np.array(hist.history.get('val_precision', []))
        r = np.array(hist.history.get('val_recall', []))
        if len(p) and len(r):
            f1 = 2*(p*r)/(p+r+1e-8)
            x = range(1, len(f1) + 1)
            ax.plot(x, f1, linestyle='--', marker=m, label=name)
    ax.set_title("Validation F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    save_and_show("figure_validation_epoch_metrics.png")

# Loss, Accuracy, Precision, Recall
plot_multi_histories(histories, 'loss',      'Loss per Epoch')
plot_multi_histories(histories, 'accuracy',  'Accuracy per Epoch')
plot_multi_histories(histories, 'precision', 'Precision per Epoch')
plot_multi_histories(histories, 'recall',    'Recall per Epoch')
# Call them:
plot_train_epoch_metrics(histories)
plot_val_epoch_metrics(histories)
