import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ae_loss(epochs, train_loss, val_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='C0')
    plt.plot(epochs, val_loss, label='Val Loss', color='C0', linestyle='--')
    plt.title('LSTM Autoencoder Reconstruction Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_nn_score_accuracy(epochs, acc_train, acc_val):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc_train, label='Train Accuracy', color='C1')
    plt.plot(epochs, acc_val, label='Val Accuracy', color='C1', linestyle='--')
    plt.title('Neural Score Accuracy', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_symbolic_score_distribution(g_sr_test, y_test):
    plt.figure(figsize=(8, 5))
    benign_scores = g_sr_test[y_test == 0]
    attack_scores = g_sr_test[y_test == 1]

    if np.std(benign_scores) > 0:
        sns.kdeplot(benign_scores, label='Benign', linestyle='--', fill=True)
    else:
        plt.axvline(benign_scores[0], color='blue', linestyle='--', label='Benign (Const)')

    if np.std(attack_scores) > 0:
        sns.kdeplot(attack_scores, label='Attack', fill=True)
    else:
        plt.axvline(attack_scores[0], color='orange', label='Attack (Const)')

    plt.title('Symbolic Score Distribution', fontsize=18)
    plt.xlabel('Symbolic Score', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_hybrid_score_distribution(f_nn, g_sr, R, y_test):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(f_nn[y_test == 1], label='Neural - Attack')
    sns.kdeplot(g_sr[y_test == 1], label='Symbolic - Attack')
    sns.kdeplot(R[y_test == 1], label='Hybrid - Attack')
    plt.title('Hybrid Risk Score Distribution', fontsize=18)
    plt.xlabel('Score', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()