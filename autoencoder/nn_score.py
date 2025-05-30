import numpy as np

def nn_score(model, X_arr):
    X3 = X_arr.reshape((-1, 1, X_arr.shape[1]))
    recon = model.predict(X3, verbose=0)
    return np.mean((X3 - recon)**2, axis=(1, 2))