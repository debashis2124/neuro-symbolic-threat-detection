import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_and_split(df, feature_cols, label_col='label'):
    X = df[feature_cols].values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_scaled, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=2/9, stratify=y_tv, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler