import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_encode_data(path):
    df = pd.read_csv(path)
    df.drop(columns=['id'], inplace=True)
    for col in ['proto','service','state']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df