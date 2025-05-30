from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

def build_generator(latent_dim, feature_dim, timesteps=1):
    z = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(z)
    x = RepeatVector(timesteps)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    out = TimeDistributed(Dense(feature_dim, activation='tanh'))(x)
    return Model(z, out, name="Generator")

def build_discriminator(timesteps, feature_dim):
    inp = Input(shape=(timesteps, feature_dim))
    x = LSTM(64, activation='relu')(inp)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name="Discriminator")