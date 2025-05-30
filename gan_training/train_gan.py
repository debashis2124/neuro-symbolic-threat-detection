import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from .gan_models import build_generator, build_discriminator

def setup_and_train_gan(X_real, latent_dim=32, epochs=500, batch_size=128,
                        lr_gen=0.001, lr_disc=0.0005):
    timesteps = X_real.shape[1]
    feature_dim = X_real.shape[2]

    generator = build_generator(latent_dim, feature_dim, timesteps)
    discriminator = build_discriminator(timesteps, feature_dim)

    opt_gen = tf.keras.optimizers.Adam(learning_rate=lr_gen, beta_1=0.5)
    opt_disc = tf.keras.optimizers.Adam(learning_rate=lr_disc, beta_1=0.5)

    discriminator.compile(optimizer=opt_disc, loss='binary_crossentropy', metrics=['accuracy'])

    discriminator.trainable = False
    z = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(z))
    gan_model = Model(z, gan_output)
    gan_model.compile(optimizer=opt_gen, loss='binary_crossentropy')

    for epoch in range(epochs):
        idx = np.random.randint(0, X_real.shape[0], batch_size)
        real = X_real[idx]

        z_noise = np.random.normal(size=(batch_size, latent_dim))
        fake = generator.predict(z_noise, verbose=0)

        y_real = np.ones((batch_size, 1)) * 0.9
        y_fake = np.zeros((batch_size, 1)) * 0.1

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real, y_real)
        d_loss_fake = discriminator.train_on_batch(fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        z_noise = np.random.normal(size=(batch_size, latent_dim))
        y_gan = np.ones((batch_size, 1))
        discriminator.trainable = False
        g_loss = gan_model.train_on_batch(z_noise, y_gan)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] D_loss: {d_loss[0]:.4f}, D_acc: {d_loss[1]:.4f}, G_loss: {g_loss:.4f}")

    return generator, discriminator, gan_model