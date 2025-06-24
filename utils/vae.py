import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae_v1(input_shape, latent_dim=64, loss_type='mse'):
    # Encoder
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense((input_shape[0]//4)*(input_shape[1]//4)*64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((input_shape[0]//4, input_shape[1]//4, 64))(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    decoder_outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE Model
    vae_inputs = encoder_inputs
    z_mean, z_log_var, z = encoder(vae_inputs)
    vae_outputs = decoder(z)
    vae = tf.keras.Model(vae_inputs, vae_outputs, name="vae")

    # VAE Loss
    if loss_type == 'mse':
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(vae_inputs, vae_outputs)
    elif loss_type == 'bce':
        reconstruction_loss = tf.keras.losses.BinaryCrossentropy()(vae_inputs, vae_outputs)
    else:
        raise ValueError("Invalid loss_type: choose 'mse' or 'bce'")
        
    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer='adam')
    return vae


def build_vae_v2(input_shape, latent_dim=64, loss_type='mse', dropout_rate=0.3, l2_reg=1e-4):
    reg = tf.keras.regularizers.l2(l2_reg)

    # Encoder
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg)(encoder_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    z_mean = tf.keras.layers.Dense(latent_dim, kernel_regularizer=reg)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, kernel_regularizer=reg)(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense((input_shape[0]//4)*(input_shape[1]//4)*64, activation='relu', kernel_regularizer=reg)(latent_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((input_shape[0]//4, input_shape[1]//4, 64))(x)

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    decoder_outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE Model
    vae_inputs = encoder_inputs
    z_mean, z_log_var, z = encoder(vae_inputs)
    vae_outputs = decoder(z)
    vae = tf.keras.Model(vae_inputs, vae_outputs, name="vae_v2")

    # VAE Loss
    if loss_type == 'mse':
        reconstruction_loss = tf.keras.losses.mean_squared_error(tf.keras.backend.flatten(vae_inputs),
                                                                tf.keras.backend.flatten(vae_outputs))
    elif loss_type == 'bce':
        reconstruction_loss = tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(vae_inputs),
                                                                tf.keras.backend.flatten(vae_outputs))
    else:
        raise ValueError("Invalid loss_type: choose 'mse' or 'bce'")

    reconstruction_loss = tf.reduce_mean(reconstruction_loss) * input_shape[0] * input_shape[1]
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0))
    return vae

