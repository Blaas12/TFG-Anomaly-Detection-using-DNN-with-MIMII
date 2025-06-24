import tensorflow as tf

def build_autoencoder_v1(input_shape, loss_type="binary_crossentropy"):
    # (64, 96, 1)
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs, outputs, name="ae")
    model.compile(optimizer='adam', loss=loss_type)

    return model


def build_autoencoder_v2(input_shape, loss_type="binary_crossentropy", dropout_rate=0.3, l2_reg=1e-4):
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs, outputs, name="ae_v2")
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=loss_type)

    return model

def build_autoencoder_v3(input_shape, loss_type="binary_crossentropy"):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder con más profundidad
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2, padding='same')(x)

    # Bottleneck más comprimido
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    # Decoder con simetría y profundidad
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs, outputs, name="ae_v3")
    model.compile(optimizer='adam', loss=loss_type)

    return model
