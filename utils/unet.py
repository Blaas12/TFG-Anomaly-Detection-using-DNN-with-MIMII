import tensorflow as tf

def build_unet_v1(input_shape, loss_type="binary_crossentropy"):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder)
    x1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)

    x2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)

    # Bottleneck
    b = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    b = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(b)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D((2, 2))(b)
    u1 = tf.keras.layers.Concatenate()([u1, x2])
    x3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    x3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x3)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(x3)
    u2 = tf.keras.layers.Concatenate()([u2, x1])
    x4 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(u2)
    x4 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x4)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x4)

    model = tf.keras.Model(inputs, outputs, name="unet_v1")
    model.compile(optimizer="adam", loss=loss_type)

    return model


def conv_block(x, filters, dropout_rate=0.0, l2_reg=0.0, apply_dropout=False):
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout_rate and apply_dropout:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def build_unet_v2(input_shape, loss_type="binary_crossentropy", dropout_rate=0.3, l2_reg=1e-4):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32, dropout_rate, l2_reg)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64, dropout_rate, l2_reg)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = conv_block(p2, 128, dropout_rate, l2_reg, apply_dropout=True)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D((2, 2))(b)
    u1 = tf.keras.layers.Concatenate()([u1, c2])
    c3 = conv_block(u1, 64, dropout_rate, l2_reg, apply_dropout=True)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c1])
    c4 = conv_block(u2, 32, dropout_rate, l2_reg)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', padding='same')(c4)

    model = tf.keras.Model(inputs, outputs, name="unet_v2")
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss=loss_type)

    return model