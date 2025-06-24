import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc

K = tf.keras.backend

def compute_mse_batchwise(X, X_pred, batch_size=64):
    errors = []
    for i in range(0, len(X), batch_size):
        mse = tf.reduce_mean(tf.math.squared_difference(X[i:i+batch_size], X_pred[i:i+batch_size]), axis=(1,2,3))
        errors.extend(mse.numpy())
    return errors

def compute_bce_batchwise(X, X_pred, batch_size=64):
    errors = []
    for i in range(0, len(X), batch_size):
        for x, x_pred in zip(X[i:i+batch_size], X_pred[i:i+batch_size]):
            bce = K.binary_crossentropy(K.flatten(x), K.flatten(x_pred))
            errors.append(K.eval(bce).mean())
    return errors

def get_training_callbacks(model_path="", loss="val_loss"):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=loss,
        patience=5,
        min_delta=1e-4,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor=loss,
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=loss,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, checkpoint, reduce_lr]

def create_dataset(X, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, X))  # For autoencoder: input = target
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

