import h5py
import numpy as np
from collections import defaultdict, Counter
from numpy.random import default_rng
import tensorflow as tf

D = tf.data.Dataset

def load_normal_data(hdf5_path, normalize=True):
    """
    Load only 'normal' samples from the HDF5 file for unsupervised training.
    """
    X = []
    with h5py.File(hdf5_path, 'r') as h5f:
        for id_folder in h5f.keys():
            if 'normal' in h5f[id_folder]:
                specs = h5f[id_folder]['normal']['mel_spectrograms'][:]
                X.append(specs)
    
    X = np.concatenate(X, axis=0)
    if normalize:
        X = (X - X.min()) / (X.max() - X.min())

    X = np.expand_dims(X, axis=-1)  # Add channel dim for CNNs
    return X

def load_all_data_for_eval(hdf5_path, normalize=True, return_filenames=False):
    X = []
    y = []
    filenames = []

    with h5py.File(hdf5_path, 'r') as h5f:
        for id_folder in h5f.keys():
            for label in ['normal', 'abnormal']:
                if label in h5f[id_folder]:
                    specs = h5f[id_folder][label]['mel_spectrograms'][:]
                    X.append(specs)
                    y.extend([0 if label == 'normal' else 1] * specs.shape[0])

                    if return_filenames:
                        files = h5f[id_folder][label]['filenames'][:]
                        decoded = [f.decode() if isinstance(f, bytes) else f for f in files]
                        filenames.extend(decoded)

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    if normalize:
        X = (X - X.min()) / (X.max() - X.min())

    X = np.expand_dims(X, axis=-1)
    if return_filenames:
        return X, y, filenames
    else:
        return X, y

def prepare_balanced_eval_dataset(data_path, batch_size=64, seed=42):
    """
    Load and balance evaluation data by clip, returning a tf.data.Dataset and some stats.

    Args:
        data_path (str): Path to the evaluation data.
        batch_size (int): Batch size for the tf.data.Dataset.
        seed (int): Random seed for reproducibility.

    Returns:
        eval_dataset (tf.data.Dataset): Balanced dataset of evaluation data.
        y_eval_balanced (np.ndarray): Corresponding labels for the balanced dataset.
        filenames_balanced (list): Filenames of the balanced segments.
    """
    # Step 1: Load all evaluation data with filenames
    X_eval, y_eval, filenames_eval = load_all_data_for_eval(data_path, return_filenames=True)

    # Step 2: Create globally unique clip IDs
    clip_ids = [fname.split("_seg")[0] for fname in filenames_eval]

    # Step 3: Group segments by clip
    clip_to_indices = defaultdict(list)
    clip_to_label = {}
    for idx, (cid, label) in enumerate(zip(clip_ids, y_eval)):
        clip_to_indices[cid].append(idx)
        clip_to_label[cid] = label  # Assumes label is consistent across segments

    # Step 4: Balance the clips by label
    rng = default_rng(seed)
    normal_clips = [cid for cid, lbl in clip_to_label.items() if lbl == 0]
    abnormal_clips = [cid for cid, lbl in clip_to_label.items() if lbl == 1]

    selected_normal_clips = rng.choice(normal_clips, size=len(abnormal_clips), replace=False)
    balanced_clips = list(selected_normal_clips) + abnormal_clips

    # Step 5: Collect all segment indices from selected clips
    final_indices = [idx for cid in balanced_clips for idx in clip_to_indices[cid]]

    # Step 6: Subset data and prepare tf.data.Dataset
    X_eval_balanced = X_eval[final_indices]
    y_eval_balanced = y_eval[final_indices]
    filenames_balanced = [filenames_eval[i] for i in final_indices]

    eval_dataset = D.from_tensor_slices(X_eval_balanced).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Step 7: Output stats
    print("Evaluation data shape:", X_eval_balanced.shape)
    print("Balanced label distribution:", Counter(y_eval_balanced))
    print("Full label distribution:", Counter(y_eval))

    return eval_dataset, X_eval_balanced, y_eval_balanced, filenames_balanced

def prepare_balanced_eval_numpy(data_path, seed=42):
    """
    Prepare balanced evaluation data for classical models (e.g., GMM) using numpy arrays.

    Returns:
        X_eval_balanced (np.ndarray)
        y_eval_balanced (np.ndarray)
        filenames_balanced (list)
    """
    X_eval, y_eval, filenames_eval = load_all_data_for_eval(data_path, return_filenames=True)
    clip_ids = [fname.split("_seg")[0] for fname in filenames_eval]

    clip_to_indices = defaultdict(list)
    clip_to_label = {}

    for idx, (cid, label) in enumerate(zip(clip_ids, y_eval)):
        clip_to_indices[cid].append(idx)
        clip_to_label[cid] = label

    rng = default_rng(seed)
    normal_clips = [cid for cid, lbl in clip_to_label.items() if lbl == 0]
    abnormal_clips = [cid for cid, lbl in clip_to_label.items() if lbl == 1]

    selected_normal_clips = rng.choice(normal_clips, size=len(abnormal_clips), replace=False)
    balanced_clips = list(selected_normal_clips) + abnormal_clips

    final_indices = [idx for cid in balanced_clips for idx in clip_to_indices[cid]]

    X_eval_balanced = X_eval[final_indices]
    y_eval_balanced = y_eval[final_indices]
    filenames_balanced = [filenames_eval[i] for i in final_indices]

    print("Evaluation data shape:", X_eval_balanced.shape)
    print("Balanced label distribution:", Counter(y_eval_balanced))
    print("Full label distribution:", Counter(y_eval))

    return X_eval_balanced, y_eval_balanced, filenames_balanced
