import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial import distance
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd


def plot_loss_curve(history):
    """
    Plots the training and validation loss over epochs.

    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
    """
    import matplotlib.pyplot as plt

    train_loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if train_loss is None or val_loss is None:
        raise ValueError("History object must contain 'loss' and 'val_loss' keys.")

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_score_histogram(scores, y_eval_bal):
    """
    Histogram of GMM anomaly scores (negative log-likelihood).
    """
    normal_scores = [s for s, y in zip(scores, y_eval_bal) if y == 0]
    abnormal_scores = [s for s, y in zip(scores, y_eval_bal) if y == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal')
    plt.hist(abnormal_scores, bins=50, alpha=0.6, label='Anomaly')
    plt.xlabel("Negative Log-Likelihood")
    plt.ylabel("Frequency")
    plt.title("GMM Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_error_histogram(errors_mse, y_eval_bal):
    normal_errors = [e for e, y in zip(errors_mse, y_eval_bal) if y == 0]
    abnormal_errors = [e for e, y in zip(errors_mse, y_eval_bal) if y == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal')
    plt.hist(abnormal_errors, bins=50, alpha=0.6, label='Anomaly')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution (Balanced Set)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_segment_level_pr_curve(errors, y_eval_bal):
    precision, recall, _ = precision_recall_curve(y_eval_bal, errors)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Segment-Level Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pr_auc


def aggregate_clip_scores(filenames, scores, labels, agg_func=np.max):
    """
    Aggregates segment-level scores into clip-level scores using a specified function (e.g. max or mean).

    Args:
        filenames (list): Segment-level filenames.
        scores (list or np.ndarray): Segment-level anomaly scores.
        labels (list or np.ndarray): Segment-level ground truth labels.
        agg_func (callable): Aggregation function (e.g. np.max, np.mean).

    Returns:
        y_true (list): True clip-level labels.
        y_score (list): Aggregated clip-level anomaly scores.
    """

    clip_ids = [fname.split('_seg')[0] for fname in filenames]
    clip_to_scores = defaultdict(list)
    clip_to_label = {}

    for cid, score, label in zip(clip_ids, scores, labels):
        clip_to_scores[cid].append(score)
        clip_to_label[cid] = label  # assumes label is consistent per clip

    clip_scores = {cid: agg_func(vals) for cid, vals in clip_to_scores.items()}
    y_true = [clip_to_label[cid] for cid in clip_scores]
    y_score = [clip_scores[cid] for cid in clip_scores]

    return y_true, y_score


def plot_clip_level_pr_curve(errors, y_eval_bal, filenames_balanced):
    y_true, y_score = aggregate_clip_scores(filenames_balanced, errors, y_eval_bal)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Clip-Level Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pr_auc


def plot_reconstructions(X_eval, X_pred, y_eval, num_samples=3):
    np.random.seed(42)
    normal_indices = np.where(y_eval == 0)[0]
    abnormal_indices = np.where(y_eval == 1)[0]
    
    normal_samples = np.random.choice(normal_indices, size=num_samples, replace=False)
    abnormal_samples = np.random.choice(abnormal_indices, size=num_samples, replace=False)
    
    plt.figure(figsize=(20, 4 * num_samples))
    
    for i, idx in enumerate(normal_samples):
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(X_eval[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Normal: Original")
        plt.colorbar()
        
        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.imshow(X_pred[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Normal: Reconstruction")
        plt.colorbar()
        
    for i, idx in enumerate(abnormal_samples):
        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(X_eval[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Anomaly: Original")
        plt.colorbar()
        
        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(X_pred[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Anomaly: Reconstruction")
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def plot_error_maps(X_eval, X_pred, y_eval, num_samples=3):
    np.random.seed(42)
    normal_indices = np.where(y_eval == 0)[0]
    abnormal_indices = np.where(y_eval == 1)[0]

    normal_samples = np.random.choice(normal_indices, size=num_samples, replace=False)
    abnormal_samples = np.random.choice(abnormal_indices, size=num_samples, replace=False)

    plt.figure(figsize=(20, 4 * num_samples))

    for i, idx in enumerate(normal_samples):
        error_map = (X_eval[idx] - X_pred[idx])**2

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(X_eval[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Normal: Original")
        plt.colorbar()

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(X_pred[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Normal: Reconstruction")
        plt.colorbar()

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(error_map.squeeze(), aspect=64/96, origin='lower', cmap='hot')
        plt.title("Normal: Error Map")
        plt.colorbar()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 4 * num_samples))

    for i, idx in enumerate(abnormal_samples):
        error_map = (X_eval[idx] - X_pred[idx])**2

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(X_eval[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Anomaly: Original")
        plt.colorbar()

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(X_pred[idx].squeeze(), aspect=64/96, origin='lower')
        plt.title("Anomaly: Reconstruction")
        plt.colorbar()

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(error_map.squeeze(), aspect=64/96, origin='lower', cmap='hot')
        plt.title("Anomaly: Error Map")
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def plot_segment_latent_space_2d(z_mean, y_eval, method='pca', n_components=2, perplexity=30):
    """
    Interactive latent space visualization using plotly.
    
    Args:
        z_mean (np.ndarray): Latent space (n_samples, latent_dim).
        y_eval (np.ndarray): Labels (0 = normal, 1 = anomaly).
        method (str): 'pca' or 'tsne'.
        n_components (int): Latent projection dimensions.
        perplexity (float): t-SNE perplexity (only used for t-SNE).
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    z_proj = reducer.fit_transform(z_mean)
    
    df = {
        "Dim1": z_proj[:, 0],
        "Dim2": z_proj[:, 1],
        "Label": ["Normal" if y == 0 else "Anomaly" for y in y_eval]
    }
    
    fig = px.scatter(
        df, x="Dim1", y="Dim2", color="Label", 
        title=f"Latent Space ({method.upper()})",
        opacity=0.7
    )
    fig.update_layout(height=600, width=800)
    fig.show()


def aggregate_clip_latent_vectors(z_mean, filenames, y_eval, aggregation_func=np.mean):
    """
    Aggregates segment-level latent vectors into clip-level latent vectors.

    Args:
        z_mean (np.ndarray): Latent vectors (n_segments, latent_dim).
        filenames (list): Segment filenames to extract clip IDs.
        y_eval (np.ndarray): Segment-level labels.
        aggregation_func (callable): Aggregation function (default: np.mean).

    Returns:
        clip_latent_array (np.ndarray): (n_clips, latent_dim)
        clip_labels (np.ndarray): (n_clips,)
    """
    clip_ids = [fname.split('_seg')[0] for fname in filenames]
    
    clip_to_latents = defaultdict(list)
    clip_to_label = {}

    for cid, latent_vec, label in zip(clip_ids, z_mean, y_eval):
        clip_to_latents[cid].append(latent_vec)
        clip_to_label[cid] = label

    clip_latent_vectors = {cid: aggregation_func(vectors, axis=0) for cid, vectors in clip_to_latents.items()}
    
    clip_latent_array = np.array(list(clip_latent_vectors.values()))
    clip_labels = np.array([clip_to_label[cid] for cid in clip_latent_vectors.keys()])

    return clip_latent_array, clip_labels


def plot_clip_latent_space_2d(clip_latent_array, clip_labels, method="pca", n_components=2, perplexity=30):
    """
    Plots clip-level latent space using PCA + Plotly.
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    z_proj = reducer.fit_transform(clip_latent_array)
    
    df = {
        "Dim1": z_proj[:, 0],
        "Dim2": z_proj[:, 1],
        "Label": ["Normal" if y == 0 else "Anomaly" for y in clip_labels]
    }
    fig = px.scatter(df, x="Dim1", y="Dim2", color="Label",
                     title="Clip-Level Latent Space (PCA)",
                     opacity=0.7, height=600, width=800)
    fig.show()


def plot_segment_latent_space_3d(z_mean, y_eval, method='pca', perplexity=30):
    """
    Interactive 3D latent space visualization at segment level using plotly.

    Args:
        z_mean (np.ndarray): Latent space (n_samples, latent_dim).
        y_eval (np.ndarray): Labels (0 = normal, 1 = anomaly).
        method (str): 'pca' or 'tsne'.
        perplexity (float): t-SNE perplexity (only used for t-SNE).
    """
    if method == "pca":
        reducer = PCA(n_components=3)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    z_proj = reducer.fit_transform(z_mean)

    df = pd.DataFrame({
        "Dim1": z_proj[:, 0],
        "Dim2": z_proj[:, 1],
        "Dim3": z_proj[:, 2],
        "Label": ["Normal" if y == 0 else "Anomaly" for y in y_eval]
    })

    fig = px.scatter_3d(df, x="Dim1", y="Dim2", z="Dim3", color="Label",
                        title=f"Segment-Level Latent Space ({method.upper()}) - 3D",
                        opacity=0.7, height=700)
    fig.update_traces(marker=dict(size=4))
    fig.show()


def plot_clip_latent_space_3d(clip_latent_array, clip_labels, method="pca", perplexity=30):
    """
    Interactive 3D latent space visualization at clip level using plotly.

    Args:
        clip_latent_array (np.ndarray): (n_clips, latent_dim)
        clip_labels (np.ndarray): (n_clips,)
        method (str): 'pca' or 'tsne'
        perplexity (int): t-SNE perplexity (used only if method == 'tsne')
    """
    if method == "pca":
        reducer = PCA(n_components=3)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    z_proj = reducer.fit_transform(clip_latent_array)

    df = pd.DataFrame({
        "Dim1": z_proj[:, 0],
        "Dim2": z_proj[:, 1],
        "Dim3": z_proj[:, 2],
        "Label": ["Normal" if y == 0 else "Anomaly" for y in clip_labels]
    })

    fig = px.scatter_3d(df, x="Dim1", y="Dim2", z="Dim3", color="Label",
                        title=f"Clip-Level Latent Space ({method.upper()}) - 3D",
                        opacity=0.7, height=700)
    fig.update_traces(marker=dict(size=4))
    fig.show()


def latent_mahalanobis_anomaly_scoring(z_train, clip_latent_array, clip_labels):
    """
    Compute Mahalanobis distance anomaly scores on clip-level latent vectors.

    Args:
        z_train (np.ndarray): Latent vectors from normal training segments or clips.
        clip_latent_array (np.ndarray): Clip-level latent vectors (from evaluation set).
        clip_labels (np.ndarray): Clip-level ground truth labels (normal=0, anomaly=1).

    Returns:
        dists (np.ndarray): Mahalanobis distances (anomaly scores).
        pr_auc (float): Precision-Recall AUC.
    """
    # Compute mean and inverse covariance from normal training latent space
    mean_vec = np.mean(z_train, axis=0)
    cov_matrix = np.cov(z_train, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)

    # Mahalanobis distance for each evaluation clip
    dists = np.array([distance.mahalanobis(vec, mean_vec, inv_cov) for vec in clip_latent_array])

    # PR AUC calculation
    precision, recall, _ = precision_recall_curve(clip_labels, dists)
    pr_auc = auc(recall, precision)

    # Plot PR curve
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Latent Space Mahalanobis PR Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return dists, pr_auc


def plot_window_level_pr_curve(dists, clip_labels, window_size=30, agg_func=np.max):
    """
    Aggregates clip-level scores into window-level scores.

    Args:
        dists (np.ndarray): Clip-level anomaly scores.
        clip_labels (np.ndarray): Clip-level ground truth labels (0/1).
        window_size (int): Number of clips per window.
        agg_func (callable): Aggregation function (default: np.max).

    Returns:
        PR-Curve for the whole window
    """
    num_windows = len(dists) // window_size

    dists_window = np.array([agg_func(dists[i*window_size:(i+1)*window_size]) for i in range(num_windows)])
    labels_window = np.array([np.max(clip_labels[i*window_size:(i+1)*window_size]) for i in range(num_windows)])

    precision, recall, _ = precision_recall_curve(labels_window, dists_window)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"Window PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Aggregated Window PR Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pr_auc


def plot_gmm_scores_3d(X_reduced, y_labels, scores, title="3D GMM Anomaly Scores (PCA)"):
    """
    3D visualization of samples in PCA space colored by anomaly scores.

    Args:
        X_reduced (np.ndarray): 3D PCA-reduced data (shape: N, 3)
        y_labels (np.ndarray): Ground truth labels (0=normal, 1=anomaly)
        scores (np.ndarray): Anomaly scores (e.g., -log-likelihood)
        title (str): Plot title
    """
    df = pd.DataFrame({
        "PC1": X_reduced[:, 0],
        "PC2": X_reduced[:, 1],
        "PC3": X_reduced[:, 2],
        "Label": y_labels,
        "Score": scores
    })

    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3",
        color="Score", color_continuous_scale="Viridis",
        symbol="Label", symbol_map={0: "circle", 1: "x"},
        title=title, height=700
    )

    fig.update_traces(marker=dict(size=3))
    fig.show()


def compute_window_pr_auc(dists, clip_labels, window_size, agg_func=np.max):
    """
    Compute PR AUC score over sliding windows of clip-level scores.

    Args:
        dists (np.ndarray): Anomaly scores (use -dist for reconstruction models).
        clip_labels (np.ndarray): Ground truth clip labels (0=normal, 1=anomaly).
        window_size (int): Number of clips per window.
        agg_func (callable): Function to aggregate scores in a window (e.g. np.max).

    Returns:
        float: PR AUC score for this window size.
    """
    num_windows = len(dists) // window_size
    if num_windows == 0:
        return np.nan  # avoid crash on very large window sizes

    dists_window = np.array([
        agg_func(dists[i*window_size:(i+1)*window_size])
        for i in range(num_windows)
    ])
    labels_window = np.array([
        np.max(clip_labels[i*window_size:(i+1)*window_size])
        for i in range(num_windows)
    ])
    precision, recall, _ = precision_recall_curve(labels_window, dists_window)
    return auc(recall, precision)


def plot_pr_auc_vs_window(dists, clip_labels, window_sizes=None, agg_func=np.max):
    if window_sizes is None:
        window_sizes = list(range(5, 100, 2))

    auc_scores = [
        compute_window_pr_auc(dists, clip_labels, ws, agg_func=agg_func)
        for ws in window_sizes
    ]

    df_auc = pd.DataFrame({
        'Window Size': window_sizes,
        'PR AUC': auc_scores
    })

    fig = px.line(
        df_auc, x='Window Size', y='PR AUC',
        markers=True,
        title="PR AUC vs Window Size",
        # template="plotly_dark"
    )

    fig.update_layout(height=500, width=800)
    fig.show()