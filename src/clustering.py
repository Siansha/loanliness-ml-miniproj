"""
Clustering analysis module for Home Credit Default Risk prediction.
Performs K-Means clustering and trains separate models per cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import silhouette_score, davies_bouldin_score
from models import calculate_metrics, apply_smote
import warnings
warnings.filterwarnings('ignore')


def perform_kmeans_clustering(X, n_clusters=4, random_state=42):
    """
    Perform K-Means clustering on the dataset.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple : (kmeans_model, cluster_labels, metrics_dict)
    """
    print(f"\nPerforming K-Means clustering with k={n_clusters}...")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate clustering metrics
    silhouette = silhouette_score(X, cluster_labels)
    davies_bouldin = davies_bouldin_score(X, cluster_labels)
    
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'inertia': kmeans.inertia_
    }
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Show cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.2f}%)")
    
    return kmeans, cluster_labels, metrics


def visualize_clusters_pca(X, cluster_labels, save_path=None):
    """
    Visualize clusters using PCA dimensionality reduction.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    cluster_labels : array
        Cluster assignments
    save_path : str
        Path to save the plot
    """
    print("\nVisualizing clusters using PCA...")
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('K-Means Clustering Visualization (PCA)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA plot saved to {save_path}")
    
    plt.show()
    
    return pca, X_pca


def train_cluster_specific_model(X_cluster, y_cluster, cluster_id, model_type='lightgbm'):
    """
    Train a model specific to a cluster.
    
    Parameters:
    -----------
    X_cluster : DataFrame
        Features for the cluster
    y_cluster : Series
        Target for the cluster
    cluster_id : int
        Cluster identifier
    model_type : str
        Type of model to train ('lightgbm', 'random_forest', 'logistic')
        
    Returns:
    --------
    tuple : (model, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"Training model for Cluster {cluster_id}")
    print(f"{'='*60}")
    print(f"Cluster size: {len(X_cluster)} samples")
    print(f"Target distribution:\n{pd.Series(y_cluster).value_counts()}")
    
    # Check if cluster has both classes
    if len(np.unique(y_cluster)) < 2:
        print(f"Warning: Cluster {cluster_id} has only one class. Skipping...")
        return None, None
    
    # Split into train and validation (80-20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42, stratify=y_cluster
    )
    
    # Apply SMOTE if minority class is too small
    if (y_train == 1).sum() < 10:
        print("Minority class too small, using class weights instead of SMOTE")
        X_train_resampled, y_train_resampled = X_train, y_train
    else:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Train model based on type
    if model_type == 'lightgbm':
        scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_resampled, y_train_resampled)
        
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_resampled, y_train_resampled)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, f"Cluster {cluster_id} - {model_type}")
    metrics['cluster_id'] = cluster_id
    metrics['cluster_size'] = len(X_cluster)
    
    return model, metrics


def cluster_based_training(X, y, n_clusters=4, model_type='lightgbm'):
    """
    Perform clustering and train separate models for each cluster.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    n_clusters : int
        Number of clusters
    model_type : str
        Type of model to train
        
    Returns:
    --------
    tuple : (cluster_models, cluster_metrics, kmeans_model, cluster_labels)
    """
    print("\n" + "="*60)
    print("CLUSTER-BASED MODEL TRAINING")
    print("="*60)
    
    # Perform clustering
    kmeans, cluster_labels, clustering_metrics = perform_kmeans_clustering(X, n_clusters)
    
    # Visualize clusters
    pca, X_pca = visualize_clusters_pca(X, cluster_labels, save_path='outputs/pca_clusters.png')
    
    # Train models for each cluster
    cluster_models = {}
    cluster_metrics = []
    
    for cluster_id in range(n_clusters):
        # Get data for this cluster
        cluster_mask = cluster_labels == cluster_id
        X_cluster = X[cluster_mask]
        y_cluster = y[cluster_mask]
        
        # Train cluster-specific model
        model, metrics = train_cluster_specific_model(
            X_cluster, y_cluster, cluster_id, model_type
        )
        
        if model is not None:
            cluster_models[cluster_id] = model
            cluster_metrics.append(metrics)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(cluster_metrics)
    
    print("\n" + "="*60)
    print("CLUSTER-SPECIFIC MODEL RESULTS")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60)
    
    # Calculate weighted average performance
    if len(metrics_df) > 0:
        total_samples = metrics_df['cluster_size'].sum()
        weighted_metrics = {
            'accuracy': (metrics_df['accuracy'] * metrics_df['cluster_size']).sum() / total_samples,
            'precision': (metrics_df['precision'] * metrics_df['cluster_size']).sum() / total_samples,
            'recall': (metrics_df['recall'] * metrics_df['cluster_size']).sum() / total_samples,
            'f1_score': (metrics_df['f1_score'] * metrics_df['cluster_size']).sum() / total_samples,
            'roc_auc': (metrics_df['roc_auc'] * metrics_df['cluster_size']).sum() / total_samples
        }
        
        print("\nWeighted Average Performance Across Clusters:")
        for metric, value in weighted_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    return cluster_models, metrics_df, kmeans, cluster_labels


def predict_with_clusters(X, kmeans_model, cluster_models):
    """
    Make predictions using cluster-specific models.
    
    Parameters:
    -----------
    X : DataFrame
        Features to predict
    kmeans_model : KMeans
        Trained K-Means model
    cluster_models : dict
        Dictionary of cluster-specific models
        
    Returns:
    --------
    array : Predictions
    """
    # Assign clusters
    cluster_labels = kmeans_model.predict(X)
    
    # Initialize predictions array
    predictions = np.zeros(len(X))
    prediction_proba = np.zeros(len(X))
    
    # Make predictions for each cluster
    for cluster_id, model in cluster_models.items():
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() > 0:
            predictions[cluster_mask] = model.predict(X[cluster_mask])
            prediction_proba[cluster_mask] = model.predict_proba(X[cluster_mask])[:, 1]
    
    return predictions, prediction_proba


def compare_global_vs_cluster_models(X, y, global_model, cluster_models, kmeans_model):
    """
    Compare performance of global model vs cluster-based approach.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    global_model : model
        Globally trained model
    cluster_models : dict
        Cluster-specific models
    kmeans_model : KMeans
        K-Means clustering model
        
    Returns:
    --------
    DataFrame : Comparison results
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Global model predictions
    y_pred_global = global_model.predict(X_test)
    y_proba_global = global_model.predict_proba(X_test)[:, 1]
    
    # Cluster-based predictions
    y_pred_cluster, y_proba_cluster = predict_with_clusters(X_test, kmeans_model, cluster_models)
    
    # Calculate metrics
    global_metrics = calculate_metrics(y_test, y_pred_global, y_proba_global, "Global Model")
    cluster_metrics = calculate_metrics(y_test, y_pred_cluster, y_proba_cluster, "Cluster-Based Model")
    
    # Create comparison dataframe
    comparison = pd.DataFrame([global_metrics, cluster_metrics])
    
    print("\n" + "="*60)
    print("GLOBAL vs CLUSTER-BASED MODEL COMPARISON")
    print("="*60)
    print(comparison.to_string(index=False))
    print("="*60)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    X = pd.read_csv('data/processed_train.csv')
    y = pd.read_csv('data/processed_target.csv').values.ravel()
    
    # Perform cluster-based training
    cluster_models, cluster_metrics, kmeans, cluster_labels = cluster_based_training(
        X, y, n_clusters=4, model_type='lightgbm'
    )
    
    # Save cluster metrics
    cluster_metrics.to_csv('outputs/cluster_metrics.csv', index=False)
    print("\nCluster metrics saved to outputs/cluster_metrics.csv")