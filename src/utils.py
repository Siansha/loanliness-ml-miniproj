"""
Utility functions for visualization and metric saving.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os


def save_metrics_to_json(metrics_dict, filepath='outputs/metrics.json'):
    """
    Save metrics dictionary to JSON file.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing metrics
    filepath : str
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Metrics saved to {filepath}")


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    labels : list
        Class labels
    save_path : str
        Path to save the plot
    title : str
        Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels else ['No Default', 'Default'],
                yticklabels=labels if labels else ['No Default', 'Default'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None, title='ROC Curve'):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    save_path : str
        Path to save the plot
    title : str
        Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_multiple_roc_curves(models_dict, X_val, y_val, save_path=None):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models
    X_val : DataFrame
        Validation features
    y_val : array
        Validation target
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_pca_2d(X, y=None, save_path=None, title='PCA Visualization'):
    """
    Plot 2D PCA visualization.
    
    Parameters:
    -----------
    X : DataFrame or array
        Feature matrix
    y : array, optional
        Labels for coloring
    save_path : str
        Path to save the plot
    title : str
        Plot title
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    if y is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                            cmap='coolwarm', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA plot saved to {save_path}")
    
    plt.show()
    
    return pca, X_pca


def plot_tsne_2d(X, y=None, save_path=None, title='t-SNE Visualization', perplexity=30):
    """
    Plot 2D t-SNE visualization.
    
    Parameters:
    -----------
    X : DataFrame or array
        Feature matrix
    y : array, optional
        Labels for coloring
    save_path : str
        Path to save the plot
    title : str
        Plot title
    perplexity : int
        t-SNE perplexity parameter
    """
    print("Computing t-SNE (this may take a while)...")
    
    # Sample data if too large
    if len(X) > 5000:
        print(f"Sampling 5000 points from {len(X)} for t-SNE visualization")
        indices = np.random.choice(len(X), 5000, replace=False)
        X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        y_sample = y[indices] if y is not None else None
    else:
        X_sample = X
        y_sample = y
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=(10, 8))
    
    if y_sample is not None:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                            cmap='coolwarm', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Target')
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=20)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    plt.show()
    
    return tsne, X_tsne


def plot_metrics_comparison(metrics_df, save_path=None):
    """
    Plot comparison of metrics across models.
    
    Parameters:
    -----------
    metrics_df : DataFrame
        Dataframe containing metrics for different models
    save_path : str
        Path to save the plot
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Filter only available metrics
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(16, 4))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        axes[idx].bar(range(len(metrics_df)), metrics_df[metric], 
                     color='steelblue', alpha=0.7)
        axes[idx].set_xticks(range(len(metrics_df)))
        axes[idx].set_xticklabels(metrics_df['model'], rotation=45, ha='right')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(metrics_df[metric]):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(y, save_path=None, title='Target Class Distribution'):
    """
    Plot target class distribution.
    
    Parameters:
    -----------
    y : array or Series
        Target variable
    save_path : str
        Path to save the plot
    title : str
        Plot title
    """
    if hasattr(y, 'value_counts'):
        counts = y.value_counts()
    else:
        unique, counts_array = np.unique(y, return_counts=True)
        counts = pd.Series(counts_array, index=unique)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(counts.index, counts.values, color=['green', 'red'], alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(counts.index, ['No Default (0)', 'Default (1)'])
    
    # Add count and percentage labels
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/total*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def plot_feature_correlations(X, top_n=20, save_path=None):
    """
    Plot correlation heatmap for top features.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    top_n : int
        Number of top features to display
    save_path : str
        Path to save the plot
    """
    # Select top features by variance
    feature_vars = X.var().sort_values(ascending=False)
    top_features = feature_vars.head(top_n).index.tolist()
    
    # Calculate correlation matrix
    corr_matrix = X[top_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Heatmap (Top {top_n} Features by Variance)')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()


def create_submission_file(test_ids, predictions, filepath='outputs/submission.csv'):
    """
    Create submission file for Kaggle.
    
    Parameters:
    -----------
    test_ids : array or Series
        Test set IDs
    predictions : array
        Predicted probabilities
    filepath : str
        Path to save submission file
    """
    submission = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'TARGET': predictions
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    submission.to_csv(filepath, index=False)
    print(f"Submission file saved to {filepath}")
    print(f"Submission shape: {submission.shape}")
    print(f"Sample predictions:\n{submission.head()}")


def print_model_summary(metrics_df):
    """
    Print a formatted summary of model performance.
    
    Parameters:
    -----------
    metrics_df : DataFrame
        Dataframe containing model metrics
    """
    print("\n" + "="*80)
    print(" " * 25 + "MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    for idx, row in metrics_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  {'Accuracy:':<15} {row['accuracy']:.4f}")
        print(f"  {'Precision:':<15} {row['precision']:.4f}")
        print(f"  {'Recall:':<15} {row['recall']:.4f}")
        print(f"  {'F1-Score:':<15} {row['f1_score']:.4f}")
        print(f"  {'ROC-AUC:':<15} {row['roc_auc']:.4f}")
    
    print("\n" + "="*80)
    
    # Highlight best model for each metric
    print("\nBest Models by Metric:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        if metric in metrics_df.columns:
            best_idx = metrics_df[metric].idxmax()
            best_model = metrics_df.loc[best_idx, 'model']
            best_value = metrics_df.loc[best_idx, metric]
            print(f"  {metric.replace('_', ' ').title():<15}: {best_model} ({best_value:.4f})")
    
    print("="*80 + "\n")


def log_experiment(experiment_name, params, metrics, filepath='outputs/experiment_log.json'):
    """
    Log experiment parameters and results.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the experiment
    params : dict
        Parameters used in the experiment
    metrics : dict
        Results/metrics from the experiment
    filepath : str
        Path to log file
    """
    import datetime
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'experiment': experiment_name,
        'parameters': params,
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Load existing logs
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Append new log
    logs.append(log_entry)
    
    # Save updated logs
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=4)
    
    print(f"Experiment logged to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully!")
    print("\nAvailable functions:")
    print("  - save_metrics_to_json")
    print("  - plot_confusion_matrix")
    print("  - plot_roc_curve")
    print("  - plot_multiple_roc_curves")
    print("  - plot_pca_2d")
    print("  - plot_tsne_2d")
    print("  - plot_metrics_comparison")
    print("  - plot_class_distribution")
    print("  - plot_feature_correlations")
    print("  - create_submission_file")
    print("  - print_model_summary")
    print("  - log_experiment")