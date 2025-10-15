"""
Model training and evaluation module for Home Credit Default Risk prediction.
Implements multiple classifiers with class imbalance handling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and validation sets.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    test_size : float
        Proportion of validation set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple : X_train, X_val, y_train, y_val
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to handle class imbalance.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple : X_resampled, y_resampled
    """
    print("Applying SMOTE to handle class imbalance...")
    print(f"Original class distribution:\n{pd.Series(y_train).value_counts()}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Parameters:
    -----------
    model : trained classifier
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels

    Returns:
    --------
    dict : evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'classification_report': classification_report(y_test, y_pred)
    }
    return metrics



def train_logistic_regression(X_train, y_train, X_val, y_val, use_smote=True):
    """
    Train Logistic Regression classifier.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
    use_smote : bool
        Whether to apply SMOTE
        
    Returns:
    --------
    tuple : (model, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train model with class weights
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Logistic Regression")
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, use_smote=True):
    """
    Train Random Forest classifier.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
    use_smote : bool
        Whether to apply SMOTE
        
    Returns:
    --------
    tuple : (model, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Random Forest")
    
    return model, metrics


def train_naive_bayes(X_train, y_train, X_val, y_val, use_smote=True):
    """
    Train Naive Bayes classifier.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
    use_smote : bool
        Whether to apply SMOTE
        
    Returns:
    --------
    tuple : (model, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training Naive Bayes...")
    print("="*60)
    
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train model
    model = GaussianNB()
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Naive Bayes")
    
    return model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val, use_smote=True):
    """
    Train LightGBM classifier.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
    use_smote : bool
        Whether to apply SMOTE (usually False for LightGBM)
        
    Returns:
    --------
    tuple : (model, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training LightGBM...")
    print("="*60)
    
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum()
    
    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
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
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "LightGBM")
    
    return model, metrics


def train_mlp(X_train, y_train, X_val, y_val, use_smote=True):
    """
    Train Multi-Layer Perceptron (Neural Network) classifier.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
    use_smote : bool
        Whether to apply SMOTE
        
    Returns:
    --------
    tuple : (model, metrics_dict)
    """
    print("\n" + "="*60)
    print("Training Multi-Layer Perceptron...")
    print("="*60)
    
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train model
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "MLP Neural Network")
    
    return model, metrics


def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    y_pred_proba : array
        Predicted probabilities
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'y_pred': y_pred  # ✅ ADD THIS LINE
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def plot_feature_importance(model, feature_names, model_name, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : trained model
        Model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    save_path : str
        Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not have feature importance attribute")
        return
    
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def train_all_models(X_train, y_train, X_val, y_val):
    """
    Train all models and return results.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    X_val : DataFrame
        Validation features
    y_val : Series
        Validation target
        
    Returns:
    --------
    tuple : (models_dict, metrics_df)
    """
    models = {}
    all_metrics = []
    
    # Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['Logistic Regression'] = lr_model
    all_metrics.append(lr_metrics)
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    models['Random Forest'] = rf_model
    all_metrics.append(rf_metrics)
    
    # Naive Bayes
    nb_model, nb_metrics = train_naive_bayes(X_train, y_train, X_val, y_val)
    models['Naive Bayes'] = nb_model
    all_metrics.append(nb_metrics)
    
    # LightGBM
    lgbm_model, lgbm_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    models['LightGBM'] = lgbm_model
    all_metrics.append(lgbm_metrics)
    
    # MLP Neural Network
    mlp_model, mlp_metrics = train_mlp(X_train, y_train, X_val, y_val)
    models['MLP Neural Network'] = mlp_model
    all_metrics.append(mlp_metrics)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODELS")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60)
    
    # Find best model
    best_model_name = metrics_df.loc[metrics_df['roc_auc'].idxmax(), 'model']
    best_auc = metrics_df['roc_auc'].max()
    print(f"\nBest Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    
    return models, metrics_df


def save_metrics(metrics_df, filepath='outputs/model_metrics.csv'):
    """
    Save metrics to CSV file.
    
    Parameters:
    -----------
    metrics_df : DataFrame
        Metrics dataframe
    filepath : str
        Path to save the CSV
    """
    metrics_df.to_csv(filepath, index=False)
    print(f"\nMetrics saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    X = pd.read_csv('data/processed_train.csv')
    y = pd.read_csv('data/processed_target.csv').values.ravel()
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    
    # Train all models
    models, metrics_df = train_all_models(X_train, y_train, X_val, y_val)
    
    # Save metrics
    save_metrics(metrics_df)
    
    # Plot feature importance for tree-based models
    feature_names = X.columns.tolist()
    
    if 'Random Forest' in models:
        plot_feature_importance(
            models['Random Forest'], 
            feature_names, 
            'Random Forest',
            top_n=20,
            save_path='outputs/rf_feature_importance.png'
        )
    
    if 'LightGBM' in models:
        plot_feature_importance(
            models['LightGBM'], 
            feature_names, 
            'LightGBM',
            top_n=20,
            save_path='outputs/lgbm_feature_importance.png'
        )