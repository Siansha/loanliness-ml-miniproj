"""
Data preprocessing module for Home Credit Default Risk prediction.
Handles data loading, merging, cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path='data/raw/'):
    """
    Load all CSV files from the Home Credit dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing raw CSV files
        
    Returns:
    --------
    dict : Dictionary containing all loaded dataframes
    """
    print("Loading datasets...")
    
    datasets = {}
    
    # Load main application data
    datasets['application_train'] = pd.read_csv(f'{data_path}application_train.csv')
    datasets['application_test'] = pd.read_csv(f'{data_path}application_test.csv')
    
    # Load supplementary data
    datasets['bureau'] = pd.read_csv(f'{data_path}bureau.csv')
    datasets['bureau_balance'] = pd.read_csv(f'{data_path}bureau_balance.csv')
    datasets['credit_card_balance'] = pd.read_csv(f'{data_path}credit_card_balance.csv')
    datasets['installments_payments'] = pd.read_csv(f'{data_path}installments_payments.csv')
    datasets['pos_cash_balance'] = pd.read_csv(f'{data_path}POS_CASH_balance.csv')
    datasets['previous_application'] = pd.read_csv(f'{data_path}previous_application.csv')
    
    print(f"Loaded {len(datasets)} datasets")
    print(f"Training samples: {len(datasets['application_train'])}")
    print(f"Test samples: {len(datasets['application_test'])}")
    
    return datasets


def aggregate_bureau_data(bureau, bureau_balance):
    """
    Aggregate bureau and bureau_balance data by SK_ID_CURR.
    
    Parameters:
    -----------
    bureau : DataFrame
        Bureau credit information
    bureau_balance : DataFrame
        Monthly balance information for bureau credits
        
    Returns:
    --------
    DataFrame : Aggregated bureau features
    """
    print("Aggregating bureau data...")
    
    # Aggregate bureau_balance first
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'STATUS': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'C'
    })
    bb_agg.columns = ['_'.join(col).strip() for col in bb_agg.columns.values]
    bb_agg.reset_index(inplace=True)
    
    # Merge with bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Aggregate by SK_ID_CURR
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
        'CREDIT_TYPE': 'count'
    })
    
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.reset_index(inplace=True)
    
    return bureau_agg


def aggregate_previous_applications(prev_app):
    """
    Aggregate previous application data.
    
    Parameters:
    -----------
    prev_app : DataFrame
        Previous applications data
        
    Returns:
    --------
    DataFrame : Aggregated previous application features
    """
    print("Aggregating previous applications...")
    
    prev_agg = prev_app.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').sum()
    })
    
    prev_agg.columns = ['PREV_' + '_'.join(col).strip() for col in prev_agg.columns.values]
    prev_agg.reset_index(inplace=True)
    
    return prev_agg


def aggregate_installments(installments):
    """
    Aggregate installments payments data.
    
    Parameters:
    -----------
    installments : DataFrame
        Installments payments data
        
    Returns:
    --------
    DataFrame : Aggregated installments features
    """
    print("Aggregating installments...")
    
    # Calculate payment differences
    installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 1)
    installments['DAYS_LATE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    
    inst_agg = installments.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_DIFF': ['min', 'max', 'mean'],
        'PAYMENT_PERC': ['min', 'max', 'mean'],
        'DAYS_LATE': ['min', 'max', 'mean']
    })
    
    inst_agg.columns = ['INST_' + '_'.join(col).strip() for col in inst_agg.columns.values]
    inst_agg.reset_index(inplace=True)
    
    return inst_agg


def aggregate_credit_card(cc_balance):
    """
    Aggregate credit card balance data.
    
    Parameters:
    -----------
    cc_balance : DataFrame
        Credit card balance data
        
    Returns:
    --------
    DataFrame : Aggregated credit card features
    """
    print("Aggregating credit card balance...")
    
    cc_agg = cc_balance.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'AMT_BALANCE': ['min', 'max', 'mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean'],
        'AMT_RECIVABLE': ['min', 'max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean']
    })
    
    cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
    cc_agg.reset_index(inplace=True)
    
    return cc_agg


def aggregate_pos_cash(pos_cash):
    """
    Aggregate POS and cash balance data.
    
    Parameters:
    -----------
    pos_cash : DataFrame
        POS cash balance data
        
    Returns:
    --------
    DataFrame : Aggregated POS cash features
    """
    print("Aggregating POS cash balance...")
    
    pos_agg = pos_cash.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    })
    
    pos_agg.columns = ['POS_' + '_'.join(col).strip() for col in pos_agg.columns.values]
    pos_agg.reset_index(inplace=True)
    
    return pos_agg


def merge_all_data(datasets):
    """
    Merge all datasets into a single dataframe.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary containing all datasets
        
    Returns:
    --------
    tuple : (merged_train, merged_test) DataFrames
    """
    print("\nMerging all datasets...")
    
    # Aggregate supplementary data
    bureau_agg = aggregate_bureau_data(datasets['bureau'], datasets['bureau_balance'])
    prev_agg = aggregate_previous_applications(datasets['previous_application'])
    inst_agg = aggregate_installments(datasets['installments_payments'])
    cc_agg = aggregate_credit_card(datasets['credit_card_balance'])
    pos_agg = aggregate_pos_cash(datasets['pos_cash_balance'])
    
    # Merge with train data
    train = datasets['application_train'].copy()
    train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    train = train.merge(prev_agg, on='SK_ID_CURR', how='left')
    train = train.merge(inst_agg, on='SK_ID_CURR', how='left')
    train = train.merge(cc_agg, on='SK_ID_CURR', how='left')
    train = train.merge(pos_agg, on='SK_ID_CURR', how='left')
    
    # Merge with test data
    test = datasets['application_test'].copy()
    test = test.merge(bureau_agg, on='SK_ID_CURR', how='left')
    test = test.merge(prev_agg, on='SK_ID_CURR', how='left')
    test = test.merge(inst_agg, on='SK_ID_CURR', how='left')
    test = test.merge(cc_agg, on='SK_ID_CURR', how='left')
    test = test.merge(pos_agg, on='SK_ID_CURR', how='left')
    
    print(f"Merged train shape: {train.shape}")
    print(f"Merged test shape: {test.shape}")
    
    return train, test


def handle_missing_values(df, threshold=0.5):
    """
    Handle missing values using threshold-based dropping and mean imputation.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    threshold : float
        Drop columns with missing rate > threshold
        
    Returns:
    --------
    DataFrame : Dataframe with handled missing values
    """
    print("\nHandling missing values...")
    
    # Calculate missing percentages
    missing_pct = df.isnull().sum() / len(df)
    
    # Drop columns with high missing rate
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing values")
    df = df.drop(columns=cols_to_drop)
    
    # Impute remaining missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    print(f"Final shape after handling missing values: {df.shape}")
    
    return df


def encode_categorical_features(train, test):
    """
    Encode categorical features using Label Encoding and One-Hot Encoding.
    
    Parameters:
    -----------
    train : DataFrame
        Training dataframe
    test : DataFrame
        Test dataframe
        
    Returns:
    --------
    tuple : (encoded_train, encoded_test, encoders)
    """
    print("\nEncoding categorical features...")
    
    # Identify categorical columns
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    # Remove SK_ID_CURR if present
    if 'SK_ID_CURR' in categorical_cols:
        categorical_cols.remove('SK_ID_CURR')
    
    encoders = {}
    
    # Label encode binary categorical variables
    binary_cols = [col for col in categorical_cols if train[col].nunique() == 2]
    
    for col in binary_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le
    
    print(f"Label encoded {len(binary_cols)} binary columns")
    
    # One-hot encode remaining categorical variables
    remaining_cols = [col for col in categorical_cols if col not in binary_cols]
    
    # Only one-hot encode columns with reasonable number of categories
    onehot_cols = [col for col in remaining_cols if train[col].nunique() <= 10]
    
    if onehot_cols:
        train = pd.get_dummies(train, columns=onehot_cols, drop_first=True)
        test = pd.get_dummies(test, columns=onehot_cols, drop_first=True)
        
        # Align columns
        train, test = train.align(test, join='left', axis=1, fill_value=0)
        
        print(f"One-hot encoded {len(onehot_cols)} columns")
    
    # Label encode high-cardinality categoricals
    high_card_cols = [col for col in remaining_cols if col not in onehot_cols]
    
    for col in high_card_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le
    
    print(f"Label encoded {len(high_card_cols)} high-cardinality columns")
    print(f"Final shape after encoding - Train: {train.shape}, Test: {test.shape}")
    
    return train, test, encoders


def create_polynomial_features(train, test, degree=2, top_n=10):
    """
    Create polynomial features for top N most important features.
    
    Parameters:
    -----------
    train : DataFrame
        Training dataframe
    test : DataFrame
        Test dataframe
    degree : int
        Degree of polynomial features
    top_n : int
        Number of top features to use
        
    Returns:
    --------
    tuple : (train_poly, test_poly)
    """
    print(f"\nCreating polynomial features (degree={degree}) for top {top_n} features...")
    
    # Select top numerical features (excluding ID and target)
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if 'SK_ID_CURR' in numeric_cols:
        numeric_cols.remove('SK_ID_CURR')
    if 'TARGET' in numeric_cols:
        numeric_cols.remove('TARGET')
    
    # Use first top_n features (in practice, you'd select based on feature importance)
    selected_cols = numeric_cols[:min(top_n, len(numeric_cols))]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    
    train_poly_array = poly.fit_transform(train[selected_cols])
    test_poly_array = poly.transform(test[selected_cols])
    
    # Create column names
    poly_feature_names = [f'POLY_{i}' for i in range(train_poly_array.shape[1])]
    
    # Create dataframes
    train_poly_df = pd.DataFrame(train_poly_array, columns=poly_feature_names, index=train.index)
    test_poly_df = pd.DataFrame(test_poly_array, columns=poly_feature_names, index=test.index)
    
    # Concatenate with original data
    train_combined = pd.concat([train, train_poly_df], axis=1)
    test_combined = pd.concat([test, test_poly_df], axis=1)
    
    print(f"Added {len(poly_feature_names)} polynomial features")
    print(f"New shape - Train: {train_combined.shape}, Test: {test_combined.shape}")
    
    return train_combined, test_combined


def normalize_features(train, test):
    """
    Normalize features using StandardScaler.
    
    Parameters:
    -----------
    train : DataFrame
        Training dataframe
    test : DataFrame
        Test dataframe
        
    Returns:
    --------
    tuple : (normalized_train, normalized_test, scaler)
    """
    print("\nNormalizing features...")
    
    # Identify columns to normalize (exclude ID and TARGET)
    cols_to_normalize = train.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'SK_ID_CURR' in cols_to_normalize:
        cols_to_normalize.remove('SK_ID_CURR')
    if 'TARGET' in cols_to_normalize:
        cols_to_normalize.remove('TARGET')
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform
    train[cols_to_normalize] = scaler.fit_transform(train[cols_to_normalize])
    test[cols_to_normalize] = scaler.transform(test[cols_to_normalize])
    
    print(f"Normalized {len(cols_to_normalize)} features")
    
    return train, test, scaler


def preprocess_pipeline(data_path='data/raw/', add_poly_features=True):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to raw data directory
    add_poly_features : bool
        Whether to add polynomial features
        
    Returns:
    --------
    tuple : (X_train, y_train, X_test, test_ids, encoders, scaler)
    """
    print("="*60)
    print("Starting preprocessing pipeline...")
    print("="*60)
    
    # Load data
    datasets = load_data(data_path)
    
    # Merge all data
    train, test = merge_all_data(datasets)
    
    # Store IDs and target
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    target = train['TARGET']
    
    # Drop ID and TARGET
    train = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    test = test.drop(columns=['SK_ID_CURR'])
    
    # Handle missing values
    train = handle_missing_values(train, threshold=0.5)
    test = handle_missing_values(test, threshold=0.5)
    
    # Align columns
    train, test = train.align(test, join='inner', axis=1)
    
    # Encode categorical features
    train, test, encoders = encode_categorical_features(train, test)
    
    # Create polynomial features (optional)
    if add_poly_features:
        train, test = create_polynomial_features(train, test, degree=2, top_n=10)
    
    # Normalize features
    train, test, scaler = normalize_features(train, test)
    
    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)
    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")
    print("="*60)
    
    return train, target, test, test_ids, encoders, scaler


if __name__ == "__main__":
    # Example usage
    X_train, y_train, X_test, test_ids, encoders, scaler = preprocess_pipeline()
    
    # Save processed data
    print("\nSaving processed data...")
    X_train.to_csv('data/processed_train.csv', index=False)
    y_train.to_csv('data/processed_target.csv', index=False)
    X_test.to_csv('data/processed_test.csv', index=False)
    test_ids.to_csv('data/test_ids.csv', index=False)
    print("Saved processed data to data/ directory")