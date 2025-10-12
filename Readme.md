# Loan Repayment Prediction using Machine Learning

## Project Overview

This project implements a comprehensive machine learning pipeline to predict loan repayment ability using the **Home Credit Default Risk** dataset from Kaggle. The system evaluates borrowers' likelihood to repay loans using alternative features beyond traditional credit scores, helping lenders assess creditworthiness for underbanked populations.

## Dataset

- **Source**: Kaggle - Home Credit Default Risk
- **Size**: 307,511 loan applications with 122 initial features
- **Target**: Binary classification (0: Default, 1: Repayment)
- **Class Distribution**: 91.9% negative, 8.1% positive (highly imbalanced)

### Data Files

- `application_train.csv` - Loan application information
- `bureau.csv` - Previous credit bureau data
- `bureau_balance.csv` - Monthly credit bureau balances
- `previous_application.csv` - Previous loan applications
- `POS_CASH_balance.csv` - POS/cash loan balances
- `installments_payments.csv` - Payment history
- `credit_card_balance.csv` - Credit card balances

## Methodology

### 1. Data Preprocessing
- **Feature Concatenation**: Merged 7 datasets using client ID (SK_ID_CURR)
- **Missing Value Handling**: Removed columns with >30% missing data, imputed rest with mean
- **Feature Encoding**: Label encoded categorical variables
- **Normalization**: Applied StandardScaler to all features
- **Result**: 651 features from 207 columns

### 2. Class Imbalance Handling
- Applied down-sampling to balance dataset (24,825 positive, 24,825 negative)
- Used stratified train-test split (80/20)

### 3. Machine Learning Models

Six classification algorithms were implemented and evaluated:

1. **Logistic Regression** - Baseline linear classifier
2. **Decision Tree** - Single tree-based classifier
3. **K-Nearest Neighbors (KNN)** - Distance-based classifier (k=5)
4. **Random Forest** - Ensemble of 100 decision trees
5. **Support Vector Machine (SVM)** - RBF kernel (trained on 5K sample for efficiency)
6. **Artificial Neural Network (ANN)** - Multi-layer perceptron with GPU acceleration

### 4. Clustering Analysis
- Applied K-Means clustering with k=4
- Trained cluster-specific models
- Compared single-model vs. cluster-based approach

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (Area Under Curve)
- Confusion Matrix
- Per-class metrics for clustering analysis

### 6. Feature Importance Analysis
- Extracted feature importance from Random Forest
- Identified top predictive features

## File Structure

```
loan-repayment-ml/
├── loanliness.ipynb                    # Main Jupyter notebook
├── model_performance_analysis.png      # Performance visualizations
├── README.md                           # This file
└── REPORT.md                           # Detailed technical report
```

## Installation & Setup

### Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow (GPU support optional)
```

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

### Configuration
Edit the `OUTPUT_PATH` variable in the notebook to set your desired output directory:
```python
OUTPUT_PATH = r"C:\your\path\here"
```

## Usage

1. **Prepare Dataset**
   - Download all 7 CSV files from Kaggle Home Credit Default Risk competition
   - Place them in a single folder

2. **Run Notebook**
   ```bash
   jupyter notebook loanliness.ipynb
   ```

3. **Expected Output**
   - Model performance comparison table
   - Cluster-specific performance metrics
   - ROC curves visualization
   - Feature importance rankings
   - Performance analysis PNG file

## Results Summary

### Model Performance
- **Best Accuracy**: Logistic Regression (69.97%)
- **Best F1-Score**: Logistic Regression (70.04%)
- **Best ROC-AUC**: ANN/Logistic Regression (varies by run)

### Clustering Insights
- K-Means with k=4 identified distinct borrower groups
- Cluster-specific models achieved ~71.57% average accuracy
- Demonstrated improvement over single-model approach

### Key Features
- **Most Important**: Days Employed (NUM_DAYS_EMPLOYED)
- **Second Most Important**: Age (DAYS_BIRTH)
- Employment stability and age are primary repayment predictors

## Key Findings

1. **Employment Status** is the strongest predictor of repayment ability
2. **Age** correlates with stable repayment patterns
3. **Down-sampling** was more effective than up-sampling for handling class imbalance
4. **Logistic Regression** provided best balance of accuracy and interpretability
5. **Cluster-based approach** showed marginal improvement over single models

## Advantages

- Helps underbanked populations access credit by using alternative features
- Reduces reliance on traditional credit scores
- Identifies key factors affecting repayment ability
- Provides multiple model options for different use cases

## Limitations

- Dataset has significant class imbalance (8% positive class)
- Features from different time periods may have temporal issues
- Some external source features (EXT_SOURCE_1/2/3) remain unexplained
- GPU training requires TensorFlow/CUDA setup

## References

- Home Credit Default Risk Dataset: https://www.kaggle.com/c/home-credit-default-risk
- Dodd-Frank Wall Street Reform and Consumer Protection Act (2010)

## Author

Yiyun Liang, Xiaomeng Jin, Zihan Wang (Stanford University)

Implementation: Enhanced ML Pipeline with GPU Acceleration

## License

MIT License - Feel free to use and modify for educational purposes.

## Contact & Support

For questions or issues, please refer to the technical report (REPORT.md) or the commented code in the Jupyter notebook.