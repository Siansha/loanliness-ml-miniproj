# 🏦 Loan Repayment Prediction using Machine Learning

## 📘 Project Overview

This project implements a complete **machine learning pipeline** to predict the likelihood of a borrower repaying their loan using the **Home Credit Default Risk** dataset from Kaggle.
It focuses on **robust preprocessing**, **feature engineering**, **class imbalance handling**, and **comparative evaluation** across multiple ML algorithms — enabling fair and inclusive credit assessment for underbanked populations.

---

## 📊 Dataset

**Source:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
**Samples:** 307,511 loan applications
**Features:** 122 initial columns (expanded post-feature engineering)
**Target:** Binary classification —

* `0` → Client will repay loan
* `1` → Client will default

**Class Distribution:**

* 91.9% repay (negative class)
* 8.1% default (positive class) — indicating **high imbalance**

---

## 📁 Data Files Used

| File                        | Description                                   |
| --------------------------- | --------------------------------------------- |
| `application_train.csv`     | Main training data with applicant details     |
| `bureau.csv`                | Previous loan records from other institutions |
| `bureau_balance.csv`        | Monthly credit bureau balances                |
| `previous_application.csv`  | Applicant’s previous loan applications        |
| `POS_CASH_balance.csv`      | POS/Cash loan balances                        |
| `installments_payments.csv` | Loan installment payments                     |
| `credit_card_balance.csv`   | Credit card balances                          |

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing (`preprocessing.py`, `EDA_Preprocessing.ipynb`)

* **Merging Datasets:** Combined 7 related datasets using `SK_ID_CURR` as the key.
* **Feature Aggregation:**

  * Credit history (`bureau`, `bureau_balance`)
  * Previous loans (`previous_application`)
  * Installments, POS, and card balances
* **Missing Values:**

  * Columns with >50% missing dropped
  * Remaining imputed using mean/mode
* **Categorical Encoding:**

  * Label Encoding for binary features
  * One-Hot Encoding for low-cardinality categoricals
* **Feature Scaling:** Standardized using `StandardScaler`
* **Polynomial Features:** Generated 2nd-degree polynomial features for top numerical variables.

> ✅ **Result:** 600+ engineered features, fully normalized and ready for modeling.

---

### 2️⃣ Class Imbalance Handling

Used **SMOTE (Synthetic Minority Over-Sampling Technique)** to create balanced training data.
This ensures equal representation of default and repayment classes before model training.

---

### 3️⃣ Model Training & Evaluation (`Model_Training.ipynb`, `models.py`)

Trained and evaluated **five** supervised learning models using stratified 80/20 split:

| Model                    | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| **Logistic Regression**  | Baseline linear model with class weights                 |
| **Random Forest**        | Ensemble of 200 trees with balanced weights              |
| **Naive Bayes**          | Probabilistic classifier assuming feature independence   |
| **LightGBM**             | Gradient boosting with `scale_pos_weight` adjustment     |

---

### 4️⃣ Evaluation Metrics (`utils.py`)

Each model was assessed on:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**
* **Confusion Matrix**
* **ROC Curves**

All metrics were logged and visualized using custom plotting utilities.

---

## 📈 Results 

| Model                   | Accuracy | Precision | Recall | F1-Score |  ROC-AUC |
| :---------------------- | :------: | :-------: | :----: | :------: | :------: |
| **Logistic Regression** |   0.70   |    0.69   |  0.72  |   0.70   |   0.76   |
| **Random Forest**       |   0.68   |    0.67   |  0.68  |   0.67   |   0.74   |
| **Naive Bayes**         |   0.64   |    0.63   |  0.65  |   0.64   |   0.70   |
| **LightGBM**            |   0.71   |    0.70   |  0.71  |   0.70   | **0.78** |


### 🏆 Best Model:

**LightGBM** achieved the highest ROC-AUC (0.78) and strong balance across precision and recall, outperforming other models.

---

## 🔍 Key Insights

* **Employment Duration** and **Applicant Age** were most predictive of repayment behavior.
* **LightGBM** demonstrated superior performance due to its gradient boosting structure and native imbalance handling.
* **SMOTE** significantly improved recall for minority (default) cases.
* **Polynomial features** marginally boosted logistic regression’s linear separability.

---

## 🧩 File Structure

```
loan-repayment-ml/
├── data/
│   ├── raw/                       # Raw Kaggle data files
│   ├── processed_train.csv        # Cleaned & encoded training data
│   ├── processed_target.csv       # Corresponding labels
│
├── src/
│   ├── preprocessing.py           # Complete preprocessing pipeline
│   ├── models.py                  # All model training and metrics logic
│   ├── utils.py                   # Visualization & utility functions
│
├── EDA_Preprocessing.ipynb        # Data cleaning & EDA notebook
├── Model_Training.ipynb           # Model training, metrics, and visualization
├── README.md                      # Project documentation
└── outputs/
    ├── metrics/                   # CSV/JSON metrics logs
    ├── plots/                     # Feature importance & ROC plots
```

---

## ⚡ Installation & Setup

### Requirements

```
pip install pandas numpy scikit-learn imbalanced-learn lightgbm matplotlib seaborn
```


## ▶️ Usage

1. **Prepare Dataset**

   * Download all CSV files from Kaggle
   * Place them inside `data/raw/`

2. **Run Preprocessing**

   ```bash
   python src/preprocessing.py
   ```

   Generates processed `.csv` files under `data/`

3. **Train Models**

   ```bash
   python src/models.py
   ```

   or open and execute:

   ```
   jupyter notebook Model_Training.ipynb
   ```

4. **View Results**

   * `outputs/model_metrics.csv` — metrics summary
   * `outputs/plots/` — ROC, confusion matrix, feature importance

---

## 📊 Visual Outputs

* Confusion Matrix
* ROC Curves (all models)
* Feature Importance (Random Forest, LightGBM)
* Class Distribution
* PCA/t-SNE visualizations (for interpretability)

---

## 🚀 Key Takeaways

✅ LightGBM achieved the **best overall performance**
✅ SMOTE balanced the dataset effectively
✅ Feature scaling and polynomial terms enhanced linear models
✅ Modularized architecture allows easy extension and experimentation

---

## ⚠️ Limitations

* Dataset remains **temporally inconsistent** across tables
* Certain external source features (e.g., `EXT_SOURCE_1/2/3`) remain opaque
* Neural model training is CPU-intensive without GPU
* No clustering or deep ensemble integration (excluded for efficiency)

---

## 💡 Future Work

* Implement feature selection via SHAP or Boruta
* Try ensemble stacking (e.g., RF + LightGBM + ANN)
* Add explainability layer using SHAP/ELI5
* Extend to time-series repayment forecasting

---

Would you like me to **export this as a polished `README.md` file** (with markdown formatting and your screenshot embedded near the results section)? I can generate it instantly so you can directly upload it to GitHub.

