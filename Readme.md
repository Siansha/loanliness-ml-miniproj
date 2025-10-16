# 🏦 Home Credit Default Risk - Machine Learning Project

## 📘 Overview

This project predicts the likelihood of a loan applicant defaulting on credit based on financial history, demographics, and past repayment patterns.
It uses multiple relational datasets provided by Home Credit to build a predictive ML pipeline for data-driven lending decisions.

---

## 🧠 Problem Statement

Many loan applicants are denied credit due to insufficient history, even when capable of repayment.
Objective: build a model to predict loan default probability, supporting fairer access to credit.

---

## ⚙️ Approach

1. **Data Loading:**
   Loaded CSVs: `application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`.

2. **Feature Engineering:**

   * Aggregated numerical data from auxiliary datasets (mean, min, max per `SK_ID_CURR`).
   * Merged aggregated features into main dataset.

3. **Data Cleaning:**

   * Dropped columns with >30% missing values.
   * Imputed remaining numeric columns with mean.

4. **Encoding & Normalization:**

   * `LabelEncoder` for categorical features.
   * `StandardScaler` for numeric features.

5. **Class Balancing:**

   * Down-sampled to equalize default and non-default cases.

6. **Model Training & Evaluation:**
   Trained six models: Logistic Regression, Decision Tree, KNN, Random Forest, SVM, MLP.
   Evaluated with accuracy, precision, recall, F1-score, ROC-AUC.

7. **Unsupervised Learning (K-Means + Random Forest):**

   * Applied K-Means to identify **4 distinct applicant clusters**.
   * Trained **Random Forest** separately on each cluster.
   * Improved overall predictive performance: **accuracy 73.25%**, F1-score 0.83.
   * Each cluster captures unique borrower characteristics:

     * **Cluster 1:** High income, low risk, regular repayments.
     * **Cluster 2:** Moderate income, medium risk.
     * **Cluster 3:** Young/first-time borrowers, limited credit history.
     * **Cluster 4:** High debt, low credit score, high default risk.

---

## 📊 Results Summary

| Model                       | Accuracy   | F1-Score   | ROC-AUC    |
| --------------------------- | ---------- | ---------- | ---------- |
| Logistic Regression         | 0.6997     | 0.7004     | 0.7629     |
| Random Forest               | 0.6823     | 0.6817     | 0.7429     |
| MLP                         | 0.6930     | 0.6924     | 0.7572     |
| SVM                         | 0.6718     | 0.6710     | 0.7371     |
| Decision Tree               | 0.5910     | 0.5926     | 0.5910     |
| KNN                         | 0.5907     | 0.5996     | 0.6211     |
| **Random Forest (K-Means)** | **0.7325** | **0.8300** | **0.7400** |

**Best Model (Global):** Logistic Regression
**Best Model (Clustered):** Random Forest (K-Means)

**Most Important Features:** `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_CREDIT`.

---

## 📈 Visualizations

* Model comparison (accuracy, F1-score, ROC-AUC).
* Top 15 most important features (Random Forest).
* K-Means + RF cluster performance included in combined visualization.

---

## 🚀 Setup Instructions

### 1️⃣ Clone or Download

```bash
git clone https://github.com/<your-username>/HomeCredit-ML.git
cd HomeCredit-ML
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3️⃣ Prepare Dataset

Download Home Credit Default Risk CSVs from Kaggle and set the `base_path` in the script.

### 4️⃣ Run Script

```bash
python main_script.py
```

### 5️⃣ View Outputs

* Console shows metrics and summaries.
* Plots saved to `base_path\model_performance_analysis.png`.

---

## 🧩 Challenges

* Large datasets (memory-intensive).
* Missing values and class imbalance.
* Long training times for SVM and MLP.

---

## 🧾 Conclusions

* Logistic Regression performed best globally.
* K-Means + Random Forest captured cluster-specific patterns and improved accuracy.
* Features like `EXT_SOURCE_*` and demographics were strong predictors.
* Cluster-specific modeling improves risk assessment and interpretability.

---

## 📂 Project Structure

```
ML/
├── application_train.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── POS_CASH_balance.csv
├── installments_payments.csv
├── credit_card_balance.csv
├── main_script.py
├── model_performance_analysis.png
└── README.md
```

---

## 👨‍💻 Author

**Shreyas R. Bhat**
Machine Learning Project | October 2025
Department of Computer Science

---
