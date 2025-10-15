# 🏦 Home Credit Default Risk - Machine Learning Project

## 📘 Overview
This project predicts the likelihood of a loan applicant defaulting on credit based on their financial history, demographic details, and past repayment patterns.  
It uses multiple relational datasets provided by Home Credit to build a predictive machine learning pipeline that aids lenders in making data-driven credit decisions.

---

## 🧠 Problem Statement
Many loan applicants are denied credit due to insufficient credit history, even when they are capable of repayment.  
The objective is to build a model that predicts the probability of defaulting on a loan, helping financial institutions provide fairer access to credit.

---

## ⚙️ Approach
1. **Data Loading:**  
   Loaded multiple CSV datasets:  
   `application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, and `credit_card_balance.csv`.

2. **Feature Engineering:**  
   - Aggregated numerical data from auxiliary datasets (mean, min, max grouped by `SK_ID_CURR`).  
   - Merged all aggregated features into the main dataset (`application_train.csv`).

3. **Data Cleaning:**  
   - Dropped columns with more than 30% missing data.  
   - Imputed remaining numeric columns with mean values.

4. **Encoding & Normalization:**  
   - Applied `LabelEncoder` for categorical columns.  
   - Standardized numerical features using `StandardScaler`.

5. **Class Balancing:**  
   - Addressed target imbalance using **down-sampling** to equalize positive (defaults) and negative (non-defaults) samples.

6. **Model Training & Evaluation:**  
   Trained six models and compared performance using accuracy, precision, recall, F1-score, and ROC-AUC:
   - Logistic Regression  
   - Decision Tree  
   - K-Nearest Neighbors (KNN)  
   - Random Forest  
   - Support Vector Machine (SVM)  
   - Multi-Layer Perceptron (MLP Neural Network)

---

## 📊 Results Summary
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|-----------|----------|----------|
| **Logistic Regression** | **0.6997** | **0.7004** | **0.7629** |
| Random Forest | 0.6823 | 0.6817 | 0.7429 |
| MLP | 0.6930 | 0.6924 | 0.7572 |
| SVM | 0.6718 | 0.6710 | 0.7371 |
| Decision Tree | 0.5910 | 0.5926 | 0.5910 |
| KNN | 0.5907 | 0.5996 | 0.6211 |

**Best Model:** Logistic Regression  
**Most Important Features:**
- `EXT_SOURCE_2`, `EXT_SOURCE_3` (external credit scores)  
- `DAYS_BIRTH` (age)  
- `DAYS_EMPLOYED`  
- `DAYS_CREDIT`

---

## 📈 Visualizations
Generated a comparative visualization showing:
- Model accuracy comparison  
- Top 15 most important features (from Random Forest)

---

## 🚀 Setup Instructions

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/<your-username>/HomeCredit-ML.git
cd HomeCredit-ML
````

### 2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed, then install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3️⃣ Prepare the Dataset

Download the Home Credit Default Risk dataset from Kaggle and place the CSV files inside:
by modify the `base_path` in the script accordingly.

### 4️⃣ Run the Script

Run the main Python file:

```bash
python main_script.py
```

### 5️⃣ View Outputs

* Console will display model metrics and summaries.
* Plots will be saved to `base_path\model_performance_analysis.png`.

---

## 🧩 Challenges Faced

* Handling extremely large datasets (`bureau_balance` has ~27M rows).
* Managing memory and optimizing joins.
* Addressing missing values and class imbalance.
* Long training times for SVM and MLP models.

---

## 🧾 Conclusions

* Logistic Regression achieved the best trade-off between performance and interpretability.
* External credit scores (`EXT_SOURCE_*`) and demographic features were strong predictors.
* The pipeline demonstrates real-world data integration, preprocessing, and model comparison in a financial domain.

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
