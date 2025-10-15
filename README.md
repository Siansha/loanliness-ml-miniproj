Home Credit Default Risk Prediction
This project aims to predict whether a client will default on their loan payments, using a variety of machine learning models. The solution includes a comprehensive data preprocessing pipeline, exploratory data analysis (EDA), and a comparative analysis of different classification models.

Table of Contents
Project Overview

Prerequisites

Project Structure

File Explanations

Data Preprocessing

Exploratory Data Analysis (EDA)

Model Training and Evaluation

Results and Key Insights

How to Run the Code

Project Overview
The goal of this project is to build a machine learning model to predict whether a Home Credit loan applicant will be able to repay their loan or not. This is a binary classification problem with a highly imbalanced dataset. The solution is structured into three main phases:

Data Preprocessing: Handling multiple raw datasets, cleaning, aggregating features, and preparing the data for modeling.

Exploratory Data Analysis (EDA): Visualizing distributions, correlations, and relationships within the data to gain insights.

Model Training: Training and evaluating several classification models, including Logistic Regression, Random Forest, Naive Bayes, and LightGBM.

Analysis: Comparing the performance of the trained models using various metrics and identifying key insights.

Prerequisites
This project requires Python 3.8+ and several libraries. You can install all the necessary dependencies using pip.

Bash

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn lightgbm
Project Structure
The project is organized into the following file structure:

.
├── src/
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── models.py                # Functions for training and evaluating models
│   └── utils.py                 # Utility functions for plotting and saving data
│
├── data/
│   ├── raw/
│   │   ├── application_train.csv
│   │   ├── bureau.csv
│   │   ├── ...                  # other raw data files
│   │
│   ├── processed_train.csv      # Processed training features
│   ├── processed_target.csv     # Processed training labels
│   ├── processed_test.csv       # Processed test features
│   └── ...                      # other processed files (.npy, .txt)
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb  # Jupyter notebook for data exploration and preprocessing
│   └── 02_Model_Training.ipynb     # Jupyter notebook for training and comparing models
│
├── outputs/
│   ├── model_comparison.csv     # CSV file with metrics for all models
│   ├── roc_curves_comparison.png # Plot comparing ROC curves of all models
│   ├── lr_confusion_matrix.png  # Confusion matrix for Logistic Regression
│   └── ...                      # Other plots and metrics
│
└── README.md
File Explanations
src/preprocessing.py: This script contains the entire data preprocessing pipeline. It handles loading multiple raw datasets, cleaning, feature aggregation from supplementary tables (like bureau, previous_application, installments_payments), missing value imputation, feature engineering (e.g., polynomial features), categorical encoding (label and one-hot encoding), and feature scaling. The output of this script is ready-to-use numerical data for model training.

src/models.py: This script is responsible for training, predicting, and evaluating the machine learning models. It includes functions for splitting data, addressing class imbalance using techniques like SMOTE, training various classifiers (Logistic Regression, Random Forest, Naive Bayes, LightGBM), and calculating performance metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

src/utils.py: This module contains helper functions that are utilized across the project. This includes functions for saving model outputs, plotting visualizations (e.g., confusion matrices, ROC curves, feature importances, distribution plots), and generating comparative tables or reports.

data/raw/: This directory stores all the original, untouched .csv files provided by Home Credit. These include application_train.csv, application_test.csv, bureau.csv, bureau_balance.csv, credit_card_balance.csv, installments_payments.csv, POS_CASH_balance.csv, and previous_application.csv.

data/processed_train.csv, data/processed_target.csv, data/processed_test.csv: These files store the cleaned, engineered, and scaled datasets after they have been processed by preprocessing.py. They are ready to be loaded directly by models.py for training and evaluation.

notebooks/01_EDA_Preprocessing.ipynb: A Jupyter notebook used for initial data exploration, understanding feature distributions, identifying correlations, and interactively developing the preprocessing steps. This notebook helps in visualizing the raw data and making informed decisions about data cleaning and feature engineering.

notebooks/02_Model_Training.ipynb: A Jupyter notebook where the machine learning models are trained, evaluated, and compared. It showcases the application of models.py functions, provides interactive insights into model performance, and generates the final output plots and comparison tables.

outputs/: This directory stores all the generated plots, comparison tables, and other analysis results from the model training and evaluation phase. This includes confusion matrices, feature importance plots, ROC curves, and a CSV file summarizing model performance metrics.

README.md: This file, which you are currently reading, provides a comprehensive overview of the project, including its goals, structure, how to run the code, and a summary of the results.

Data Preprocessing
The preprocessing.py script contains a complete pipeline for preparing the raw data. The key steps performed in this module are:

Data Loading: All supplementary .csv files (bureau.csv, previous_application.csv, credit_card_balance.csv, etc.) are loaded along with the main training and testing data.

Feature Aggregation: The supplementary datasets are aggregated to create new features for each client (SK_ID_CURR) before merging them with the main application data.

For example, bureau.csv and bureau_balance.csv are aggregated to create features like the mean, max, and min of DAYS_CREDIT, AMT_CREDIT_SUM, and MONTHS_BALANCE for each client.

The previous_application data is aggregated to create features such as the mean AMT_ANNUITY and the number of approved contracts (NAME_CONTRACT_STATUS=='Approved') for each client.

Missing Value Handling: Columns with a missing value rate above a certain threshold (e.g., 50%) are dropped. The remaining missing numerical values are imputed with the mean, while categorical missing values are filled with the mode or 'Unknown'.

Feature Engineering: Polynomial features of degree 2 are created for the top 10 numerical features to capture non-linear relationships and interactions.

Categorical Encoding: Binary categorical variables are label-encoded, while other categorical variables with a reasonable number of unique values (up to 10) are one-hot encoded. High-cardinality categorical features are also label-encoded.

Feature Scaling: All numerical features are normalized using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1, which is essential for models like Logistic Regression and Neural Networks.

The preprocessed data is then saved as NumPy arrays (.npy) and a list of feature names is saved to a text file for later use in model training.

Exploratory Data Analysis (EDA)
The EDA phase, typically conducted within 01_EDA_Preprocessing.ipynb, involves visualizing various aspects of the data to understand its characteristics, identify patterns, and detect potential issues.

Feature Distributions
Image 8: Distribution of NAME_CONTRACT_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, and AMT_CREDIT
This image displays histograms of several key features.

The distributions show the frequency of different values for NAME_CONTRACT_TYPE, FLAG_OWN_CAR, and FLAG_OWN_REALTY, indicating the prevalence of different contract types, car ownership, and property ownership among applicants.

CNT_CHILDREN shows the number of children, often with a peak at 0 or 1.

AMT_INCOME_TOTAL and AMT_CREDIT distributions help understand the spread of income and loan amounts, which are crucial for assessing repayment capacity. These often show a right-skewed distribution, typical for financial data.

Correlation Heatmap
Image 1: Correlation Heatmap (Sample Features)
This heatmap visualizes the Pearson correlation coefficients between a selection of important features and the TARGET variable.

Red cells indicate a strong positive correlation, while blue cells indicate a strong negative correlation. Lighter shades mean weaker correlations.

Correlations with TARGET are particularly important, as they show which features are most associated with loan default. For instance, DAYS_EMPLOYED (number of days employed) might have a significant negative correlation with TARGET, suggesting that longer employment reduces default risk. DAYS_BIRTH also tends to be correlated, with older applicants possibly having different default rates.

Income and Credit Distribution by Target
Image 2: Income Distribution by Target and Credit Amount Distribution by Target
These box plots compare the distribution of Income and Credit Amount between clients who repaid their loans (Target=0) and those who defaulted (Target=1).

The "Income Distribution by Target" plot shows whether there's a significant difference in income levels between repaying and defaulting clients. Often, defaulting clients might have lower median incomes or a wider spread of incomes.

The "Credit Amount Distribution by Target" plot similarly compares the loan amounts. This can reveal if clients with higher or lower credit amounts are more prone to defaulting. Outliers in these plots indicate extreme values in the data.

Model Training and Evaluation
The models.py script handles the training and evaluation of various classification models. The key aspects of this module are:

Data Splitting: The preprocessed training data is split into a training set and a validation set to evaluate model performance on unseen data.

Class Imbalance: The dataset is highly imbalanced, with the target variable TARGET=1 (default) making up only about 8% of the training data. This issue is addressed by using SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class and create a balanced dataset for training.

Model Selection: The following models are trained and evaluated:

Logistic Regression: A linear model with class_weight='balanced' to handle imbalance.

Random Forest: An ensemble tree-based model with hyperparameters tuned to prevent overfitting.

Naive Bayes: A probabilistic classifier based on Bayes' theorem.

LightGBM: A gradient boosting framework that uses tree-based learning algorithms. scale_pos_weight is used for imbalance handling, and early stopping is applied to prevent overfitting.


Performance Metrics: Models are evaluated using a variety of metrics, including:

Accuracy: The proportion of correctly classified instances.

Precision: The proportion of positive identifications that were actually correct.

Recall: The proportion of actual positives that were correctly identified.

F1-Score: The harmonic mean of precision and recall.

ROC-AUC: The Area Under the Receiver Operating Characteristic Curve, a key metric for imbalanced datasets.

Results and Key Insights
Based on the model comparison and individual model analyses, we can draw the following conclusions:

Model Performance Comparison
Image 6: Accuracy, Precision, Recall, and F1-Score Comparison Bar Charts
This image presents four bar charts comparing the key performance metrics across Logistic Regression, Random Forest, Naive Bayes, and LightGBM.

Accuracy Comparison: Shows LightGBM as the highest (0.920), followed by Random Forest (0.887). Naive Bayes has the lowest (0.211), suggesting it might be over-classifying one class.

Precision Comparison: LightGBM leads in precision (0.573), meaning when it predicts a default, it's more likely to be correct.

Recall Comparison: Naive Bayes stands out with a remarkably high recall (0.922), indicating it identifies most actual defaults, though potentially at the cost of high false positives. Logistic Regression also shows good recall (0.682).

F1-Score Comparison: Logistic Regression has the highest F1-Score (0.271), which is a balanced measure of precision and recall, crucial for imbalanced datasets.

Overall, the model comparison shows that LightGBM achieved the highest ROC-AUC (0.7754) and highest accuracy/precision, making it strong in identifying true positives with fewer false positives. Logistic Regression achieved the highest F1-Score (0.2708) and good recall, indicating a good balance for detecting defaults. Naive Bayes had an extremely high recall but very low precision, making it suitable if minimizing false negatives is the absolute top priority, even with many false alarms.

Confusion Matrices
Confusion matrices provide a detailed breakdown of correct and incorrect predictions for each class.

Image 3: LightGBM - Confusion Matrix

True Negatives (No Default, No Default): 56,387 correctly predicted non-defaulters.

False Positives (No Default, Default): 151 clients who did not default were incorrectly predicted as defaulters. This is a very low number, indicating high precision for the "Default" class.

False Negatives (Default, No Default): 4,762 clients who defaulted were incorrectly predicted as non-defaulters. This is a significant number, indicating LightGBM struggles with recall for the default class despite high overall accuracy.

True Positives (Default, Default): 203 correctly predicted defaulters.

LightGBM excels at correctly identifying non-defaulters (high true negatives) and has very few false alarms (low false positives), but it misses a large number of actual defaulters (high false negatives).

Image 5: Logistic Regression - Confusion Matrix

True Negatives (No Default, No Default): 39,876 correctly predicted non-defaulters.

False Positives (No Default, Default): 16,662 clients who did not default were incorrectly predicted as defaulters. This is a relatively high number of false alarms.

False Negatives (Default, No Default): 1,578 clients who defaulted were incorrectly predicted as non-defaulters. This number is significantly lower than LightGBM's, indicating better recall.

True Positives (Default, Default): 3,387 correctly predicted defaulters.

Logistic Regression, with its class_weight='balanced' setting, shows a better balance in identifying defaulters (higher true positives and lower false negatives) compared to LightGBM, but it also generates many more false positives. This aligns with its higher F1-score and recall values.

Image 7: Naive Bayes - Confusion Matrix

True Negatives (No Default, No Default): 8,411 correctly predicted non-defaulters. This is a very low number compared to other models.

False Positives (No Default, Default): 48,127 clients who did not default were incorrectly predicted as defaulters. This is an extremely high number of false alarms.

False Negatives (Default, No Default): 386 clients who defaulted were incorrectly predicted as non-defaulters. This is a very low number, highlighting its high recall.

True Positives (Default, Default): 4,579 correctly predicted defaulters.

Naive Bayes demonstrates an aggressive approach to identifying defaults, resulting in extremely high recall (very few missed defaults) but at the severe cost of precision, as it flags a vast number of non-defaulters as defaulters. This matches its high recall and low precision from the comparison metrics.

Feature Importance
Feature importance plots help understand which features contribute most to a model's predictions.

Image 4: Top 20 Feature Importance - LightGBM
This bar chart shows the top 20 most important features for the LightGBM model.

Features like EXT_SOURCE_3, EXT_SOURCE_2, CODE_GENDER_M, and various aggregated BUREAU and INST_PAYMENT features are prominent.

The EXT_SOURCE features (external data sources) are consistently high, suggesting they are very strong predictors of default risk.

Features derived from bureau (past credit history) and installments_payments (payment behavior) also play a crucial role, providing insights into a client's historical financial conduct.

Image 10: Top 20 Feature Importance - Random Forest
This bar chart shows the top 20 most important features for the Random Forest model.

Similar to LightGBM, EXT_SOURCE_3 and EXT_SOURCE_2 are among the top predictors, reinforcing their significance.

Other important features include BUREAU_AMT_CREDIT_MAX_OVERDUE_mean (mean of max overdue amount from other credits), NAME_EDUCATION_TYPE_Secondary / secondary special, NAME_EDUCATION_TYPE_Higher education, and INST_PAYMENT_PERC_min.

This indicates that credit history with other institutions, education level, and minimum payment percentages are strong indicators for Random Forest's predictions.
