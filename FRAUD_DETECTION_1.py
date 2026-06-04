# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:30:30 2026

@author: notif
"""

# ==========================================================
# FINANCIAL FRAUD DETECTION PROJECT
# DATASET: PaySim Financial Fraud Dataset
# ==========================================================

#  IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import sqlite3
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
#  LOAD DATASET
# ==========================================================

print("Loading Dataset...")

data = pd.read_csv(
    r"C:\Users\notif\Downloads\archive (3).zip"
)

print("Dataset Loaded Successfully")
print("Shape:", data.shape)

# ==========================================================
# BASIC DATA INSPECTION
# ==========================================================

print("\nFirst 5 Rows")
print(data.head())

print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

print("\nRemoving Duplicate Records")

data.drop_duplicates(inplace=True)

print("New Shape:", data.shape)

# ==========================================================
#  FRAUD DISTRIBUTION
# ==========================================================

fraud_counts = data['isFraud'].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    fraud_counts,
    labels=['Not Fraud','Fraud'],
    autopct='%1.2f%%'
)

plt.title("Fraud Distribution Percentage")

plt.show()

# ==========================================================
#  FEATURE ENGINEERING
# ==========================================================

print("\nCreating New Features")

# Sender balance difference

data['org_balance_diff'] = (
    data['oldbalanceOrg']
    - data['newbalanceOrig']
)

# Receiver balance difference

data['dest_balance_diff'] = (
    data['newbalanceDest']
    - data['oldbalanceDest']
)

# Amount ratio

data['amount_ratio'] = (
    data['amount']
    /
    (data['oldbalanceOrg'] + 1)
)

print("Feature Engineering Completed")

# ==========================================================
#  ENCODE TRANSACTION TYPE
# ==========================================================

print("\nEncoding Transaction Type")

encoder = LabelEncoder()

data['type'] = encoder.fit_transform(
    data['type']
)

print("Encoding Completed")

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    data.corr(numeric_only=True),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================================================
# CREATE SAMPLE DATASET
# ==========================================================
# IMPORTANT:
# Full dataset has millions of rows.
# Use sample for faster training.
# ==========================================================

print("\nCreating Sample Dataset")

sample_data = data.sample(
    n=100000,
    random_state=42
)

print("Sample Shape:", sample_data.shape)

# ==========================================================
# SELECT FEATURES
# ==========================================================

X = sample_data[
[
'type',
'amount',
'oldbalanceOrg',
'newbalanceOrig',
'oldbalanceDest',
'newbalanceDest',
'org_balance_diff',
'dest_balance_diff',
'amount_ratio'
]
]

y = sample_data['isFraud']

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================================
#  LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Logistic Regression")

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# ==========================================================
# DECISION TREE
# ==========================================================

print("\nTraining Decision Tree")

dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\nTraining Random Forest")

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# ==========================================================
# MODEL COMPARISON
# ==========================================================

print("\nMODEL COMPARISON")

models = {
    "Logistic Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred
}

for name, pred in models.items():

    print("\n=================================")
    print(name)
    print("=================================")

    print(
        "Accuracy:",
        accuracy_score(y_test, pred)
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            pred,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            pred,
            zero_division=0
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_test,
            pred,
            zero_division=0
        )
    )

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nRandom Forest Classification Report")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==========================================================
#  FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({

'Feature': X.columns,

'Importance': rf.feature_importances_

})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")

print(importance)

# ==========================================================
#  SAVE MODEL
# ==========================================================

joblib.dump(
    rf,
    "fraud_model.pkl"
)

print("\nModel Saved Successfully")

# ==========================================================
#  SAVE DATA TO SQLITE DATABASE
# ==========================================================

conn = sqlite3.connect(
    "fraud_detection.db"
)

sample_data.to_sql(
    "transactions",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("\nDatabase Created Successfully")

# ==========================================================
# TEST FRAUD PREDICTION
# ==========================================================

print("\nTesting Model")

model = joblib.load(
    "fraud_model.pkl"
)

sample_transaction = pd.DataFrame({

'type':[4],

'amount':[500000],

'oldbalanceOrg':[600000],

'newbalanceOrig':[100000],

'oldbalanceDest':[0],

'newbalanceDest':[500000],

'org_balance_diff':[500000],

'dest_balance_diff':[500000],

'amount_ratio':[0.83]

})

prediction = model.predict(
    sample_transaction
)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Fraud Transaction Detected")
else:
    print("Legitimate Transaction")


########### Model F1 Score Comparision
comparison = pd.DataFrame({
    'Model':['Logistic Regression','Decision Tree','Random Forest'],
    'Accuracy':[0.9991,0.99935,0.99995],
    'Precision':[0.8571,0.8000,1.0000],
    'Recall':[0.4286,0.7143,0.9643],
    'F1 Score':[0.5714,0.7547,0.9818]
})

print(comparison)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.bar(
    comparison['Model'],
    comparison['F1 Score']
)

plt.title("Model Comparison using F1 Score")

plt.xlabel("Machine Learning Models")

plt.ylabel("F1 Score")

plt.ylim(0,1)

for i,v in enumerate(comparison['F1 Score']):
    plt.text(i, v+0.02, str(round(v,3)))

plt.tight_layout()

plt.savefig(
    "model_comparison_chart.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

comparison.to_csv("model_comparison.csv",index=False)

########### Feature Importance Bar Chart
plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

########## ROC Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

rf_probs = rf.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(
    y_test,
    rf_probs
)

roc_auc = auc(fpr,tpr)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png")

plt.show()

############# Streamlit Dashboard Screenshots

# ==========================================================
# PROJECT COMPLETED
# ==========================================================

print("\nFINANCIAL FRAUD DETECTION PROJECT COMPLETED")