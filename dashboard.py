# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:25:55 2026

@author: notif
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# Load Dataset
data = pd.read_csv(
    r"C:\Users\notif\Downloads\archive (3).zip"
)

# Dashboard Title
st.title("Financial Fraud Detection Dashboard")

st.write("Machine Learning Based Fraud Detection System")

# =========================
# KPI SECTION
# =========================

total_transactions = len(data)

fraud_transactions = int(
    data['isFraud'].sum()
)

fraud_rate = round(
    (
        fraud_transactions /
        total_transactions
    ) * 100,
    3
)

col1,col2,col3 = st.columns(3)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Transactions",
    fraud_transactions
)

col3.metric(
    "Fraud Rate %",
    fraud_rate
)

# =========================
# FRAUD PIE CHART
# =========================

st.subheader(
    "Fraud Distribution"
)

fraud_counts = (
    data['isFraud']
    .value_counts()
)

fig1 = px.pie(
    values=fraud_counts.values,
    names=[
        "Not Fraud",
        "Fraud"
    ]
)

st.plotly_chart(fig1)

# =========================
# TRANSACTION TYPE CHART
# =========================

st.subheader(
    "Transaction Type Analysis"
)

fig2 = px.histogram(
    data,
    x='type'
)

st.plotly_chart(fig2)

# =========================
# FRAUD BY TYPE
# =========================

st.subheader(
    "Fraud by Transaction Type"
)

fraud_type = (
    data.groupby('type')
    ['isFraud']
    .sum()
    .reset_index()
)

fig3 = px.bar(
    fraud_type,
    x='type',
    y='isFraud'
)

st.plotly_chart(fig3)

# =========================
# AMOUNT DISTRIBUTION
# =========================

st.subheader(
    "Transaction Amount Distribution"
)

fig4 = px.histogram(
    data,
    x='amount',
    nbins=50
)

st.plotly_chart(fig4)

# =========================
# FRAUD TRANSACTIONS TABLE
# =========================

st.subheader(
    "Top Fraud Transactions"
)

fraud_df = data[
    data['isFraud']==1
]

st.dataframe(
    fraud_df.head(20)
)

st.success(
    "Dashboard Loaded Successfully"
)
