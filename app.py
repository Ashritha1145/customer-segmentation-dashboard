import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧑‍🤝‍🧑 Customer Segmentation Dashboard")

# Load data
df = pd.read_csv("customers.csv")

# Select features
X = df[["AnnualIncome", "SpendingScore"]]

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Income", round(df["AnnualIncome"].mean(), 2))
col3.metric("Average Spending", round(df["SpendingScore"].mean(), 2))

st.markdown("---")

# Scatter Plot
fig = px.scatter(
    df,
    x="AnnualIncome",
    y="SpendingScore",
    color=df["Cluster"].astype(str),
    size="Age",
    hover_data=["CustomerID"],
    title="Customer Segments"
)

st.plotly_chart(fig, use_container_width=True)

# Cluster Count
fig2 = px.histogram(
    df,
    x="Cluster",
    color=df["Cluster"].astype(str),
    title="Customers per Segment"
)

st.plotly_chart(fig2, use_container_width=True)

# Data Table
st.subheader("Customer Data")
st.dataframe(df)
