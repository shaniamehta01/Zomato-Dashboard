import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("Zomato Growth Intelligence Dashboard (Advanced)")

data = pd.read_csv("data.csv")

st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select Problem",
    ["Churn Analysis", "Segmentation & Conversion", "Demand Forecast"]
)

# ---- CHURN ----
if option == "Churn Analysis":
    st.header("🔴 Churn Prediction + CLV")

    data["churn"] = (data["last_order_days"] > 10).astype(int)

    # CLV (simple regression proxy)
    X = data[["order_count"]]
    y = data["avg_order_value"]
    model = LinearRegression().fit(X, y)
    data["CLV"] = model.predict(X) * data["order_count"]

    col1, col2 = st.columns(2)
    col1.subheader("Churn Distribution")
    col1.bar_chart(data["churn"].value_counts())

    col2.subheader("Top High CLV Users")
    col2.write(data.sort_values("CLV", ascending=False).head())

# ---- SEGMENTATION ----
elif option == "Segmentation & Conversion":
    st.header("🔵 Customer Segmentation")

    kmeans = KMeans(n_clusters=2, random_state=0)
    data["cluster"] = kmeans.fit_predict(data[["order_count","avg_order_value"]])

    st.subheader("Cluster Distribution")
    st.bar_chart(data["cluster"].value_counts())

    st.subheader("Cluster Details")
    st.write(data[["user_id","cluster","order_count","avg_order_value"]])

# ---- DEMAND ----
elif option == "Demand Forecast":
    st.header("🟢 Demand Forecast")

    demand = data.groupby("city")["total_orders"].sum()
    st.bar_chart(demand)

    data["delay_risk"] = np.where(data["order_count"] < 3, "High", "Low")
    st.subheader("Delay Risk")
    st.write(data[["user_id","city","delay_risk"]])
