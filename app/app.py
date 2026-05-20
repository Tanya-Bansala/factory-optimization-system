import pandas as pd
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Factory Optimization AI System", layout="wide")

st.title("Factory Reallocation & Shipping Optimization AI System")

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "Nassau_Candy_Distributor_updated.csv")

df = pd.read_csv(file_path)

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

st.sidebar.header("Filters")

product = st.sidebar.selectbox("Product", df["Product Name"].unique())
region = st.sidebar.selectbox("Region", df["Region"].unique())
ship_mode = st.sidebar.selectbox("Ship Mode", df["Ship Mode"].unique())

filtered_df = df[
    (df["Product Name"] == product) &
    (df["Region"] == region) &
    (df["Ship Mode"] == ship_mode)
]

st.subheader("Selected Data View")
st.dataframe(filtered_df.head())

st.subheader("Key Performance Indicators")

total_sales = df["Sales"].sum()
total_profit = df["Gross Profit"].sum()
avg_lead_time = df["Lead_Time"].mean()
total_orders = df["Order ID"].nunique()

profit_margin = (total_profit / total_sales) * 100
avg_order_value = total_sales / total_orders
efficiency_score = total_profit / (avg_lead_time + 1)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Sales", f"{total_sales:,.0f}")
col2.metric("Profit", f"{total_profit:,.0f}")
col3.metric("Lead Time", f"{avg_lead_time:.2f}")
col4.metric("Orders", total_orders)

col5, col6 = st.columns(2)
col5.metric("Profit Margin", f"{profit_margin:.2f}%")
col6.metric("Avg Order Value", f"{avg_order_value:.2f}")

st.metric("Efficiency Score", f"{efficiency_score:.2f}")

st.divider()

st.subheader("Lead Time Distribution")

fig1, ax1 = plt.subplots()
ax1.hist(df["Lead_Time"], bins=20)
ax1.set_title("Lead Time Distribution")
ax1.set_xlabel("Days")
ax1.set_ylabel("Frequency")

st.pyplot(fig1)

st.subheader("Lead Time Prediction Model")

features = ["Sales", "Units", "Cost", "Gross Profit"]
df_model = df.dropna(subset=features + ["Lead_Time"])

@st.cache_data
def train_model(data):
    X = data[features]
    y = data["Lead_Time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model

model = train_model(df_model)

st.success("Model trained successfully")

st.subheader("Predict Shipping Lead Time")

sales = st.number_input("Sales", float(df["Sales"].mean()))
units = st.number_input("Units", float(df["Units"].mean()))
cost = st.number_input("Cost", float(df["Cost"].mean()))
profit = st.number_input("Gross Profit", float(df["Gross Profit"].mean()))

input_data = np.array([[sales, units, cost, profit]])
predicted_lead_time = model.predict(input_data)[0]

st.metric("Predicted Lead Time", f"{predicted_lead_time:.2f}")

st.subheader("Factory Optimization Engine")

factories = df["Region"].unique()
results = []

for f in factories:
    temp = df[df["Region"] == f]

    if len(temp) > 0:
        avg_lt = temp["Lead_Time"].mean()
        profit_val = temp["Gross Profit"].sum()

        score = (profit_val * 0.7) - (avg_lt * 0.3)

        results.append([f, avg_lt, profit_val, score])

rec_df = pd.DataFrame(results, columns=[
    "Region", "Avg Lead Time", "Total Profit", "Optimization Score"
])

rec_df = rec_df.sort_values(by="Optimization Score", ascending=False)

st.dataframe(rec_df)

best_region = rec_df.iloc[0]["Region"]

st.success(f"Best Recommended Region: {best_region}")

st.subheader("Factory Performance Comparison")

fig2, ax2 = plt.subplots()
ax2.bar(rec_df["Region"], rec_df["Optimization Score"])
ax2.set_title("Optimization Score")
ax2.set_xlabel("Region")
ax2.set_ylabel("Score")

plt.xticks(rotation=45)

st.pyplot(fig2)

st.subheader("System Insights")

st.write("""
Gradient Boosting model predicts shipping lead time.
Factory recommendation uses profit and efficiency trade-off.
KPI dashboard shows operational performance.
Efficiency score highlights productivity.
System supports factory reallocation decisions.
""")