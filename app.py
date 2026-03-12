import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Retention Model", layout="wide")

st.title("Intelligent Retention & Pricing Model")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel("Online Retail.xlsx")

df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# Create Customer Features
# -----------------------------
customer = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'InvoiceDate': 'max'
}).reset_index()

customer.rename(columns={
    'TotalPrice': 'TotalSpending',
    'InvoiceNo': 'TotalOrders',
    'InvoiceDate': 'LastPurchase'
}, inplace=True)

reference_date = df['InvoiceDate'].max()
customer['Recency'] = (reference_date - customer['LastPurchase']).dt.days

# -----------------------------
# Predict Churn Probability
# -----------------------------
X = customer[['TotalSpending','TotalOrders','Recency']]
customer['Churn_Probability'] = model.predict_proba(X)[:,1]

st.subheader("Top High Risk Customers")

top_risk = customer.sort_values(
    by="Churn_Probability",
    ascending=False
).head(10)

st.dataframe(
    top_risk[[
        "CustomerID",
        "TotalSpending",
        "TotalOrders",
        "Recency",
        "Churn_Probability"
    ]]
)
st.subheader("Churn Probability Distribution")

st.bar_chart(customer["Churn_Probability"].head(50))
# -----------------------------
# User Input
# -----------------------------
st.write("Select a customer to see churn risk and pricing recommendation.")
cust_id = st.selectbox(
    "Select Customer ID",
    sorted(customer['CustomerID'].unique())
)
row = customer[customer['CustomerID'] == cust_id].iloc[0]
if cust_id != 0:

    if cust_id in customer['CustomerID'].values:

        row = customer[customer['CustomerID'] == cust_id].iloc[0]

        st.subheader("Customer Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Spending", round(row['TotalSpending'],2))
        col2.metric("Total Orders", int(row['TotalOrders']))
        col3.metric("Recency (Days)", int(row['Recency']))

        churn_prob = row['Churn_Probability']
        st.metric("Churn Probability", round(churn_prob,3))

        # Risk Logic
        if churn_prob > 0.8:
            discount = 25
            st.error("High Risk 🔴 - Urgent retention campaign")
        elif churn_prob > 0.6:
            discount = 15
            st.warning("Medium Risk 🟡 - Special personalized offer")
        else:
            discount = 5
            st.success("Low Risk 🟢 - Loyalty offer")

        dynamic_price = row['TotalSpending'] * (1 - discount/100)

        st.write("Recommended Discount:", discount, "%")
        st.write("Dynamic Price:", round(dynamic_price,2))

    else:
        st.warning("Customer ID not found.")