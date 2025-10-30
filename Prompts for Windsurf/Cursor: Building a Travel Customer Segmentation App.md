
# Prompts for Windsurf/Cursor: Building a Travel Customer Segmentation App

This document provides a series of prompts to guide you (as Windsurf/Cursor) in building a Streamlit web application for RFM and CLTV analysis of travel customer data. The application will use the "Online Retail II" dataset, contextualized for a travel booking scenario.

**Application Idea:**

A marketing analytics dashboard for a travel agency called "Wanderlust Travel". The dashboard will help marketing managers identify their most valuable customers, segment them based on their booking behavior, and calculate their lifetime value. This will enable targeted marketing campaigns to increase customer retention and profitability.

---

### Prompt 1: Project Setup and Data Loading

**You are an expert Python developer specializing in data analysis and Streamlit. Your task is to start building the Wanderlust Travel customer segmentation app.**

1.  **Create a new Python script named `app.py`.**
2.  **Import the following libraries:** `streamlit`, `pandas`, `numpy`, `datetime`, and `plotly.express`.
3.  **Set up the Streamlit page configuration:**
    *   Set the page title to "Wanderlust Travel - Customer Segmentation".
    *   Use a wide layout.
4.  **Load the dataset:**
    *   The dataset is located at: `https://github.com/aflorentin001/wanderlusttravel/raw/main/Wanderlust_Travel_Bookings.csv`
    *   Load the CSV file into a pandas DataFrame called `df`.
5.  **Display a title and a sample of the data in the Streamlit app:**
    *   Add a header: "Customer Booking Data".
    *   Show the first 10 rows of the DataFrame.

```python
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px

st.set_page_config(layout="wide", page_title="Wanderlust Travel - Customer Segmentation")

@st.cache_data
def load_data():
    df = pd.read_csv('https://github.com/aflorentin001/wanderlusttravel/raw/main/Wanderlust_Travel_Bookings.csv')
    return df

df = load_data()

st.title("Wanderlust Travel - Customer Segmentation Dashboard")

st.header("Raw Customer Booking Data")
st.dataframe(df.head(10))

```

---

### Prompt 2: Data Cleaning and Preparation

**Now, let's clean and prepare the data for analysis. Your task is to add a data cleaning section to the `app.py` script.**

1.  **Create a function `clean_data(df)` that performs the following steps:**
    *   Drop rows with missing `CustomerID`.
    *   Ensure `Quantity` is greater than 0.
    *   Ensure `UnitPrice` is greater than 0.
    *   Create a `TotalPrice` column by multiplying `Quantity` and `UnitPrice`.
    *   Convert `InvoiceDate` to datetime objects.
2.  **Apply this function to the DataFrame.**
3.  **In the Streamlit app, display a summary of the cleaned data:**
    *   Add a header: "Cleaned & Prepared Data".
    *   Show the number of unique customers.
    *   Display the date range of the transactions.
    *   Show the first 10 rows of the cleaned DataFrame.

```python
# (Add this after the initial data loading)

@st.cache_data
def clean_data(_df):
    df = _df.copy()
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["BookingAmount"] > 0]
    df = df[df["Status"] == "Confirmed"]  # Only confirmed bookings
    df["BookingDate"] = pd.to_datetime(df["BookingDate"])
    # Rename columns to match RFM analysis expectations
    df["InvoiceDate"] = df["BookingDate"]
    df["InvoiceNo"] = df["BookingReference"]
    df["TotalPrice"] = df["BookingAmount"]
    return df

df_cleaned = clean_data(df)

st.header("Cleaned & Prepared Data")
st.metric("Unique Customers", df_cleaned["CustomerID"].nunique())
st.metric("Date Range", f"{df_cleaned["InvoiceDate"].min().strftime('%Y-%m-%d')} to {df_cleaned['InvoiceDate'].max().strftime('%Y-%m-%d')}")
st.dataframe(df_cleaned.head(10))
```

---

### Prompt 3: RFM Analysis

**Next, let's perform RFM (Recency, Frequency, Monetary) analysis to segment the customers. Add the following functionality to your `app.py` script.**

1.  **Create a function `calculate_rfm(df)` that takes the cleaned DataFrame and returns an RFM DataFrame.**
    *   Calculate `Recency` as the number of days from the last invoice date to a snapshot date (the day after the last invoice date).
    *   Calculate `Frequency` as the number of unique invoices for each customer.
    *   Calculate `Monetary` as the sum of `TotalPrice` for each customer.
2.  **Create RFM scores:**
    *   Score `Recency` from 1 to 5, where 5 is the most recent.
    *   Score `Frequency` from 1 to 5, where 5 is the most frequent.
    *   Score `Monetary` from 1 to 5, where 5 is the highest spending.
3.  **Create customer segments based on the RFM scores**, using the logic from the article (Champions, Loyal Customers, etc.).
4.  **In the Streamlit app, display the RFM analysis results:**
    *   Add a header: "RFM Analysis & Customer Segments".
    *   Show the first 10 rows of the RFM DataFrame.
    *   Display a bar chart showing the distribution of customers across the different segments.

```python
# (Add this after the data cleaning section)

@st.cache_data
def calculate_rfm(_df):
    df = _df.copy()
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)

    # Create RFM scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Create customer segments
    def segment_customer(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 2 and row['F_Score'] >= 2:
            return 'Loyal Customers'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 1:
            return 'Potential Loyalists'
        elif row['R_Score'] >= 4:
            return 'Recent Customers'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 4:
            return 'At Risk'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'Hibernating'
        else:
            return 'Lost'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

rfm_df = calculate_rfm(df_cleaned)

st.header("RFM Analysis & Customer Segments")
st.dataframe(rfm_df.head(10))

st.subheader("Customer Segment Distribution")
segment_counts = rfm_df['Segment'].value_counts()
fig = px.bar(segment_counts, x=segment_counts.index, y=segment_counts.values, labels={'x': 'Segment', 'y': 'Number of Customers'})
st.plotly_chart(fig)

```

---

### Prompt 4: CLTV Analysis

**Now, let's calculate the Customer Lifetime Value (CLTV). This will help us understand the long-term value of each customer. Add this functionality to your `app.py` script.**

1.  **Create a function `calculate_cltv(df)` that takes the cleaned DataFrame and returns a CLTV DataFrame.**
    *   Calculate `CustomerLifespan` as the number of days between the first and last purchase for each customer.
    *   Calculate `AverageOrderValue` (AOV).
    *   Calculate `PurchaseFrequency`.
    *   Calculate `CLTV` using the formula: `CLTV = (AOV * PurchaseFrequency) / ChurnRate` where `ChurnRate` is the inverse of `CustomerLifespan`.
2.  **Merge the CLTV data with the RFM data.**
3.  **In the Streamlit app, display the CLTV analysis results:**
    *   Add a header: "CLTV Analysis".
    *   Show the top 10 customers by CLTV.
    *   Display a scatter plot of CLTV vs. Recency, colored by RFM segment.

```python
# (Add this after the RFM section)

@st.cache_data
def calculate_cltv(_df):
    df = _df.copy()
    cltv = df.groupby('CustomerID').agg(
        first_purchase_date=('InvoiceDate', 'min'),
        last_purchase_date=('InvoiceDate', 'max'),
        total_transaction=('InvoiceNo', 'nunique'),
        total_price=('TotalPrice', 'sum')
    )

    cltv['CustomerLifespan'] = (cltv['last_purchase_date'] - cltv['first_purchase_date']).dt.days
    cltv['AverageOrderValue'] = cltv['total_price'] / cltv['total_transaction']
    cltv['PurchaseFrequency'] = cltv['total_transaction'] / (cltv['CustomerLifespan'] / 365)
    
    # Handle cases where lifespan is 0
    cltv['PurchaseFrequency'] = cltv['PurchaseFrequency'].replace([np.inf, -np.inf], 0)
    cltv.fillna(0, inplace=True)

    # Simplified Churn Rate and CLTV
    cltv['ChurnRate'] = 1 / (cltv['CustomerLifespan'] + 1) # Add 1 to avoid division by zero
    cltv['CLTV'] = (cltv['AverageOrderValue'] * cltv['PurchaseFrequency']) / cltv['ChurnRate']
    cltv['CLTV'] = cltv['CLTV'].replace([np.inf, -np.inf], 0)
    cltv.fillna(0, inplace=True)

    return cltv

cltv_df = calculate_cltv(df_cleaned)
rfm_cltv_df = rfm_df.merge(cltv_df[['CLTV']], on='CustomerID')

st.header("CLTV Analysis")
st.subheader("Top 10 Customers by CLTV")
st.dataframe(rfm_cltv_df.sort_values(by='CLTV', ascending=False).head(10))

st.subheader("CLTV vs. Recency")
fig_cltv = px.scatter(rfm_cltv_df, x='Recency', y='CLTV', color='Segment', hover_data=['CustomerID'])
st.plotly_chart(fig_cltv)

```

---

### Prompt 5: Deployment to Streamlit Cloud

**Finally, let's prepare the application for deployment so you can share it with the world. You will need a GitHub account and a Streamlit Cloud account.**

1.  **Create a `requirements.txt` file.** This file lists all the Python libraries your app needs to run.
    ```
    streamlit
    pandas
    numpy
    plotly
    openpyxl
    ```

2.  **Create a new public GitHub repository.**

3.  **Upload the following files to your new GitHub repository:**
    *   `app.py`
    *   `requirements.txt`

4.  **Sign up for a free Streamlit Cloud account** at [https://streamlit.io/cloud](https://streamlit.io/cloud).

5.  **From your Streamlit Cloud dashboard, click "New app" and connect your GitHub account.**

6.  **Select the repository you just created, and make sure the main file is set to `app.py`.**

7.  **Click "Deploy!"**

Your application will now be deployed and accessible via a public URL.

---

