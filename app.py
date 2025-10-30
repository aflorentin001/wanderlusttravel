import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Wanderlust Travel - Customer Segmentation",
    page_icon="✈️"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">✈️ Wanderlust Travel - Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Identify your most valuable customers, predict churn, and optimize marketing spend**")

# Data loading function
@st.cache_data
def load_data():
    """Load the Wanderlust Travel Bookings dataset"""
    df = pd.read_csv('https://github.com/aflorentin001/wanderlusttravel/raw/main/Wanderlust_Travel_Bookings.csv')
    return df

# Data cleaning function
@st.cache_data
def clean_data(_df):
    """Clean and prepare the data for analysis"""
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

# RFM calculation function
@st.cache_data
def calculate_rfm(_df):
    """Calculate RFM metrics and segment customers"""
    df = _df.copy()
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
    rfm.reset_index(inplace=True)

    # Create RFM scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

    # Convert to int for calculations
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Create customer segments
    def segment_customer(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 2 and row['F_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 3 and row['F_Score'] <= 3:
            return 'Potential Loyalists'
        elif row['R_Score'] >= 4:
            return 'Recent Customers'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 2:
            return 'At Risk'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 3:
            return 'Cant Lose Them'
        elif row['R_Score'] <= 2:
            return 'Lost'
        else:
            return 'Others'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

# CLTV calculation function
@st.cache_data
def calculate_cltv(_df):
    """Calculate Customer Lifetime Value"""
    df = _df.copy()
    cltv = df.groupby('CustomerID').agg(
        first_purchase_date=('InvoiceDate', 'min'),
        last_purchase_date=('InvoiceDate', 'max'),
        total_transaction=('InvoiceNo', 'nunique'),
        total_price=('TotalPrice', 'sum')
    )

    cltv['CustomerLifespan'] = (cltv['last_purchase_date'] - cltv['first_purchase_date']).dt.days
    cltv['AverageOrderValue'] = cltv['total_price'] / cltv['total_transaction']
    cltv['PurchaseFrequency'] = cltv['total_transaction'] / ((cltv['CustomerLifespan'] + 1) / 365)
    
    # Handle edge cases
    cltv['PurchaseFrequency'] = cltv['PurchaseFrequency'].replace([np.inf, -np.inf], 0)
    cltv.fillna(0, inplace=True)

    # Simplified CLTV calculation
    cltv['ChurnRate'] = 1 / (cltv['CustomerLifespan'] + 1)
    cltv['CLTV'] = (cltv['AverageOrderValue'] * cltv['PurchaseFrequency']) / (cltv['ChurnRate'] + 0.01)
    cltv['CLTV'] = cltv['CLTV'].replace([np.inf, -np.inf], 0)
    cltv.fillna(0, inplace=True)

    return cltv

# KMeans clustering function
@st.cache_data
def perform_kmeans(_rfm_df, n_clusters=4):
    """Perform KMeans clustering on RFM data"""
    rfm_df = _rfm_df.copy()
    X = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    X['Monetary'] = np.log1p(X['Monetary'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return rfm_df, kmeans.inertia_

# Main app
def main():
    # Load and clean data
    with st.spinner("Loading data..."):
        df = load_data()
        df_cleaned = clean_data(df)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Display data overview
    st.sidebar.subheader("Data Overview")
    st.sidebar.metric("Total Transactions", f"{len(df_cleaned):,}")
    st.sidebar.metric("Unique Customers", f"{df_cleaned['CustomerID'].nunique():,}")
    st.sidebar.metric("Date Range", f"{df_cleaned['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df_cleaned['InvoiceDate'].max().strftime('%Y-%m-%d')}")
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate to:", ["Overview", "RFM Analysis", "CLTV Analysis", "Customer Segments", "KMeans Clustering"])
    
    # Calculate RFM and CLTV
    rfm_df = calculate_rfm(df_cleaned)
    cltv_df = calculate_cltv(df_cleaned)
    rfm_cltv_df = rfm_df.merge(cltv_df[['CLTV', 'AverageOrderValue', 'PurchaseFrequency']], on='CustomerID', how='left')
    
    # Page routing
    if page == "Overview":
        show_overview(df_cleaned, rfm_cltv_df)
    elif page == "RFM Analysis":
        show_rfm_analysis(rfm_cltv_df)
    elif page == "CLTV Analysis":
        show_cltv_analysis(rfm_cltv_df)
    elif page == "Customer Segments":
        show_segments(rfm_cltv_df)
    elif page == "KMeans Clustering":
        show_kmeans(rfm_cltv_df)

def show_overview(df_cleaned, rfm_cltv_df):
    """Display overview page"""
    st.header("📈 Business Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"£{df_cleaned['TotalPrice'].sum():,.2f}")
    with col2:
        st.metric("Average Order Value", f"£{df_cleaned.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}")
    with col3:
        st.metric("Total Customers", f"{df_cleaned['CustomerID'].nunique():,}")
    with col4:
        st.metric("Avg CLTV", f"£{rfm_cltv_df['CLTV'].mean():,.2f}")
    
    st.markdown("---")
    
    # Revenue over time
    st.subheader("📊 Revenue Trend Over Time")
    revenue_by_month = df_cleaned.groupby(df_cleaned['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum().reset_index()
    revenue_by_month['InvoiceDate'] = revenue_by_month['InvoiceDate'].astype(str)
    fig = px.line(revenue_by_month, x='InvoiceDate', y='TotalPrice', 
                  labels={'InvoiceDate': 'Month', 'TotalPrice': 'Revenue (£)'},
                  title='Monthly Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top customers
    st.subheader("🏆 Top 10 Customers by Total Spend")
    top_customers = df_cleaned.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
    top_customers.columns = ['Customer ID', 'Total Spend (£)']
    top_customers['Total Spend (£)'] = top_customers['Total Spend (£)'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(top_customers, use_container_width=True)

def show_rfm_analysis(rfm_cltv_df):
    """Display RFM analysis page"""
    st.header("🎯 RFM Analysis")
    st.markdown("**Recency, Frequency, Monetary analysis to understand customer behavior**")
    
    # RFM distribution
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Recency (days)", f"{rfm_cltv_df['Recency'].mean():.0f}")
    with col2:
        st.metric("Avg Frequency", f"{rfm_cltv_df['Frequency'].mean():.1f}")
    with col3:
        st.metric("Avg Monetary", f"£{rfm_cltv_df['Monetary'].mean():,.2f}")
    
    st.markdown("---")
    
    # RFM histograms
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(rfm_cltv_df, x='Recency', nbins=50, title='Recency Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(rfm_cltv_df, x='Frequency', nbins=50, title='Frequency Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(rfm_cltv_df, x=np.log1p(rfm_cltv_df['Monetary']), nbins=50, title='Monetary Distribution (Log Scale)')
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM scatter plot
    st.subheader("📊 RFM Scatter Plot")
    fig = px.scatter(rfm_cltv_df, x='Recency', y='Frequency', size='Monetary', color='Segment',
                     hover_data=['CustomerID', 'CLTV'], title='Customer Distribution by RFM Metrics')
    st.plotly_chart(fig, use_container_width=True)
    
    # RFM data table
    st.subheader("📋 RFM Data Table")
    display_df = rfm_cltv_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'Segment']].copy()
    display_df['Monetary'] = display_df['Monetary'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(display_df.head(20), use_container_width=True)

def show_cltv_analysis(rfm_cltv_df):
    """Display CLTV analysis page"""
    st.header("💰 Customer Lifetime Value Analysis")
    st.markdown("**Understanding the long-term value of your customers**")
    
    # CLTV metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CLTV", f"£{rfm_cltv_df['CLTV'].sum():,.2f}")
    with col2:
        st.metric("Average CLTV", f"£{rfm_cltv_df['CLTV'].mean():,.2f}")
    with col3:
        st.metric("Median CLTV", f"£{rfm_cltv_df['CLTV'].median():,.2f}")
    with col4:
        st.metric("Max CLTV", f"£{rfm_cltv_df['CLTV'].max():,.2f}")
    
    st.markdown("---")
    
    # CLTV distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CLTV Distribution")
        cltv_clean = rfm_cltv_df[rfm_cltv_df['CLTV'] <= rfm_cltv_df['CLTV'].quantile(0.99)]
        fig = px.histogram(cltv_clean, x='CLTV', nbins=50, title='CLTV Distribution (99th Percentile)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Pareto Chart")
        sorted_cltv = rfm_cltv_df.sort_values('CLTV', ascending=False).reset_index(drop=True)
        sorted_cltv['CumulativePercent'] = (sorted_cltv['CLTV'].cumsum() / sorted_cltv['CLTV'].sum()) * 100
        sorted_cltv['CustomerPercent'] = ((sorted_cltv.index + 1) / len(sorted_cltv)) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_cltv['CustomerPercent'], y=sorted_cltv['CumulativePercent'],
                                 mode='lines', name='Cumulative CLTV'))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Perfect Equality',
                                 line=dict(dash='dash')))
        fig.add_vline(x=20, line_dash="dot", line_color="red", annotation_text="Top 20%")
        fig.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="80% of Value")
        fig.update_layout(title='Cumulative CLTV Distribution', xaxis_title='% of Customers', yaxis_title='% of Total CLTV')
        st.plotly_chart(fig, use_container_width=True)
    
    # CLTV vs Recency
    st.subheader("🔍 CLTV vs Recency by Segment")
    fig = px.scatter(rfm_cltv_df, x='Recency', y='CLTV', color='Segment', hover_data=['CustomerID'],
                     title='Customer Lifetime Value vs Recency')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top customers by CLTV
    st.subheader("🏆 Top 20 Customers by CLTV")
    top_cltv = rfm_cltv_df.sort_values(by='CLTV', ascending=False).head(20)[['CustomerID', 'CLTV', 'Segment', 'Recency', 'Frequency', 'Monetary']].copy()
    top_cltv['CLTV'] = top_cltv['CLTV'].apply(lambda x: f"£{x:,.2f}")
    top_cltv['Monetary'] = top_cltv['Monetary'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(top_cltv, use_container_width=True)

def show_segments(rfm_cltv_df):
    """Display customer segments page"""
    st.header("👥 Customer Segments")
    st.markdown("**Understanding different customer groups and their characteristics**")
    
    # Segment distribution
    st.subheader("📊 Segment Distribution")
    segment_counts = rfm_cltv_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(segment_counts, use_container_width=True)
    
    with col2:
        fig = px.bar(segment_counts, x='Segment', y='Count', title='Number of Customers by Segment',
                     color='Segment')
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment characteristics
    st.subheader("📈 Segment Characteristics")
    segment_stats = rfm_cltv_df.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CLTV': 'mean',
        'CustomerID': 'count'
    }).round(2).reset_index()
    segment_stats.columns = ['Segment', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Avg CLTV', 'Customer Count']
    segment_stats['Avg Monetary'] = segment_stats['Avg Monetary'].apply(lambda x: f"£{x:,.2f}")
    segment_stats['Avg CLTV'] = segment_stats['Avg CLTV'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(segment_stats, use_container_width=True)
    
    # Revenue by segment
    st.subheader("💰 Revenue Contribution by Segment")
    segment_revenue = rfm_cltv_df.groupby('Segment')['Monetary'].sum().reset_index()
    segment_revenue.columns = ['Segment', 'Total Revenue']
    fig = px.pie(segment_revenue, values='Total Revenue', names='Segment', title='Revenue Distribution by Segment')
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment filter
    st.subheader("🔍 Explore Specific Segment")
    selected_segment = st.selectbox("Select a segment to explore:", rfm_cltv_df['Segment'].unique())
    segment_data = rfm_cltv_df[rfm_cltv_df['Segment'] == selected_segment][['CustomerID', 'Recency', 'Frequency', 'Monetary', 'CLTV']].copy()
    segment_data['Monetary'] = segment_data['Monetary'].apply(lambda x: f"£{x:,.2f}")
    segment_data['CLTV'] = segment_data['CLTV'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(segment_data, use_container_width=True)

def show_kmeans(rfm_cltv_df):
    """Display KMeans clustering page"""
    st.header("🔬 KMeans Clustering Analysis")
    st.markdown("**Discovering natural customer groups using machine learning**")
    
    # Number of clusters selector
    n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=8, value=4)
    
    # Perform clustering
    clustered_df, inertia = perform_kmeans(rfm_cltv_df, n_clusters)
    
    # Cluster statistics
    st.subheader("📊 Cluster Characteristics")
    cluster_stats = clustered_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CLTV': 'mean',
        'CustomerID': 'count'
    }).round(2).reset_index()
    cluster_stats.columns = ['Cluster', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Avg CLTV', 'Customer Count']
    cluster_stats['Avg Monetary'] = cluster_stats['Avg Monetary'].apply(lambda x: f"£{x:,.2f}")
    cluster_stats['Avg CLTV'] = cluster_stats['Avg CLTV'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(cluster_stats, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("📈 3D Cluster Visualization")
    fig = px.scatter_3d(clustered_df, x='Recency', y='Frequency', z=np.log1p(clustered_df['Monetary']),
                        color='Cluster', hover_data=['CustomerID', 'CLTV'],
                        title='Customer Clusters in 3D Space',
                        labels={'z': 'Log(Monetary)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Customer Distribution")
        cluster_counts = clustered_df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig = px.bar(cluster_counts, x='Cluster', y='Count', title='Customers per Cluster')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue by Cluster")
        cluster_revenue = clustered_df.groupby('Cluster')['Monetary'].sum().reset_index()
        cluster_revenue.columns = ['Cluster', 'Total Revenue']
        fig = px.bar(cluster_revenue, x='Cluster', y='Total Revenue', title='Revenue per Cluster')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
