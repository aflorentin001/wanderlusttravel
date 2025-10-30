# Google Colab Notebook Template for RFM & CLTV Analysis

This is a template you can copy into Google Colab to perform RFM and CLTV analysis.

## Instructions:
1. Go to https://colab.research.google.com
2. Create a new notebook
3. Copy each code cell below into separate cells in your Colab notebook
4. Run the cells in order

---

## Cell 1: Install and Import Libraries

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")
```

---

## Cell 2: Load and Inspect Data

```python
# Load the Wanderlust Travel Bookings dataset
df = pd.read_csv('https://github.com/aflorentin001/wanderlusttravel/raw/main/Wanderlust_Travel_Bookings.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
```

---

## Cell 3: Clean the Data

```python
# Data cleaning
df = df.dropna(subset=['CustomerID'])
df = df[df['BookingAmount'] > 0]
df = df[df['Status'] == 'Confirmed']  # Only confirmed bookings
df['BookingDate'] = pd.to_datetime(df['BookingDate'])
# Rename columns to match RFM analysis expectations
df['InvoiceDate'] = df['BookingDate']
df['InvoiceNo'] = df['BookingReference']
df['TotalPrice'] = df['BookingAmount']

print(f"Clean Dataset: {df.shape[0]} transactions")
print(f"Unique Customers: {df['CustomerID'].nunique()}")
print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
```

---

## Cell 4: Calculate RFM Metrics

```python
# Calculate RFM
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("RFM Summary Statistics:")
print(rfm.describe())
print("\nFirst 10 customers:")
print(rfm.head(10))
```

---

## Cell 5: Create RFM Scores and Segments

```python
# Create RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['RFM_Segment'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)

# Define customer segments
def segment_customer(row):
    if row['RFM_Segment'] >= 9 and row['R_Score'] >= 4:
        return 'Champions'
    elif row['RFM_Segment'] >= 6 and row['R_Score'] >= 3:
        return 'Loyal Customers'
    elif row['F_Score'] >= 3 and row['R_Score'] >= 3:
        return 'Potential Loyalists'
    elif row['R_Score'] >= 4:
        return 'Recent Customers'
    elif row['RFM_Segment'] >= 6 and row['R_Score'] <= 2:
        return 'At Risk'
    elif row['F_Score'] >= 2 and row['R_Score'] <= 2:
        return 'Cant Lose Them'
    elif row['R_Score'] <= 2:
        return 'Lost'
    else:
        return 'Others'

rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)

print("\nCustomer Segments:")
print(rfm['Customer_Segment'].value_counts())
```

---

## Cell 6: Visualize RFM Distributions

```python
# Create RFM visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RFM Analysis: Distribution of Key Metrics', fontsize=16, fontweight='bold')

# Recency
axes[0, 0].hist(rfm['Recency'], bins=50, color='#FF7043', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Recency Distribution (Days Since Last Purchase)', fontweight='bold')
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].axvline(rfm['Recency'].median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {rfm["Recency"].median():.0f} days')
axes[0, 0].legend()

# Frequency
axes[0, 1].hist(rfm['Frequency'], bins=50, color='#00BFA5', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Frequency Distribution (Number of Purchases)', fontweight='bold')
axes[0, 1].set_xlabel('Number of Transactions')
axes[0, 1].set_ylabel('Number of Customers')
axes[0, 1].axvline(rfm['Frequency'].median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {rfm["Frequency"].median():.0f} purchases')
axes[0, 1].legend()

# Monetary
axes[1, 0].hist(np.log10(rfm['Monetary']), bins=50, color='#FFC107', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Monetary Distribution (Log Scale)', fontweight='bold')
axes[1, 0].set_xlabel('Log10(Total Spend)')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].axvline(np.log10(rfm['Monetary'].median()), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: £{rfm["Monetary"].median():.2f}')
axes[1, 0].legend()

# Segments
segment_counts = rfm['Customer_Segment'].value_counts()
colors_palette = ['#FF7043', '#00BFA5', '#FFC107', '#42A5F5', '#AB47BC', '#66BB6A']
axes[1, 1].barh(segment_counts.index, segment_counts.values, color=colors_palette[:len(segment_counts)])
axes[1, 1].set_title('Customer Segments Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Number of Customers')
axes[1, 1].set_ylabel('Segment')
for i, v in enumerate(segment_counts.values):
    axes[1, 1].text(v, i, f' {v}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Cell 7: Calculate CLTV

```python
# Calculate Customer Lifetime Value
cltv_data = df.groupby('CustomerID').agg(
    first_purchase_date=('InvoiceDate', 'min'),
    last_purchase_date=('InvoiceDate', 'max'),
    total_transaction=('InvoiceNo', 'nunique'),
    total_price=('TotalPrice', 'sum')
)

cltv_data['CustomerLifespan'] = (cltv_data['last_purchase_date'] - cltv_data['first_purchase_date']).dt.days
cltv_data['AvgOrderValue'] = cltv_data['total_price'] / cltv_data['total_transaction']
cltv_data['PurchaseFrequency'] = cltv_data['total_transaction'] / ((cltv_data['CustomerLifespan'] + 1) / 365)

# Handle edge cases
cltv_data['PurchaseFrequency'] = cltv_data['PurchaseFrequency'].replace([np.inf, -np.inf], 0)
cltv_data.fillna(0, inplace=True)

# Calculate CLTV
cltv_data['ChurnRate'] = 1 / (cltv_data['CustomerLifespan'] + 1)
cltv_data['CLTV'] = (cltv_data['AvgOrderValue'] * cltv_data['PurchaseFrequency']) / (cltv_data['ChurnRate'] + 0.01)
cltv_data['CLTV'] = cltv_data['CLTV'].replace([np.inf, -np.inf], 0)
cltv_data.fillna(0, inplace=True)

print("CLTV Summary:")
print(f"Total CLTV: £{cltv_data['CLTV'].sum():,.2f}")
print(f"Average CLTV: £{cltv_data['CLTV'].mean():,.2f}")
print(f"Median CLTV: £{cltv_data['CLTV'].median():,.2f}")
print(f"Max CLTV: £{cltv_data['CLTV'].max():,.2f}")

print("\nTop 10 Customers by CLTV:")
print(cltv_data.sort_values('CLTV', ascending=False).head(10))
```

---

## Cell 8: Visualize CLTV

```python
# CLTV visualizations
cltv_data_clean = cltv_data[cltv_data['CLTV'] <= cltv_data['CLTV'].quantile(0.99)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Lifetime Value (CLTV) Analysis', fontsize=16, fontweight='bold')

# CLTV Distribution
axes[0, 0].hist(cltv_data_clean['CLTV'], bins=50, color='#00BFA5', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('CLTV Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Customer Lifetime Value (£)')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].axvline(cltv_data_clean['CLTV'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: £{cltv_data_clean["CLTV"].mean():.2f}')
axes[0, 0].axvline(cltv_data_clean['CLTV'].median(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: £{cltv_data_clean["CLTV"].median():.2f}')
axes[0, 0].legend()

# AOV vs Purchase Frequency
scatter = axes[0, 1].scatter(cltv_data_clean['AvgOrderValue'], cltv_data_clean['PurchaseFrequency'], 
                             c=cltv_data_clean['CLTV'], cmap='YlOrRd', alpha=0.6, s=50)
axes[0, 1].set_title('Average Order Value vs Purchase Frequency', fontweight='bold')
axes[0, 1].set_xlabel('Average Order Value (£)')
axes[0, 1].set_ylabel('Purchase Frequency (purchases/year)')
plt.colorbar(scatter, ax=axes[0, 1], label='CLTV (£)')

# Value Segments
cltv_quartiles = pd.qcut(cltv_data_clean['CLTV'], q=4, labels=['Low Value', 'Medium Value', 'High Value', 'Top Value'])
segment_counts = cltv_quartiles.value_counts().sort_index()
colors = ['#FFC107', '#FF7043', '#42A5F5', '#00BFA5']
axes[1, 0].bar(segment_counts.index, segment_counts.values, color=colors)
axes[1, 0].set_title('Customer Value Segments', fontweight='bold')
axes[1, 0].set_xlabel('CLTV Segment')
axes[1, 0].set_ylabel('Number of Customers')
for i, v in enumerate(segment_counts.values):
    axes[1, 0].text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')

# Pareto Chart
sorted_cltv = cltv_data_clean.sort_values('CLTV', ascending=False)
sorted_cltv['CumulativePercent'] = (sorted_cltv['CLTV'].cumsum() / sorted_cltv['CLTV'].sum()) * 100
sorted_cltv['CustomerPercent'] = (np.arange(1, len(sorted_cltv) + 1) / len(sorted_cltv)) * 100
axes[1, 1].plot(sorted_cltv['CustomerPercent'], sorted_cltv['CumulativePercent'], color='#00BFA5', linewidth=2)
axes[1, 1].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Equality')
axes[1, 1].set_title('Cumulative CLTV Distribution (Pareto)', fontweight='bold')
axes[1, 1].set_xlabel('Cumulative % of Customers')
axes[1, 1].set_ylabel('Cumulative % of Total CLTV')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(20, color='red', linestyle=':', alpha=0.5)
axes[1, 1].axhline(80, color='red', linestyle=':', alpha=0.5)
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## Cell 9: KMeans Clustering - Elbow Method

```python
# Prepare data for clustering
X = rfm[['Recency', 'Frequency', 'Monetary']]
X_transformed = X.copy()
X_transformed['Monetary'] = np.log1p(X_transformed['Monetary'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

# Elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.title('Elbow Method: Finding Optimal Number of Clusters', fontweight='bold', fontsize=14)
plt.xlabel('Number of Clusters (K)', fontweight='bold')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold')
plt.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Suggested K=4')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

---

## Cell 10: Apply KMeans Clustering

```python
# Apply KMeans with optimal K
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Cluster analysis
cluster_summary = rfm.groupby('KMeans_Cluster').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median'],
    'CustomerID': 'count'
}).round(2)

print("Cluster Analysis:")
print(cluster_summary)
```

---

## Cell 11: Visualize Clusters

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 6))

# 3D scatter plot
ax1 = fig.add_subplot(131, projection='3d')
colors = ['#FF7043', '#00BFA5', '#FFC107', '#42A5F5']
for i in range(optimal_k):
    cluster_data = rfm[rfm['KMeans_Cluster'] == i]
    ax1.scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                np.log1p(cluster_data['Monetary']), c=colors[i], label=f'Cluster {i}', alpha=0.6, s=30)
ax1.set_xlabel('Recency (days)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_zlabel('Log(Monetary)', fontweight='bold')
ax1.set_title('KMeans Clustering (3D View)', fontweight='bold', fontsize=12)
ax1.legend()

# Customer distribution
ax2 = fig.add_subplot(132)
cluster_counts = rfm['KMeans_Cluster'].value_counts().sort_index()
ax2.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
ax2.set_xticks(range(len(cluster_counts)))
ax2.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
ax2.set_title('Customer Distribution by Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Number of Customers', fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    ax2.text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')

# Revenue by cluster
ax3 = fig.add_subplot(133)
cluster_revenue = rfm.groupby('KMeans_Cluster')['Monetary'].sum().sort_index()
ax3.bar(range(len(cluster_revenue)), cluster_revenue.values, color=colors)
ax3.set_xticks(range(len(cluster_revenue)))
ax3.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
ax3.set_title('Total Revenue by Cluster', fontweight='bold', fontsize=12)
ax3.set_ylabel('Total Revenue (£)', fontweight='bold')
for i, v in enumerate(cluster_revenue.values):
    ax3.text(i, v, f'£{v/1000:.0f}K', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Cell 12: Export Results

```python
# Merge RFM and CLTV data
final_results = rfm.merge(cltv_data[['CLTV']], on='CustomerID', how='left')

# Save to CSV
final_results.to_csv('customer_segmentation_results.csv', index=False)

print("✅ Analysis complete!")
print(f"Results saved to 'customer_segmentation_results.csv'")
print(f"\nTotal customers analyzed: {len(final_results)}")
print(f"Total CLTV: £{final_results['CLTV'].sum():,.2f}")
print(f"Average CLTV: £{final_results['CLTV'].mean():,.2f}")
```

---

## 🎉 Done!

You've successfully completed RFM and CLTV analysis! 

### Next Steps:
1. Download the results CSV file
2. Share this notebook by clicking "Share" in the top right
3. Make the notebook public or share the link with your team
4. Consider building a Streamlit app for interactive exploration

### To Share Your Colab:
1. Click "Share" button in top right
2. Change access to "Anyone with the link"
3. Copy the link and share it

**Example shared link format:**
`https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID`
