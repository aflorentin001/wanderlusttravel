
How to interpret your CLTV results:

Our analysis shows:

Total CLTV: £6,059,859.33 (the combined lifetime value of your entire customer base)
Average CLTV: £1,411.24 per customer
Median CLTV: £663.73 per customer

The gap between average (£1,411) and median (£663) tells you that a small number of high-value customers are pulling up the average. This is the 80/20 rule in action.

Business implications:

If your customer acquisition cost (CAC) is £100, your average customer is worth 14x their acquisition cost. Excellent unit economics.
If CAC is £500, you’re in trouble. Only customers above £500 CLTV are profitable.
The median customer (£663) represents your “typical” customer, not the average. Use this for realistic forecasting.
Step 6: Visualize CLTV Insights
cltv_data_clean = cltv_data[cltv_data['CLTV'] <= cltv_data['CLTV'].quantile(0.99)]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Lifetime Value (CLTV) Analysis', fontsize=16, fontweight='bold')
axes[0, 0].hist(cltv_data_clean['CLTV'], bins=50, color='#00BFA5', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('CLTV Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Customer Lifetime Value (£)')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].axvline(cltv_data_clean['CLTV'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: £{cltv_data_clean["CLTV"].mean():.2f}')
axes[0, 0].axvline(cltv_data_clean['CLTV'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: £{cltv_data_clean["CLTV"].median():.2f}')
axes[0, 0].legend()
scatter = axes[0, 1].scatter(cltv_data_clean['AvgOrderValue'], cltv_data_clean['PurchaseFrequency'], c=cltv_data_clean['CLTV'], cmap='YlOrRd', alpha=0.6, s=50)
axes[0, 1].set_title('Average Order Value vs Purchase Frequency', fontweight='bold')
axes[0, 1].set_xlabel('Average Order Value (£)')
axes[0, 1].set_ylabel('Purchase Frequency (purchases/year)')
plt.colorbar(scatter, ax=axes[0, 1], label='CLTV (£)')
cltv_quartiles = pd.qcut(cltv_data_clean['CLTV'], q=4, labels=['Low Value', 'Medium Value', 'High Value', 'Top Value'])
segment_counts = cltv_quartiles.value_counts().sort_index()
colors = ['#FFC107', '#FF7043', '#42A5F5', '#00BFA5']
axes[1, 0].bar(segment_counts.index, segment_counts.values, color=colors)
axes[1, 0].set_title('Customer Value Segments', fontweight='bold')
axes[1, 0].set_xlabel('CLTV Segment')
axes[1, 0].set_ylabel('Number of Customers')
for i, v in enumerate(segment_counts.values):
    axes[1, 0].text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
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

How to read these visualizations:

CLTV Distribution (top-left): Most customers cluster around the median (£663). The long right tail shows a few extremely valuable customers. These outliers deserve personal attention.

AOV vs Frequency (top-right): The scatter plot reveals four customer types:

High AOV, High Frequency: Your whales. Treat them like royalty.
High AOV, Low Frequency: Big spenders who buy rarely. Increase their frequency.
Low AOV, High Frequency: Frequent small buyers. Upsell them.
Low AOV, Low Frequency: Least valuable. Low-touch automation.

Value Segments (bottom-left): Each quartile contains roughly 1,000 customers, but they don’t contribute equally to revenue. The “Top Value” segment likely generates 50%+ of your revenue.

Pareto Chart (bottom-right): This is the money chart. The dotted lines show the 80/20 rule. In our data, the top 20% of customers contribute approximately 60% of total CLTV. This is actually quite distributed (many businesses see top 20% contribute 80%+).

Actionable insight: If the curve is steep (approaching the top-left corner quickly), you’re heavily dependent on a few customers. Diversification or retention focus is critical.

Step 7: Apply KMeans Clustering (Finding Hidden Patterns)

RFM gives you predefined segments. KMeans finds natural groupings in your data that you might never have considered.

First, find the optimal number of clusters using the Elbow Method:

X = rfm[['Recency', 'Frequency', 'Monetary']]
X_transformed = X.copy()
X_transformed['Monetary'] = np.log1p(X_transformed['Monetary'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
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

Why this matters: The Elbow Method helps you avoid over-segmentation (too many clusters) or under-segmentation (too few). You’re looking for the “elbow” where adding more clusters gives diminishing returns.

How to interpret the Elbow graph:

Look at the percentage decrease in inertia:

K=2 to K=3: 30% decrease
K=3 to K=4: 30% decrease
K=4 to K=5: 23% decrease
K=5 to K=6: 15% decrease

The elbow occurs around K=4 or K=5, where the decrease starts to flatten. We’ll use K=4 because:

The improvement drops significantly after K=4
Four segments are manageable for marketing campaigns
More segments create operational complexity
Step 8: Apply KMeans and Interpret Clusters
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
cluster_summary = rfm.groupby('KMeans_Cluster').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'],
    'Monetary': ['mean', 'median'],
    'CustomerID': 'count'
}).round(2)
print("Cluster Analysis:")
print(cluster_summary)

The four clusters discovered:

Cluster 0 (2,059 customers): Regular Customers

Average Recency: 53 days
Average Frequency: 2.2 purchases
Average Monetary: £586.88
Interpretation: Your bread and butter. Consistent but moderate engagement. They’re not going anywhere, but they’re not Champions either.

Cluster 1 (1,004 customers): Low Engagement

Average Recency: 253 days
Average Frequency: 1.5 purchases
Average Monetary: £408.62
Interpretation: Likely churned. Haven’t bought in 8+ months. Low engagement when they were active. May not be worth aggressive win-back unless your margins support it.

Cluster 2 (1,253 customers): Frequent Buyers

Average Recency: 30 days
Average Frequency: 8.7 purchases
Average Monetary: £4,471.95
Interpretation: Your loyal customer base. Bought recently, buy often, spend significantly. Focus retention and upselling here.

Cluster 3 (22 customers): VIP Champions

Average Recency: 6 days
Average Frequency: 77.5 purchases
Average Monetary: £76,791.63
Interpretation: Your whales. Just 0.5% of customers but generating massive revenue. These 22 people deserve white-glove treatment. Assign account managers. Send birthday cards. Never let them leave.

Critical insight: The algorithm found something your RFM analysis missed. Those 22 VIP Champions are in a class of their own. They buy 9x more frequently than your next-best segment and spend 17x more. Losing even one of them is catastrophic.

Step 9: Visualize KMeans Clusters
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(131, projection='3d')
colors = ['#FF7043', '#00BFA5', '#FFC107', '#42A5F5']
for i in range(optimal_k):
    cluster_data = rfm[rfm['KMeans_Cluster'] == i]
    ax1.scatter(cluster_data['Recency'], cluster_data['Frequency'], np.log1p(cluster_data['Monetary']), c=colors[i], label=f'Cluster {i}', alpha=0.6, s=30)
ax1.set_xlabel('Recency (days)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_zlabel('Log(Monetary)', fontweight='bold')
ax1.set_title('KMeans Clustering (3D View)', fontweight='bold', fontsize=12)
ax1.legend()
ax2 = fig.add_subplot(132)
cluster_counts = rfm['KMeans_Cluster'].value_counts().sort_index()
ax2.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
ax2.set_xticks(range(len(cluster_counts)))
ax2.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
ax2.set_title('Customer Distribution by Cluster', fontweight='bold', fontsize=12)
ax2.set_ylabel('Number of Customers', fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    ax2.text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
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

How to read the 3D scatter plot: Each point is a customer. The distance between points shows similarity. Notice how Cluster 3 (VIP Champions in blue) sits in its own space, far from all other clusters. They’re genuinely different.

Distribution by cluster (middle chart): Clusters 0 and 2 contain the majority of customers (76% combined). These are your operational focus.

Revenue by cluster (right chart): Despite having only 22 customers, Cluster 3 generates £1.69M in revenue (about 19% of total). Cluster 2 (Frequent Buyers) generates £5.6M (62% of total). These two clusters (29% of customers) drive 81% of revenue.

Business implication: If you lose Cluster 2 or 3, your business collapses. All retention and engagement efforts should prioritize these segments.

Turning Analysis Into Action: Your Strategic Playbook

You’ve now segmented your customers three different ways. Here’s how to use these insights:

For Champions and VIP Champions (Clusters 2 & 3):

Send personalized thank-you notes or gifts
Offer exclusive early access to new products
Create a VIP tier with special perks
Assign dedicated account managers for Cluster 3
Quarterly check-ins to ensure satisfaction
Cost: High touch, but these customers justify it

For Regular Customers (Cluster 0):

Automated email sequences to increase purchase frequency
“Customers who bought X also bought Y” recommendations
Loyalty programs with point accumulation
Free shipping thresholds to increase order value
Cost: Low touch, high automation

For At Risk and Low Engagement (Cluster 1 and RFM “At Risk”):

Win-back email campaigns with time-limited discounts
Survey to understand why they stopped buying
Retargeting ads on social platforms
“We miss you” campaigns with 20% off next purchase
Cost: Medium touch, test and measure ROI

For Lost Customers (RFM “Lost” and “Can’t Lose Them”):

Deep discount offers (30–50% off) for reactivation
“What did we do wrong?” surveys with incentive
Consider removing from regular email lists (save costs)
One last “final chance” campaign before archiving
Cost: Low touch, accept that many won’t return

Resource Allocation Based on CLTV:

Customers with CLTV > £2,000: Personal outreach
CLTV £500 — £2,000: Automated nurture campaigns
CLTV < £500: Minimal investment unless margins support it
The Business Case: Why This Matters

Let’s run the numbers on our example dataset:

Current State: