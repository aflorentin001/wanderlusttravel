# Wanderlust Travel - Customer Segmentation Dashboard

A Streamlit web application for performing RFM (Recency, Frequency, Monetary) and CLTV (Customer Lifetime Value) analysis on travel customer data.

## 🎯 Project Overview

This application helps travel agencies and businesses identify their most valuable customers, segment them based on booking behavior, and calculate their lifetime value. It enables targeted marketing campaigns to increase customer retention and profitability.

## 📊 Features

- **RFM Analysis**: Segment customers based on Recency, Frequency, and Monetary value
- **CLTV Calculation**: Calculate Customer Lifetime Value for strategic planning
- **Customer Segmentation**: Automatically categorize customers into actionable segments (Champions, At Risk, Lost, etc.)
- **KMeans Clustering**: Discover natural customer groups using machine learning
- **Interactive Visualizations**: Explore data through charts, graphs, and 3D scatter plots
- **Business Insights**: Get actionable recommendations for each customer segment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/aflorentin001/wanderlusttravel.git
   cd wanderlusttravel
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
.
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── Wanderlust_Travel_Bookings.csv           # Dataset
├── README.md                                 # This file
├── Google Colab Notebook Template.md        # Colab analysis template
├── Reflection on RFM, CLTV, and Application Development.md
└── 📦 Project Deliverables Summary.md
```

## 📚 Dataset

The application uses the **Wanderlust Travel Bookings** dataset, which contains:
- Real travel booking transactions
- Multiple unique customers with repeat bookings
- Date range: 2023 booking data
- Complete booking details including CustomerID, BookingDate, BookingReference, BookingAmount, Destination, ServiceType, and Status

**Dataset URL**: https://github.com/aflorentin001/wanderlusttravel/raw/main/Wanderlust_Travel_Bookings.csv

## 🎨 Application Pages

### 1. Overview
- Key business metrics (Total Revenue, AOV, Customer Count)
- Revenue trend over time
- Top 10 customers by spend

### 2. RFM Analysis
- RFM metric distributions
- Customer scatter plots
- Detailed RFM scores and segments

### 3. CLTV Analysis
- Customer Lifetime Value metrics
- CLTV distribution and Pareto analysis
- Top customers by CLTV

### 4. Customer Segments
- Segment distribution and characteristics
- Revenue contribution by segment
- Detailed segment exploration

### 5. KMeans Clustering
- Machine learning-based customer grouping
- 3D cluster visualization
- Cluster characteristics and revenue analysis

## 🌐 Deployment to Streamlit Cloud

1. **Push your code to GitHub** (already done if you cloned this repo)

2. **Sign up for Streamlit Cloud** at https://streamlit.io/cloud

3. **Deploy your app:**
   - Click "New app" in Streamlit Cloud
   - Connect your GitHub account
   - Select this repository
   - Set main file to `app.py`
   - Click "Deploy!"

Your app will be live at: `https://[your-app-name].streamlit.app`

## 📖 Key Concepts

### RFM Analysis
- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Number of purchases
- **Monetary (M)**: Total amount spent

### Customer Segments
- **Champions**: Best customers (high R, F, M)
- **Loyal Customers**: Regular purchasers
- **At Risk**: Previously good customers who haven't purchased recently
- **Lost**: Haven't purchased in a long time
- **Recent Customers**: New customers with potential
- **Potential Loyalists**: Customers showing promise
- **Cant Lose Them**: High-value customers at risk

### CLTV Formula
```
CLTV = (Average Order Value × Purchase Frequency) / Churn Rate
```

## 🎓 Learning Resources

- Original Article: [Turn Customer Data Into Cash](https://drlee.io/turn-customer-data-into-cash-master-cltv-rfm-analysis-and-kmeans-clustering-in-google-colab-c0b88bafe450)
- Streamlit Documentation: https://docs.streamlit.io
- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn Documentation: https://scikit-learn.org/stable/

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created as part of a data analytics course project at Miami Dade College.

## 🙏 Acknowledgments

- Dr. Ernesto Lee for the original RFM/CLTV methodology
- Streamlit for the amazing web framework
