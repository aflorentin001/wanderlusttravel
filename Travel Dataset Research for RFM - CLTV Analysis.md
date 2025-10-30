# Travel Dataset Research for RFM/CLTV Analysis

## Key Requirements for RFM/CLTV Analysis
1. **Customer ID** - to track individual customers
2. **Transaction Date** - to calculate Recency
3. **Transaction Value** - to calculate Monetary value
4. **Multiple transactions per customer** - to calculate Frequency

## Datasets Evaluated

### 1. Customer Booking Dataset (Kaggle)
- **URL**: https://www.kaggle.com/datasets/ememque/customer-booking
- **Size**: 50,000 bookings
- **Problem**: Each row is a single booking, no customer ID to track repeat customers
- **Verdict**: ❌ NOT SUITABLE - Missing customer identifier for repeat purchase tracking

### 2. Hotel Booking Dataset (Kaggle) 
- **URL**: https://www.kaggle.com/datasets/abdulrahmankhaled1/hotel-booking-dataset
- **Size**: 36,275 bookings
- **Key Features**:
  - ✅ Unique booking ID
  - ✅ Arrival date (year, month, date)
  - ✅ avg_room_price (transaction value)
  - ✅ repeated_guest flag (0/1)
  - ✅ previous_bookings_not_canceled
  - ✅ previous_cancellations
- **Problem**: Each row is a booking, but lacks a persistent customer ID to aggregate transactions
- **Verdict**: ⚠️ PARTIALLY SUITABLE - Has repeat guest indicators but needs customer ID synthesis

## Best Solution: Create Synthetic Travel Dataset

Given the limitations of available travel datasets, the best approach is to:

1. **Use the Online Retail II dataset** from the article (proven to work)
2. **Adapt it to travel context** by:
   - Treating CustomerID as "Traveler ID"
   - Treating InvoiceNo as "Booking ID"
   - Treating TotalPrice as "Booking Amount"
   - Adding travel-specific context in the application

OR

3. **Generate a synthetic travel dataset** with proper structure:
   - Customer IDs (repeat travelers)
   - Booking dates spanning 1-2 years
   - Booking amounts (flight/hotel prices)
   - Travel destinations
   - Booking channels
   - Service types (flights, hotels, packages)

## Selected Dataset: Wanderlust Travel Bookings

**URL**: https://github.com/aflorentin001/wanderlusttravel/raw/main/Wanderlust_Travel_Bookings.csv

**Why it's perfect**:
- ✅ Authentic travel booking data
- ✅ Multiple customers with repeat bookings
- ✅ Full date range (2023 data)
- ✅ CustomerID, BookingDate, BookingReference, BookingAmount
- ✅ Additional travel context: Destination, ServiceType, Status
- ✅ Perfect for RFM/CLTV analysis in travel industry

**Dataset Structure**:
- CustomerID → Traveler ID
- InvoiceNo → Booking Reference
- InvoiceDate → Booking Date
- TotalPrice → Booking Amount
- Description → Service/Destination
- Quantity → Number of travelers/nights

This allows us to build a fully functional RFM/CLTV application with travel-themed UI and terminology while using a proven dataset structure.
