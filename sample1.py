import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

# Set page config
st.set_page_config(page_title="Comprehensive Orders Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Read the CSV file
    df = pd.read_csv("orders_dataset.csv")
    
    # Convert date columns with robust error handling
    date_columns = ['orderDate', 'deliveryDate', 'dateOfBirth', 'creationDate']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
        if col == 'deliveryDate':
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
    
    # Calculate delivery delay with error handling
    df['delivery_delay'] = (df['deliveryDate'] - df['orderDate']).dt.days
    df['delivery_delay'] = df['delivery_delay'].fillna('Unknown')
    
    # Calculate customer age
    def calculate_age(born):
        if pd.isnull(born):
            return None
        today = datetime.now()
        try:
            age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            return age
        except:
            return None
    
    df['customer_age'] = df['dateOfBirth'].apply(calculate_age)
    
    # Additional preprocessing
    df['order_year_month'] = df['orderDate'].dt.to_period('M')
    
    return df

# Load the data
df = load_data()

def main():
    # Sidebar for filters
    st.sidebar.title("📊 Dashboard Filters")
    
    # Multi-select filters with error handling
    try:
        selected_states = st.sidebar.multiselect(
            "Select States", 
            options=df['state'].dropna().unique(),
            default=df['state'].dropna().unique()
        )
        
        selected_months = st.sidebar.multiselect(
            "Select Months", 
            options=sorted(df['order_year_month'].astype(str).unique()),
            default=[df['order_year_month'].astype(str).max()]
        )
    except Exception as e:
        st.sidebar.error(f"Error in filter selection: {e}")
        selected_states = df['state'].dropna().unique()
        selected_months = [df['order_year_month'].astype(str).max()]
    
    # Filter dataframe
    df_filtered = df[
        (df['state'].isin(selected_states)) & 
        (df['order_year_month'].astype(str).isin(selected_months))
    ]
    
    # Dashboard Title
    st.title("🛍️ Comprehensive Orders Analysis")
    
    # Key Metrics with error handling
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", f"{len(df_filtered):,}")
        
        with col2:
            st.metric("Total Revenue", f"€{df_filtered['price'].sum():,.2f}")
        
        with col3:
            st.metric("Average Order Value", f"€{df_filtered['price'].mean():,.2f}")
        
        with col4:
            st.metric("Return Rate", f"{(df_filtered['returnShipment'].mean() * 100):.2f}%")
    except Exception as e:
        st.error(f"Error calculating key metrics: {e}")
    
    # Visualization Section
    st.header("📈 Detailed Insights")
    
    # First Row of Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Order Distribution by Color
        st.subheader("Orders by Color")
        try:
            color_orders = df_filtered['color'].value_counts()
            fig_color = px.pie(
                values=color_orders.values, 
                names=color_orders.index, 
                title="Color Distribution"
            )
            st.plotly_chart(fig_color, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating color distribution chart: {e}")
    
    with col2:
        # Order Distribution by Size
        st.subheader("Orders by Size")
        try:
            size_orders = df_filtered['size'].value_counts()
            fig_size = px.bar(
                x=size_orders.index, 
                y=size_orders.values, 
                title="Size Distribution",
                labels={'x':'Size', 'y':'Number of Orders'}
            )
            st.plotly_chart(fig_size, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating size distribution chart: {e}")
    
    # Second Row of Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by Manufacturer
        st.subheader("Revenue by Manufacturer")
        try:
            manufacturer_revenue = df_filtered.groupby('manufacturerID')['price'].sum().sort_values(ascending=False)
            fig_manufacturer = px.bar(
                x=manufacturer_revenue.index, 
                y=manufacturer_revenue.values, 
                title="Total Revenue by Manufacturer",
                labels={'x':'Manufacturer ID', 'y':'Total Revenue'}
            )
            st.plotly_chart(fig_manufacturer, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating manufacturer revenue chart: {e}")
    
    with col2:
        # Delivery Performance
        st.subheader("Delivery Performance")
        try:
            # Handle cases with '?' in delivery date
            df_delivery = df_filtered[df_filtered['deliveryDate'].notna()]
            delivery_performance = df_delivery.groupby(df_delivery['orderDate'].dt.month)['delivery_delay'].mean()
            fig_delivery = px.line(
                x=delivery_performance.index, 
                y=delivery_performance.values, 
                title="Average Delivery Delay by Month",
                labels={'x':'Month', 'y':'Average Delay (Days)'}
            )
            st.plotly_chart(fig_delivery, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating delivery performance chart: {e}")
    
    # Advanced Analytics
    st.header("🔍 Advanced Analytics")
    
    # Return Rate Analysis
    st.subheader("Return Rate Analysis")
    try:
        return_rate_by_state = df_filtered.groupby('state')['returnShipment'].mean() * 100
        fig_returns = px.bar(
            x=return_rate_by_state.index, 
            y=return_rate_by_state.values, 
            title="Return Rate by State",
            labels={'x':'State', 'y':'Return Rate (%)'}
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating return rate analysis: {e}")
    
    # Customer Age Distribution
    st.subheader("Customer Age Distribution")
    try:
        # Safely handle age calculation and binning
        age_bins = [0, 30, 45, 60, 100]
        age_labels = ['18-30', '31-45', '46-60', '60+']
        
        df_filtered['age_group'] = pd.cut(
            df_filtered['customer_age'].fillna(-1), 
            bins=[-1] + age_bins, 
            labels=['Unknown'] + age_labels, 
            right=False
        )
        
        age_distribution = df_filtered['age_group'].value_counts()
        
        fig_age = px.pie(
            values=age_distribution.values, 
            names=age_distribution.index, 
            title="Customer Age Distribution"
        )
        st.plotly_chart(fig_age, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating age distribution chart: {e}")
    
    # Detailed Insights
    st.header("📋 Detailed Order Insights")
    
    # Detailed Table with Filtering
    st.subheader("Order Details")
    try:
        # Columns to display
        display_columns = [
            'orderDate', 'deliveryDate', 'itemID', 'color', 'size', 
            'manufacturerID', 'price', 'state', 'returnShipment'
        ]
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_price = st.number_input("Minimum Price", min_value=0.0, value=0.0)
        with col2:
            max_price = st.number_input("Maximum Price", min_value=0.0, value=df_filtered['price'].max())
        with col3:
            show_returns_only = st.checkbox("Show Returns Only")
        
        # Apply additional filtering
        df_table = df_filtered[
            (df_filtered['price'].between(min_price, max_price)) & 
            (df_filtered['returnShipment'] == 1 if show_returns_only else True)
        ]
        
        st.dataframe(df_table[display_columns])
    except Exception as e:
        st.error(f"Error creating detailed order table: {e}")

# Run the main function with error handling
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
