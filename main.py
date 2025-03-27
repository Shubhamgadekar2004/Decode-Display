import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime

# Set page config as the first Streamlit command
st.set_page_config(page_title="Comprehensive Orders Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        df = pd.read_csv("orders_dataset.csv")
        
        # Convert date columns
        df['orderDate'] = pd.to_datetime(df['orderDate'], format='%d-%m-%Y', errors='coerce')
        df['deliveryDate'] = pd.to_datetime(df['deliveryDate'], format='%Y-%m-%d', errors='coerce')
        df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], errors='coerce')
        df['creationDate'] = pd.to_datetime(df['creationDate'], format='%d-%m-%Y', errors='coerce')
        
        # Create additional useful columns
        df['order_year_month'] = df['orderDate'].dt.to_period('M').astype(str)
        df['delivery_delay'] = (df['deliveryDate'] - df['orderDate']).dt.days
        
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
        
        # Handle missing values
        df.fillna({'color': 'Unknown', 'size': 'Unknown', 'state': 'Unknown'}, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Main dashboard function
    def main():
        # Sidebar
        st.sidebar.title("Filters")
        
        # Multi-select filters
        selected_states = st.sidebar.multiselect(
            "Select States", 
            options=df['state'].unique(),
            default=df['state'].unique()
        )
        
        selected_months = st.sidebar.multiselect(
            "Select Months", 
            options=sorted(df['order_year_month'].unique()),
            default=[df['order_year_month'].max()]
        )
        
        selected_colors = st.sidebar.multiselect(
            "Select Colors", 
            options=df['color'].unique(),
            default=df['color'].unique()
        )
        
        selected_sizes = st.sidebar.multiselect(
            "Select Sizes", 
            options=df['size'].unique(),
            default=df['size'].unique()
        )
        
        # Filter dataframe
        df_filtered = df[
            (df['state'].isin(selected_states)) & 
            (df['order_year_month'].isin(selected_months)) &
            (df['color'].isin(selected_colors)) &
            (df['size'].isin(selected_sizes))
        ]
        
        # Dashboard Title
        st.title("🛍️ Comprehensive Orders Dashboard")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", f"{len(df_filtered):,}")
        
        with col2:
            st.metric("Total Revenue", f"€{df_filtered['price'].sum():,.2f}")
        
        with col3:
            st.metric("Average Order Value", f"€{df_filtered['price'].mean():,.2f}" if len(df_filtered) > 0 else "€0.00")
        
        with col4:
            st.metric("Return Rate", f"{(df_filtered['returnShipment'].mean() * 100):.2f}%" if len(df_filtered) > 0 else "0.00%")
        
        # Visualization Section
        st.header("Detailed Insights")
        
        # First Row of Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Order Distribution by Color
            st.subheader("Orders by Color")
            color_orders = df_filtered['color'].value_counts()
            fig_color = px.pie(
                values=color_orders.values, 
                names=color_orders.index, 
                title="Color Distribution",
                hole=0.3
            )
            st.plotly_chart(fig_color, use_container_width=True)
        
        with col2:
            # Order Distribution by Size
            st.subheader("Orders by Size")
            size_orders = df_filtered['size'].value_counts()
            fig_size = px.bar(
                x=size_orders.index, 
                y=size_orders.values, 
                title="Size Distribution",
                labels={'x':'Size', 'y':'Number of Orders'}
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        # Second Row of Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Manufacturer
            st.subheader("Revenue by Manufacturer")
            manufacturer_revenue = df_filtered.groupby('manufacturerID')['price'].sum().sort_values(ascending=False)
            fig_manufacturer = px.bar(
                x=manufacturer_revenue.index, 
                y=manufacturer_revenue.values, 
                title="Total Revenue by Manufacturer",
                labels={'x':'Manufacturer ID', 'y':'Total Revenue'}
            )
            st.plotly_chart(fig_manufacturer, use_container_width=True)
        
        with col2:
            # Delivery Performance
            st.subheader("Delivery Performance")
            delivery_performance = df_filtered[df_filtered['delivery_delay'] > 0].groupby(df_filtered['orderDate'].dt.month)['delivery_delay'].mean()
            fig_delivery = px.line(
                x=delivery_performance.index, 
                y=delivery_performance.values, 
                title="Average Delivery Delay by Month",
                labels={'x':'Month', 'y':'Average Delay (Days)'}
            )
            st.plotly_chart(fig_delivery, use_container_width=True)
        
        # Additional Insights
        st.header("Advanced Analytics")
        
        # Return Rate Analysis
        st.subheader("Return Rate Analysis")
        return_rate_by_state = df_filtered.groupby('state')['returnShipment'].mean() * 100
        fig_returns = px.bar(
            x=return_rate_by_state.index, 
            y=return_rate_by_state.values, 
            title="Return Rate by State",
            labels={'x':'State', 'y':'Return Rate (%)'}
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Customer Age Distribution
        st.subheader("Customer Age Distribution")
        age_bins = [0, 30, 45, 60, 100]
        age_labels = ['18-30', '31-45', '46-60', '60+']
        
        # Safely handle age calculation and binning
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

        # Detailed Table
        st.header("Detailed Order Insights")
        st.subheader("Order Details")
        st.dataframe(df_filtered[['orderDate', 'color', 'size', 'manufacturerID', 'price', 'state', 'returnShipment']].head(20))

    # Run the main function
    if __name__ == "__main__":
        main()
else:
    st.error("Data could not be loaded. Please check the dataset.")