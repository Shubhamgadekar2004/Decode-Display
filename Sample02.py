# Complete Streamlit Orders Dashboard with Forecast & Return Prediction

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
from prophet import Prophet
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 🎨 Dashboard Theme & Page Setup
# -------------------------------
st.set_page_config(
    page_title="📦 Orders Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for theme
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .block-container { padding-top: 2rem; }
        .css-18e3th9 { padding: 1.5rem; background-color: #ffffff; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Data Loading & Preprocessing
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("orders_dataset.csv")
    df['orderDate'] = pd.to_datetime(df['orderDate'], format='%d-%m-%Y', errors='coerce')
    df['deliveryDate'] = pd.to_datetime(df['deliveryDate'], format='%Y-%m-%d', errors='coerce')
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], errors='coerce')
    df['creationDate'] = pd.to_datetime(df['creationDate'], format='%d-%m-%Y', errors='coerce')
    df['order_year_month'] = df['orderDate'].dt.to_period('M').astype(str)
    df['delivery_delay'] = (df['deliveryDate'] - df['orderDate']).dt.days
    df['customer_age'] = df['dateOfBirth'].apply(lambda dob: datetime.now().year - dob.year if pd.notnull(dob) else None)
    df.fillna({'color': 'Unknown', 'size': 'Unknown', 'state': 'Unknown'}, inplace=True)
    return df

df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------

st.sidebar.header("🔍 Filters")

states = st.sidebar.multiselect("State", df['state'].unique(), default=df['state'].unique())
months = st.sidebar.multiselect("Order Month", sorted(df['order_year_month'].unique()), default=df['order_year_month'].unique())
colors = st.sidebar.multiselect("Color", df['color'].unique(), default=df['color'].unique())
sizes = st.sidebar.multiselect("Size", df['size'].unique(), default=df['size'].unique())

df_filtered = df[
    (df['state'].isin(states)) &
    (df['order_year_month'].isin(months)) &
    (df['color'].isin(colors)) &
    (df['size'].isin(sizes))
]

# -------------------------------
# KPIs
# -------------------------------

st.title("📦 Orders Intelligence Dashboard")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total Orders", f"{len(df_filtered):,}")
kpi2.metric("Total Revenue", f"€{df_filtered['price'].sum():,.2f}")
kpi3.metric("Avg Order Value", f"€{df_filtered['price'].mean():.2f}" if len(df_filtered) > 0 else "€0.00")
kpi4.metric("Return Rate", f"{(df_filtered['returnShipment'].mean() * 100):.2f}%" if len(df_filtered) > 0 else "0.00%")

# -------------------------------
# 📊 Charts
# -------------------------------

st.subheader("🟣 Product Insights")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.pie(df_filtered, names='color', title="Orders by Color")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df_filtered['size'].value_counts().reset_index(), x='index', y='size', title="Orders by Size",
                  labels={'index': 'Size', 'size': 'Count'})
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("🟢 Manufacturer & Delivery Insights")
col3, col4 = st.columns(2)

with col3:
    rev_manufacturer = df_filtered.groupby('manufacturerID')['price'].sum().sort_values(ascending=False)
    fig3 = px.bar(rev_manufacturer, title="Revenue by Manufacturer")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    df_delays = df_filtered[df_filtered['delivery_delay'] > 0]
    if not df_delays.empty:
        delays = df_delays.groupby(df_delays['orderDate'].dt.month)['delivery_delay'].mean()
        fig4 = px.line(x=delays.index, y=delays.values, title="Avg Delivery Delay by Month",
                       labels={'x': 'Month', 'y': 'Avg Delay (days)'})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No delivery delay data available.")

# -------------------------------
# Return Rate Analysis
# -------------------------------

st.subheader("🔴 Return Rate by State")
returns = df_filtered.groupby('state')['returnShipment'].mean() * 100
fig5 = px.bar(returns, title="Return Rate by State")
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# 🎯 Time Series Forecast (Prophet)
# -------------------------------

st.header("📈 Interactive Time Series Forecasting (Orders)")

df_forecast = df_filtered.groupby('orderDate').size().reset_index(name='orders')
df_forecast.rename(columns={'orderDate': 'ds', 'orders': 'y'}, inplace=True)

if len(df_forecast) > 30:
    model = Prophet()
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig6 = plot_plotly = plotly.graph_objs.Figure()
    fig6 = model.plot(forecast)
    st.pyplot(fig6)
else:
    st.warning("Not enough data for forecasting. Please select more data from the filters.")

# -------------------------------
# 🤖 Return Prediction (XGBoost)
# -------------------------------

st.header("🤖 Return Prediction Model")

# Prepare simple model
features = ['price', 'delivery_delay', 'customer_age']
df_model = df_filtered.dropna(subset=features + ['returnShipment'])

X = df_model[features]
y = df_model['returnShipment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

y_pred = model_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Return Prediction Model Accuracy: {acc*100:.2f}%")

# Optional: Display feature importance
importance = model_xgb.feature_importances_
fig7 = px.bar(x=features, y=importance, title="Feature Importance for Return Prediction")
st.plotly_chart(fig7, use_container_width=True)

# -------------------------------
# 🟡 Detailed Data Table
# -------------------------------

st.subheader("📄 Sample Order Records")
st.dataframe(df_filtered.head(20))

# -------------------------------
# End of Dashboard
# -------------------------------

