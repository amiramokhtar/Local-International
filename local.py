import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

# 1. App Title
st.title("🏨 Hotel Performance Analysis – Local and International Markets")

# 2. Upload CSV
uploaded_file = st.sidebar.file_uploader("📤 Upload Cleaned_Hotel_Booking.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(""https://raw.githubusercontent.com/amiramokhtar/Local-International/master/Cleaned_Hotel_Booking.csv")

    # 3. Data Cleaning and Preprocessing
    df['Arrival date'] = pd.to_datetime(df['Arrival date'], errors='coerce')
    df['Year'] = df['Arrival date'].dt.year
    df['Month'] = df['Arrival date'].dt.month
    df['YearMonth'] = df['Arrival date'].dt.to_period('M').astype(str)
    df['Market_Type'] = df['Country'].apply(lambda x: 'Local' if str(x).strip().lower() == 'egypt' else 'International')

    st.subheader("🧾 Data Sample")
    st.write(df.head())

    # 4. Bookings 2018 (Bar Chart)
    df_2018 = df[df['Year'] == 2018]
    pivot_2018 = df_2018.groupby(['Month', 'Market_Type'])['No Of Rooms'].sum().reset_index()
    fig1 = px.bar(pivot_2018, x='Month', y='No Of Rooms', color='Market_Type', barmode='group',
                  title='📊 Local and International Bookings in 2018')
    st.plotly_chart(fig1)

    # 5. Occupancy Rate (Line Chart)
    occupancy = df.groupby(['YearMonth', 'Market_Type'])['Occupancy Rate %'].mean().reset_index()
    fig2 = px.line(occupancy, x='YearMonth', y='Occupancy Rate %', color='Market_Type',
                   title='📈 Monthly Occupancy Rate by Market')
    st.plotly_chart(fig2)

    # 6. ADR & RevPAR (Table)
    st.subheader("📌 Average ADR and RevPAR by Market")
    adr_revpar = df.groupby('Market_Type')[['ADR', 'RevPAR']].mean().reset_index()
    st.dataframe(adr_revpar)

    # 7. Monthly Revenue (Area Chart)
    market_monthly = df.groupby(['YearMonth', 'Market_Type'])['total rate net'].sum().reset_index()
    fig3 = px.area(market_monthly, x='YearMonth', y='total rate net', color='Market_Type',
                   title='📊 Monthly Revenue – Local vs International',
                   template='plotly_white', groupnorm='fraction')
    st.plotly_chart(fig3)

    # 8. Revenue Share (Pie Chart)
    market_rev = df.groupby('Market_Type')['total rate net'].sum().reset_index()
    fig4 = px.pie(market_rev, names='Market_Type', values='total rate net', hole=0.5,
                  title='🎯 Market Share by Revenue')
    st.plotly_chart(fig4)

    # 9. Prophet Forecast – Local Bookings
    st.subheader("📅 Forecasting Local Market Bookings")
    monthly_local = df[df['Market_Type'] == 'Local'].groupby('YearMonth')['No Of Rooms'].sum().reset_index()
    monthly_local['ds'] = pd.to_datetime(monthly_local['YearMonth'])
    monthly_local['y'] = monthly_local['No Of Rooms']
    monthly_local = monthly_local[['ds', 'y']]

    model = Prophet()
    model.fit(monthly_local)
    future = model.make_future_dataframe(periods=6, freq='ME')  # Use 'ME' to avoid FutureWarning
    forecast = model.predict(future)

    fig5 = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Expected Room Bookings'},
                   title='🔮 Forecast: Local Market Bookings (Next 6 Months)')
    st.plotly_chart(fig5)

    # 10. ML Model – Predict ADR
    st.subheader("🧠 Predict ADR (Average Daily Rate) using Machine Learning")
    ml_df = df.dropna(subset=['ADR'])

    # Features and Target
    X = ml_df[['No Of Rooms', 'Adult', 'Child', 'T.Pax', 'Room Type', 'Market_Type', 'arrival_day', 'arrival_month', 'booking_day_diff']]
    y = ml_df['ADR']

    cat_features = ['Room Type', 'Market_Type']
    num_features = ['No Of Rooms', 'Adult', 'Child', 'T.Pax', 'arrival_day', 'arrival_month', 'booking_day_diff']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.success(f"✅ Model Trained – MAE: {round(mae, 2)}")

    # 11. User Input for Prediction
    st.markdown("### 🔍 Try a Custom Prediction")
    user_input = {
        'No Of Rooms': st.number_input('No Of Rooms', min_value=1, max_value=10, value=2),
        'Adult': st.number_input('Adults', min_value=1, max_value=10, value=2),
        'Child': st.number_input('Children', min_value=0, max_value=10, value=0),
        'T.Pax': st.number_input('Total Pax', min_value=1, max_value=20, value=2),
        'Room Type': st.selectbox('Room Type', options=df['Room Type'].dropna().unique()),
        'Market_Type': st.selectbox('Market', options=['Local', 'International']),
        'arrival_day': st.number_input('Arrival Day', min_value=1, max_value=31, value=15),
        'arrival_month': st.number_input('Arrival Month', min_value=1, max_value=12, value=6),
        'booking_day_diff': st.number_input('Booking Lead Time (days)', min_value=0, max_value=365, value=30)
    }

    user_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(user_df)[0]
    st.success(f"🔮 Predicted ADR: {round(prediction, 2)} EGP")

else:
    st.info("📥 Please upload your cleaned hotel booking CSV file to begin.")
