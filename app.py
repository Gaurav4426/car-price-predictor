import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Second-Hand Car Price Predictor")

# ==============================
# BRAND + MODELS
# ==============================

brand_models = {
    'Maruti Suzuki': ['Alto', 'Swift', 'Baleno', 'Dzire', 'Brezza', 'Ertiga', 'WagonR'],
    'Hyundai': ['i10', 'i20', 'Aura', 'Verna', 'Creta', 'Venue'],
    'Tata': ['Tiago', 'Punch', 'Nexon', 'Harrier', 'Safari'],
    'Mahindra': ['Bolero', 'Scorpio', 'XUV300', 'XUV700'],
    'Kia': ['Sonet', 'Seltos', 'Carens']
}

# ==============================
# BASE PRICES
# ==============================

new_car_price = {
    'Alto': 400000, 'Swift': 700000, 'Baleno': 800000,
    'Dzire': 750000, 'Brezza': 900000, 'Ertiga': 1100000, 'WagonR': 600000,
    'i10': 600000, 'i20': 800000, 'Aura': 700000,
    'Verna': 1300000, 'Creta': 1500000, 'Venue': 900000,
    'Tiago': 500000, 'Punch': 1100000, 'Nexon': 900000,
    'Harrier': 1800000, 'Safari': 2000000,
    'Bolero': 1000000, 'Scorpio': 1800000,
    'XUV300': 1200000, 'XUV700': 2500000,
    'Sonet': 900000, 'Seltos': 1400000, 'Carens': 1200000
}

fuel_types = ['Petrol', 'Diesel', 'Petrol+CNG']
transmissions = ['Manual', 'Automatic']

# ==============================
# DATA GENERATION
# ==============================

data = []

for _ in range(300):
    brand = np.random.choice(list(brand_models.keys()))
    model = np.random.choice(brand_models[brand])

    year = np.random.randint(2010, 2026)
    km = np.random.randint(5000, 150000)
    mileage = np.random.uniform(12, 25)
    engine = np.random.randint(800, 2000)
    insurance = np.random.choice([0, 1])
    transmission = np.random.choice(transmissions)
    fuel = np.random.choice(fuel_types)

    base_price = new_car_price.get(model, 800000)

    age = 2026 - year
    if age == 0:
        price = base_price
    elif age == 1:
        price = base_price * 0.90
    elif age == 2:
        price = base_price * 0.80
    else:
        price = base_price * 0.80 * (0.90 ** (age - 2))

    price -= km * 0.8
    price += engine * 1.5
    price += mileage * 500
    price += insurance * 8000

    if transmission == 'Automatic':
        price += 10000

    if fuel == 'Petrol+CNG':
        price += 20000

    price = min(price, base_price)
    price = max(price, 50000)

    data.append([
        brand, model, year, km, mileage, engine,
        insurance, transmission, fuel, price
    ])

df = pd.DataFrame(data, columns=[
    'brand','model','year','km','mileage','engine',
    'insurance','transmission','fuel','price'
])

# ==============================
# FIXED ENCODING (ERROR SOLVED)
# ==============================

df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
df_encoded = df_encoded.dropna()

X = df_encoded.drop('price', axis=1).astype(float)
y = df_encoded['price'].astype(float)

# ==============================
# MODEL
# ==============================

model_ml = RandomForestRegressor(n_estimators=100, random_state=42)
model_ml.fit(X, y)

# ==============================
# UI INPUT
# ==============================

brand = st.selectbox("Select Brand", list(brand_models.keys()))
model_name = st.selectbox("Select Model", brand_models[brand])

year = st.slider("Year", 2010, 2025, 2020)
km = st.number_input("KM Driven", 0, 200000, 50000)
mileage = st.number_input("Mileage", 10.0, 30.0, 18.0)
engine = st.number_input("Engine CC", 800, 2000, 1200)

insurance = st.selectbox("Insurance", [0,1])
transmission = st.selectbox("Transmission", transmissions)
fuel = st.selectbox("Fuel Type", fuel_types)

# ==============================
# PREDICTION
# ==============================

input_data = {
    'brand': list(brand_models.keys()).index(brand),
    'model': brand_models[brand].index(model_name),
    'year': year,
    'km': km,
    'mileage': mileage,
    'engine': engine,
    'insurance': insurance,
    'transmission': transmissions.index(transmission),
    'fuel': fuel_types.index(fuel)
}

input_df = pd.DataFrame([input_data]).astype(float)

if st.button("Predict Price"):
    pred = model_ml.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {int(pred)}")
    st.info(f"📊 Range: ₹ {int(pred*0.9)} - ₹ {int(pred*1.1)}")
