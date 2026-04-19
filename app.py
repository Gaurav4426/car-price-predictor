import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction System")

# ==============================
# DATA GENERATION (SAME AS YOUR CODE)
# ==============================

data=[]

brands_sh=['Maruti','Hyundai','Tata','Honda','Toyota']

models_sh = {
'Maruti':['Swift','Baleno','WagonR','Dzire','Brezza'],
'Hyundai':['i20','Creta','Venue','Verna','Grand i10'],
'Tata':['Nexon','Punch','Harrier','Tiago','Altroz'],
'Honda':['City','Amaze','Jazz','WRV','Civic'],
'Toyota':['Innova','Fortuner','Glanza','Etios','Corolla']
}

model_price_sh = {
'Swift':600000,'Baleno':750000,'WagonR':550000,'Dzire':700000,'Brezza':900000,
'i20':800000,'Creta':1500000,'Venue':1200000,'Verna':1400000,'Grand i10':650000,
'Nexon':1100000,'Punch':850000,'Harrier':1900000,'Tiago':600000,'Altroz':850000,
'City':1300000,'Amaze':850000,'Jazz':900000,'WRV':1100000,'Civic':1800000,
'Innova':2200000,'Fortuner':3500000,'Glanza':800000,'Etios':700000,'Corolla':1700000
}

fuel_sh=['Petrol','Diesel','Petrol+CNG','Electric']
trans_sh=['Manual','Automatic']

for _ in range(150):
    brand=np.random.choice(brands_sh)
    model=np.random.choice(models_sh[brand])

    year=np.random.randint(2010,2027)
    km=np.random.randint(5000,180000)
    mileage=np.random.uniform(10,25)
    engine=np.random.randint(800,2500)

    insurance=np.random.choice([0,1])
    transmission=np.random.choice(trans_sh)
    fuel=np.random.choice(fuel_sh)

    base_price=model_price_sh[model]

    age_penalty=(2026-year)*40000
    km_penalty=km*3

    price=(
        base_price
        -age_penalty
        -km_penalty
        +(25000 if transmission=='Automatic' else 0)
        +(60000 if fuel=='Electric' else 20000 if fuel=='Petrol+CNG' else 0)
        +insurance*20000
        +engine*4
        +mileage*2000
    )

    price=max(price,50000)

    data.append([
        0,brand,model,year,km,mileage,engine,
        insurance,transmission,fuel,
        0,0,0,0,'Black','No',price
    ])

# ==============================
# DATAFRAME + ENCODING
# ==============================

columns=[
'car_type','brand','model','year','km_driven','mileage','engine_cc',
'insurance','transmission','fuel_type',
'engine_power','interior','safety','custom_paint',
'color','sunroof','price'
]

df=pd.DataFrame(data,columns=columns)

df_encoded=df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype=='object':
        df_encoded[col]=df_encoded[col].astype('category').cat.codes

X=df_encoded.drop('price',axis=1)
y=df_encoded['price']

model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X,y)

# ==============================
# UI
# ==============================

st.subheader("Select Car Details")

brand = st.selectbox("Brand", brands_sh)
model_name = st.selectbox("Model", models_sh[brand])

year = st.slider("Year", 2010, 2026, 2020)
km = st.number_input("KM Driven", 0, 200000, 50000)
mileage = st.number_input("Mileage", 10.0, 30.0, 18.0)
engine = st.number_input("Engine CC", 800, 2500, 1200)

insurance = st.selectbox("Insurance", [0,1])
transmission = st.selectbox("Transmission", trans_sh)
fuel = st.selectbox("Fuel Type", fuel_sh)

# ==============================
# ENCODING INPUT (IMPORTANT)
# ==============================

brand_code = brands_sh.index(brand)
model_code = models_sh[brand].index(model_name)
trans_code = trans_sh.index(transmission)
fuel_code = fuel_sh.index(fuel)

input_data=[
    0,
    brand_code,
    model_code,
    year,
    km,
    mileage,
    engine,
    insurance,
    trans_code,
    fuel_code,
    0,0,0,0,0,0
]

input_df=pd.DataFrame([input_data],columns=X.columns)

# ==============================
# PREDICTION
# ==============================

if st.button("Predict Price"):
    pred=model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {int(pred)}")
    st.info(f"📊 Range: ₹ {int(pred*0.9)} - ₹ {int(pred*1.1)}")
