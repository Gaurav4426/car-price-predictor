import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Car Price Prediction System")
st.markdown("### AI-based Used Car Valuation (Normal + Luxury)")

# ==============================
# BRANDS + MODELS
# ==============================

brands = [
    'Maruti','Hyundai','Tata','Honda','Toyota',
    'Mahindra',
    'BMW','Audi','Porsche'
]

models = {
'Maruti':['Swift','Baleno','WagonR','Dzire','Brezza'],
'Hyundai':['i20','Creta','Venue','Verna','Grand i10'],
'Tata':['Nexon','Punch','Harrier','Tiago','Altroz'],
'Honda':['City','Amaze','Jazz','WRV','Civic'],
'Toyota':['Innova','Fortuner','Glanza','Etios','Corolla'],
'Mahindra':['Thar','Scorpio','XUV700','Bolero','XUV300','XEV BE 6'],
'BMW':['3 Series','5 Series','X1'],
'Audi':['A4','A6','Q3'],
'Porsche':['Macan','Cayenne']
}

# ==============================
# BASE PRICE
# ==============================

base_price_dict = {
'Swift':600000,'Baleno':750000,'WagonR':550000,'Dzire':700000,'Brezza':900000,
'i20':800000,'Creta':1500000,'Venue':1200000,'Verna':1400000,'Grand i10':650000,
'Nexon':1100000,'Punch':850000,'Harrier':1900000,'Tiago':600000,'Altroz':850000,
'City':1300000,'Amaze':850000,'Jazz':900000,'WRV':1100000,'Civic':1800000,
'Innova':2200000,'Fortuner':3500000,'Glanza':800000,'Etios':700000,'Corolla':1700000,

# Mahindra
'Thar':1500000,'Scorpio':1400000,'XUV700':1800000,'Bolero':900000,'XUV300':1200000,'XEV BE 6':2500000,

'3 Series':6000000,'5 Series':9000000,'X1':4500000,
'A4':5500000,'A6':8000000,'Q3':4500000,
'Macan':9000000,'Cayenne':15000000
}

fuel_types=['Petrol','Diesel','Petrol+CNG','Electric']
transmissions=['Manual','Automatic']

# ==============================
# DATA GENERATION
# ==============================

data=[]

for _ in range(300):
    brand=np.random.choice(brands)
    model=np.random.choice(models[brand])

    year=np.random.randint(2010,2026)
    km=np.random.randint(5000,180000)
    mileage=np.random.uniform(1,30)   # CHANGED HERE
    engine=np.random.randint(800,2500)

    insurance=np.random.choice([0,1])
    transmission=np.random.choice(transmissions)
    fuel=np.random.choice(fuel_types)

    base_price = base_price_dict.get(model, 800000)

    price=(
        base_price
        -(2026-year)*50000
        -km*3
        +engine*4
        +mileage*1000
        +insurance*20000
        +(30000 if transmission=='Automatic' else 0)
        +(30000 if fuel=='Petrol+CNG' else 0)
        +(50000 if fuel=='Electric' else 0)
    )

    price=max(price,50000)

    data.append([
        brand,model,year,km,mileage,engine,
        insurance,transmission,fuel,price
    ])

df=pd.DataFrame(data,columns=[
'brand','model','year','km','mileage','engine',
'insurance','transmission','fuel','price'
])

# ==============================
# ENCODING (SAFE)
# ==============================

df_encoded = pd.get_dummies(df)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# ==============================
# MODEL
# ==============================

model_ml = RandomForestRegressor(n_estimators=100, random_state=42)
model_ml.fit(X, y)

# ==============================
# CAR IMAGES
# ==============================

images={
'Swift':"https://imgd.aeplcdn.com/664x374/n/cw/ec/159099/swift.jpeg",
'Baleno':"https://imgd.aeplcdn.com/664x374/n/cw/ec/146183/baleno.jpeg",
'Creta':"https://imgd.aeplcdn.com/664x374/n/cw/ec/131825/creta.jpeg",
'Punch':"https://imgd.aeplcdn.com/664x374/n/cw/ec/39015/punch.jpeg",
'3 Series':"https://imgd.aeplcdn.com/664x374/n/cw/ec/192443/3series.jpeg",
'A4':"https://imgd.aeplcdn.com/664x374/n/cw/ec/39445/a4.jpeg",
'Macan':"https://imgd.aeplcdn.com/664x374/n/cw/ec/39232/macan.jpeg"
}

# ==============================
# TABS
# ==============================

tab1, tab2 = st.tabs(["🔍 Predict Price", "⚖️ Compare Cars"])

# ==============================
# PREDICT TAB
# ==============================

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", brands)
        model_name = st.selectbox("Model", models[brand])

        if model_name in images:
            st.image(images[model_name], width=300)

    with col2:
        year = st.slider("Year", 2010, 2025, 2020)
        km = st.number_input("KM Driven", 0, 200000, 50000)
        mileage = st.number_input("Mileage", 1.0, 30.0, 18.0)  # CHANGED HERE
        engine = st.number_input("Engine CC", 800, 2500, 1200)

        insurance = st.selectbox("Insurance", [0,1])
        transmission = st.selectbox("Transmission", transmissions)
        fuel = st.selectbox("Fuel Type", fuel_types)

    input_df=pd.DataFrame([{
        'brand':brand,
        'model':model_name,
        'year':year,
        'km':km,
        'mileage':mileage,
        'engine':engine,
        'insurance':insurance,
        'transmission':transmission,
        'fuel':fuel
    }])

    input_encoded=pd.get_dummies(input_df)
    input_encoded=input_encoded.reindex(columns=X.columns,fill_value=0)

    if st.button("Predict Price"):
        pred=model_ml.predict(input_encoded)[0]
        st.success(f"💰 Estimated Price: ₹ {int(pred)}")
        st.info(f"📊 Range: ₹ {int(pred*0.9)} - ₹ {int(pred*1.1)}")

# ==============================
# COMPARE TAB
# ==============================

with tab2:
    st.subheader("Compare Two Cars")

    colA, colB = st.columns(2)

    def get_input(col, key):
        brand = col.selectbox("Brand", brands, key=key+"b")
        model_name = col.selectbox("Model", models[brand], key=key+"m")

        year = col.slider("Year", 2010, 2025, 2020, key=key+"y")
        km = col.number_input("KM", 0, 200000, 50000, key=key+"k")

        mileage = col.number_input("Mileage", 1.0, 30.0, 18.0, key=key+"mi")  # CHANGED HERE
        engine = col.number_input("Engine", 800, 2500, 1200, key=key+"e")

        insurance = col.selectbox("Insurance", [0,1], key=key+"i")
        transmission = col.selectbox("Transmission", transmissions, key=key+"t")
        fuel = col.selectbox("Fuel", fuel_types, key=key+"f")

        df=pd.DataFrame([{
            'brand':brand,'model':model_name,'year':year,'km':km,
            'mileage':mileage,'engine':engine,'insurance':insurance,
            'transmission':transmission,'fuel':fuel
        }])

        df=pd.get_dummies(df)
        df=df.reindex(columns=X.columns,fill_value=0)

        return df

    input1=get_input(colA,"A")
    input2=get_input(colB,"B")

    if st.button("Compare"):
        p1=model_ml.predict(input1)[0]
        p2=model_ml.predict(input2)[0]

        colA.success(f"₹ {int(p1)}")
        colB.success(f"₹ {int(p2)}")
