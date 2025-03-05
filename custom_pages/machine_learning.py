import streamlit as st
import pandas as pd
from models.ml_models import run_classification

# --------------------------
# การเตรียมข้อมูล
# --------------------------

def Null_Avg(df):
    for column in df.columns:
        if df[column].isnull().any() and df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

# --------------------------
# Iuput ของ user
# --------------------------

def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        engine_size = st.number_input('Engine Size (Liters)', min_value=0.0, value=2.5)
        horsepower = st.number_input('Horsepower (HP)', min_value=0, value=150)
        wheelbase = st.number_input('Wheelbase (inches)', min_value=0, value=110)

    with col2:
        width = st.number_input('Width (inches)', min_value=0, value=70)
        length = st.number_input('Length (inches)', min_value=0, value=190)
        curb_weight = st.number_input('Curb Weight (lbs)', min_value=0, value=3000)

    with col3:
        fuel_capacity = st.number_input('Fuel Capacity (gallons)', min_value=0, value=16)
        fuel_efficiency = st.number_input('Fuel Efficiency (mpg)', min_value=0.0, value=25.0)
        power_perf_factor = st.number_input('Power Performance Factor', min_value=0.0, value=50.0)

    data = {
        'Engine_size': engine_size,
        'Horsepower': horsepower,
        'Wheelbase': wheelbase,
        'Width': width,
        'Length': length,
        'Curb_weight': curb_weight/1000,
        'Fuel_capacity': fuel_capacity,
        'Fuel_efficiency': fuel_efficiency,
        'Power_perf_factor': power_perf_factor
    }
    return pd.DataFrame(data, index=[0])

# --------------------------
# การทำนายข้อมูลของ ML
# --------------------------

def predict_vehicle_type(model):
    user_data = user_input_features()
    st.write("ข้อมูลของ User", user_data)
    user_data_scaled = st.session_state.scaler.transform(user_data)

    if model == 'KNN':
        pred = st.session_state.knn.predict(user_data_scaled)
    elif model == 'SVM':
        pred = st.session_state.svm.predict(user_data_scaled)
    elif model == 'Decision Tree':
        pred = st.session_state.dt.predict(user_data_scaled)
    elif model == 'Ensemble':
        pred = st.session_state.ensemble.predict(user_data_scaled)
    else:
        st.error("เลือกไม่ถูก!!")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("ของมูลของ User ที่ปรับแล้ว ", user_data_scaled)
    with col2:
        st.write("ค่าที่ได้จากโมเดล", pred)

    vehicle_type = st.session_state.label_encoder.inverse_transform(pred)[0]
    st.divider()
    st.header(f"โมเดลคาดเดาประเภทรถยนต์ ({model}) ➡ {vehicle_type}")

    model_metrics = st.session_state.metrics.get(model)
    st.write(f"**ความแม่นยำของโมเดล :** 🎯 {model_metrics['accuracy']*100} %")

# --------------------------
# หน้า UI Streamlit
# --------------------------

st.title("การจำแนกประเภทของรถยนต์ (Vehicle type)")
model = st.selectbox("เลือกโมเดล", ['KNN', 'SVM', 'Decision Tree', 'Ensemble'])

df = pd.read_csv("car_dataset_large.csv", encoding='ISO-8859-1')
df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors='coerce')
df = Null_Avg(df)

if 'knn' not in st.session_state:
    knn, svm, dt, ensemble, scaler, label_encoder,metrics = run_classification(df)
    st.session_state.knn = knn
    st.session_state.svm = svm
    st.session_state.dt = dt
    st.session_state.ensemble = ensemble
    st.session_state.scaler = scaler
    st.session_state.label_encoder = label_encoder
    st.session_state.metrics = metrics

predict_vehicle_type(model)