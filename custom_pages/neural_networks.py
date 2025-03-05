import streamlit as st
import os
import numpy as np
import tensorflow as tf
from models.nn_models import load_dataset, train_and_predict, load_model

st.title("การทำนายผลจากข้อมูลพื้นฐานใน TensorFlow")
# --------------------------
# Iuput ของ user
# --------------------------
col1, col2 = st.columns(2)
with col1 :
    dataset_name = st.selectbox("เลือกเซ็ตข้อมูล", ["mnist", "fashion_mnist", "cifar10"])
    if dataset_name == "mnist":
        st.write("**MNIST** คือกลุ่มข้อมูลของรูปภาพการเขียนตัวเลข (0-9) ด้วยมือ")
    elif dataset_name == "fashion_mnist":
        st.write("**Fashion MNIST** คือกลุ่มข้อมูลของรูปภาพของแฟชั่นต่างๆ เช่น เสื้อ,กางเกง เป็นต้น")
    elif dataset_name == "  ":
        st.write("**CIFAR-10** คือกลุ่มข้อมูลของรูปภาพของประเภทของจำนวน 10 ประเภทของรูปสี")
with col2 : 
    epochs = st.number_input("ใส่จำนวน epochs สำหรับการ training", min_value=1, max_value=100, value=1)
# --------------------------
# โหลดข้อมูลตามที่ user ต้องการ
# --------------------------
st.divider()
@st.cache_data
def load_data(dataset_name):
    train_data, test_data = load_dataset(dataset_name)
    return train_data, test_data
train_data, test_data = load_data(dataset_name)
# --------------------------
# การแสดงผลลัพธ์ของโมเดล
# --------------------------
col1, col2 = st.columns(2)
with col1 :
    index = st.number_input("ใส่ลำดับของรูปภาพที่ต้องการ (0-9999)", min_value=0, max_value=9999, value=130)
    image, label = train_data[0][index], train_data[1][index]
    st.image(image, caption=f"Selected Image (Label: {label})", width=250)
with col2 : 
    model_path = f"model_{dataset_name}.keras"
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.header(f"โมเดลคาดเดาข้อมูล")
        st.write("โมเดลนี้ได้มาจากการ Train ครั้งก่อน")
        
        image_input = np.expand_dims(train_data[0][index], axis=0)
        prediction = model.predict(image_input)
        predicted_label = np.argmax(prediction, axis=1) 
        st.subheader(f"ข้อมูลจริง : {label}")
        st.subheader(f"ข้อมูลที่ได้จากโมเดล : {predicted_label[0]}")

        loss, accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
        st.write(f"**ความแม่นยำของโมเดล :** 🎯 {accuracy * 100:.2f}%")
    else:
        model = None
        st.subheader("Model not found. It will be trained.")
# --------------------------
# ส่วนของการ Train โมเดลใหม่หา user ต้องการ
# --------------------------
if st.button("Train โมเดลใหม่") or model is None:
    with st.spinner("🔄 โมเดลกำลัง Train..."):
        accuracy, loss, predicted_df, model = train_and_predict(dataset_name, train_data, test_data, epochs=epochs, save_model=True)
        
        st.success("✅ Train โมเดลสำเร็จ!")
        st.subheader(f"ความเม้นยำของโมเดลกับข้อมูล Train : 🎯 {accuracy * 100:.2f}%")
        st.subheader("-> Predicted Results <-")
        st.dataframe(predicted_df)
    model.save(model_path)  
    st.write("โมเดลถูกบันทึกแล้ว")
    model = load_model(model_path)
    
    loss, accuracy = model.evaluate(test_data[0], test_data[1], verbose=0)
    st.subheader(f"ความเม้นยำของโมเดลกับข้อมูล Test : 🎯 {accuracy * 100:.2f}%")