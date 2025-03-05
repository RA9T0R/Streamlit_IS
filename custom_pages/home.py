import streamlit as st
import pandas as pd
import tensorflow as tf

st.title("ยินดีตอนรับเข้าสู่ Website ML & Neural Netwoorks🤖")
st.write(
    "ในเว็ปนี้คุณสามารถเข้ามาลองใช้งาน Machine Learing ได้สำหรับการคำนายประเภทของรถยนต์ อีกทั้งยังสามารถลองเล่น Neural Netwoorks ที่จะช่วยจำแนกรูปภาพของ Tensorflow ด้วย ใช้แถบด้านข้างเพื่อนำทางไปยังส่วนต่างๆ 🚀"
)
st.markdown("---")

col1, col2 = st.columns(2)
CarSales = pd.read_csv("Car_sales.csv")

@st.cache_data
def load_datasets():
    mnist = tf.keras.datasets.mnist.load_data()
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
    cifar10 = tf.keras.datasets.cifar10.load_data()
    return mnist, fashion_mnist, cifar10

(mnist_train, mnist_test), (fashion_train, fashion_test), (cifar_train, cifar_test) = load_datasets()

with col1:
    st.header("Dataset 1 : Car Detail")
    st.write("ข้อมูลเบื้องต้นเกี่ยวกับรถยนต์")
    st.dataframe(CarSales)
with col2:
    st.header("Dataset 2 : MNIST , Fashion MNIST , CIFAR-10")
    st.write("ข้อมูลเบื้องต้นของ Tensorflow")

    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.image(mnist_train[0][i], width=100)
            st.image(fashion_train[0][i], width=100)
            st.image(cifar_train[0][i], width=100)


st.markdown("---")
st.header("โมเดลที่ใช้งาน Project นี้")
col1, col2 = st.columns(2)
with col1:
    st.subheader("KNN : K-Nearest Neighbor")
    st.image("custom_pages/KNN.png", caption="[รูปภาพจาก](https://medium.com/@sachinsoni600517/k-nearest-neighbours-introduction-to-machine-learning-algorithms-9dbc9d9fb3b2)",width=700)
    st.subheader("D3 : Decision Tree")
    st.image("custom_pages/D3.png", caption="[รูปภาพจาก](https://k21academy.com/datascience-blog/decision-tree-algorithm/)",width=700)
with col2:
    st.subheader("SVM : Support Vector Machine")
    st.image("custom_pages/SVM.png", caption="[รูปภาพจาก](https://vitalflux.com/classification-model-svm-classifier-python-example/)",width=700)
    st.subheader("Ensemble")
    st.image("custom_pages/ensemble.png", caption="[รูปภาพจาก](https://intuitivetutorial.com/2023/05/12/ensemble-models-in-machine-learning/)",width=700)
    
    st.subheader("Neural Networks (CNN)")
    st.image("custom_pages/neuralnetworks.png", caption="[รูปภาพจาก](https://en.m.wikipedia.org/wiki/File:Typical_cnn.png)",width=700)


st.write("Created by Phongphat Bangkha - [GITHUB](https://github.com/RA9T0R)")

