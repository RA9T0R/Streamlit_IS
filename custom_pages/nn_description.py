import streamlit as st
import tensorflow as tf
import numpy as np

st.title("รายละเอียดของ Neural Networks การจำแนกผลจากข้อมูล TensorFlow")

# --------------------------
# ส่วนของแสดงข้อมูลเบื้องต้น
# --------------------------
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "mnist":
        return tf.keras.datasets.mnist.load_data()
    elif dataset_name == "fashion_mnist":
        return tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == "cifar10":
        return tf.keras.datasets.cifar10.load_data()

st.header("ข้อมูลเริ่มต้น")
dataset_name = st.selectbox(
    "เลือกเซ็ตข้อมูล", 
    ["mnist", "fashion_mnist", "cifar10"]
)

train_data, test_data = load_data(dataset_name)

col1, col2 = st.columns(2)
with col1 :
    index = st.number_input("ใส่ลำดับของรูปภาพที่ต้องการ (0-9999)", min_value=0, max_value=9999, value=0)
    image, label = train_data[0][index], train_data[1][index]
with col2 :
    st.image(image, caption=f"Selected Image (Label: {label})", width=250)
# --------------------------
# ส่วนของข้อมูลเพิ่มเติมเกี่ยวกับข้อมูล
# --------------------------
with st.expander("ข้อมูลเพิ่มเติมเกี่ยวกับชุดข้อมูล"):
    st.subheader("1.แหล่งที่มาของข้อมูล")
    st.write("""
    ข้อมูลชุดนี้มาจาก **TensorFlow Datasets** ซึ่งรวมถึงชุดข้อมูลที่ได้รับความนิยมในการฝึกอบรมโมเดลเรียนรู้ของเครื่อง เช่น **MNIST**, **Fashion MNIST**, และ **CIFAR-10** โดยชุดข้อมูลนี้ถูกนำมาใช้ในงานวิจัยหลายๆ งาน เพื่อทดสอบการทำงานของโมเดลต่างๆ โดยในโมเดลนี้จะเป็นการใช้งาน Model Architecture แบบเดียวเพื่อเป็นการแสดงให้เห็นว่าแต่ละข้อมูลนั้นจะมีความเหมาะสมกับ Model Architecture ที่ต่างกันไป [ลิ้งค์นี้](https://www.tensorflow.org/datasets/catalog/overview?hl=th)
    """)
    

    st.subheader("2.คำอธิบายของแต่ละชุดข้อมูล")
    if dataset_name == "mnist":
        st.write("""
        **MNIST** คือชุดข้อมูลของรูปภาพที่แสดงตัวเลขจาก 0 ถึง 9 จะเขียนด้วยมือ """)
    elif dataset_name == "fashion_mnist":
        st.write("""
        **Fashion MNIST** คือชุดข้อมูลของรูปภาพที่แสดงแฟชั่นต่างๆ เช่น เสื้อผ้า รองเท้า กระเป๋า """)
    elif dataset_name == "cifar10":
        st.write("""
        **CIFAR-10** คือชุดข้อมูลที่ประกอบด้วยรูปภาพจาก 10 คลาสต่างๆ ซึ่งรวมถึงประเภทของรูปภาพเช่น เครื่องบิน, รถยนต์, นก, สุนัข ฯลฯ""")
    

    st.subheader("3.โครงสร้างของข้อมูล")
    st.write("""
    ทุกชุดข้อมูลมีโครงสร้างที่คล้ายกัน ประกอบด้วย:
    - **Train Data**: รูปภาพและป้ายกำกับ (label) ของข้อมูลฝึกอบรม
    - **Test Data**: รูปภาพและป้ายกำกับของข้อมูลทดสอบ
    - **Image Shape**: รูปภาพในแต่ละชุดข้อมูลมีขนาดต่างกัน เช่น MNIST เป็น 28x28 พิกเซล, CIFAR-10 เป็น 32x32 พิกเซล
    """)


    st.subheader("4.การแสดงผลภาพ")
    st.write("ต่อไปนี้แสดงถึงภาพบางส่วนในชุดข้อมูลที่เลือก")
    columns = st.columns(10)
    for i in range(10):
        image, label = train_data[0][i], train_data[1][i]
        with columns[i]: 
            st.image(image, caption=f"Label: {label}", width=100)


    st.subheader("5.ข้อมูลเชิงสถิติของชุดข้อมูล")
    st.write("""ข้อมูลเชิงสถิติเบื้องต้นของชุดข้อมูลเหล่านี้จะช่วยให้เราเข้าใจลักษณะต่างๆ เช่น ขนาดภาพ การกระจายของป้ายกำกับ ฯลฯ""")

    st.write(f"Train Data Shape: {train_data[0].shape}")
    st.write(f"Test Data Shape: {test_data[0].shape}")

    col1, col2 = st.columns(2)
    with col1 :
        st.write("ข้อมูลสำหรับฝึก")
        label_count1 = np.bincount(train_data[1].flatten()) 
        st.bar_chart(label_count1)
    with col2 :
        st.write("ข้อมูลสำหรับทดสอบ")
        label_count2 = np.bincount(test_data[1].flatten()) 
        st.bar_chart(label_count2)
# --------------------------
# ส่วนของการอธิบายการเตรียมข้อมูล
# --------------------------
st.header("Exploratory Data Analysis (EDA) : การเตรียมข้อมูล")
st.subheader("1.นำเข้าข้อมูลจาก Tensorflow")
st.caption("เป็นการโหลดข้อมูล Dataset ตามที่ผู้ใช้ต้องการโดยจะมีการแบ่งข้อมูลออกตัวของ (train_images, train_labels), (test_images, test_labels) แล้วส่งข้อมูลออกไปใช้งานในส่วนอื่นๆ")
code_normalization = '''
def load_dataset(name):
    if name == "mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    elif name == "fashion_mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    elif name == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)'''
st.code(code_normalization, language="python")

st.subheader("2.Normalize ข้อมูลของ Tensorflow")
st.caption("เตรียมข้อมูลภาพและ label สำหรับการฝึกโดยแปลงให้เป็น tf.data.Datasetทำให้เป็นมาตรฐานและแบ่งกลุ่มเพื่อการประมวลผลที่มีประสิทธิภาพระหว่างการฝึกอบรม")
col1, col2 = st.columns(2)
with col1 :
    code_normalization = '''
    def normalize_img(image, label):
        image = tf.cast(image, tf.float32) 
        image = image / 255.0 
        return image, label'''
    st.code(code_normalization, language="python")
with col2 :
    code_normalization = '''
    def preprocess_data(data):
        images, labels = data
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))  
        dataset = dataset.map(normalize_img)  
        dataset = dataset.batch(32)  
        return dataset'''
    st.code(code_normalization, language="python")
# --------------------------
# ส่วนของการอธิบายการออกแบบโมเดล
# --------------------------
st.header("📝Design & Compile Model")
st.subheader("1.ออกแบบโมเดลและกำหนด Heyper parameter")
st.caption("การสร้างโมเดล Architecture ของโมเดลนั้นมีความสำคัญมากสำหรับความสามารถของโมเดล โดยโมเดลนี้เป็น CNN ที่ประกอบด้วยชั้น Conv2D และ MaxPooling2D สำหรับดึงคุณลักษณะจากภาพ, ชั้น GlobalAveragePooling2D เพื่อลดขนาดข้อมูล, ชั้น Dense ขนาด 64 ใช้ ReLU, และชั้น Dense ขนาด 10 ใช้ Softmax สำหรับจำแนกประเภท 10 คลาส")
code_normalization = '''
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model'''
st.code(code_normalization, language="python")

st.subheader("2.ขั้นตอนการ Train")
st.caption("โดยเป็นการนำโมเดลที่ออกแบบไว้มา train หรือการใช้คำสั่ง fit โดยจะมีจุดสำคัญอยู่คือค่าของ epochs โดยในที่นี้จะใช้ค่าจาก user")
code_normalization = '''
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)'''
st.code(code_normalization, language="python")
# --------------------------
# ส่วนคำอธิบายความหมายโมเดล
# --------------------------
st.header("🤖Neural Networks")
st.subheader("CNN : Convolutional Neural Network")
st.write("Convolutional Neural Network นั้นเป็น Deep-Leaning model ที่ใช้กันแพร่หลายในการจำแนกรูปภาพ โดยจะมีส่วนประกอบอยู่ 4 Layers 1.Convolution layer - เป็นการใช้ Filters เพื่อแยก Feature ของรูปภาพ 2.Pooling layer - ลดขนาดของ feature 3.Flatten layer - แปลงข้อมูลจาก 2D เป็น 1D 4.Fully Connected layer (ANN) - ใช้ข้อมูล 1D มาเรียนรูปในการจำแนกรูปภาพ")
st.markdown("##### - จุดสำคัญของโมเดล CNN : การเลือกจำนวน epochs หรือ รอบการ train และ Model Architecture ให้เหมาะสมกับข้อมูล")