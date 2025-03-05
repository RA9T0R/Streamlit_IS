import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("รายละเอียดของ Model การจำแนกประเภทของรถยนต์ (Vehicle type)")

# --------------------------
# ส่วนของแสดงข้อมูลเบื้องต้น
# --------------------------
dfStart = pd.read_csv("Car_sales.csv")
df = pd.read_csv("car_dataset_large.csv")

st.header("ข้อมูลเริ่มต้น")
st.dataframe(dfStart)
# --------------------------
# ส่วนของข้อมูลเพิ่มเติมเกี่ยวกับข้อมูล
# --------------------------
with st.expander("ข้อมูลเพิ่มเติมเกี่ยวกับข้อมูล"):
    st.subheader("1. แหล่งที่มาของข้อมูล")
    st.write("""
    ข้อมูลนี้มาจาก **Kaggle** ซึ่งสามารถเข้าถึงได้ที่ [ลิ้งค์นี้](https://www.kaggle.com/datasets/gagandeep16/car-sales)
    ข้อมูลนี้เป็นข้อมูลจากเว็บไซต์ Kaggle ซึ่งเกี่ยวข้องกับข้อมูลของรถยนต์
    โดยที่เรานั้นต้องการข้อมูลที่ไม่สมบูรณ์ จึงได้ใช้เทคโนโลยี AI เช่น ChatGPT ในการแปลงข้อมูลส่วนใหญ่ให้เหมาะสมกับการนำมาใช้
    ให้มีความไม่สมบูรณ์ เช่น การทำให้ข้อมูลบางตัวเป็นค่าว่าง และ ลบ columns ออกไปเลยเป็นต้น
    """)

    st.subheader("2. Columns ของข้อมูล")
    st.write("""
    ข้อมูลนี้ประกอบด้วย Columns ดังต่อไปนี้:
    """)

    columns = [
        ("Manufacturer", "ผู้ผลิตของรถยนต์ เช่น Toyota, BMW, Ford, เป็นต้น"),
        ("Model", "รุ่นของรถยนต์ที่ผลิตออกมา เช่น Camry, X5, Mustang, เป็นต้น"),
        ("Sales_in_thousands", "ยอดขายของรถยนต์ในหน่วยพัน (k)"),
        ("__year_resale_value", "มูลค่าการขายต่อในปีนั้นๆ"),
        ("Vehical_type", "ประเภทของรถยนต์ เช่น Sedan, SUV, Truck"),
        ("Price_in_thousands", "ราคาของรถยนต์ในหน่วยพัน (k)"),
        ("Engine_size", "ขนาดของเครื่องยนต์ (โดยวัดเป็นลิตร เช่น 2.5L, 3.0L)"),
        ("Horsepower", "กำลังเครื่องยนต์ (แรงม้าที่เครื่องยนต์สามารถผลิตได้)"),
        ("Wheelbase", "ระยะฐานล้อ (ระยะห่างระหว่างล้อหน้ากับล้อหลัง)"),
        ("Width", "ความกว้างของรถยนต์"),
        ("Length", "ความยาวของรถยนต์"),
        ("Curb_weight", "น้ำหนักของรถยนต์เมื่อไม่ได้บรรทุกสิ่งของหรือผู้โดยสาร"),
        ("Fuel_capacity", "ความจุของถังน้ำมัน (จำนวนลิตรที่สามารถบรรจุได้)"),
        ("Fuel_efficiency", "อัตราการใช้น้ำมัน (ประสิทธิภาพในการใช้น้ำมัน เช่น กิโลเมตรต่อลิตร)"),
        ("Latest_Launch", "วันที่ล่าสุดที่รถยนต์รุ่นนี้ถูกเปิดตัวในตลาด"),
        ("Power_perf_factor", "ปัจจัยที่คำนวณจากสมรรถนะของรถยนต์ เช่น ความเร็ว การเร่ง ความแรง")
    ]

    for column, description in columns:
        st.markdown(f" {column} : {description}")

    st.subheader("3. การแก้ไขข้อมูล")
    st.write("""
    จากที่เห็นใน columns ของ Vehicle_type นั้นจะมีข้อมูลแค่ Passenger กับ Car เท่านั้น และด้วยที่ผมมีความตั่งใจที่จะทำ Machine Learning แบบ Classification ผมจึงไปใช้งาน AI อย่าง ChatGPT ในการแก้ไขและดัดแปลงข้อมูลให้เหมาะสมกับการนำมาใช้งานแทน เพราะถ้าหากใช้ข้อมูลดิบจาก dataset เดิมจะทำให้ไม่เห็นความสามารถของ model ได้จริงและเพิ่มข้อมูลเพิ่มเติมเช่น การลบ columns ที่ไม่จำเป็นออกไปโดยจะลบ Manufacturer,Model,Sales_in_thousands,__year_resale_value,Latest_Launch แต่ยังคงมีความไม่สมบูรณ์ของข้อมูลอยู่
    """) 
# --------------------------
# ส่วนของการอธิบายการเตรียมข้อมูล
# --------------------------
st.header("Exploratory Data Analysis (EDA) : การเตรียมข้อมูล")
st.subheader("1.จัดการกับค่าว่างใน Dataset 🧽")
null_check = df.isnull().any()

df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors='coerce')
def Null_Avg(df):
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype in ['float64', 'int64']:
                df[column].fillna(df[column].mean(), inplace=True)
    return df

col1, col2 = st.columns([1, 6])
with col1 :
    st.write("เช็คว่ามี Columns ว่างไหม", null_check)
with col2 : 
    df_filled = Null_Avg(df)
    st.write("ข้อมูลหลังจากการแทนค่าข้อมูลที่ว่าง (null) โดยค่าเฉลี่ย:")
    st.dataframe(df_filled)

st.write("โดยใช้คำสั่ง Null_Avg ")
code = '''def Null_Avg(df):
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype in ['float64', 'int64']:
                df[column].fillna(df[column].mean(), inplace=True)
    return df'''
st.code(code, language="python")

st.subheader('2. Data Encoding')
st.caption("การทำ One-hot Encoding เป็นเทคนิคของ data encoding เพื่อแยกข้อมูลที่มี 2 ประเภทออกมาเป็น 2 ค่าโดยแต่ละค่าจะมีค่าไบนารี โดยได้มีการใช้ LabelEncoder เพื่อที่จะแปลงค่า")
code_normalization = '''
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)'''
st.code(code_normalization, language="python")

st.subheader('3. แยก Features (X) และ Target (y)')
st.caption("การแยกข้อมูลเป็น Features และ Target นั้นจะแบ่งสิ่งที่โมเดลจะเรียนรู้และหาคำตอบ ")
st.caption("X คือ Features ที่ใช้ในการทำนาย โดยจะมีค่าดังนี้ Price_in_thousands, Engine_size, Horsepower, Wheelbase, Width,Length, Curb_weight, Fuel_capacity, Fuel_efficiency, Power_perf_factor")
st.caption("Y คือ Target ที่เราจะทำนาย โดยจะเป็นค่าของ Vehicle_type")
code_normalization = '''
    X = df[['Engine_size', 'Horsepower', 'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency', 'Power_perf_factor']]
    y = df['Vehicle_type'] '''
st.code(code_normalization, language="python")

st.subheader('4. แบ่งข้อมูล Train และ Test')
st.caption("ต่อมาเราจะทำการแบ่งข้อมูลออกมา 2 ส่วนคือ 1.Train เป็นส่วนที่จะใช้ในการให้โมเดลฝึกฝน 80% 2.Test เป็นส่วนที่จะใช้ในการทดสอบโมเดล 20% สุดท้ายrandom_state = 42 เพื่อให้ผลลัพธ์ออกมาเหมือนเดิมทุกครั้งที่รันโมเดล")
code_normalization = '''
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 42)'''
st.code(code_normalization, language="python")

st.subheader('5. Scale ข้อมูล')
st.caption("การปรับขนาดข้อมูลให้มีค่าอยู่ในช่วงที่กำหนด เช่น ตั้งแต่ 0 ถึง 1 โดยการใช้สูตร")
st.latex(r'''\text{X\_scaled} = \frac{X - \min(X)}{\max(X) - \min(X)}''')
code_normalization = '''
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)'''
st.code(code_normalization, language="python")
# --------------------------
# ส่วนแสดงข้อมูลในรูปแบบ Visualization
# --------------------------
st.subheader("👀Data Analytics and Visualization")
st.caption("ลองนำข้อมูลที่จะการแล้วมีโชว์ในรูปแบบต่างๆ")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("การกระจายขนาดของเครื่องยนต์")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Engine_size'], kde=True, bins=30)
    st.pyplot(plt)

with col2:
    st.markdown("การกระจายราคาตามประเภทของรถยนต์")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Vehicle_type', y='Price_in_thousands')
    st.pyplot(plt)

with col3:
    st.markdown("Width ↔ Length")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Width', y='Length', hue='Vehicle_type')
    st.pyplot(plt)

with col4:
    st.markdown("ราคาตามประเภทรถยนต์")
    df_sales_by_manufacturer = df.groupby('Vehicle_type')['Price_in_thousands'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_sales_by_manufacturer, x='Vehicle_type', y='Price_in_thousands')
    plt.xticks(rotation=90) 
    st.pyplot(plt)
# --------------------------
# ส่วนคำอธิบายความหมายโมเดล
# --------------------------
st.header("🤖Machine Lerning - Ensemble methods")
st.caption("ใน project นี้ของ machine lerning ได้ใช้แบบ Supervise learning Classification โดยมีจุดประสงค์ในการจำแนกประเภท คือการใช้ model ที่เรียนรู้จากข้อมูลที่เรากำหนด(labelled) และนำไปใช้เพื่อคาดการณ์ประเภทของจุดข้อมูลใหม่")

st.subheader("1️⃣KNN : K-Nearest Neighbor")
st.write("K-Nearest Neighbor นั้นเป็น non-parametric หรือ lazy learning โดยโมเดลนี้นั้นจะเก็บทั้งข้อมูลที่เรากำหนดไว้ (Train) เพื่อเป็นตัวอ้างอิง ในการคำนวนหา 'ระยะห่าง' ระหว่างข้อมูลนั้นกับข้อมูล Train โดยจะโดยอิงตามจำนวน Neighbor ที่ใกล้ที่สุด โดยที่การคำนวนระยะห่างนั้นก็จะใช้ Distance Calculation : Euclidean distance ในการคำนวน")
st.markdown("##### - จุดสำคัญของโมเดล KNN : การเลือก hyperparameters หรือ K หรือ จำนวน Neighbor ให้เหมาะสมกับข้อมูล")

st.subheader("2️⃣SVM : Support Vector Machine")
st.write("Support Vector Machine นั้นมีเพื่อค้นหา hyperplane ใน N มิติ โดยที่จะเป็นการวัดระยะจากจุดข้อมูลกับ hyperplane โดยจะมีค่า margin ในการแบ่ง และมันนั้นยังจะมี The Regularization parameter(C)คือการบอกว่าเราอยากหลีกเลี่ยงการจัดประเภทตัวอย่างผิดแค่ไหน โดยในโมเดลนี้นั้นจะมีทางเลือกในเรื่องของ Kernel อีกมันคือการทำให้เส้นหลักมีความยืดหยุ่นมาขึ้น ")
st.markdown("##### - จุดสำคัญของโมเดล SVM : การเลือก kernel,degree,C ให้เหมาะสมกับข้อมูล")

st.subheader("3️⃣D3 : Decision Tree")
st.write("Decision Tree นั้นเปรียบเสมือนกับต้นไม้กลับหัวโดยโมเดลนี้นั้นจะหาฟีเจอร์ที่ดีที่สุดในชุดข้อมูลโดยใช้ Information Gain และมีการแบ่งข้อมูลออกมาให้เป็น subsets นั้นจะเก็บค่าที่เป็นไปได้สำหรับคุณสมบัติที่ดีที่สุด จากนั้นมันก็จะลงลึกลงไปเรื่อยๆจนได้คำตอบ")
st.markdown("##### - จุดสำคัญของโมเดล D3 : การเลือก Depth ให้เหมาะสมกับข้อมูล")

st.subheader("4️⃣Ensemble")
st.write("Ensemble นั้นเกิดมาจาก Law of Large Numbers คือการที่ยิ่งใช้ข้อมูลมากขึ้นผลลัพธ์ที่ได้จะเช้าใกล้ความเป็นไปได้มาขึ้น เช่น การโยนเหรียญ โดยโมเดลนี้นั้นเปรียบเสมือนป่าที่ในป่าจะมีโมเดลอื่นๆอีก และเมื่อแต่ละโมเดลได้คำตอบของมันเอง Ensemble ก็จำทำการ Majority-Voting หรือก็คือการหาคำตอบที่โมเดลเลือกเยอะที่สุดนั้นเอง")
st.markdown("##### - จุดสำคัญของโมเดล Ensemble : การที่แต่ละโมเดลที่อยู่ในป่าดี")
# --------------------------
# ส่วนแสดงประสิทธิภาพของโมเดล
# --------------------------
st.header("🧮การวัดประสิทธิภาพของโมเดล (performance metrics)")
metrics = pd.DataFrame( {
    'Model': ["KNN", "SVM", "D3", "Ensemble"],
    'Accuracy': [0.755, 0.885, 0.995, 0.98],
    'Precision': [0.751094, 0.884952, 0.995138, 0.979985],
    'Recall': [0.755, 0.885, 0.995, 0.98],
    'F1-Score': [0.729762, 0.884099, 0.994988, 0.979856],
    'MAE': [0.425, 0.2, 0.005, 0.03],
    'MSE': [0.785, 0.37, 0.005, 0.05],
    'RMSE': [0.886002, 0.608276, 0.070711, 0.223607],
})
metrics.index = metrics.index + 1
st.write(metrics)
# --------------------------
# ส่วนแสดงสูตรการคำนวนของการวัดระสิทธิภาพโมเดล
# --------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy")
    st.write("ความแม่นยำวัดสัดส่วนของข้อมูลที่คาดการณ์ถูกต้องจากข้อมูลทั้งหมด")
    st.latex(r'''\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}''')
    st.divider()
    st.subheader("Recall (Sensitivity)")
    st.write("Recall จะวัดสัดส่วนของผลบวกที่เป็นจริงเทียบกับผลบวกที่แท้จริงทั้งหมด")
    st.latex(r'''\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}''')
    st.divider()
    st.subheader("Mean Absolute Error (MAE)")
    st.write("MAE วัดความแตกต่างสัมบูรณ์เฉลี่ยระหว่างค่าที่คาดการณ์กับค่าจริง")
    st.latex(r'''\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|''')
    st.divider()
    st.subheader("Root Mean Squared Error (RMSE)")
    st.write("RMSE คือรากที่สองของ MSE และให้การวัดค่าเฉลี่ยของขนาดข้อผิดพลาด")
    st.latex(r'''\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}''')
with col2:
    st.subheader("Precision")
    st.write("Precision วัดสัดส่วนของคำทำนายเชิงบวกที่แท้จริงจากคำทำนายเชิงบวกทั้งหมด")
    st.latex(r'''\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}''')
    st.divider()
    st.subheader("F1-Score")
    st.write("F1-Score คือค่าเฉลี่ยฮาร์มอนิกของ precision และ recall")
    st.latex(r'''\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}''')
    st.divider()
    st.subheader("Mean Squared Error (MSE)")
    st.write("MSE วัดความแตกต่างกำลังสองเฉลี่ยระหว่างค่าที่คาดการณ์และค่าที่แท้จริง")
    st.latex(r'''\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2''')
    st.divider()
    st.subheader("Confusion Matrix")
    st.write("Confusion Matrix คือ ตารางที่สรุปประสิทธิภาพของแบบจำลองการจำแนกประเภท")
    st.latex(r'''\text{Confusion Matrix} = \begin{bmatrix}
    \text{True Positives (TP)} & \text{False Positives (FP)} \\
    \text{False Negatives (FN)} & \text{True Negatives (TN)}
    \end{bmatrix}''')