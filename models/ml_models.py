from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import numpy as np
import streamlit as st

def run_classification(df):
    X = df[['Engine_size', 'Horsepower', 'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency', 'Power_perf_factor']]
    y = df['Vehicle_type'] 

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=7)
    svm = SVC(kernel='poly',degree=4,C=1, probability=True)  
    dt = DecisionTreeClassifier(max_depth=10,random_state=42)
    ensemble = VotingClassifier(estimators=[('knn', knn), ('svm', svm), ('dt', dt)], voting='soft')

    knn.fit(X_train_scaled, y_train)
    svm.fit(X_train_scaled, y_train)
    dt.fit(X_train_scaled, y_train)
    ensemble.fit(X_train_scaled, y_train)

    st.session_state.knn = knn
    st.session_state.svm = svm
    st.session_state.dt = dt
    st.session_state.ensemble = ensemble
    st.session_state.scaler = scaler
    st.session_state.label_encoder = label_encoder

    knn_pred = knn.predict(X_test_scaled)
    svm_pred = svm.predict(X_test_scaled)
    dt_pred = dt.predict(X_test_scaled)
    ensemble_pred = ensemble.predict(X_test_scaled)

    knn_proba = knn.predict_proba(X_test_scaled)
    svm_proba = svm.predict_proba(X_test_scaled)
    dt_proba = dt.predict_proba(X_test_scaled)
    ensemble_proba = ensemble.predict_proba(X_test_scaled)

    #คำนวน accuracy ของแต่ละโมเดล
    knn_accuracy = accuracy_score(y_test, knn_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    #คำนวน precision recall and F1-score MAE MSE RMSE
    knn_precision = precision_score(y_test, knn_pred, average='weighted')
    svm_precision = precision_score(y_test, svm_pred, average='weighted')
    dt_precision = precision_score(y_test, dt_pred, average='weighted')
    ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted')

    knn_recall = recall_score(y_test, knn_pred, average='weighted')
    svm_recall = recall_score(y_test, svm_pred, average='weighted')
    dt_recall = recall_score(y_test, dt_pred, average='weighted')
    ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')

    knn_f1 = f1_score(y_test, knn_pred, average='weighted')
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')
    dt_f1 = f1_score(y_test, dt_pred, average='weighted')
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

    knn_mae = mean_absolute_error(y_test, knn_proba.argmax(axis=1))
    svm_mae = mean_absolute_error(y_test, svm_proba.argmax(axis=1))
    dt_mae = mean_absolute_error(y_test, dt_proba.argmax(axis=1))
    ensemble_mae = mean_absolute_error(y_test, ensemble_proba.argmax(axis=1))

    knn_mse = mean_squared_error(y_test, knn_proba.argmax(axis=1))
    svm_mse = mean_squared_error(y_test, svm_proba.argmax(axis=1))
    dt_mse = mean_squared_error(y_test, dt_proba.argmax(axis=1))
    ensemble_mse = mean_squared_error(y_test, ensemble_proba.argmax(axis=1))

    knn_rmse = np.sqrt(knn_mse)
    svm_rmse = np.sqrt(svm_mse)
    dt_rmse = np.sqrt(dt_mse)
    ensemble_rmse = np.sqrt(ensemble_mse)

    # สรุปข้อมูลประสิทธิภาพ
    metrics = {
        'KNN': {
            'accuracy': knn_accuracy,
            'precision': knn_precision,
            'recall': knn_recall,
            'f1': knn_f1,
            'mae': knn_mae,
            'mse': knn_mse,
            'rmse': knn_rmse,
        },
        'SVM': {
            'accuracy': svm_accuracy,
            'precision': svm_precision,
            'recall': svm_recall,
            'f1': svm_f1,
            'mae': svm_mae,
            'mse': svm_mse,
            'rmse': svm_rmse,
        },
        'Decision Tree': {
            'accuracy': dt_accuracy,
            'precision': dt_precision,
            'recall': dt_recall,
            'f1': dt_f1,
            'mae': dt_mae,
            'mse': dt_mse,
            'rmse': dt_rmse,
        },
        'Ensemble': {
            'accuracy': ensemble_accuracy,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'f1': ensemble_f1,
            'mae': ensemble_mae,
            'mse': ensemble_mse,
            'rmse': ensemble_rmse,
        }
    }
    return knn, svm, dt, ensemble, scaler, label_encoder, metrics