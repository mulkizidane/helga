import streamlit as st
import pandas as pd
import joblib

# Memuat model dan label encoders
model = joblib.load('random_forest_model.pkl')  # Ganti dengan path model Anda
label_encoders = joblib.load('label_encoders.pkl')  # Ganti dengan path label encoders Anda

# Judul aplikasi
st.title("Prediksi Stroke")

# Input dari pengguna
st.header("Masukkan Data Anda")

# Input fitur
work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
smoking_status = st.selectbox("Status Merokok", ["never smoked", "smokes", "formerly smoked"])
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
ever_married = st.selectbox("Status Pernikahan", ["Yes", "No"])
bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=22.0)
residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Rata-rata Tingkat Glukosa", min_value=0.0, value=85.0)
heart_disease = st.selectbox("Riwayat Penyakit Jantung", ["Yes", "No"])
hypertension = st.selectbox("Tekanan Darah Tinggi", ["Yes", "No"])

# Mengonversi input ke format yang sesuai
input_data = pd.DataFrame({
    'work_type': [work_type],
    'gender': [gender],
    'smoking_status': [smoking_status],
    'age': [age],
    'ever_married': [ever_married],
    'bmi': [bmi],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'heart_disease': [1 if heart_disease == "Yes" else 0],
    'hypertension': [1 if hypertension == "Yes" else 0]
})

# Mengencode fitur kategorikal menggunakan label encoders
for column in ['work_type', 'gender', 'smoking_status', 'ever_married', 'Residence_type']:
    input_data[column] = label_encoders[column].transform(input_data[column])

# Menampilkan data input
st.write("Data yang dimasukkan:")
st.write(input_data)

# Prediksi
if st.button("Prediksi Stroke"):
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    # Menampilkan hasil prediksi dan probabilitas
    if prediction[0] == 1:
        st.success("Prediksi: Risiko Stroke Tinggi")
    else:
        st.success("Prediksi: Risiko Stroke Rendah")
    
    st.write(f"Probabilitas Risiko Stroke Rendah (Class 0): {probabilities[0][0]:.2f}")
    st.write(f"Probabilitas Risiko Stroke Tinggi (Class 1): {probabilities[0][1]:.2f}")
