import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load model yang telah dilatih
model = joblib.load('random_forest_model.pkl')

# Load label encoders
label_encoders = joblib.load('label_encoders.pkl')

# Load scaler
scaler = joblib.load('scaler.pkl')

# Fungsi untuk melakukan prediksi
def predict(avg_glucose_level, Residence_type, ever_married, bmi, age, gender, smoking_status, work_type):
    # Mengubah input menjadi DataFrame
    input_data = pd.DataFrame({
        'avg_glucose_level': [avg_glucose_level],
        'Residence_type': [Residence_type],
        'ever_married': [ever_married],
        'bmi': [bmi],
        'age': [age],
        'gender': [gender],
        'smoking_status': [smoking_status],
        'work_type': [work_type]
    })
    
    # Encode fitur kategorikal
    for column in input_data.select_dtypes(include=['object']).columns:
        input_data[column] = label_encoders[column].transform(input_data[column])
    
    # Normalisasi fitur
    input_data_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi
    prediction = model.predict(input_data_scaled)
    
    return prediction[0]

# Judul aplikasi
st.title('Prediksi Risiko Stroke dengan Model Random Forest')

# Input dari pengguna
avg_glucose_level = st.number_input('Tingkat Glukosa Rata-rata', min_value=0.0, max_value=300.0, value=100.0)
Residence_type = st.selectbox('Tipe Tempat Tinggal', ['Urban', 'Rural'])
ever_married = st.selectbox('Pernah Menikah', ['Yes', 'No'])
bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=50.0, value=22.0)
age = st.number_input('Usia', min_value=0, max_value=100, value=30)
gender = st.selectbox('Jenis Kelamin', ['Male', 'Female', 'Other'])
smoking_status = st.selectbox('Status Merokok', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
work_type = st.selectbox('Jenis Pekerjaan', ['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'])

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
    prediction = predict(avg_glucose_level, Residence_type, ever_married, bmi, age, gender, smoking_status, work_type)
    if prediction == 0:
        st.success('Prediksi: Tidak Berisiko Stroke')
    else:
        st.error('Prediksi: Berisiko Stroke')
